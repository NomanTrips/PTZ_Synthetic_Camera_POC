"""Episode writer utilities for logging PTZ capture sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Tuple

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_VIDEO_KEY = "observation.image"


def _metadata_template(*, fps: float, chunk_size: int) -> Dict[str, object]:
    """Return the default metadata structure expected by LeRobot."""

    return {
        "codebase_version": "v2.0",
        "robot_type": "unknown",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": int(chunk_size),
        "fps": float(fps),
        "splits": {"train": "0:1000000"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.image": {
                "dtype": "video",
                "shape": [256, 256, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [3],
                "names": {"motors": ["pan", "tilt", "zoom"]},
            },
            "action": {
                "dtype": "float32",
                "shape": [3],
                "names": {"motors": ["dpan", "dtilt", "dzoom"]},
            },
            "episode_index": {"dtype": "int64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "next.reward": {"dtype": "float32", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "next.success": {"dtype": "bool", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }


@dataclass
class EpisodeWriter:
    """Collect frame data for a single episode and write it to disk."""

    episode_index: int
    data_path: Path
    video_path: Path
    fps: float
    chunk_index: int

    _video_writer: imageio.core.format.Format = field(init=False, repr=False)
    _rows: Dict[str, List[object]] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)

        self._rows = {
            "episode_index": [],
            "frame_index": [],
            "timestamp": [],
            "observation.state": [],
            "action": [],
            "task_index": [],
            "next.reward": [],
            "next.done": [],
            "next.success": [],
            "index": [],
        }

        # Configure FFmpeg writer via imageio. Enforce yuv420p for compatibility.
        self._video_writer = imageio.get_writer(
            str(self.video_path),
            fps=float(self.fps),
            codec="libx264",
            format="FFMPEG",
            output_params=["-pix_fmt", "yuv420p"],
        )

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------
    def append(
        self,
        frame: np.ndarray,
        *,
        frame_index: int,
        timestamp: float,
        state: Tuple[float, float, float],
        action: Tuple[float, float, float],
    ) -> None:
        """Append a single observation/action pair to the episode."""

        if self._closed:
            raise RuntimeError("Cannot append to a closed EpisodeWriter")

        frame_u8 = np.asarray(frame)
        if frame_u8.dtype != np.uint8:
            frame_u8 = np.clip(frame_u8, 0, 255).astype(np.uint8)
        if frame_u8.ndim != 3 or frame_u8.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")

        self._video_writer.append_data(frame_u8)

        self._rows["episode_index"].append(int(self.episode_index))
        self._rows["frame_index"].append(int(frame_index))
        self._rows["timestamp"].append(float(timestamp))
        self._rows["observation.state"].append(tuple(float(x) for x in state))
        self._rows["action"].append(tuple(float(x) for x in action))
        self._rows["task_index"].append(0)
        self._rows["next.reward"].append(0.0)
        self._rows["next.done"].append(False)
        self._rows["next.success"].append(False)
        self._rows["index"].append(int(frame_index))

    # ------------------------------------------------------------------
    # Finalisation helpers
    # ------------------------------------------------------------------
    def close(self) -> int:
        """Flush buffered data to disk and return the number of frames."""

        if self._closed:
            return len(self._rows["frame_index"])

        self._video_writer.close()
        self._closed = True

        frame_count = len(self._rows["frame_index"])
        if frame_count == 0:
            # Remove empty video containers if nothing was logged.
            if self.video_path.exists():
                self.video_path.unlink()
            return 0

        table = self._build_table()
        pq.write_table(table, self.data_path)
        return frame_count

    def discard(self) -> None:
        """Abort the episode and clean up any partially written artefacts."""

        if not self._closed:
            self._video_writer.close()
            self._closed = True
        if self.video_path.exists():
            self.video_path.unlink()
        if self.data_path.exists():
            self.data_path.unlink()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_table(self) -> pa.Table:
        """Construct a :class:`pyarrow.Table` from the buffered data."""

        frame_count = len(self._rows["frame_index"])
        if frame_count == 0:
            raise RuntimeError("Cannot build table without any frames")

        states = np.asarray(self._rows["observation.state"], dtype=np.float32)
        actions = np.asarray(self._rows["action"], dtype=np.float32)

        arrays = {
            "episode_index": pa.array(self._rows["episode_index"], type=pa.int64()),
            "frame_index": pa.array(self._rows["frame_index"], type=pa.int64()),
            "timestamp": pa.array(self._rows["timestamp"], type=pa.float32()),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(states.reshape(-1), type=pa.float32()),
                3,
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(actions.reshape(-1), type=pa.float32()),
                3,
            ),
            "task_index": pa.array(self._rows["task_index"], type=pa.int64()),
            "next.reward": pa.array(self._rows["next.reward"], type=pa.float32()),
            "next.done": pa.array(self._rows["next.done"], type=pa.bool_()),
            "next.success": pa.array(self._rows["next.success"], type=pa.bool_()),
            "index": pa.array(self._rows["index"], type=pa.int64()),
        }

        return pa.table(arrays)


class DatasetManager:
    """Handle dataset directory layout and metadata bookkeeping."""

    def __init__(
        self,
        root: Path,
        *,
        fps: float,
        chunk_size: int = 1000,
        video_key: str = DEFAULT_VIDEO_KEY,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_dir = self.root / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.meta_dir / "info.json"

        self.fps = float(fps)
        self.chunk_size = int(chunk_size)
        self.video_key = video_key

        self._meta_base = _metadata_template(fps=self.fps, chunk_size=self.chunk_size)
        self._write_meta(self._meta_base)

        self.total_frames = 0
        self.total_episodes = 0
        self.used_chunks: set[int] = set()

    # ------------------------------------------------------------------
    # Episode management helpers
    # ------------------------------------------------------------------
    def create_episode_writer(self, episode_index: int) -> EpisodeWriter:
        """Initialise an :class:`EpisodeWriter` for ``episode_index``."""

        chunk_index = episode_index // self.chunk_size

        data_dir = self.root / "data" / f"chunk-{chunk_index:03d}"
        video_dir = self.root / "videos" / f"chunk-{chunk_index:03d}" / self.video_key

        writer = EpisodeWriter(
            episode_index=episode_index,
            data_path=data_dir / f"episode_{episode_index:06d}.parquet",
            video_path=video_dir / f"episode_{episode_index:06d}.mp4",
            fps=self.fps,
            chunk_index=chunk_index,
        )
        return writer

    def complete_episode(self, writer: EpisodeWriter, frames_logged: int) -> None:
        """Update dataset statistics after a successful episode."""

        if frames_logged <= 0:
            return
        self.used_chunks.add(writer.chunk_index)
        self.total_frames += int(frames_logged)
        self.total_episodes += 1

    def finalize(self) -> None:
        """Persist metadata with up-to-date statistics."""

        meta = dict(self._meta_base)
        meta["total_frames"] = int(self.total_frames)
        meta["total_episodes"] = int(self.total_episodes)
        meta["total_videos"] = int(self.total_episodes)
        meta["total_chunks"] = max(len(self.used_chunks), 1)

        self._write_meta(meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _write_meta(self, payload: Dict[str, object]) -> None:
        """Write ``payload`` to ``meta/info.json`` with indentation."""

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


__all__ = ["EpisodeWriter", "DatasetManager", "DEFAULT_VIDEO_KEY"]

