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


def _metadata_template(*, fps: int, chunk_size: int, video_key: str) -> Dict[str, object]:
    """Return the default metadata structure expected by LeRobot v3.0."""

    non_video_feature_fps = int(fps)

    return {
        "codebase_version": "v3.0",
        "robot_type": "unknown",
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 1,
        "data_files_size_in_mb": 0,
        "video_files_size_in_mb": 0,
        "chunks_size": int(chunk_size),
        "fps": int(fps),
        "splits": {"train": "0:1000000"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": f"videos/{video_key}/chunk-{{chunk_index:03d}}/file-{{file_index:03d}}.mp4",
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
                "fps": non_video_feature_fps,
            },
            "action": {
                "dtype": "float32",
                "shape": [3],
                "names": {"motors": ["dpan", "dtilt", "dzoom"]},
                "fps": non_video_feature_fps,
            },
            "episode_index": {"dtype": "int64", "shape": [1], "fps": non_video_feature_fps},
            "frame_index": {"dtype": "int64", "shape": [1], "fps": non_video_feature_fps},
            "timestamp": {"dtype": "float32", "shape": [1], "fps": non_video_feature_fps},
            "next.reward": {"dtype": "float32", "shape": [1], "fps": non_video_feature_fps},
            "next.done": {"dtype": "bool", "shape": [1], "fps": non_video_feature_fps},
            "next.success": {"dtype": "bool", "shape": [1], "fps": non_video_feature_fps},
            "index": {"dtype": "int64", "shape": [1], "fps": non_video_feature_fps},
            "task_index": {"dtype": "int64", "shape": [1], "fps": non_video_feature_fps},
        },
    }


@dataclass
class EpisodeWriter:
    """Collect frame data for a single episode and register it with the dataset."""

    episode_index: int
    dataset: "DatasetManager"
    fps: float

    data_path: Path
    video_path: Path
    chunk_index: int

    _closed: bool = field(default=False, init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)
    _states: List[Tuple[float, float, float]] = field(default_factory=list, init=False, repr=False)
    _actions: List[Tuple[float, float, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataset_from_index = self.dataset.total_frames

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

        self.dataset._append_frame(
            frame_u8,
            episode_index=self.episode_index,
            frame_index=frame_index,
            timestamp=timestamp,
            state=state,
            action=action,
        )

        self._frame_count += 1
        self._states.append(tuple(float(x) for x in state))
        self._actions.append(tuple(float(x) for x in action))

    # ------------------------------------------------------------------
    # Finalisation helpers
    # ------------------------------------------------------------------
    def close(self) -> int:
        """Flush buffered data to disk and return the number of frames."""

        if self._closed:
            return self._frame_count

        self._closed = True

        if self._frame_count == 0:
            return 0

        self.dataset._register_episode(
            episode_index=self.episode_index,
            frame_count=self._frame_count,
            dataset_from_index=self._dataset_from_index,
            states=self._states,
            actions=self._actions,
        )
        return self._frame_count

    def discard(self) -> None:
        """Abort the episode and clean up any partially written artefacts."""

        if self._closed:
            return
        self._closed = True
        if self._frame_count:
            raise RuntimeError(
                "Discarding non-empty episodes is not supported with the shared v3 writer"
            )
        self._frame_count = 0


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

        self.data_path = self.root / "data" / "chunk-000" / "file-000.parquet"
        self.video_path = (
            self.root
            / "videos"
            / self.video_key
            / "chunk-000"
            / "file-000.mp4"
        )

        self.tasks_path = self.meta_dir / "tasks.parquet"
        self.episodes_path = (
            self.meta_dir / "episodes" / "chunk-000" / "episodes_000.parquet"
        )
        self.stats_path = self.meta_dir / "stats.json"

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.tasks_path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes_path.parent.mkdir(parents=True, exist_ok=True)

        self._meta_base = _metadata_template(
            fps=int(self.fps),
            chunk_size=self.chunk_size,
            video_key=self.video_key,
        )
        self._write_meta(self._meta_base)

        self.total_frames = 0
        self.total_episodes = 0

        self._data_rows: Dict[str, List[object]] = {
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

        self._video_writer: imageio.core.format.Format | None = None
        self._episode_records: List[Dict[str, object]] = []

        self._image_min: np.ndarray | None = None
        self._image_max: np.ndarray | None = None
        self._image_sum = np.zeros(3, dtype=np.float64)
        self._image_sum_sq = np.zeros(3, dtype=np.float64)
        self._image_pixel_count = 0
        self._image_frame_count = 0

    # ------------------------------------------------------------------
    # Episode management helpers
    # ------------------------------------------------------------------
    def create_episode_writer(self, episode_index: int) -> EpisodeWriter:
        """Initialise an :class:`EpisodeWriter` for ``episode_index``."""

        return EpisodeWriter(
            episode_index=episode_index,
            dataset=self,
            fps=self.fps,
            data_path=self.data_path,
            video_path=self.video_path,
            chunk_index=0,
        )

    def complete_episode(self, writer: EpisodeWriter, frames_logged: int) -> None:
        """Compatibility shim for legacy call sites."""

        # Episode statistics are registered by :meth:`EpisodeWriter.close` via
        # :meth:`_register_episode`. This method is kept for backwards
        # compatibility and does not need to do additional work.
        return None

    def finalize(self) -> None:
        """Persist dataset files and metadata in the LeRobot v3.0 layout."""

        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None

        if self.total_frames:
            table = self._build_dataset_table()
            pq.write_table(table, self.data_path)
        elif self.data_path.exists():
            self.data_path.unlink()

        if not self.total_frames and self.video_path.exists():
            self.video_path.unlink()

        self._write_tasks_table()
        self._write_episodes_table()
        self._write_stats_file()

        meta = dict(self._meta_base)
        meta["total_frames"] = int(self.total_frames)
        meta["total_episodes"] = int(self.total_episodes)

        if self.data_path.exists():
            meta["data_files_size_in_mb"] = round(
                self.data_path.stat().st_size / (1024 * 1024), 6
            )
        if self.video_path.exists():
            meta["video_files_size_in_mb"] = round(
                self.video_path.stat().st_size / (1024 * 1024), 6
            )

        self._write_meta(meta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_video_writer(self) -> imageio.core.format.Format:
        if self._video_writer is None:
            self._video_writer = imageio.get_writer(
                str(self.video_path),
                fps=float(self.fps),
                codec="libx264",
                format="FFMPEG",
                output_params=["-pix_fmt", "yuv420p"],
            )
        return self._video_writer

    def _append_frame(
        self,
        frame: np.ndarray,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        state: Tuple[float, float, float],
        action: Tuple[float, float, float],
    ) -> None:
        writer = self._ensure_video_writer()
        writer.append_data(frame)

        frame_float = frame.astype(np.float32) / 255.0
        channel_min = frame_float.min(axis=(0, 1)).astype(np.float64)
        channel_max = frame_float.max(axis=(0, 1)).astype(np.float64)
        channel_sum = frame_float.sum(axis=(0, 1))
        channel_sum_sq = np.square(frame_float).sum(axis=(0, 1))
        pixel_count = frame_float.shape[0] * frame_float.shape[1]

        if self._image_min is None:
            self._image_min = channel_min
            self._image_max = channel_max
        else:
            self._image_min = np.minimum(self._image_min, channel_min)
            self._image_max = np.maximum(self._image_max, channel_max)

        self._image_sum += channel_sum.astype(np.float64)
        self._image_sum_sq += channel_sum_sq.astype(np.float64)
        self._image_pixel_count += int(pixel_count)
        self._image_frame_count += 1

        self._data_rows["episode_index"].append(int(episode_index))
        self._data_rows["frame_index"].append(int(frame_index))
        self._data_rows["timestamp"].append(float(timestamp))
        self._data_rows["observation.state"].append(tuple(float(x) for x in state))
        self._data_rows["action"].append(tuple(float(x) for x in action))
        self._data_rows["task_index"].append(0)
        self._data_rows["next.reward"].append(0.0)
        self._data_rows["next.done"].append(False)
        self._data_rows["next.success"].append(False)
        self._data_rows["index"].append(int(frame_index))

        self.total_frames += 1

    def _register_episode(
        self,
        *,
        episode_index: int,
        frame_count: int,
        dataset_from_index: int,
        states: List[Tuple[float, float, float]],
        actions: List[Tuple[float, float, float]],
    ) -> None:
        dataset_to_index = dataset_from_index + frame_count
        video_from_timestamp = dataset_from_index / float(self.fps)
        video_to_timestamp = dataset_to_index / float(self.fps)

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)

        self._episode_records.append(
            {
                "episode_index": int(episode_index),
                "frame_count": int(frame_count),
                "dataset_from_index": int(dataset_from_index),
                "dataset_to_index": int(dataset_to_index),
                "video_from_timestamp": float(video_from_timestamp),
                "video_to_timestamp": float(video_to_timestamp),
                "states": states_np,
                "actions": actions_np,
            }
        )

        self.total_episodes = len(self._episode_records)

    def _build_dataset_table(self) -> pa.Table:
        if self.total_frames == 0:
            raise RuntimeError("Cannot build dataset table without any frames")

        states = np.asarray(self._data_rows["observation.state"], dtype=np.float32)
        actions = np.asarray(self._data_rows["action"], dtype=np.float32)

        arrays = {
            "episode_index": pa.array(self._data_rows["episode_index"], type=pa.int64()),
            "frame_index": pa.array(self._data_rows["frame_index"], type=pa.int64()),
            "timestamp": pa.array(self._data_rows["timestamp"], type=pa.float32()),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(states.reshape(-1), type=pa.float32()),
                3,
            ),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(actions.reshape(-1), type=pa.float32()),
                3,
            ),
            "task_index": pa.array(self._data_rows["task_index"], type=pa.int64()),
            "next.reward": pa.array(self._data_rows["next.reward"], type=pa.float32()),
            "next.done": pa.array(self._data_rows["next.done"], type=pa.bool_()),
            "next.success": pa.array(self._data_rows["next.success"], type=pa.bool_()),
            "index": pa.array(self._data_rows["index"], type=pa.int64()),
        }

        return pa.table(arrays)

    def _write_tasks_table(self) -> None:
        table = pa.table(
            {
                "__index_level_0__": pa.array(
                    ["Center and zoom on the target."], type=pa.string()
                ),
                "task_index": pa.array([0], type=pa.int64()),
            }
        )
        pq.write_table(table, self.tasks_path)

    def _write_episodes_table(self) -> None:
        if not self._episode_records:
            if self.episodes_path.exists():
                self.episodes_path.unlink()
            return

        table = pa.table(
            {
                "episode_index": pa.array(
                    [rec["episode_index"] for rec in self._episode_records],
                    type=pa.int64(),
                ),
                "data/chunk_index": pa.array(
                    [0 for _ in self._episode_records], type=pa.int64()
                ),
                "data/file_index": pa.array(
                    [0 for _ in self._episode_records], type=pa.int64()
                ),
                "dataset_from_index": pa.array(
                    [rec["dataset_from_index"] for rec in self._episode_records],
                    type=pa.int64(),
                ),
                "dataset_to_index": pa.array(
                    [rec["dataset_to_index"] for rec in self._episode_records],
                    type=pa.int64(),
                ),
                "videos/observation.image/chunk_index": pa.array(
                    [0 for _ in self._episode_records], type=pa.int64()
                ),
                "videos/observation.image/file_index": pa.array(
                    [0 for _ in self._episode_records], type=pa.int64()
                ),
                "videos/observation.image/from_timestamp": pa.array(
                    [rec["video_from_timestamp"] for rec in self._episode_records],
                    type=pa.float32(),
                ),
                "videos/observation.image/to_timestamp": pa.array(
                    [rec["video_to_timestamp"] for rec in self._episode_records],
                    type=pa.float32(),
                ),
                "tasks": pa.array(
                    [["Center and zoom on the target."] for _ in self._episode_records],
                    type=pa.list_(pa.string()),
                ),
                "length": pa.array(
                    [rec["frame_count"] for rec in self._episode_records],
                    type=pa.int64(),
                ),
            }
        )
        pq.write_table(table, self.episodes_path)

    def _write_stats_file(self) -> None:
        if not self.total_frames:
            if self.stats_path.exists():
                self.stats_path.unlink()
            return

        def _to_list(value: np.ndarray | np.generic) -> List[object]:
            arr = np.asarray(value)
            if arr.ndim == 0:
                return [arr.item()]
            return arr.tolist()

        def _compute_stats(array: np.ndarray) -> Dict[str, object]:
            arr = np.asarray(array)
            arr_float = arr.astype(np.float64)
            min_val = arr.min(axis=0)
            max_val = arr.max(axis=0)
            mean_val = arr_float.mean(axis=0)
            std_val = arr_float.std(axis=0, ddof=0)
            count = int(arr.shape[0]) if arr.ndim > 0 else int(arr.size)
            return {
                "min": _to_list(min_val),
                "max": _to_list(max_val),
                "mean": _to_list(mean_val),
                "std": _to_list(std_val),
                "count": [count],
            }

        def _compute_bool_stats(array: np.ndarray) -> Dict[str, object]:
            arr = np.asarray(array, dtype=bool)
            arr_float = arr.astype(np.float64)
            min_val = arr.min(axis=0)
            max_val = arr.max(axis=0)
            mean_val = arr_float.mean(axis=0)
            std_val = arr_float.std(axis=0, ddof=0)
            count = int(arr.shape[0]) if arr.ndim > 0 else int(arr.size)
            return {
                "min": _to_list(min_val),
                "max": _to_list(max_val),
                "mean": _to_list(mean_val),
                "std": _to_list(std_val),
                "count": [count],
            }

        stats_payload: Dict[str, object] = {}

        index_array = np.array(self._data_rows["index"], dtype=np.int64)
        stats_payload["index"] = _compute_stats(index_array.reshape(-1, 1))

        next_success = np.array(self._data_rows["next.success"], dtype=bool)
        stats_payload["next.success"] = _compute_bool_stats(next_success.reshape(-1, 1))

        state_array = np.array(self._data_rows["observation.state"], dtype=np.float32)
        stats_payload["observation.state"] = _compute_stats(state_array)

        next_done = np.array(self._data_rows["next.done"], dtype=bool)
        stats_payload["next.done"] = _compute_bool_stats(next_done.reshape(-1, 1))

        timestamp_array = np.array(self._data_rows["timestamp"], dtype=np.float32)
        stats_payload["timestamp"] = _compute_stats(timestamp_array.reshape(-1, 1))

        episode_index_array = np.array(self._data_rows["episode_index"], dtype=np.int64)
        stats_payload["episode_index"] = _compute_stats(episode_index_array.reshape(-1, 1))

        frame_index_array = np.array(self._data_rows["frame_index"], dtype=np.int64)
        stats_payload["frame_index"] = _compute_stats(frame_index_array.reshape(-1, 1))

        action_array = np.array(self._data_rows["action"], dtype=np.float32)
        stats_payload["action"] = _compute_stats(action_array)

        task_index_array = np.array(self._data_rows["task_index"], dtype=np.int64)
        stats_payload["task_index"] = _compute_stats(task_index_array.reshape(-1, 1))

        next_reward_array = np.array(self._data_rows["next.reward"], dtype=np.float32)
        stats_payload["next.reward"] = _compute_stats(next_reward_array.reshape(-1, 1))

        if self._image_frame_count and self._image_min is not None and self._image_max is not None:
            mean = self._image_sum / float(self._image_pixel_count)
            variance = self._image_sum_sq / float(self._image_pixel_count) - mean**2
            variance = np.clip(variance, 0.0, None)
            std = np.sqrt(variance)

            stats_payload["observation.image"] = {
                "min": _to_list(self._image_min.reshape(3, 1, 1)),
                "max": _to_list(self._image_max.reshape(3, 1, 1)),
                "mean": _to_list(mean.reshape(3, 1, 1)),
                "std": _to_list(std.reshape(3, 1, 1)),
                "count": [int(self._image_frame_count)],
            }

        with self.stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)
            f.write("\n")

    def _write_meta(self, payload: Dict[str, object]) -> None:
        """Write ``payload`` to ``meta/info.json`` with indentation."""

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


__all__ = ["EpisodeWriter", "DatasetManager", "DEFAULT_VIDEO_KEY"]

