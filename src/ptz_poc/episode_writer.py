"""Episode writer utilities for logging PTZ capture sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


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
                "shape": [7],
                "names": {
                    "pose": [
                        "pan_deg",
                        "tilt_deg",
                        "zoom_norm",
                        "x",
                        "y",
                        "yaw_deg",
                        "pitch_deg",
                    ]
                },
                "fps": non_video_feature_fps,
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": {
                    "pose_deltas": [
                        "dpan",
                        "dtilt",
                        "dzoom",
                        "forward",
                        "strafe",
                        "dyaw",
                        "dpitch",
                    ]
                },
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
    _states: List[Tuple[float, ...]] = field(default_factory=list, init=False, repr=False)
    _actions: List[Tuple[float, ...]] = field(default_factory=list, init=False, repr=False)
    _start_timestamp: float | None = field(default=None, init=False, repr=False)

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
        state: Tuple[float, ...],
        action: Tuple[float, ...],
    ) -> None:
        """Append a single observation/action pair to the episode."""

        if self._closed:
            raise RuntimeError("Cannot append to a closed EpisodeWriter")

        frame_u8 = np.asarray(frame)
        if frame_u8.dtype != np.uint8:
            frame_u8 = np.clip(frame_u8, 0, 255).astype(np.uint8)
        if frame_u8.ndim != 3 or frame_u8.shape[2] != 3:
            raise ValueError("frame must have shape (H, W, 3)")

        if self._start_timestamp is None:
            self._start_timestamp = float(timestamp)

        relative_timestamp = max(0.0, float(timestamp) - self._start_timestamp)

        self.dataset._append_frame(
            frame_u8,
            episode_index=self.episode_index,
            frame_index=frame_index,
            timestamp=relative_timestamp,
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
        append: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.meta_dir = self.root / "meta"
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.meta_dir / "info.json"

        self.fps = float(fps)
        self.chunk_size = int(chunk_size)
        self.video_key = video_key
        self.append = bool(append)

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

        existing_meta: Optional[Dict[str, object]] = None
        if self.append and self.meta_path.exists():
            try:
                with self.meta_path.open("r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except json.JSONDecodeError:
                existing_meta = None

        if existing_meta is not None:
            self._meta_base = existing_meta
            self.total_frames = int(existing_meta.get("total_frames", 0))
            self.total_episodes = int(existing_meta.get("total_episodes", 0))
            image_shape = (
                existing_meta.get("features", {})
                .get("observation.image", {})
                .get("shape", [256, 256, 3])
            )
        else:
            self._meta_base = _metadata_template(
                fps=int(self.fps),
                chunk_size=self.chunk_size,
                video_key=self.video_key,
            )
            self.total_frames = 0
            self.total_episodes = 0
            image_shape = self._meta_base["features"]["observation.image"].get(
                "shape", [256, 256, 3]
            )

        self._image_shape = tuple(int(x) for x in image_shape)

        self._write_meta(self._meta_base)

        self._existing_stats: Optional[Dict[str, object]] = None
        if self.append and self.stats_path.exists():
            try:
                with self.stats_path.open("r", encoding="utf-8") as f:
                    self._existing_stats = json.load(f)
            except json.JSONDecodeError:
                self._existing_stats = None

        self._new_frame_count = 0

        self._episode_records: List[Dict[str, object]] = []
        if self.append and self.episodes_path.exists():
            self._episode_records.extend(self._load_existing_episode_records())
        if not self._episode_records:
            self.total_episodes = 0
        else:
            self.total_episodes = len(self._episode_records)

        self._video_temp_path: Optional[Path] = None
        if self.append and self.video_path.exists():
            self._video_temp_path = self.video_path.with_name(
                f"{self.video_path.stem}.append{self.video_path.suffix}"
            )
            if self._video_temp_path.exists():
                self._video_temp_path.unlink()
        self._video_output_path = self._video_temp_path or self.video_path

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
            video_path=self._video_output_path,
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

        new_frame_count = self._new_frame_count

        if self.append:
            if new_frame_count:
                new_table = self._build_dataset_table()
                if self.data_path.exists():
                    existing_table = pq.read_table(self.data_path)
                    if existing_table.num_rows:
                        combined_table = pa.concat_tables([existing_table, new_table])
                    else:
                        combined_table = new_table
                else:
                    combined_table = new_table
                pq.write_table(combined_table, self.data_path)
        else:
            if new_frame_count:
                table = self._build_dataset_table()
                pq.write_table(table, self.data_path)
            elif self.data_path.exists():
                self.data_path.unlink()

        if self.append:
            if new_frame_count and self._video_temp_path is not None:
                self._merge_videos(self.video_path, self._video_temp_path)
            elif not new_frame_count and self._video_temp_path is not None and self._video_temp_path.exists():
                self._video_temp_path.unlink()
        else:
            if not new_frame_count and self.video_path.exists():
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
                str(self._video_output_path),
                fps=float(self.fps),
                codec="libx264",
                format="FFMPEG",
                output_params=["-pix_fmt", "yuv420p"],
            )
        return self._video_writer

    def _merge_videos(self, existing_path: Path, new_path: Path) -> None:
        if not new_path.exists():
            return

        if not existing_path.exists():
            if new_path != existing_path:
                new_path.replace(existing_path)
            return

        temp_output = existing_path.with_name(
            f"{existing_path.stem}.tmp{existing_path.suffix}"
        )
        if temp_output.exists():
            temp_output.unlink()

        writer = imageio.get_writer(
            str(temp_output),
            fps=float(self.fps),
            codec="libx264",
            format="FFMPEG",
            output_params=["-pix_fmt", "yuv420p"],
        )

        try:
            existing_reader = imageio.get_reader(str(existing_path))
            try:
                for frame in existing_reader:
                    writer.append_data(frame)
            finally:
                existing_reader.close()

            new_reader = imageio.get_reader(str(new_path))
            try:
                for frame in new_reader:
                    writer.append_data(frame)
            finally:
                new_reader.close()
        finally:
            writer.close()

        existing_path.unlink()
        temp_output.replace(existing_path)

        if new_path.exists() and new_path != existing_path:
            new_path.unlink()

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
        # ``timestamp`` is relative to the start of the episode (in seconds).
        self._data_rows["timestamp"].append(float(timestamp))
        self._data_rows["observation.state"].append(tuple(float(x) for x in state))
        self._data_rows["action"].append(tuple(float(x) for x in action))
        self._data_rows["task_index"].append(0)
        self._data_rows["next.reward"].append(0.0)
        self._data_rows["next.done"].append(False)
        self._data_rows["next.success"].append(False)
        self._data_rows["index"].append(int(frame_index))

        self.total_frames += 1
        self._new_frame_count += 1

    def _register_episode(
        self,
        *,
        episode_index: int,
        frame_count: int,
        dataset_from_index: int,
        states: List[Tuple[float, ...]],
        actions: List[Tuple[float, ...]],
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

    def _load_existing_episode_records(self) -> List[Dict[str, object]]:
        records: List[Dict[str, object]] = []
        try:
            table = pq.read_table(self.episodes_path)
        except (FileNotFoundError, OSError):
            return records

        for row in table.to_pylist():
            frame_count = int(row.get("length", 0))
            dataset_from_index = int(row.get("dataset_from_index", 0))
            dataset_to_index = int(row.get("dataset_to_index", dataset_from_index + frame_count))
            video_from_timestamp = float(
                row.get("videos/observation.image/from_timestamp", 0.0)
            )
            video_to_timestamp = float(
                row.get("videos/observation.image/to_timestamp", video_from_timestamp)
            )

            records.append(
                {
                    "episode_index": int(row.get("episode_index", len(records))),
                    "frame_count": frame_count,
                    "dataset_from_index": dataset_from_index,
                    "dataset_to_index": dataset_to_index,
                    "video_from_timestamp": video_from_timestamp,
                    "video_to_timestamp": video_to_timestamp,
                    "states": None,
                    "actions": None,
                }
            )

        return records

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
        task_texts = ["Center and zoom on the target."]
        task_indices = list(range(len(task_texts)))

        df = pd.DataFrame({"task_index": task_indices}, index=task_texts)
        df.to_parquet(self.tasks_path, index=True)

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
        if self.append and self._new_frame_count == 0:
            return

        if not self.append and not self._data_rows["episode_index"]:
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
            min_val = arr.min(axis=0) if arr.size else np.array([0])
            max_val = arr.max(axis=0) if arr.size else np.array([0])
            mean_val = arr_float.mean(axis=0) if arr.size else np.zeros_like(min_val, dtype=np.float64)
            std_val = arr_float.std(axis=0, ddof=0) if arr.size else np.zeros_like(min_val, dtype=np.float64)
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
            min_val = arr.min(axis=0) if arr.size else np.array([False])
            max_val = arr.max(axis=0) if arr.size else np.array([False])
            mean_val = arr_float.mean(axis=0) if arr.size else np.zeros_like(min_val, dtype=np.float64)
            std_val = arr_float.std(axis=0, ddof=0) if arr.size else np.zeros_like(min_val, dtype=np.float64)
            count = int(arr.shape[0]) if arr.ndim > 0 else int(arr.size)
            return {
                "min": _to_list(min_val),
                "max": _to_list(max_val),
                "mean": _to_list(mean_val),
                "std": _to_list(std_val),
                "count": [count],
            }

        def _extract_count(stats: Dict[str, object] | None) -> int:
            if not stats:
                return 0
            count_val = stats.get("count", [])
            if isinstance(count_val, list) and count_val:
                return int(count_val[0])
            if isinstance(count_val, (int, float)):
                return int(count_val)
            return 0

        def _combine_numeric_stats(
            existing: Optional[Dict[str, object]], new: Dict[str, object]
        ) -> Dict[str, object]:
            existing_count = _extract_count(existing)
            new_count = _extract_count(new)

            if existing_count == 0:
                return new
            if new_count == 0:
                return existing if existing is not None else new

            existing_mean = np.asarray(existing["mean"], dtype=np.float64)
            new_mean = np.asarray(new["mean"], dtype=np.float64)
            combined_count = existing_count + new_count

            combined_mean = (
                existing_mean * existing_count + new_mean * new_count
            ) / combined_count

            existing_var = np.square(np.asarray(existing["std"], dtype=np.float64))
            new_var = np.square(np.asarray(new["std"], dtype=np.float64))
            combined_var = (
                existing_count
                * (existing_var + np.square(existing_mean - combined_mean))
                + new_count * (new_var + np.square(new_mean - combined_mean))
            ) / combined_count

            combined_std = np.sqrt(np.clip(combined_var, 0.0, None))

            combined_min = np.minimum(
                np.asarray(existing["min"]), np.asarray(new["min"])
            )
            combined_max = np.maximum(
                np.asarray(existing["max"]), np.asarray(new["max"])
            )

            return {
                "min": _to_list(combined_min),
                "max": _to_list(combined_max),
                "mean": _to_list(combined_mean),
                "std": _to_list(combined_std),
                "count": [combined_count],
            }

        def _combine_image_stats(
            existing: Optional[Dict[str, object]], new: Dict[str, object]
        ) -> Dict[str, object]:
            existing_frames = _extract_count(existing)
            new_frames = _extract_count(new)

            if existing_frames == 0:
                return new
            if new_frames == 0:
                return existing if existing is not None else new

            frame_pixels = int(self._image_shape[0]) * int(self._image_shape[1])
            existing_samples = existing_frames * frame_pixels
            new_samples = new_frames * frame_pixels

            existing_mean = np.asarray(existing["mean"], dtype=np.float64)
            new_mean = np.asarray(new["mean"], dtype=np.float64)
            combined_samples = existing_samples + new_samples

            combined_mean = (
                existing_mean * existing_samples + new_mean * new_samples
            ) / combined_samples

            existing_var = np.square(np.asarray(existing["std"], dtype=np.float64))
            new_var = np.square(np.asarray(new["std"], dtype=np.float64))
            combined_var = (
                existing_samples
                * (existing_var + np.square(existing_mean - combined_mean))
                + new_samples * (new_var + np.square(new_mean - combined_mean))
            ) / combined_samples

            combined_std = np.sqrt(np.clip(combined_var, 0.0, None))
            combined_min = np.minimum(
                np.asarray(existing["min"]), np.asarray(new["min"])
            )
            combined_max = np.maximum(
                np.asarray(existing["max"]), np.asarray(new["max"])
            )

            return {
                "min": _to_list(combined_min),
                "max": _to_list(combined_max),
                "mean": _to_list(combined_mean),
                "std": _to_list(combined_std),
                "count": [existing_frames + new_frames],
            }

        existing_stats = self._existing_stats if self.append else None

        stats_payload: Dict[str, object] = {}

        index_array = np.array(self._data_rows["index"], dtype=np.int64)
        index_stats = _compute_stats(index_array.reshape(-1, 1))
        stats_payload["index"] = _combine_numeric_stats(
            existing_stats.get("index") if existing_stats else None, index_stats
        )

        next_success = np.array(self._data_rows["next.success"], dtype=bool)
        next_success_stats = _compute_bool_stats(next_success.reshape(-1, 1))
        stats_payload["next.success"] = _combine_numeric_stats(
            existing_stats.get("next.success") if existing_stats else None,
            next_success_stats,
        )

        state_array = np.array(self._data_rows["observation.state"], dtype=np.float32)
        state_stats = _compute_stats(state_array)
        stats_payload["observation.state"] = _combine_numeric_stats(
            existing_stats.get("observation.state") if existing_stats else None,
            state_stats,
        )

        next_done = np.array(self._data_rows["next.done"], dtype=bool)
        next_done_stats = _compute_bool_stats(next_done.reshape(-1, 1))
        stats_payload["next.done"] = _combine_numeric_stats(
            existing_stats.get("next.done") if existing_stats else None,
            next_done_stats,
        )

        timestamp_array = np.array(self._data_rows["timestamp"], dtype=np.float32)
        timestamp_stats = _compute_stats(timestamp_array.reshape(-1, 1))
        stats_payload["timestamp"] = _combine_numeric_stats(
            existing_stats.get("timestamp") if existing_stats else None,
            timestamp_stats,
        )

        episode_index_array = np.array(self._data_rows["episode_index"], dtype=np.int64)
        episode_index_stats = _compute_stats(episode_index_array.reshape(-1, 1))
        stats_payload["episode_index"] = _combine_numeric_stats(
            existing_stats.get("episode_index") if existing_stats else None,
            episode_index_stats,
        )

        frame_index_array = np.array(self._data_rows["frame_index"], dtype=np.int64)
        frame_index_stats = _compute_stats(frame_index_array.reshape(-1, 1))
        stats_payload["frame_index"] = _combine_numeric_stats(
            existing_stats.get("frame_index") if existing_stats else None,
            frame_index_stats,
        )

        action_array = np.array(self._data_rows["action"], dtype=np.float32)
        action_stats = _compute_stats(action_array)
        stats_payload["action"] = _combine_numeric_stats(
            existing_stats.get("action") if existing_stats else None,
            action_stats,
        )

        task_index_array = np.array(self._data_rows["task_index"], dtype=np.int64)
        task_index_stats = _compute_stats(task_index_array.reshape(-1, 1))
        stats_payload["task_index"] = _combine_numeric_stats(
            existing_stats.get("task_index") if existing_stats else None,
            task_index_stats,
        )

        next_reward_array = np.array(self._data_rows["next.reward"], dtype=np.float32)
        next_reward_stats = _compute_stats(next_reward_array.reshape(-1, 1))
        stats_payload["next.reward"] = _combine_numeric_stats(
            existing_stats.get("next.reward") if existing_stats else None,
            next_reward_stats,
        )

        if (
            self._image_frame_count
            and self._image_min is not None
            and self._image_max is not None
        ):
            mean = self._image_sum / float(self._image_pixel_count)
            variance = self._image_sum_sq / float(self._image_pixel_count) - mean**2
            variance = np.clip(variance, 0.0, None)
            std = np.sqrt(variance)

            image_stats = {
                "min": _to_list(self._image_min.reshape(3, 1, 1)),
                "max": _to_list(self._image_max.reshape(3, 1, 1)),
                "mean": _to_list(mean.reshape(3, 1, 1)),
                "std": _to_list(std.reshape(3, 1, 1)),
                "count": [int(self._image_frame_count)],
            }

            stats_payload["observation.image"] = _combine_image_stats(
                existing_stats.get("observation.image") if existing_stats else None,
                image_stats,
            )
        elif existing_stats and "observation.image" in existing_stats:
            stats_payload["observation.image"] = existing_stats["observation.image"]

        if existing_stats:
            for key, value in existing_stats.items():
                if key not in stats_payload:
                    stats_payload[key] = value

        with self.stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats_payload, f, indent=2)
            f.write("\n")

    def _write_meta(self, payload: Dict[str, object]) -> None:
        """Write ``payload`` to ``meta/info.json`` with indentation."""

        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


__all__ = ["EpisodeWriter", "DatasetManager", "DEFAULT_VIDEO_KEY"]

