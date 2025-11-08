from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import types

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "ptz_poc" / "episode_writer.py"

MODULE_NAME = "ptz_poc.episode_writer_for_tests"
if "ptz_poc" not in sys.modules:
    package = types.ModuleType("ptz_poc")
    package.__path__ = []  # type: ignore[attr-defined]
    sys.modules["ptz_poc"] = package
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec is not None and spec.loader is not None
episode_writer = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = episode_writer
spec.loader.exec_module(episode_writer)
DatasetManager = episode_writer.DatasetManager

import numpy as np
import pyarrow.parquet as pq
import pytest


def _record_episode(
    manager: DatasetManager,
    *,
    episode_index: int,
    start_timestamp: float,
    fps: float,
    frame_count: int,
) -> list[int]:
    """Record ``frame_count`` frames for ``episode_index`` starting at ``start_timestamp``."""

    writer = manager.create_episode_writer(episode_index)
    expected_values: list[int] = []

    for frame_idx in range(frame_count):
        absolute_ts = start_timestamp + frame_idx / fps
        pixel_value = int(40 * episode_index + frame_idx * 5)
        frame = np.full((8, 8, 3), pixel_value, dtype=np.uint8)
        state = (float(frame_idx), float(frame_idx + 0.5), float(frame_idx + 1.0))
        action = (
            float(frame_idx) * 0.1,
            float(frame_idx) * 0.2,
            float(frame_idx) * 0.3,
        )

        writer.append(
            frame,
            frame_index=frame_idx,
            timestamp=absolute_ts,
            state=state,
            action=action,
        )
        expected_values.append(pixel_value)

    assert writer.close() == frame_count
    return expected_values


@pytest.fixture(scope="function")
def recorded_dataset(tmp_path: Path) -> dict[str, object]:
    fps = 15.0
    frame_count = 3
    pause_seconds = 0.75
    dataset_root = tmp_path / "dataset"
    manager = DatasetManager(dataset_root, fps=fps)

    expected_colors = {
        0: _record_episode(
            manager,
            episode_index=0,
            start_timestamp=1.25,
            fps=fps,
            frame_count=frame_count,
        ),
        1: _record_episode(
            manager,
            episode_index=1,
            start_timestamp=1.25 + pause_seconds + frame_count / fps,
            fps=fps,
            frame_count=frame_count,
        ),
    }

    manager.finalize()

    data_table = pq.read_table(manager.data_path).to_pandas()
    episodes_table = pq.read_table(manager.episodes_path).to_pandas()

    return {
        "manager": manager,
        "fps": fps,
        "data_table": data_table,
        "episodes_table": episodes_table,
        "expected_colors": expected_colors,
    }


def test_episode_timestamps_reset(recorded_dataset: dict[str, object]) -> None:
    fps = recorded_dataset["fps"]
    df = recorded_dataset["data_table"]

    for _, group in df.groupby("episode_index"):
        group = group.sort_values("frame_index").reset_index(drop=True)
        assert group.loc[0, "timestamp"] == pytest.approx(0.0, abs=1e-6)

        expected_ts = np.arange(len(group), dtype=np.float64) / fps
        np.testing.assert_allclose(group["timestamp"].to_numpy(), expected_ts, atol=1e-4)


def test_lerobot_decodes_without_overshoot(recorded_dataset: dict[str, object]) -> None:
    manager: DatasetManager = recorded_dataset["manager"]
    fps = recorded_dataset["fps"]
    df = recorded_dataset["data_table"]
    episodes_table = recorded_dataset["episodes_table"]
    expected_colors = recorded_dataset["expected_colors"]

    try:
        import importlib

        video_utils = importlib.import_module("lerobot.datasets.video_utils")
        importlib.import_module("torch")
        importlib.import_module("torchvision")
    except ModuleNotFoundError:
        pytest.skip("lerobot (and its video decoding dependencies) are not available")

    tolerance = 1.0 / (2.0 * fps)

    for _, episode_row in episodes_table.iterrows():
        episode_index = int(episode_row["episode_index"])
        group = (
            df[df["episode_index"] == episode_index]
            .sort_values("frame_index")
            .reset_index(drop=True)
        )
        from_timestamp = float(episode_row["videos/observation.image/from_timestamp"])
        query_ts = [from_timestamp + float(ts) for ts in group["timestamp"]]

        frames = video_utils.decode_video_frames(
            manager.video_path,
            query_ts,
            tolerance_s=tolerance,
            backend="pyav",
        )
        assert frames.shape[0] == len(group)

        frames_np = (frames.numpy() * 255.0).round().astype(np.uint8)
        for frame_idx, pixel_value in enumerate(expected_colors[episode_index]):
            assert frames_np[frame_idx, 0, 0, 0] == pytest.approx(pixel_value, abs=3)
