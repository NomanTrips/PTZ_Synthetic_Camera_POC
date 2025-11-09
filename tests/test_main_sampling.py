from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types

import numpy as np
import pyarrow.parquet as pq
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EPISODE_WRITER_PATH = SRC_DIR / "ptz_poc" / "episode_writer.py"
episode_writer_spec = importlib.util.spec_from_file_location(
    "ptz_poc.episode_writer_for_sampling_test", EPISODE_WRITER_PATH
)
assert episode_writer_spec is not None and episode_writer_spec.loader is not None
episode_writer_module = importlib.util.module_from_spec(episode_writer_spec)
sys.modules["ptz_poc.episode_writer_for_sampling_test"] = episode_writer_module
episode_writer_spec.loader.exec_module(episode_writer_module)

DatasetManager = episode_writer_module.DatasetManager
EpisodeWriter = episode_writer_module.EpisodeWriter


class _StubHUDRenderer:
    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def draw(self, *_args, **_kwargs) -> None:
        return None


class _StubInputHandler:
    def __init__(self, *_args, **_kwargs) -> None:
        return None

    def poll(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            quit_requested=False,
            toggle_recording=False,
            dpan=0.0,
            dtilt=0.0,
            dzoom=0.0,
        )


class _StubRig:
    def __init__(self, *_args, **_kwargs) -> None:
        self.state = types.SimpleNamespace(pan_deg=0.0, tilt_deg=0.0, zoom_norm=0.0)

    def render(self, frame):
        return frame

    def apply(self, *_args, **_kwargs) -> None:
        return None


class _StubVideoReader:
    def __init__(self, *_args, **_kwargs) -> None:
        self.info = types.SimpleNamespace(fps=None)

    def __iter__(self):
        return iter(())


ptz_poc_stub = types.ModuleType("ptz_poc")
ptz_poc_stub.DatasetManager = DatasetManager
ptz_poc_stub.EpisodeWriter = EpisodeWriter
ptz_poc_stub.HUDRenderer = _StubHUDRenderer
ptz_poc_stub.InputHandler = _StubInputHandler
ptz_poc_stub.Rig = _StubRig
ptz_poc_stub.VideoReader = _StubVideoReader

sys.modules["ptz_poc"] = ptz_poc_stub


def _import_main_with_stubbed_pygame(monkeypatch: pytest.MonkeyPatch):
    stub_screen = types.SimpleNamespace(fill=lambda *_args, **_kwargs: None)
    stub_display = types.SimpleNamespace(
        set_mode=lambda *_args, **_kwargs: stub_screen,
        set_caption=lambda *_args, **_kwargs: None,
        flip=lambda: None,
    )
    stub_transform = types.SimpleNamespace(smoothscale=lambda surface, _size: surface)
    stub_surfarray = types.SimpleNamespace(make_surface=lambda frame: frame)
    stub_clock = types.SimpleNamespace(tick=lambda _fps: None)
    stub_time = types.SimpleNamespace(Clock=lambda: stub_clock)

    pygame_stub = types.ModuleType("pygame")
    pygame_stub.init = lambda: None
    pygame_stub.quit = lambda: None
    pygame_stub.display = stub_display
    pygame_stub.transform = stub_transform
    pygame_stub.surfarray = stub_surfarray
    pygame_stub.time = stub_time

    monkeypatch.setitem(sys.modules, "pygame", pygame_stub)

    if "main" in sys.modules:
        del sys.modules["main"]

    main_spec = importlib.util.spec_from_file_location("main", PROJECT_ROOT / "main.py")
    assert main_spec is not None and main_spec.loader is not None
    main_module = importlib.util.module_from_spec(main_spec)
    sys.modules["main"] = main_module
    main_spec.loader.exec_module(main_module)
    return main_module


def test_sampling_respects_source_playback_and_target_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    main = _import_main_with_stubbed_pygame(monkeypatch)

    source_fps = 30.0
    target_fps = 10.0
    playback_fps = main.resolve_playback_fps(source_fps, target_fps)
    assert playback_fps == pytest.approx(source_fps)

    sampler = main.FrameSampler(target_fps)

    recorded_pts: list[float] = []
    total_source_frames = int(source_fps * 3.0)
    for frame_idx in range(total_source_frames):
        pts = frame_idx / source_fps
        if sampler.is_due(pts):
            sampler.mark_logged(pts)
            recorded_pts.append(pts)

    expected_frames = int(target_fps * 3.0)
    assert len(recorded_pts) == expected_frames
    assert recorded_pts[0] == pytest.approx(0.0, abs=1e-9)
    observed_rate = (len(recorded_pts) - 1) / (recorded_pts[-1] - recorded_pts[0])
    assert observed_rate == pytest.approx(target_fps, rel=1e-3)

    dataset_root = tmp_path / "dataset"
    manager = DatasetManager(dataset_root, fps=target_fps)
    writer = manager.create_episode_writer(0)

    for frame_index, pts in enumerate(recorded_pts):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        writer.append(
            frame,
            frame_index=frame_index,
            timestamp=pts,
            state=(0.0, 0.0, 0.0),
            action=(0.0, 0.0, 0.0),
        )

    assert writer.close() == expected_frames
    manager.finalize()

    data_table = pq.read_table(manager.data_path).to_pandas()
    assert len(data_table) == expected_frames
    assert data_table["frame_index"].tolist() == list(range(expected_frames))
    assert data_table["timestamp"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert data_table["timestamp"].iloc[-1] == pytest.approx(
        recorded_pts[-1] - recorded_pts[0], abs=1e-6
    )

    episodes_table = pq.read_table(manager.episodes_path).to_pandas()
    assert int(episodes_table.loc[0, "length"]) == expected_frames
    assert episodes_table.loc[0, "videos/observation.image/to_timestamp"] == pytest.approx(
        expected_frames / target_fps, abs=1e-6
    )
