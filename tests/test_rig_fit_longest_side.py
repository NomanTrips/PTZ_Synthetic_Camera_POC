from __future__ import annotations

import pathlib
import sys

import importlib.util

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

class _Cv2Stub:
    INTER_LINEAR = 1

    @staticmethod
    def resize(image: np.ndarray, dsize: tuple[int, int], interpolation: int | None = None) -> np.ndarray:  # noqa: ARG002
        target_w, target_h = dsize
        src_h, src_w = image.shape[:2]

        channels = image.shape[2] if image.ndim == 3 else 1
        output = np.zeros((target_h, target_w, channels), dtype=float)

        x_edges = np.linspace(0, src_w, target_w + 1)
        y_edges = np.linspace(0, src_h, target_h + 1)

        for yi in range(target_h):
            y0 = int(np.floor(y_edges[yi]))
            y1 = int(np.ceil(y_edges[yi + 1]))
            for xi in range(target_w):
                x0 = int(np.floor(x_edges[xi]))
                x1 = int(np.ceil(x_edges[xi + 1]))

                patch = image[y0:y1, x0:x1]
                if patch.size:
                    output[yi, xi] = patch.mean(axis=(0, 1))

        return output.astype(image.dtype)


sys.modules.setdefault("cv2", _Cv2Stub())

RIG_MODULE_PATH = SRC_DIR / "ptz_poc" / "rig.py"
rig_spec = importlib.util.spec_from_file_location("ptz_poc.rig_under_test", RIG_MODULE_PATH)
assert rig_spec is not None and rig_spec.loader is not None
rig_module = importlib.util.module_from_spec(rig_spec)
sys.modules[rig_spec.name] = rig_module
rig_spec.loader.exec_module(rig_module)
Rig = rig_module.Rig


def test_fit_longest_side_letterboxes_and_preserves_edges() -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, 0] = np.array([255, 0, 0], dtype=np.uint8)
    frame[:, -1] = np.array([0, 0, 255], dtype=np.uint8)

    rig = Rig(
        fov_x_deg=80.0,
        fov_y_deg=60.0,
        output_size=(2, 2),
        fit_longest_side=True,
    )

    rig.reset()
    viewport = rig.render(frame)

    assert viewport.shape == (2, 2, 3)

    first_column_mean = viewport[:, 0].mean(axis=0)
    second_column_mean = viewport[:, 1].mean(axis=0)

    assert first_column_mean[0] > 50 and first_column_mean[2] < 50
    assert second_column_mean[2] > 50 and second_column_mean[0] < 50
