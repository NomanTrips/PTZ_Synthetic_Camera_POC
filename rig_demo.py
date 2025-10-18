"""Minimal demonstration script for the synthetic PTZ rig."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import imageio.v3 as imageio

from ptz_poc import Rig, VideoReader


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("rig_demo_outputs"),
        help="Directory where demo frames will be written",
    )
    parser.add_argument("--size", type=int, default=256, help="Output viewport edge length in pixels")
    parser.add_argument("--fov_x", type=float, default=60.0, help="Horizontal field-of-view in degrees")
    parser.add_argument("--fov_y", type=float, default=40.0, help="Vertical field-of-view in degrees")
    return parser.parse_args(argv)


def generate_sample_actions() -> List[tuple[float, float, float]]:
    """Return a handful of pan/tilt/zoom deltas for demonstration purposes."""

    return [
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-20.0, 5.0, 0.1),
        (15.0, -10.0, 0.2),
        (0.0, 10.0, -0.15),
    ]


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    reader = VideoReader(args.video)
    try:
        frame, _ = next(iter(reader))
    except StopIteration:  # pragma: no cover - guard clause
        raise RuntimeError(f"Video {args.video} does not contain any decodable frames")

    rig = Rig(
        fov_x_deg=args.fov_x,
        fov_y_deg=args.fov_y,
        output_size=(args.size, args.size),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rig.reset()
    imageio.imwrite(args.out_dir / "frame_000.png", rig.render(frame))

    for index, (dpan, dtilt, dzoom) in enumerate(generate_sample_actions(), start=1):
        state = rig.apply(dpan, dtilt, dzoom)
        viewport = rig.render(frame)
        output_path = args.out_dir / f"frame_{index:03d}.png"
        imageio.imwrite(output_path, viewport)
        print(
            f"Frame {index:03d}: pan={state.pan_deg:.2f}°, tilt={state.tilt_deg:.2f}°, zoom={state.zoom_norm:.2f} → {output_path}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
