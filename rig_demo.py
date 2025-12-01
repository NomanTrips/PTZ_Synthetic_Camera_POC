"""Minimal demonstration script for the synthetic PTZ rig."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

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


def generate_sample_actions() -> List[tuple[float, float, float, float, float, float, float]]:
    """Return a handful of pan/tilt/zoom/pose deltas for demonstration purposes."""

    return [
        (0.0, 0.0, 0.0, 0.02, 0.0, 15.0, 0.0),
        (10.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0),
        (-20.0, 5.0, 0.1, 0.01, -0.01, -10.0, 2.5),
        (15.0, -10.0, 0.2, -0.02, 0.0, 0.0, -2.5),
        (0.0, 10.0, -0.15, 0.0, 0.0, 20.0, 0.0),
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

    for index, (dpan, dtilt, dzoom, forward, strafe, dyaw, dpitch) in enumerate(
        generate_sample_actions(), start=1
    ):
        state = rig.apply(dpan, dtilt, dzoom, forward, strafe, dyaw, dpitch)
        viewport = rig.render(frame)
        output_path = args.out_dir / f"frame_{index:03d}.png"
        imageio.imwrite(output_path, viewport)
        print(
            "Frame {idx:03d}: pan={pan:.2f}°, tilt={tilt:.2f}°, yaw={yaw:.2f}°, "
            "pos=({x:+.3f}, {y:+.3f}), zoom={zoom:.2f} → {path}".format(
                idx=index,
                pan=state.pan_deg,
                tilt=state.tilt_deg,
                yaw=state.yaw_deg,
                x=state.x,
                y=state.y,
                zoom=state.zoom_norm,
                path=output_path,
            )
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
