"""Command-line entry point for the PTZ Synthetic Camera proof-of-concept."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from ptz_poc import DatasetManager, EpisodeWriter, HUDRenderer, InputHandler, Rig, VideoReader


class FrameSampler:
    """Track when frames should be logged based on a target sampling rate."""

    def __init__(self, target_fps: float) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self._interval = 1.0 / float(target_fps)
        self._last_logged_timestamp: float | None = None

    def reset(self) -> None:
        """Reset internal state so the next frame is logged immediately."""

        self._last_logged_timestamp = None

    def is_due(self, pts_sec: float) -> bool:
        """Return ``True`` when ``pts_sec`` satisfies the sampling interval."""

        if self._last_logged_timestamp is None:
            return True
        return (float(pts_sec) - self._last_logged_timestamp) >= self._interval - 1e-9

    def mark_logged(self, pts_sec: float) -> None:
        """Record that a frame at ``pts_sec`` was written to the dataset."""

        self._last_logged_timestamp = float(pts_sec)


def resolve_playback_fps(source_fps: float | None, target_fps: float) -> float:
    """Return the display rate, prioritising the source FPS when available."""

    if source_fps is not None and source_fps > 0:
        return float(source_fps)
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    return float(target_fps)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video")
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Directory for dataset output"
    )
    parser.add_argument("--size", type=int, default=256, help="Viewport/window size in pixels")
    parser.add_argument("--fps", type=int, default=24, help="Target output FPS")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append captured data to an existing dataset instead of starting fresh",
    )
    parser.add_argument(
        "--skip-noop-frames",
        action="store_true",
        help=(
            "When set, omit frames from the recording if the action is effectively a no-op."
        ),
    )
    parser.add_argument(
        "--pan-speed",
        type=float,
        default=2.5,
        help=(
            "Multiplier applied to horizontal inputs; higher values move the camera faster."
        ),
    )
    parser.add_argument(
        "--tilt-speed",
        type=float,
        default=2.5,
        help=(
            "Multiplier applied to vertical inputs; higher values move the camera faster."
        ),
    )
    parser.add_argument(
        "--zoom-step",
        type=float,
        default=0.05,
        help="Amount of zoom applied per mouse-wheel notch.",
    )
    parser.add_argument(
        "--log-on-motion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled, immediately log a frame whenever the action indicates motion. "
            "Use --no-log-on-motion to keep the legacy interval-only behaviour."
        ),
    )
    parser.add_argument(
        "--viewport-position",
        choices=("center", "left", "right"),
        default="center",
        help=(
            "Initial horizontal viewport placement on the input video. "
            "Defaults to the center."
        ),
    )
    return parser.parse_args(argv)


def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    """Convert an RGB numpy frame into a pygame surface."""

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame with shape (H, W, 3)")

    frame = np.transpose(frame, (1, 0, 2))
    return pygame.surfarray.make_surface(frame)


def has_motion(action: tuple[float, float, float], eps: float = 1e-6) -> bool:
    """Return True if any component of the action indicates motion beyond eps."""

    return any(abs(component) > eps for component in action)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    reader = VideoReader(args.video)
    playback_fps = resolve_playback_fps(reader.info.fps, float(args.fps))
    frame_sampler = FrameSampler(float(args.fps))
    frame_iter = iter(reader)
    try:
        full_frame, pts_sec = next(frame_iter)
    except StopIteration:
        raise RuntimeError(f"Video {args.video} does not contain any decodable frames")

    captured_episodes = 0
    total_logged_frames = 0

    pygame.init()
    try:
        window_size = (args.size, args.size)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("PTZ Synthetic Camera POC")

        rig = Rig(fov_x_deg=80.0, fov_y_deg=60.0, output_size=(args.size, args.size))
        if args.viewport_position == "center":
            initial_pan = 0.0
        elif args.viewport_position == "left":
            initial_pan = rig.pan_limits[0]
        else:
            initial_pan = rig.pan_limits[1]
        rig.reset(pan_deg=initial_pan)
        input_handler = InputHandler(
            pan_speed=args.pan_speed,
            tilt_speed=args.tilt_speed,
            zoom_step=args.zoom_step,
        )
        hud = HUDRenderer(window_size)

        dataset = DatasetManager(args.out_dir, fps=args.fps, append=args.append)

        recording = False
        base_episode_index = dataset.total_episodes
        episode_index = base_episode_index
        frame_index = 0
        episode_writer: EpisodeWriter | None = None

        clock = pygame.time.Clock()
        while True:
            viewport = rig.render(full_frame)
            frame_surface = frame_to_surface(viewport)
            frame_surface = pygame.transform.smoothscale(frame_surface, window_size)

            command = input_handler.poll()
            if command.quit_requested:
                if recording and episode_writer is not None:
                    frames = episode_writer.close()
                    dataset.complete_episode(episode_writer, frames)
                    total_logged_frames += frames
                    episode_index += int(frames > 0)
                    recording = False
                    episode_writer = None
                break

            if command.toggle_recording:
                if recording:
                    frames = 0
                    if episode_writer is not None:
                        frames = episode_writer.close()
                        dataset.complete_episode(episode_writer, frames)
                        total_logged_frames += frames
                        episode_writer = None
                    if frames > 0:
                        episode_index += 1
                    recording = False
                    frame_index = 0
                else:
                    episode_writer = dataset.create_episode_writer(episode_index)
                    recording = True
                    frame_index = 0
                    frame_sampler.reset()

            state_before = rig.state
            action = (command.dpan, command.dtilt, command.dzoom)

            if recording:
                should_log = frame_sampler.is_due(pts_sec)
                if (
                    args.log_on_motion
                    and not should_log
                    and has_motion(action)
                ):
                    frame_sampler.reset()
                    should_log = True
                if should_log and args.skip_noop_frames:
                    should_log = frame_index == 0 or has_motion(action)

                if should_log and episode_writer is not None:
                    episode_writer.append(
                        viewport,
                        frame_index=frame_index,
                        timestamp=pts_sec,
                        state=(
                            state_before.pan_deg,
                            state_before.tilt_deg,
                            state_before.zoom_norm,
                        ),
                        action=action,
                    )
                    frame_sampler.mark_logged(pts_sec)
                    frame_index += 1

            rig.apply(*action)

            screen.fill((0, 0, 0))
            hud.draw(screen, frame_surface, state_before)
            pygame.display.flip()
            clock.tick(playback_fps)

            try:
                full_frame, pts_sec = next(frame_iter)
            except StopIteration:
                if recording and episode_writer is not None:
                    frames = episode_writer.close()
                    dataset.complete_episode(episode_writer, frames)
                    total_logged_frames += frames
                    episode_index += int(frames > 0)
                    recording = False
                    episode_writer = None
                break
        captured_episodes = episode_index - base_episode_index
        dataset.finalize()
    finally:
        pygame.quit()

    if captured_episodes:
        print(f"Captured {captured_episodes} episode(s) with {total_logged_frames} frame(s)")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
