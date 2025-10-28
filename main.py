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
    return parser.parse_args(argv)


def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    """Convert an RGB numpy frame into a pygame surface."""

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected frame with shape (H, W, 3)")

    frame = np.transpose(frame, (1, 0, 2))
    return pygame.surfarray.make_surface(frame)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    reader = VideoReader(args.video)
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
        input_handler = InputHandler(pan_speed=2.5, tilt_speed=2.5, zoom_step=0.05)
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

            state_before = rig.state
            action = (command.dpan, command.dtilt, command.dzoom)

            if recording:
                if episode_writer is not None:
                    episode_writer.append(
                        viewport,
                        frame_index=frame_index,
                        timestamp=pts_sec,
                        state=(state_before.pan_deg, state_before.tilt_deg, state_before.zoom_norm),
                        action=action,
                    )
                frame_index += 1

            rig.apply(*action)

            screen.fill((0, 0, 0))
            hud.draw(screen, frame_surface, state_before)
            pygame.display.flip()
            clock.tick(args.fps)

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
