"""Command-line entry point for the PTZ Synthetic Camera proof-of-concept."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from ptz_poc import VideoReader


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for dataset output")
    parser.add_argument("--size", type=int, default=256, help="Viewport/window size in pixels")
    parser.add_argument("--fps", type=int, default=24, help="Target output FPS")
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
        first_frame, _ = next(frame_iter)
    except StopIteration:
        raise RuntimeError(f"Video {args.video} does not contain any decodable frames")

    pygame.init()
    try:
        window_size = (args.size, args.size)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("PTZ Synthetic Camera POC")

        frame_surface = frame_to_surface(first_frame)
        frame_surface = pygame.transform.smoothscale(frame_surface, window_size)

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))
            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        pygame.quit()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())