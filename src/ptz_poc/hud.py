"""Heads-up display (HUD) helpers for rendering overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame

from .rig import RigState


@dataclass(slots=True)
class HUDStyle:
    """Visual configuration for :class:`HUDRenderer`."""

    crosshair_color: Tuple[int, int, int] = (255, 255, 255)
    crosshair_thickness: int = 1
    crosshair_length_ratio: float = 0.2
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_bg_color: Tuple[int, int, int, int] = (0, 0, 0, 160)
    font_name: str | None = None
    font_size: int = 16
    text_padding: int = 6


class HUDRenderer:
    """Render overlays (crosshair + telemetry text) on top of the viewport."""

    def __init__(self, window_size: Tuple[int, int], style: HUDStyle | None = None) -> None:
        self.window_width, self.window_height = window_size
        self.style = style or HUDStyle()

        pygame.font.init()
        if self.style.font_name:
            self.font = pygame.font.SysFont(self.style.font_name, self.style.font_size)
        else:
            self.font = pygame.font.SysFont("monospace", self.style.font_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def draw(self, screen: pygame.Surface, frame_surface: pygame.Surface, state: RigState) -> None:
        """Draw the viewport and HUD overlays to ``screen``."""

        screen.blit(frame_surface, (0, 0))
        self._draw_crosshair(screen)
        self._draw_state_text(screen, state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _draw_crosshair(self, screen: pygame.Surface) -> None:
        length = int(min(self.window_width, self.window_height) * self.style.crosshair_length_ratio)
        half_length = length // 2
        cx = self.window_width // 2
        cy = self.window_height // 2

        pygame.draw.line(
            screen,
            self.style.crosshair_color,
            (cx - half_length, cy),
            (cx + half_length, cy),
            self.style.crosshair_thickness,
        )
        pygame.draw.line(
            screen,
            self.style.crosshair_color,
            (cx, cy - half_length),
            (cx, cy + half_length),
            self.style.crosshair_thickness,
        )

    def _draw_state_text(self, screen: pygame.Surface, state: RigState) -> None:
        lines = [
            "PAN {:+06.2f}°  TILT {:+06.2f}°  YAW {:+06.2f}°  PITCH {:+06.2f}°".format(
                state.pan_deg, state.tilt_deg, state.yaw_deg, state.pitch_deg
            ),
            "X {:+06.3f}  Y {:+06.3f}  ZOOM {:0.2f}".format(state.x, state.y, state.zoom_norm),
        ]

        padding = self.style.text_padding
        text_surfaces = [self.font.render(line, True, self.style.text_color) for line in lines]
        width = max(surface.get_width() for surface in text_surfaces) + padding * 2
        height = sum(surface.get_height() for surface in text_surfaces) + padding * (len(text_surfaces) + 1)

        padded_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        padded_surface.fill(self.style.text_bg_color)

        y_cursor = padding
        for surface in text_surfaces:
            padded_surface.blit(surface, (padding, y_cursor))
            y_cursor += surface.get_height() + padding

        x = padding
        y = self.window_height - padded_surface.get_height() - padding

        screen.blit(padded_surface, (x, y))
