"""Input handling utilities for the PTZ synthetic camera demo."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass
class InputCommand:
    """Represents the control signals produced for a single tick."""

    dpan: float = 0.0
    dtilt: float = 0.0
    dzoom: float = 0.0
    dyaw: float = 0.0
    dpitch: float = 0.0
    toggle_recording: bool = False
    quit_requested: bool = False


class InputHandler:
    """Translate pygame events into PTZ rig control commands."""

    def __init__(
        self,
        *,
        pan_speed: float = 2.0,
        tilt_speed: float = 2.0,
        zoom_step: float = 0.1,
        fps_zoom_speed: float | None = None,
        fps_controls: bool = False,
    ) -> None:
        if pan_speed < 0 or tilt_speed < 0 or zoom_step <= 0:
            raise ValueError("Input speed parameters must be non-negative")
        if fps_zoom_speed is not None and fps_zoom_speed <= 0:
            raise ValueError("fps_zoom_speed must be positive when provided")

        self.pan_speed = float(pan_speed)
        self.tilt_speed = float(tilt_speed)
        self.zoom_step = float(zoom_step)
        self.fps_zoom_speed = float(fps_zoom_speed) if fps_zoom_speed is not None else self.zoom_step
        self.fps_controls = bool(fps_controls)
        self._pending_zoom_delta = 0.0

    def poll(self) -> InputCommand:
        """Consume pygame events and convert them into a command."""

        command = InputCommand()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                command.quit_requested = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    command.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    command.toggle_recording = True
            elif self.fps_controls and event.type == pygame.MOUSEMOTION:
                dx, dy = event.rel
                command.dyaw += float(dx)
                command.dpitch -= float(dy)
            elif event.type == pygame.MOUSEWHEEL:
                self._pending_zoom_delta += event.y * self.zoom_step

        pressed = pygame.key.get_pressed()

        if self.fps_controls:
            if pressed[pygame.K_w]:
                command.dzoom += self.fps_zoom_speed
            if pressed[pygame.K_s]:
                command.dzoom -= self.fps_zoom_speed
        else:
            if pressed[pygame.K_a]:
                command.dpan -= self.pan_speed
            if pressed[pygame.K_d]:
                command.dpan += self.pan_speed
            if pressed[pygame.K_w]:
                command.dtilt += self.tilt_speed
            if pressed[pygame.K_s]:
                command.dtilt -= self.tilt_speed

        if self._pending_zoom_delta:
            command.dzoom += self._pending_zoom_delta
            self._pending_zoom_delta = 0.0

        return command
