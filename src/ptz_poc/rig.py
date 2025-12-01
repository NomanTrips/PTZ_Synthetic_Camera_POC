"""Synthetic pan/tilt/zoom rig for cropping viewports from full frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import cv2
import numpy as np


@dataclass
class RigState:
    """Simple container representing the rig's pose.

    The planar ``x``/``y`` offsets use the rig's configured ``position_basis``
    (pixels, degrees, or normalized units relative to frame size).
    """

    pan_deg: float
    tilt_deg: float
    zoom_norm: float
    x: float = 0.0
    y: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


class Rig:
    """Virtual pan/tilt/zoom camera that renders cropped viewports.

    The rig tracks both angular orientation (pan/tilt/yaw/pitch) and planar
    translation relative to the input frame's centre. The planar offsets can be
    expressed in pixels, degrees, or normalized units to match downstream
    control schemes.
    """

    def __init__(
        self,
        *,
        fov_x_deg: float,
        fov_y_deg: float,
        output_size: Tuple[int, int],
        zoom_alpha: float = 1.5,
        pan_limits: Tuple[float, float] | None = None,
        tilt_limits: Tuple[float, float] | None = None,
        zoom_limits: Tuple[float, float] = (0.0, 1.0),
        max_deltas: Tuple[float, float, float] = (5.0, 5.0, 0.1),
        position_limits: Tuple[float, float] | None = None,
        position_speed: float = 12.0,
        position_basis: Literal["pixels", "degrees", "normalized"] = "pixels",
        yaw_sensitivity: float = 0.1,
        pitch_sensitivity: float = 0.1,
    ) -> None:
        if fov_x_deg <= 0:
            raise ValueError("fov_x_deg must be positive")
        if fov_y_deg <= 0:
            raise ValueError("fov_y_deg must be positive")
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output_size must be positive")
        if zoom_alpha <= 0:
            raise ValueError("zoom_alpha must be positive")
        if position_limits is not None and position_limits[0] >= position_limits[1]:
            raise ValueError("position_limits must be an ordered (min, max) tuple")
        if position_speed < 0:
            raise ValueError("position_speed must be non-negative")
        if position_basis not in {"pixels", "degrees", "normalized"}:
            raise ValueError("position_basis must be one of: pixels, degrees, normalized")
        if yaw_sensitivity < 0 or pitch_sensitivity < 0:
            raise ValueError("angular sensitivities must be non-negative")

        self.fov_x_deg = float(fov_x_deg)
        self.fov_y_deg = float(fov_y_deg)
        self.output_height, self.output_width = output_size
        self.zoom_alpha = float(zoom_alpha)

        self.pan_limits = pan_limits or (-self.fov_x_deg / 2.0, self.fov_x_deg / 2.0)
        self.tilt_limits = tilt_limits or (-self.fov_y_deg / 2.0, self.fov_y_deg / 2.0)
        self.zoom_limits = zoom_limits
        self.max_deltas = max_deltas
        self.position_limits = position_limits or (-float("inf"), float("inf"))
        self.position_speed = float(position_speed)
        self.position_basis = position_basis
        self.yaw_sensitivity = float(yaw_sensitivity)
        self.pitch_sensitivity = float(pitch_sensitivity)

        self._state = RigState(
            pan_deg=0.0,
            tilt_deg=0.0,
            zoom_norm=float(np.clip(0.0, *self.zoom_limits)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def state(self) -> RigState:
        return RigState(
            pan_deg=self._state.pan_deg,
            tilt_deg=self._state.tilt_deg,
            zoom_norm=self._state.zoom_norm,
            x=self._state.x,
            y=self._state.y,
            yaw_deg=self._state.yaw_deg,
            pitch_deg=self._state.pitch_deg,
        )

    def reset(
        self,
        *,
        pan_deg: float = 0.0,
        tilt_deg: float = 0.0,
        zoom_norm: float = 0.0,
        x: float = 0.0,
        y: float = 0.0,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
    ) -> None:
        """Reset the rig pose to the provided values."""

        self._state = RigState(
            pan_deg=self._clamp(pan_deg, *self.pan_limits),
            tilt_deg=self._clamp(tilt_deg, *self.tilt_limits),
            zoom_norm=self._clamp(zoom_norm, *self.zoom_limits),
            x=self._clamp(x, *self.position_limits),
            y=self._clamp(y, *self.position_limits),
            yaw_deg=self._wrap_angle(yaw_deg),
            pitch_deg=self._clamp(pitch_deg, -89.9, 89.9),
        )

    def apply(
        self,
        dpan: float,
        dtilt: float,
        dzoom: float,
        forward: float = 0.0,
        strafe: float = 0.0,
        dyaw: float = 0.0,
        dpitch: float = 0.0,
    ) -> RigState:
        """Apply incremental deltas to the current pose."""

        max_dpan, max_dtilt, max_dzoom = self.max_deltas
        dpan = self._clamp(dpan, -max_dpan, max_dpan)
        dtilt = self._clamp(dtilt, -max_dtilt, max_dtilt)
        dzoom = self._clamp(dzoom, -max_dzoom, max_dzoom)

        pan = self._clamp(self._state.pan_deg + dpan, *self.pan_limits)
        tilt = self._clamp(self._state.tilt_deg + dtilt, *self.tilt_limits)
        zoom = self._clamp(self._state.zoom_norm + dzoom, *self.zoom_limits)

        yaw = self._wrap_angle(self._state.yaw_deg + dyaw * self.yaw_sensitivity)
        pitch = self._clamp(self._state.pitch_deg + dpitch * self.pitch_sensitivity, -89.9, 89.9)

        yaw_rad = np.deg2rad(yaw)
        dx = (forward * np.sin(yaw_rad) + strafe * np.cos(yaw_rad)) * self.position_speed
        dy = (forward * np.cos(yaw_rad) - strafe * np.sin(yaw_rad)) * self.position_speed

        x = self._clamp(self._state.x + dx, *self.position_limits)
        y = self._clamp(self._state.y + dy, *self.position_limits)

        self._state = RigState(
            pan_deg=pan,
            tilt_deg=tilt,
            zoom_norm=zoom,
            x=x,
            y=y,
            yaw_deg=yaw,
            pitch_deg=pitch,
        )
        return self.state

    def render(self, full_frame: np.ndarray) -> np.ndarray:
        """Render the current viewport from ``full_frame``."""

        if full_frame.ndim != 3 or full_frame.shape[2] != 3:
            raise ValueError("full_frame must have shape (H, W, 3)")

        frame = np.asarray(full_frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        h_full, w_full, _ = frame.shape
        aspect = self.output_width / self.output_height

        base_width = min(w_full, h_full * aspect)
        viewport_width = max(1.0, base_width * float(np.exp(-self.zoom_alpha * self._state.zoom_norm)))
        viewport_height = max(1.0, viewport_width / aspect)
        if viewport_height > h_full:
            viewport_height = float(h_full)
            viewport_width = viewport_height * aspect

        half_w = viewport_width / 2.0
        half_h = viewport_height / 2.0

        pixels_per_deg_x = w_full / self.fov_x_deg
        pixels_per_deg_y = h_full / self.fov_y_deg

        pan_offset_px = self._state.pan_deg * pixels_per_deg_x
        tilt_offset_px = self._state.tilt_deg * pixels_per_deg_y

        base_center_x = (w_full / 2.0) + pan_offset_px
        base_center_y = (h_full / 2.0) - tilt_offset_px

        planar_dx_px, planar_dy_px = self._position_offsets_px(
            w_full, h_full, pixels_per_deg_x, pixels_per_deg_y
        )

        planar_dx_px = self._clamp(
            planar_dx_px,
            half_w - base_center_x,
            (w_full - half_w) - base_center_x,
        )
        planar_dy_px = self._clamp(
            planar_dy_px,
            half_h - base_center_y,
            (h_full - half_h) - base_center_y,
        )

        center_x = base_center_x + planar_dx_px
        center_y = base_center_y + planar_dy_px

        self._update_planar_state_from_pixels(
            planar_dx_px,
            planar_dy_px,
            w_full,
            h_full,
            pixels_per_deg_x,
            pixels_per_deg_y,
        )

        width_int = max(1, int(round(viewport_width)))
        height_int = max(1, int(round(viewport_height)))

        left = int(round(center_x)) - width_int // 2
        top = int(round(center_y)) - height_int // 2

        left = int(self._clamp(left, 0, max(0, w_full - width_int)))
        top = int(self._clamp(top, 0, max(0, h_full - height_int)))

        right = left + width_int
        bottom = top + height_int

        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            raise RuntimeError("Viewport crop is empty; check rig configuration")

        resized = cv2.resize(crop, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR)
        return resized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _position_offsets_px(
        self,
        width: float,
        height: float,
        pixels_per_deg_x: float,
        pixels_per_deg_y: float,
    ) -> Tuple[float, float]:
        if self.position_basis == "pixels":
            return float(self._state.x), float(self._state.y)
        if self.position_basis == "degrees":
            return (
                float(self._state.x * pixels_per_deg_x),
                float(self._state.y * pixels_per_deg_y),
            )
        return float(self._state.x * width), float(self._state.y * height)

    def _update_planar_state_from_pixels(
        self,
        dx_px: float,
        dy_px: float,
        width: float,
        height: float,
        pixels_per_deg_x: float,
        pixels_per_deg_y: float,
    ) -> None:
        if self.position_basis == "pixels":
            x_units, y_units = dx_px, dy_px
        elif self.position_basis == "degrees":
            x_units = dx_px / pixels_per_deg_x
            y_units = dy_px / pixels_per_deg_y
        else:
            x_units = dx_px / width
            y_units = dy_px / height

        self._state = RigState(
            pan_deg=self._state.pan_deg,
            tilt_deg=self._state.tilt_deg,
            zoom_norm=self._state.zoom_norm,
            x=float(x_units),
            y=float(y_units),
            yaw_deg=self._state.yaw_deg,
            pitch_deg=self._state.pitch_deg,
        )

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return float(np.clip(value, min_value, max_value))

    @staticmethod
    def _wrap_angle(angle_deg: float) -> float:
        wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
        return wrapped

