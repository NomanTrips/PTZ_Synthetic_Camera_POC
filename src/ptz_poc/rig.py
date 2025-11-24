"""Synthetic pan/tilt/zoom rig for cropping viewports from full frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class RigState:
    """Simple container representing the rig's pose."""

    pan_deg: float
    tilt_deg: float
    zoom_norm: float


class Rig:
    """Virtual pan/tilt/zoom camera that renders cropped viewports."""

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
    ) -> None:
        if fov_x_deg <= 0:
            raise ValueError("fov_x_deg must be positive")
        if fov_y_deg <= 0:
            raise ValueError("fov_y_deg must be positive")
        if output_size[0] <= 0 or output_size[1] <= 0:
            raise ValueError("output_size must be positive")
        if zoom_alpha <= 0:
            raise ValueError("zoom_alpha must be positive")

        self.fov_x_deg = float(fov_x_deg)
        self.fov_y_deg = float(fov_y_deg)
        self.output_height, self.output_width = output_size
        self.zoom_alpha = float(zoom_alpha)

        self.pan_limits = pan_limits or (-self.fov_x_deg / 2.0, self.fov_x_deg / 2.0)
        self.tilt_limits = tilt_limits or (-self.fov_y_deg / 2.0, self.fov_y_deg / 2.0)
        self.zoom_limits = zoom_limits
        self.max_deltas = max_deltas

        self._state = RigState(pan_deg=0.0, tilt_deg=0.0, zoom_norm=float(np.clip(0.0, *self.zoom_limits)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def state(self) -> RigState:
        return RigState(
            pan_deg=self._state.pan_deg,
            tilt_deg=self._state.tilt_deg,
            zoom_norm=self._state.zoom_norm,
        )

    def reset(self, *, pan_deg: float = 0.0, tilt_deg: float = 0.0, zoom_norm: float = 0.0) -> None:
        """Reset the rig pose to the provided values."""

        self._state = RigState(
            pan_deg=self._clamp(pan_deg, *self.pan_limits),
            tilt_deg=self._clamp(tilt_deg, *self.tilt_limits),
            zoom_norm=self._clamp(zoom_norm, *self.zoom_limits),
        )

    def apply(self, dpan: float, dtilt: float, dzoom: float) -> RigState:
        """Apply incremental deltas to the current pose."""

        max_dpan, max_dtilt, max_dzoom = self.max_deltas
        dpan = self._clamp(dpan, -max_dpan, max_dpan)
        dtilt = self._clamp(dtilt, -max_dtilt, max_dtilt)
        dzoom = self._clamp(dzoom, -max_dzoom, max_dzoom)

        pan = self._clamp(self._state.pan_deg + dpan, *self.pan_limits)
        tilt = self._clamp(self._state.tilt_deg + dtilt, *self.tilt_limits)
        zoom = self._clamp(self._state.zoom_norm + dzoom, *self.zoom_limits)

        self._state = RigState(pan_deg=pan, tilt_deg=tilt, zoom_norm=zoom)
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

        center_x = (w_full / 2.0) + (self._state.pan_deg * pixels_per_deg_x)
        center_y = (h_full / 2.0) - (self._state.tilt_deg * pixels_per_deg_y)

        center_x = self._clamp(center_x, half_w, w_full - half_w)
        center_y = self._clamp(center_y, half_h, h_full - half_h)

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
    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return float(np.clip(value, min_value, max_value))

