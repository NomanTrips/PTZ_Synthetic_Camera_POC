"""Utilities for decoding video frames via PyAV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator, Optional, Tuple

import av
import numpy as np


@dataclass(frozen=True)
class VideoStreamInfo:
    """Metadata about the primary video stream."""

    width: int
    height: int
    fps: Optional[float]


class VideoReader(Iterable[Tuple[np.ndarray, float]]):
    """Iterate over frames from a video file using PyAV.

    Parameters
    ----------
    path:
        Path to the video file. The video stream is decoded to RGB ``uint8``
        frames. Each iteration yields a tuple ``(frame, pts_sec)`` where
        ``frame`` has shape ``(H, W, 3)`` and ``pts_sec`` is the presentation
        timestamp in seconds.
    stream_index:
        Optionally force the index of the video stream to decode. By default
        the first video stream is used.
    """

    def __init__(self, path: Path | str, *, stream_index: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

        self._stream_index = stream_index
        self._info = self._probe_stream_info()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def info(self) -> VideoStreamInfo:
        """Return metadata describing the decoded video stream."""

        return self._info

    @property
    def width(self) -> int:
        return self._info.width

    @property
    def height(self) -> int:
        return self._info.height

    @property
    def fps(self) -> Optional[float]:
        return self._info.fps

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float]]:
        return self._frame_generator()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _probe_stream_info(self) -> VideoStreamInfo:
        with av.open(str(self.path)) as container:
            stream = self._select_stream(container)
            width = stream.codec_context.width or stream.width
            height = stream.codec_context.height or stream.height
            fps = self._resolve_fps(stream)
        return VideoStreamInfo(width=width, height=height, fps=fps)

    def _frame_generator(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        with av.open(str(self.path)) as container:
            stream = self._select_stream(container)
            time_base = float(stream.time_base) if stream.time_base is not None else None

            frame_index = 0
            for frame in container.decode(stream):
                rgb_frame = frame.to_rgb().to_ndarray()
                if rgb_frame.dtype != np.uint8:
                    rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)

                pts_sec = self._frame_timestamp_seconds(
                    frame=frame,
                    time_base=time_base,
                    frame_index=frame_index,
                )
                frame_index += 1
                yield rgb_frame, pts_sec

    def _select_stream(self, container: av.container.InputContainer) -> av.video.stream.VideoStream:
        if self._stream_index is not None:
            try:
                stream = container.streams.video[self._stream_index]
            except IndexError as exc:  # pragma: no cover - guard clause
                raise ValueError(
                    f"Video stream index {self._stream_index} not present in {self.path}"
                ) from exc
            return stream

        try:
            return next(stream for stream in container.streams if stream.type == "video")
        except StopIteration as exc:  # pragma: no cover - guard clause
            raise ValueError(f"No video stream found in {self.path}") from exc

    def _resolve_fps(self, stream: av.video.stream.VideoStream) -> Optional[float]:
        if stream.average_rate is not None and stream.average_rate.denominator != 0:
            return float(stream.average_rate)
        if stream.base_rate is not None and stream.base_rate.denominator != 0:
            return float(stream.base_rate)
        return None

    def _frame_timestamp_seconds(
        self,
        *,
        frame: av.video.frame.VideoFrame,
        time_base: Optional[float],
        frame_index: int,
    ) -> float:
        if frame.time is not None:
            return float(frame.time)
        if frame.pts is not None and time_base:
            return float(frame.pts * time_base)
        if self.fps:
            return float(frame_index / self.fps)
        return float(frame_index)