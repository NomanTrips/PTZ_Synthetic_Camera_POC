"""PTZ Synthetic Camera proof-of-concept package."""

from .episode_writer import DatasetManager, EpisodeWriter, DEFAULT_VIDEO_KEY
from .hud import HUDRenderer, HUDStyle
from .input_handler import InputCommand, InputHandler
from .rig import Rig, RigState
from .video_reader import VideoReader

__all__ = [
    "VideoReader",
    "Rig",
    "RigState",
    "InputHandler",
    "InputCommand",
    "HUDRenderer",
    "HUDStyle",
    "EpisodeWriter",
    "DatasetManager",
    "DEFAULT_VIDEO_KEY",
]
