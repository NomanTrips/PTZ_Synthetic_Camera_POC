"""PTZ Synthetic Camera proof-of-concept package."""

from .rig import Rig, RigState
from .video_reader import VideoReader

__all__ = ["VideoReader", "Rig", "RigState"]
