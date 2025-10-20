"""PTZ Synthetic Camera proof-of-concept package."""

from .input_handler import InputCommand, InputHandler
from .rig import Rig, RigState
from .video_reader import VideoReader

__all__ = ["VideoReader", "Rig", "RigState", "InputHandler", "InputCommand"]
