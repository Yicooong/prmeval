"""Built-in remote inference baselines."""

from .gvl import GVLRemote
from .progress_test import ProgressTestRemote
from .rbm import RBMRemote
from .rewind import ReWiNDRemote
from .rlvlmf import RLVLMFRemote
from .robodopamine import RoboDopamineRemote
from .roboreward import RoboRewardRemote
from .topreward import TopRewardRemote
from .vlac import VLACRemote

__all__ = [
    "GVLRemote",
    "ProgressTestRemote",
    "RBMRemote",
    "RLVLMFRemote",
    "ReWiNDRemote",
    "RoboDopamineRemote",
    "RoboRewardRemote",
    "TopRewardRemote",
    "VLACRemote",
]
