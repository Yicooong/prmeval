"""Built-in inference baselines; importing this package registers each implementation."""

from .gvl import GVL
from .progress_test import ProgressTestModel
from .rbm import RBMModel
from .rlvlmf import RLVLMF
from .robodopamine import RoboDopamine
from .roboreward import RoboReward
from .sole_r1 import SoleR1
from .topreward import TopReward
from .vlac import VLAC

__all__ = [
    "GVL",
    "RLVLMF",
    "VLAC",
    "ProgressTestModel",
    "RBMModel",
    "RoboDopamine",
    "RoboReward",
    "SoleR1",
    "TopReward",
]
