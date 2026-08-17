"""Built-in local-first baseline package."""

from ..model import register_preference_model, register_progress_model
from .gvl import GVL
from .progress_test import ProgressTestModel
from .rbm_model import RBMModel
from .rlvlmf import RLVLMF
from .robodopamine import RoboDopamine
from .roboreward import RoboReward
from .topreward import TopReward
from .vlac import VLAC

__all__ = [
    "GVL",
    "ProgressTestModel",
    "RBMModel",
    "RLVLMF",
    "RoboDopamine",
    "RoboReward",
    "TopReward",
    "VLAC",
]


def register_all() -> None:
    register_progress_model("progress_test")(ProgressTestModel)
    register_progress_model("gvl")(GVL)
    register_progress_model("rbm")(RBMModel)
    register_progress_model("rewind")(RBMModel)
    register_preference_model("rlvlmf")(RLVLMF)
    register_progress_model("robodopamine")(RoboDopamine)
    register_progress_model("roboreward")(RoboReward)
    register_progress_model("topreward")(TopReward)
    register_progress_model("vlac")(VLAC)


register_all()
