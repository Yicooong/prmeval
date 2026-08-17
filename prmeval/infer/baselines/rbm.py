from ...core.registry import register_infer
from .common import RewardHeadProgressRemote


@register_infer("rbm")
class RBMRemote(RewardHeadProgressRemote):
    baseline_name = "RBM"
