from ...core.registry import register_infer
from .common import RewardHeadProgressRemote


@register_infer("rewind")
class ReWiNDRemote(RewardHeadProgressRemote):
    baseline_name = "ReWiND"
