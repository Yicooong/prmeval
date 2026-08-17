"""Stage 2 remote infer transports and built-in model adapters."""

from ..core.config import InferConfig
from ..core.registry import INFERS
from . import baselines as baselines
from .base import RemoteError, RemoteInfer

__all__ = ["RemoteError", "RemoteInfer", "create_infer"]


def create_infer(config: InferConfig):
    infer_cls = INFERS.get(config.name)
    if config.transport and infer_cls.transport != config.transport:
        raise ValueError(f"Infer '{config.name}' uses transport '{infer_cls.transport}', not '{config.transport}'")
    return infer_cls(config)
