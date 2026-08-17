"""Stage 2 local and remote inference construction."""

from ..core.config import InferConfig
from ..core.registry import INFERS
from . import baselines as baselines
from .base import Infer, RemoteError, RemoteInfer
from .model import (
    PreferenceModel,
    PreferenceResult,
    ProgressModel,
    ProgressResult,
    RemoteContext,
    register_preference_model,
    register_progress_model,
)

__all__ = [
    "Infer",
    "PreferenceModel",
    "PreferenceResult",
    "ProgressModel",
    "ProgressResult",
    "RemoteContext",
    "RemoteError",
    "RemoteInfer",
    "create_infer",
    "register_preference_model",
    "register_progress_model",
]


def create_infer(config: InferConfig):
    infer_cls = INFERS.get(config.name)
    if config.mode not in infer_cls.supported_modes:
        supported = ", ".join(sorted(infer_cls.supported_modes))
        raise ValueError(f"Infer '{config.name}' does not support mode '{config.mode}'; supported: {supported}")
    if config.batch_size > 1 and not infer_cls.supports_batch_mode(str(config.mode)):
        raise ValueError(f"Infer '{config.name}' does not support native batching; use infer.batch_size=1")
    if config.max_concurrency > 1 and not infer_cls.thread_safe_mode(str(config.mode)):
        raise ValueError(f"Infer '{config.name}' is not thread-safe; use infer.max_concurrency=1")
    infer = infer_cls(config)
    if config.transport and infer.transport != config.transport:
        raise ValueError(f"Infer '{config.name}' uses transport '{infer.transport}', not '{config.transport}'")
    return infer
