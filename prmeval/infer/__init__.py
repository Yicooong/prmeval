"""Stage 2 remote infer transports and built-in model adapters."""

from .adapters import create_infer
from .base import RemoteInfer, RemoteError

__all__ = ["RemoteInfer", "RemoteError", "create_infer"]
