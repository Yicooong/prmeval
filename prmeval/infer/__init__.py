"""Stage 2 inference interfaces and built-in registrations."""

from . import baselines as baselines
from .base import Infer, RemoteError

__all__ = [
    "Infer",
    "RemoteError",
]
