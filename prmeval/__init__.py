"""PRMEval local and remote evaluation framework."""

from .core.config import EvalConfig
from .core.runner import Evaluator

__all__ = [
    "EvalConfig",
    "Evaluator",
]
