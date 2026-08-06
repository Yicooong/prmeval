"""PRMEval remote evaluation framework."""

from .core.config import EvalConfig
from .core.schemas import (
    BaselineIdentity,
    DatasetIdentity,
    EvaluationRecord,
    EvaluationIdentity,
    ExecutionInfo,
    FrameReference,
    PreferencePrediction,
    PreferenceSample,
    ProgressPrediction,
    ProgressSample,
    RecordInput,
    RecordInputItem,
    RunManifest,
    SourceInfo,
    Trajectory,
    ValuePayload,
)
from .evaluation.runner import Evaluator

__all__ = [
    "EvalConfig",
    "Evaluator",
    "BaselineIdentity",
    "DatasetIdentity",
    "EvaluationRecord",
    "EvaluationIdentity",
    "ExecutionInfo",
    "FrameReference",
    "PreferencePrediction",
    "PreferenceSample",
    "ProgressPrediction",
    "ProgressSample",
    "RecordInput",
    "RecordInputItem",
    "RunManifest",
    "SourceInfo",
    "Trajectory",
    "ValuePayload",
]
