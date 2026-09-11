"""Prediction validation and assembly of inference records."""

from __future__ import annotations

from typing import Any

from .config import InferConfig
from .schemas import (
    EvaluationRecord,
    EvaluationSample,
    Prediction,
    PreferencePrediction,
    PreferenceSample,
    ProgressPrediction,
    ProgressSample,
)


def validate_prediction_for_sample(sample: EvaluationSample, prediction: Prediction) -> None:
    """Check prediction identity, type, and length against the hydrated sample."""
    if prediction.sample_id != sample.sample_id:
        raise ValueError(f"Prediction sample_id mismatch: expected {sample.sample_id}, got {prediction.sample_id}")
    if isinstance(sample, ProgressSample):
        if not isinstance(prediction, ProgressPrediction):
            raise TypeError(f"Progress sample requires ProgressPrediction, got {type(prediction).__name__}")
        expected = len(sample.trajectory.frames)
        actual = len(prediction.progress)
        if actual != expected:
            raise ValueError(f"Progress length mismatch: expected {expected}, got {actual}")
    elif isinstance(sample, PreferenceSample) and not isinstance(prediction, PreferencePrediction):
        raise TypeError(f"Preference sample requires PreferencePrediction, got {type(prediction).__name__}")


def build_inference_record(
    source: EvaluationRecord,
    config: InferConfig,
    prediction: Prediction | None = None,
    error: str | None = None,
    error_response: Any = None,
) -> EvaluationRecord:
    """Attach the original prediction or failure information without converting the sample."""
    failed = error is not None
    return EvaluationRecord(
        eval_type=source.eval_type,
        sample=source.sample,
        prediction=prediction,
        execution={
            "infer_name": config.name,
            "status": "error" if failed else "success",
            "model": (config.model_id or config.model_path or config.name) if failed else None,
            "error": error,
            "raw_response": error_response,
        },
    )
