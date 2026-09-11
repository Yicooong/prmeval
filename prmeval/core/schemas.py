"""评估数据结构及记录状态约束。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameworkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Trajectory(FrameworkModel):
    id: str
    task: str
    frames: Any
    data_source: str = "unknown"
    is_robot: bool = False
    is_simulation: bool = False
    quality_label: str | None = None
    partial_success: float | None = None
    preference_group_id: str | None = None
    preference_rank: int | float | None = None
    target_progress: list[float] | None = None
    frame_indices: list[int] | None = None
    num_frames_total: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProgressSample(FrameworkModel):
    sample_id: str
    dataset_name: str
    trajectory: Trajectory


class PreferenceSample(FrameworkModel):
    sample_id: str
    dataset_name: str
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory


EvaluationSample = ProgressSample | PreferenceSample


class FrameReference(FrameworkModel):
    """Portable reference to a materialized frame array in a sample bundle."""

    type: Literal["npz"] = "npz"
    path: str
    key: str = "frames"
    num_frames: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)


class ProgressPrediction(FrameworkModel):
    sample_id: str
    progress: list[Annotated[float, Field(ge=0, le=1)]] = Field(min_length=1)
    model: str


class PreferencePrediction(FrameworkModel):
    sample_id: str
    chosen_probability: float = Field(ge=0, le=1)
    preference: Literal["chosen", "rejected", "tie"]
    model: str


Prediction = ProgressPrediction | PreferencePrediction


class ExecutionInfo(FrameworkModel):
    status: Literal["success", "error"]
    infer_name: str = Field(description="Registered infer implementation used for this execution")
    model: str | None = Field(default=None, description="Attempted model on failure; success uses prediction.model")
    error: str | None = None
    raw_response: Any = Field(default=None, description="Unparsed backend response retained when inference fails")


class EvaluationRecord(FrameworkModel):
    """A sampled request, its optional prediction, and execution information."""

    eval_type: str
    sample: EvaluationSample
    prediction: Prediction | None = None
    execution: ExecutionInfo | None = None

    @model_validator(mode="after")
    def validate_inference_state(self) -> EvaluationRecord:
        if self.execution is None:
            if self.prediction is not None:
                raise ValueError("A sampled record need execution info")
            return self
        if self.execution.status == "error":
            if not self.execution.error or not self.execution.error.strip():
                raise ValueError("An error record requires a non-empty error message")
            return self
        if self.prediction is None:
            raise ValueError("A successful record requires a prediction")
        if self.prediction.sample_id != self.sample.sample_id:
            raise ValueError("Prediction sample_id must match sample.sample_id")
        expected_type = ProgressPrediction if isinstance(self.sample, ProgressSample) else PreferencePrediction
        if not isinstance(self.prediction, expected_type):
            raise ValueError(f"{type(self.sample).__name__} requires {expected_type.__name__}")
        return self
