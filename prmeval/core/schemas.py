from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameworkModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


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
    trajectory: Trajectory
    sample_type: Literal["progress"] = "progress"
    eval_type: str


class PreferenceSample(FrameworkModel):
    sample_id: str
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory
    sample_type: Literal["preference"] = "preference"
    eval_type: str


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
    model_version: str | None = None
    raw_response: Any = None


class PreferencePrediction(FrameworkModel):
    sample_id: str
    chosen_probability: float = Field(ge=0, le=1)
    preference: Literal["chosen", "rejected", "tie"]
    model: str
    model_version: str | None = None
    raw_response: Any = None


Prediction = ProgressPrediction | PreferencePrediction


class DatasetIdentity(FrameworkModel):
    """Dataset identity used for metric slicing, independent of loading details."""

    name: str = Field(description="Canonical dataset name, for example rbm-1m-ood")
    source: str | None = Field(default=None, description="Optional subset or original source name")


class EvaluationIdentity(FrameworkModel):
    type: str = Field(description="Evaluation type, for example progress or policy_ranking")
    dataset: DatasetIdentity = Field(description="Dataset dimensions associated with this sample")


class RecordInputItem(FrameworkModel):
    """One media item in a request; preference evaluation can contain chosen and rejected items."""

    role: str = Field(default="trajectory", description="Input role such as trajectory, chosen, or rejected")
    frames: Any = Field(description="FrameReference on disk; temporarily hydrated to an array at runtime")
    frame_indices: list[int] = Field(default_factory=list, description="Sampled indices in the source sequence")
    source_id: str | None = Field(default=None, description="Optional source ID for audit and debugging only")
    data: dict[str, Any] = Field(default_factory=dict, description="Non-core extensions for this input item")


class RecordInput(FrameworkModel):
    task: str = Field(description="Natural-language task supplied to the model")
    items: list[RecordInputItem] = Field(min_length=1, description="Input items used by this model request")


class ValuePayload(FrameworkModel):
    """Extensible target/prediction payload validated by each metric according to kind."""

    kind: str = Field(description="Payload kind such as progress, rank, or preference")
    values: list[float] | None = Field(default=None, description="Sequence values such as a progress curve")
    value: float | None = Field(default=None, description="Single numeric value such as a ground-truth rank")
    label: str | None = Field(default=None, description="Discrete label such as chosen or successful")
    probability: float | None = Field(default=None, ge=0, le=1, description="Optional probability prediction")


class InferIdentity(FrameworkModel):
    name: str = Field(description="Registered infer name in the evaluation framework")
    model: str = Field(description="Local checkpoint or remote model identity used for inference")
    version: str | None = Field(default=None, description="Optional model or deployment version")


class ExecutionInfo(FrameworkModel):
    status: Literal["success", "error"] = Field(description="Inference status for this sample")
    error: str | None = Field(default=None, description="Error summary when inference fails")
    raw_response: Any = Field(default=None, description="Unparsed backend response retained when inference fails")


class EvaluationRecord(FrameworkModel):
    """Unified sample/inference record consumed, but not mutated, by metrics."""

    # schema_version controls protocol compatibility; execution presence tracks inference state.
    schema_version: Literal["bench.record.v1"] = Field(
        default="bench.record.v1", description="Unified record protocol version"
    )
    sample_id: str = Field(description="Unique sample ID preserved across all three stages")

    # evaluation/input/target originate in Stage 1 and must be preserved by Stage 2.
    evaluation: EvaluationIdentity = Field(description="Evaluation type and dataset dimensions")
    input: RecordInput = Field(description="Model input and sampling information")
    target: ValuePayload | None = Field(default=None, description="Metric target; never sent to the remote model")

    # infer/prediction/execution are populated by Stage 2 and forbidden on sampled records.
    infer: InferIdentity | None = Field(default=None, description="Identity of the predicting infer/model")
    prediction: ValuePayload | None = Field(default=None, description="Model output normalized by its baseline")
    execution: ExecutionInfo | None = Field(default=None, description="Inference status and optional error response")

    @model_validator(mode="after")
    def validate_result_state(self):
        if self.execution is None:
            if self.infer is not None or self.prediction is not None:
                raise ValueError("A sampled record cannot contain infer or prediction results")
            return self
        if self.infer is None:
            raise ValueError("An inferred record requires infer information")
        if self.execution.status == "success" and self.prediction is None:
            raise ValueError("A successful inferred record requires a prediction")
        if self.execution.status == "error":
            if not self.execution.error:
                raise ValueError("An error inferred record requires an error message")
            if self.prediction is not None:
                raise ValueError("An error inferred record cannot contain a prediction")
        return self


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
