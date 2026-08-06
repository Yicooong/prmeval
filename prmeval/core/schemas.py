from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrameworkModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Trajectory(FrameworkModel):
    id: str
    task: str
    frames: Any
    data_source: str = "unknown"
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
    split: str | None = Field(default=None, description="Optional dataset split, for example test")
    source: str | None = Field(default=None, description="Optional subset or original source name")


class EvaluationIdentity(FrameworkModel):
    type: str = Field(description="Evaluation type, for example reward_alignment or policy_ranking")
    dataset: DatasetIdentity = Field(description="Dataset dimensions associated with this sample")


class RecordInputItem(FrameworkModel):
    """One media item in a request; preference evaluation can contain chosen and rejected items."""

    role: str = Field(default="trajectory", description="Input role such as trajectory, chosen, or rejected")
    frames: Any = Field(description="FrameReference on disk; temporarily hydrated to an array at runtime")
    frame_indices: list[int] = Field(default_factory=list, description="Sampled indices in the source sequence")
    num_frames_total: int | None = Field(default=None, description="Source frame count before sampling")
    source_id: str | None = Field(default=None, description="Optional source ID for audit and debugging only")
    data: dict[str, Any] = Field(default_factory=dict, description="Non-core extensions for this input item")


class RecordInput(FrameworkModel):
    task: str = Field(description="Natural-language task supplied to the model")
    task_id: str | None = Field(default=None, description="Grouping key for task-level metrics")
    items: list[RecordInputItem] = Field(min_length=1, description="Input items used by this model request")


class ValuePayload(FrameworkModel):
    """Extensible target/prediction payload validated by each metric according to kind."""

    kind: str = Field(description="Payload kind such as progress, rank, or preference")
    values: list[float] | None = Field(default=None, description="Sequence values such as a progress curve")
    value: float | None = Field(default=None, description="Single numeric value such as a ground-truth rank")
    label: str | None = Field(default=None, description="Discrete label such as chosen or successful")
    probability: float | None = Field(default=None, ge=0, le=1, description="Optional probability prediction")
    data: dict[str, Any] = Field(default_factory=dict, description="Extensions specific to this payload kind")


class BaselineIdentity(FrameworkModel):
    name: str = Field(description="Registered baseline name in the evaluation framework")
    model: str = Field(description="Model name used by the remote service")
    version: str | None = Field(default=None, description="Optional model or deployment version")


class ExecutionInfo(FrameworkModel):
    status: Literal["success", "error"] = Field(description="Inference status for this sample")
    attempts: int = Field(default=1, ge=1, description="Number of requests including retries")
    latency_seconds: float | None = Field(default=None, ge=0, description="Total inference latency")
    error: str | None = Field(default=None, description="Error summary when inference fails")


class SourceInfo(FrameworkModel):
    id: str | None = Field(default=None, description="Optional source ID; not a cross-stage primary key")
    data: dict[str, Any] = Field(default_factory=dict, description="Extended source provenance information")


class EvaluationRecord(FrameworkModel):
    """Unified sample/inference record consumed, but not mutated, by metrics."""

    # schema_version controls protocol compatibility; stage tracks per-record inference state.
    schema_version: Literal["bench.record.v1"] = Field(
        default="bench.record.v1", description="Unified record protocol version"
    )
    stage: Literal["sampled", "inferred"] = Field(
        description="sampled awaits inference; inferred contains a successful or failed inference result"
    )
    sample_id: str = Field(description="Unique sample ID preserved across all three stages")

    # evaluation/input/target originate in Stage 1 and must be preserved by Stage 2.
    evaluation: EvaluationIdentity = Field(description="Evaluation type and dataset dimensions")
    input: RecordInput = Field(description="Model input and sampling information")
    target: ValuePayload | None = Field(default=None, description="Metric target; never sent to the remote model")

    # baseline/prediction/execution are populated by Stage 2 and forbidden on sampled records.
    baseline: BaselineIdentity | None = Field(default=None, description="Identity of the predicting baseline/model")
    prediction: ValuePayload | None = Field(default=None, description="Model output normalized by its adapter")
    execution: ExecutionInfo | None = Field(default=None, description="Inference status, retries, latency, and error")

    source: SourceInfo = Field(default_factory=SourceInfo, description="Optional source provenance")
    extensions: dict[str, Any] = Field(
        default_factory=dict, description="Dataset, baseline, or experiment extensions outside the core protocol"
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_record(cls, value):
        """Read legacy flat EvaluationRecord rows while all new writes use bench.record.v1."""
        if not isinstance(value, dict) or "evaluation" in value:
            return value
        if "eval_type" not in value:
            return value
        prediction = value.get("prediction")
        normalized_prediction = None
        baseline_model = value.get("baseline") or "unknown"
        baseline_version = None
        if isinstance(prediction, BaseModel):
            prediction = prediction.model_dump()
        if isinstance(prediction, dict):
            baseline_model = str(prediction.get("model") or baseline_model)
            baseline_version = prediction.get("model_version")
            if "progress" in prediction:
                normalized_prediction = {
                    "kind": "progress", "values": prediction["progress"],
                    "data": {"raw_response": prediction.get("raw_response")},
                }
            elif "preference" in prediction:
                normalized_prediction = {
                    "kind": "preference",
                    "label": prediction["preference"],
                    "probability": prediction.get("chosen_probability"),
                    "data": {"raw_response": prediction.get("raw_response")},
                }
        eval_type = str(value["eval_type"])
        metadata = dict(value.get("metadata") or {})
        target = None
        if value.get("target_progress") is not None:
            target = {"kind": "progress", "values": value["target_progress"]}
        elif eval_type == "quality_preference":
            target = {"kind": "preference", "label": "chosen"}
        elif eval_type == "policy_ranking":
            rank = value.get("partial_success")
            if rank is None:
                rank = value.get("preference_rank")
            if rank is None:
                rank = {
                    "successful": 2.0, "suboptimal": 1.0, "failure": 0.0, "failed": 0.0
                }.get(value.get("quality_label"))
            target = {"kind": "rank", "value": rank, "label": value.get("quality_label")}
        elif eval_type == "confusion_matrix":
            target = {
                "kind": "task_match",
                "value": float(metadata.get("lang_task") == metadata.get("video_task")),
                "data": {
                    "lang_task": metadata.get("lang_task"),
                    "video_task": metadata.get("video_task"),
                },
            }
        source_id = value.get("trajectory_id")
        status = value.get("status", "success")
        return {
            "schema_version": "bench.record.v1",
            "stage": "inferred",
            "sample_id": value["sample_id"],
            "evaluation": {
                "type": eval_type,
                "dataset": {"name": str(value.get("dataset") or "unknown")},
            },
            "input": {
                "task": str(value.get("task") or ""),
                "task_id": str(value.get("task") or ""),
                "items": [{
                    "role": "trajectory",
                    "frames": [],
                    "frame_indices": metadata.get("frame_indices") or [],
                    "source_id": source_id,
                }],
            },
            "target": target,
            "baseline": {
                "name": str(value.get("baseline") or "unknown"),
                "model": baseline_model,
                "version": baseline_version,
            },
            "prediction": normalized_prediction,
            "execution": {
                "status": status,
                "attempts": value.get("attempts", 1),
                "latency_seconds": value.get("latency_seconds"),
                "error": value.get("error"),
            },
            "source": {"id": source_id},
            "extensions": {
                "metadata": metadata,
                "quality_label": value.get("quality_label"),
                "partial_success": value.get("partial_success"),
                "preference_rank": value.get("preference_rank"),
            },
        }

    @model_validator(mode="after")
    def validate_result_state(self):
        # Centralize stage constraints so Optional fields are not unconditionally optional.
        if self.stage == "sampled" and any(
            value is not None for value in (self.baseline, self.prediction, self.execution)
        ):
            raise ValueError("A sampled record cannot contain baseline, prediction, or execution results")
        if self.stage == "inferred" and self.execution is None:
            raise ValueError("An inferred record requires execution information")
        if self.execution and self.execution.status == "success":
            if self.prediction is None or self.baseline is None:
                raise ValueError("A successful inferred record requires baseline and prediction")
        if self.execution and self.execution.status == "error" and not self.execution.error:
            raise ValueError("An error inferred record requires an error message")
        return self


class RunManifest(FrameworkModel):
    run_id: str
    fingerprint: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    status: Literal["running", "completed"] = "running"
    config: dict[str, Any]
    environment: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] | None = None


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
