from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from ..core.schemas import EvaluationSample, PreferencePrediction, PreferenceSample, ProgressPrediction, ProgressSample
from .base import RemoteError, RemoteInfer, image_data_url


class RemoteTrajectory(BaseModel):
    id: str
    frames: list[dict[str, Any]]


class SpecializedRequest(BaseModel):
    model: str
    request_id: str
    prediction_type: Literal["progress", "preference", "instruction_likelihood"]
    task: str
    trajectories: list[RemoteTrajectory]
    options: dict[str, Any] = Field(default_factory=dict)


class SpecializedPrediction(BaseModel):
    trajectory_id: str | None = None
    progress: list[Annotated[float, Field(ge=0, le=1)]] | None = None
    preference_probability: float | None = Field(default=None, ge=0, le=1)


class SpecializedResponse(BaseModel):
    id: str
    model: str
    model_version: str | None = None
    predictions: list[SpecializedPrediction]
    usage: dict[str, Any] = Field(default_factory=dict)


class SpecializedInfer(RemoteInfer):
    transport = "specialized"
    capabilities = {"progress", "preference"}
    progress_prediction_type: Literal["progress", "instruction_likelihood"] = "progress"

    @staticmethod
    def _trajectory(traj) -> RemoteTrajectory:
        return RemoteTrajectory(
            id=traj.id,
            frames=[{"type": "image_url", "image_url": {"url": image_data_url(frame)}} for frame in traj.frames],
        )

    def predict(self, sample: EvaluationSample):
        if isinstance(sample, ProgressSample):
            request = SpecializedRequest(
                model=self.config.model_id,
                request_id=sample.sample_id,
                prediction_type=self.progress_prediction_type,
                task=sample.trajectory.task,
                trajectories=[self._trajectory(sample.trajectory)],
                options={"return_per_frame": True, **self.config.options},
            )
        elif isinstance(sample, PreferenceSample):
            request = SpecializedRequest(
                model=self.config.model_id,
                request_id=sample.sample_id,
                prediction_type="preference",
                task=sample.chosen_trajectory.task,
                trajectories=[self._trajectory(sample.chosen_trajectory), self._trajectory(sample.rejected_trajectory)],
                options=self.config.options,
            )
        else:
            raise TypeError(f"Unsupported sample: {type(sample)!r}")
        raw = self._post_json("/v1/evaluations", request.model_dump())
        response = SpecializedResponse.model_validate(raw)
        if response.id != sample.sample_id:
            raise RemoteError(f"Response id mismatch: expected {sample.sample_id}, got {response.id}")
        if len(response.predictions) != 1:
            raise RemoteError("Specialized endpoint must return exactly one prediction per request")
        prediction = response.predictions[0]
        if isinstance(sample, ProgressSample):
            if not prediction.progress:
                raise RemoteError("Specialized endpoint returned no progress")
            return ProgressPrediction(
                sample_id=sample.sample_id,
                progress=prediction.progress,
                model=response.model,
                model_version=response.model_version,
                raw_response=raw,
            )
        probability = prediction.preference_probability
        if probability is None or not 0 <= probability <= 1:
            raise RemoteError("Invalid preference_probability")
        label = "chosen" if probability > 0.5 else "rejected" if probability < 0.5 else "tie"
        return PreferencePrediction(
            sample_id=sample.sample_id,
            chosen_probability=probability,
            preference=label,
            model=response.model,
            model_version=response.model_version,
            raw_response=raw,
        )
