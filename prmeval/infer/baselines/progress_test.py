from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer, vision_content
from ..openai import OpenAIChatClient, progress_schema


@register_infer("progress_test")
class ProgressTestModel(Infer):
    """Generic OpenAI-compatible VLM for end-to-end remote pipeline tests."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        self.client = OpenAIChatClient(config)

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        del reference_video_path
        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._prompt(task_description, len(frames_array), self.config.options)}
        ]
        content.extend(vision_content(frames_array))
        parsed, _ = self.client.chat(
            [{"role": "user", "content": content}],
            progress_schema(len(frames_array)),
        )
        return np.asarray(parsed["progress"], dtype=float)

    def predict(self, sample: EvaluationSample) -> Prediction:
        if not isinstance(sample, ProgressSample):
            raise TypeError(f"{self.config.name} only supports progress samples")
        reference_path = sample.trajectory.metadata.get("reference_video_path")
        values = np.asarray(
            self.compute_progress(
                np.asarray(sample.trajectory.frames),
                sample.trajectory.task,
                str(reference_path) if reference_path else None,
            ),
            dtype=float,
        ).reshape(-1)
        expected = len(sample.trajectory.frames)
        if len(values) != expected:
            raise ValueError(f"Progress length mismatch: expected {expected}, got {len(values)}")
        if not np.isfinite(values).all():
            raise ValueError("Progress values must be finite")
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Progress values must be in [0, 1]")
        return ProgressPrediction(
            sample_id=sample.sample_id,
            progress=values.tolist(),
            model=self.config.model_id or self.config.model_path or self.config.name,
            model_version=self.config.model_version,
        )

    def begin_prediction(self) -> None:
        self.client.begin_prediction()

    def attempts(self) -> int:
        return self.client.attempts()

    @staticmethod
    def _prompt(task_description: str, num_frames: int, options: dict[str, Any]) -> str:
        custom_prompt = options.get("prompt")
        if custom_prompt:
            return str(custom_prompt).replace("{task}", task_description).replace("{num_frames}", str(num_frames))
        return (
            "Estimate the robot's task progress in every image. "
            f"Task: {task_description}. "
            f"Return exactly {num_frames} progress values in image order. "
            "Each value must be between 0 and 1, where 0 means no task progress and 1 means the task is fully "
            "completed. Return the values in the progress field required by the response JSON schema."
        )
