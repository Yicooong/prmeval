from __future__ import annotations

import math
import re
from typing import Any, ClassVar

import numpy as np

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer, RemoteError, image_data_url
from ..openai import OpenAIChatClient

SYSTEM_PROMPT = (
    "You are an expert roboticist predicting task progress from an ordered sequence of robot observations. "
    "Reason about the visual change before giving the current progress. Enclose reasoning in <think></think> "
    "and the numeric task-progress percentage in <answer></answer>. The answer may be an integer or decimal. "
    "Example: <think>detailed reasoning</think><answer>25%</answer>"
)

ANSWER_RE = re.compile(
    r"<answer>\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*%?\s*</answer\s*>",
    re.IGNORECASE | re.DOTALL,
)


@register_infer("sole_r1")
class SoleR1(Infer):
    """Remote-only SOLE-R1 progress inference over Stage 1 frame sequences."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        self.client = OpenAIChatClient(config)

    @staticmethod
    def _question(task_description: str, previous_progress: float) -> str:
        return (
            "The three supplied observations are, in order: the first timestep, the previous timestep, and "
            "the current timestep. "
            f"The robot task is: {task_description or 'Complete the task.'} "
            "The task progress at the first timestep is 0%. "
            f"The task progress at the previous timestep is {previous_progress:g}%. "
            "Predict the task progress at the current timestep."
        )

    @staticmethod
    def _image_item(frame: Any) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": image_data_url(frame), "detail": "low"},
        }

    @classmethod
    def _messages(
        cls,
        frames_array: np.ndarray,
        frame_index: int,
        task_description: str,
        previous_progress: float,
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for label, frame in (
            ("First timestep:", frames_array[0]),
            ("Previous timestep:", frames_array[frame_index - 1]),
            ("Current timestep:", frames_array[frame_index]),
        ):
            content.append({"type": "text", "text": label})
            content.append(cls._image_item(frame))
        content.append(
            {
                "type": "text",
                "text": cls._question(task_description, previous_progress),
            }
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

    @staticmethod
    def _completion_text(response: dict[str, Any]) -> str:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RemoteError(
                "SOLE-R1 response is missing choices[0].message",
                raw_response=response,
            ) from exc

        for field in ("content", "reasoning_content"):
            value = message.get(field)
            if isinstance(value, str) and value.strip():
                return value
        raise RemoteError("SOLE-R1 response has no text content", raw_response=response)

    @staticmethod
    def _parse_progress_percent(completion: str, response: dict[str, Any] | None = None) -> float:
        match = ANSWER_RE.search(completion)
        if match is None:
            raise RemoteError(
                "SOLE-R1 response is missing a numeric <answer>...</answer>",
                raw_response=response,
            )
        value = float(match.group(1))
        if not math.isfinite(value):
            raise RemoteError("SOLE-R1 progress must be finite", raw_response=response)
        return value

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        del reference_video_path
        num_frames = len(frames_array)
        if num_frames == 0:
            raise ValueError("SOLE-R1 requires at least one Stage 1 frame")

        percentages = [0.0]
        for frame_index in range(1, num_frames):
            previous_progress = percentages[-1]
            messages = self._messages(
                frames_array,
                frame_index,
                task_description,
                previous_progress,
            )
            response = self.client.completion(messages)
            completion = self._completion_text(response)
            progress = self._parse_progress_percent(completion, response)
            if not 0 <= progress <= 100:
                raise RemoteError(
                    f"SOLE-R1 progress {progress:g}% is outside PRMEval's supported 0%-100% range",
                    raw_response=response,
                )
            percentages.append(progress)
        return np.asarray(percentages, dtype=np.float64) / 100.0

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
