from __future__ import annotations

from typing import Any, ClassVar

from ...core.schemas import ProgressPrediction, ProgressSample
from ..base import vision_content
from ..openai import OpenAIChatInfer, progress_schema


def prediction(sample, values, config, raw):
    return ProgressPrediction(
        sample_id=sample.sample_id,
        progress=[float(value) for value in values],
        model=config.model_id,
        model_version=config.model_version,
        raw_response=raw,
    )


def progress_content(prompt: str, frames) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": prompt}]
    content.extend(vision_content(frames))
    return content


def progress_sample(sample, name: str) -> ProgressSample:
    if not isinstance(sample, ProgressSample):
        raise TypeError(f"{name} only supports progress samples")
    return sample


class RewardHeadProgressRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}
    baseline_name = "reward model"

    def predict(self, sample):
        sample = progress_sample(sample, self.baseline_name)
        frames = list(sample.trajectory.frames)
        prompt = (
            f"Run the {self.baseline_name} progress head for the ordered robot trajectory. "
            f"Task: {sample.trajectory.task}. Return exactly one normalized progress value per frame."
        )
        schema = progress_schema(len(frames))
        parsed, raw = self._chat([{"role": "user", "content": progress_content(prompt, frames)}], schema)
        return prediction(sample, parsed["progress"], self.config, raw)
