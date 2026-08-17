from typing import ClassVar

from ...core.registry import register_infer
from ...core.schemas import ProgressSample
from ..openai import OpenAIChatInfer


@register_infer("progress_test")
class ProgressTestRemote(OpenAIChatInfer):
    """Generic OpenAI-compatible VLM used to smoke-test progress inference."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def progress_prompt(self, sample: ProgressSample) -> str:
        custom_prompt = self.config.options.get("prompt")
        if custom_prompt:
            return (
                str(custom_prompt)
                .replace("{task}", sample.trajectory.task)
                .replace("{num_frames}", str(len(sample.trajectory.frames)))
            )
        return (
            "Estimate the robot's task progress in every image. "
            f"Task: {sample.trajectory.task}. "
            f"Return exactly {len(sample.trajectory.frames)} progress values in image order. "
            "Each value must be between 0 and 1, where 0 means no task progress and 1 means the task is fully "
            "completed. Return the values in the progress field required by the response JSON schema."
        )

    def predict(self, sample):
        if not isinstance(sample, ProgressSample):
            raise TypeError("progress_test only supports progress samples")
        return super().predict(sample)
