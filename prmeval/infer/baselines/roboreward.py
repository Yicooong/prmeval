from typing import ClassVar

from ...core.registry import register_infer
from ..openai import OpenAIChatInfer
from .common import prediction, progress_content, progress_sample


@register_infer("roboreward")
class RoboRewardRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def predict(self, sample):
        sample = progress_sample(sample, "RoboReward")
        prompt = (
            "You are a robot trajectory evaluator. Assess the robot's task progress by examining the video "
            "frames and assign exactly one discrete score. 1 - No Success: no goal-relevant change. "
            "2 - Minimal Progress: movement toward the goal without meaningful completion. "
            "3 - Partial Completion: clear progress but a major requirement remains unmet. "
            "4 - Near Completion: correct region and intent with one minor unmet requirement. "
            "5 - Perfect Completion: the final state satisfies all requirements. Judge the final state without "
            f"a time limit. Task: {sample.trajectory.task}"
        )
        content = progress_content(prompt, sample.trajectory.frames)
        schema = {
            "name": "roboreward_score",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
                "additionalProperties": False,
            },
        }
        parsed, raw = self._chat([{"role": "user", "content": content}], schema)
        score = int(parsed["score"])
        return prediction(sample, [(score - 1) / 4] * len(sample.trajectory.frames), self.config, raw)
