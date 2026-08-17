import hashlib
from typing import ClassVar

from ...core.registry import register_infer
from ...core.schemas import PreferencePrediction, PreferenceSample
from ..base import vision_content
from ..openai import PREFERENCE_SCHEMA, OpenAIChatInfer


@register_infer("rlvlmf")
class RLVLMFRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"preference"}

    def predict(self, sample):
        if not isinstance(sample, PreferenceSample):
            raise TypeError("RL-VLM-F only supports preference samples")
        chosen_first = bool(int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:2], 16) % 2)
        chosen = sample.chosen_trajectory.frames[-1]
        rejected = sample.rejected_trajectory.frames[-1]
        frames = [chosen, rejected] if chosen_first else [rejected, chosen]
        task = sample.chosen_trajectory.task
        prompt = (
            "Each image is the final frame of a robot trajectory. Think causally and use image comparison to "
            "distinguish the robot base from the end effector. Describe Image A and Image B, then decide where "
            f"the goal is better achieved. The goal is {task}. Return A, B, or tie; use tie only when there is "
            "no discernible progress or difference. Also return the probability that Image A is better."
        )
        content = [{"type": "text", "text": prompt}]
        content.extend(vision_content(frames, "Image"))
        parsed, raw = self._chat([{"role": "user", "content": content}], PREFERENCE_SCHEMA)
        label = parsed["preference"]
        probability_a = float(parsed["probability_a"])
        if label == "tie":
            preference, chosen_probability = "tie", 0.5
        elif (label == "A") == chosen_first:
            preference, chosen_probability = "chosen", probability_a if chosen_first else 1 - probability_a
        else:
            preference, chosen_probability = "rejected", probability_a if chosen_first else 1 - probability_a
        return PreferencePrediction(
            sample_id=sample.sample_id,
            chosen_probability=chosen_probability,
            preference=preference,
            model=self.config.model_id,
            model_version=self.config.model_version,
            raw_response=raw,
        )
