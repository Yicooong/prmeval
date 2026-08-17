import hashlib
import random
from typing import Any, ClassVar

from ...core.registry import register_infer
from ..base import RemoteError, vision_content
from ..openai import OpenAIChatInfer
from .common import prediction, progress_sample


@register_infer("gvl")
class GVLRemote(OpenAIChatInfer):
    """GVL's shuffled-frame protocol, transported through chat completions."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def predict(self, sample):
        sample = progress_sample(sample, "GVL")
        frames = list(sample.trajectory.frames)
        order = list(range(len(frames)))
        random.Random(int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:16], 16)).shuffle(order)
        task = sample.trajectory.task
        prompt = (
            f"You are an expert roboticist tasked to predict task completion percentages for frames of a robot "
            f"for the task of {task}. Task completion percentages are between 0 and 100, where 100 is full "
            "completion. The query frames are in random order, so judge every frame independently. The initial "
            "robot scene shown first has 0 percent completion. For every numbered query frame, return its frame "
            "number, a short visual description, and task completion percentage in the frames field."
        )
        content = [{"type": "text", "text": prompt}, {"type": "text", "text": "Initial robot scene:"}]
        content.extend(vision_content(frames[:1], "Initial frame"))
        content.append({"type": "text", "text": "Randomly ordered query frames:"})
        content.extend(vision_content([frames[index] for index in order]))
        schema = {
            "name": "gvl_progress_prediction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "frames": {
                        "type": "array",
                        "minItems": len(frames),
                        "maxItems": len(frames),
                        "items": {
                            "type": "object",
                            "properties": {
                                "frame_number": {"type": "integer", "minimum": 1, "maximum": len(frames)},
                                "frame_description": {"type": "string"},
                                "task_completion_percentage": {"type": "number", "minimum": 0, "maximum": 100},
                            },
                            "required": ["frame_number", "frame_description", "task_completion_percentage"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["frames"],
                "additionalProperties": False,
            },
        }
        expected_numbers = set(range(1, len(frames) + 1))

        def validate_frame_numbers(value: dict[str, Any]) -> None:
            numbers = [int(item["frame_number"]) for item in value["frames"]]
            if set(numbers) != expected_numbers or len(numbers) != len(set(numbers)):
                raise RemoteError("GVL response contains missing or duplicate frame numbers")

        parsed, raw = self._chat([{"role": "user", "content": content}], schema, validator=validate_frame_numbers)
        by_number = {int(item["frame_number"]): item for item in parsed["frames"]}
        values = [0.0] * len(frames)
        for presented, original in enumerate(order):
            values[original] = float(by_number[presented + 1]["task_completion_percentage"]) / 100.0
        return prediction(sample, values, self.config, raw)
