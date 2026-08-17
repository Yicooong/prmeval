from typing import ClassVar

from ...core.registry import register_infer
from ..base import RemoteError
from ..openai import OpenAIChatInfer
from .common import prediction, progress_content, progress_sample


@register_infer("vlac")
class VLACRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def predict(self, sample):
        sample = progress_sample(sample, "VLAC")
        frames = list(sample.trajectory.frames)
        prompt = (
            "Act as the VLAC pair-wise trajectory critic. Evaluate the ordered robot trajectory for the task "
            f"'{sample.trajectory.task}' and return the critic progress values in frame order. Values may use "
            "either the [0,1] rich-value scale or percentage scale [0,100]."
        )
        schema = {
            "name": "vlac_progress",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "progress": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "number", "minimum": 0, "maximum": 100},
                    }
                },
                "required": ["progress"],
                "additionalProperties": False,
            },
        }
        parsed, raw = self._chat([{"role": "user", "content": progress_content(prompt, frames)}], schema)
        values = [float(value) for value in parsed["progress"]]
        if max(values) > 1.0:
            values = [value / 100.0 for value in values]
        if any(not 0 <= value <= 1 for value in values):
            raise RemoteError(f"VLAC returned values outside [0,1]: {values}")
        if len(values) < len(frames):
            values.extend([values[-1]] * (len(frames) - len(values)))
        elif len(values) > len(frames):
            values = values[: len(frames)]
        return prediction(sample, values, self.config, raw)
