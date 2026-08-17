from typing import Any, ClassVar

import numpy as np

from ...core.registry import register_infer
from ..base import RemoteError, image_data_url
from ..openai import OpenAIChatInfer
from .common import prediction, progress_sample

ROBODOPAMINE_PROMPT = """
You are a rigorous, impartial vision evaluator for robot task progress. Judge whether the AFTER image set moves
closer to the task objective than the BEFORE image set, using the reference examples only as visual anchors.

Task: {task}

The images are supplied in this exact order:
1. REFERENCE START — Robot Front Image (task just starting)
2. REFERENCE END — Robot Front Image (task fully completed; a neutral placeholder means no goal image is available)
3. BEFORE Robot Front Image
4. BEFORE Robot Left Wrist Image
5. BEFORE Robot Right Wrist Image
6. AFTER Robot Front Image
7. AFTER Robot Left Wrist Image
8. AFTER Robot Right Wrist Image

Compare BEFORE and AFTER and judge whether AFTER moves closer to accomplishing the task. Calibrate REFERENCE START
as just beginning and REFERENCE END as fully completed. AFTER better than BEFORE is positive; regression is negative;
essentially unchanged or genuinely ambiguous is zero. Normalize the change to an integer percentage in [-100, 100].
For improvements, scale relative to what remained from BEFORE to END. For regressions, scale relative to how far
BEFORE had progressed from START.

Use task alignment, completeness, pose, contact, placement, orientation, grasp quality, collisions, and stability.
Use the front view for global geometry and the wrist views for fine-grained grasp/contact evidence. A decisive failure
in any view overrides apparent progress. Ignore lighting, color shifts, clutter, and watermarks.
""".strip()


def _image(frame) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_data_url(frame), "detail": "low"}}


@register_infer("robodopamine")
class RoboDopamineRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def _content(self, task: str, reference_start, reference_end, before, after) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": ROBODOPAMINE_PROMPT.format(task=task)}]
        for label, frame in (
            ("REFERENCE START — Robot Front Image", reference_start),
            ("REFERENCE END — Robot Front Image", reference_end),
            ("BEFORE Robot Front Image", before),
            ("BEFORE Robot Left Wrist Image", before),
            ("BEFORE Robot Right Wrist Image", before),
            ("AFTER Robot Front Image", after),
            ("AFTER Robot Left Wrist Image", after),
            ("AFTER Robot Right Wrist Image", after),
        ):
            content.extend(({"type": "text", "text": f"{label}:"}, _image(frame)))
        return content

    def predict(self, sample):
        sample = progress_sample(sample, "RoboDopamine")
        frames = list(sample.trajectory.frames)
        if not frames:
            raise RemoteError("RoboDopamine received no frames")
        mode = str(self.config.options.get("eval_mode", "incremental")).lower()
        if mode not in {"incremental", "forward", "backward"}:
            raise ValueError("RoboDopamine options.eval_mode must be incremental, forward, or backward")
        interval = self.config.options.get("frame_interval", 1)
        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            raise ValueError("RoboDopamine options.frame_interval must be a positive integer")
        indices = list(range(0, len(frames), interval))
        if indices[-1] != len(frames) - 1:
            indices.append(len(frames) - 1)

        reference_end = np.full((224, 224, 3), 128, dtype=np.uint8)
        schema = {
            "name": "robodopamine_change",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"relative_change_percent": {"type": "number", "minimum": -100, "maximum": 100}},
                "required": ["relative_change_percent"],
                "additionalProperties": False,
            },
        }
        compact_values = [0.0]
        raw_responses = []
        previous = 0.0
        for transition, after_index in enumerate(indices[1:]):
            if mode == "incremental":
                before = frames[indices[transition]]
            elif mode == "forward":
                before = frames[indices[0]]
            else:
                before = reference_end
            content = self._content(sample.trajectory.task, frames[0], reference_end, before, frames[after_index])
            parsed, raw = self._chat([{"role": "user", "content": content}], schema)
            change = float(parsed["relative_change_percent"]) / 100.0
            if mode == "incremental":
                if transition == 0:
                    current = change
                elif change >= 0:
                    current = previous + (1 - previous) * change
                else:
                    current = previous + previous * change
            elif mode == "forward":
                current = change
            else:
                current = 1.0 + change
            compact_values.append(current)
            previous = current
            raw_responses.append(
                {
                    "transition": transition,
                    "before_index": None if mode == "backward" else indices[transition] if mode == "incremental" else 0,
                    "after_index": after_index,
                    "response": raw,
                }
            )
        values = np.clip(np.asarray(compact_values, dtype=float), 0.0, 1.0).tolist()
        if len(values) < len(frames):
            values.extend([values[-1]] * (len(frames) - len(values)))
        elif len(values) > len(frames):
            values = values[: len(frames)]
        return prediction(sample, values, self.config, raw_responses)
