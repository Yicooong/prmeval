from __future__ import annotations

import hashlib
import random
from typing import Any, ClassVar

import numpy as np

from ..core.config import InferConfig
from ..core.registry import INFERS, register_infer
from ..core.schemas import PreferencePrediction, PreferenceSample, ProgressPrediction, ProgressSample
from .base import RemoteError, image_data_url, vision_content
from .openai import PREFERENCE_SCHEMA, OpenAIChatInfer, progress_schema


def _prediction(sample, values, config, raw):
    return ProgressPrediction(
        sample_id=sample.sample_id,
        progress=[float(v) for v in values],
        model=config.model_id,
        model_version=config.model_version,
        raw_response=raw,
    )


def _progress_content(prompt: str, frames) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": prompt}]
    content.extend(vision_content(frames))
    return content


def _image(frame) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image_data_url(frame), "detail": "low"}}


def _progress_sample(sample, name: str) -> ProgressSample:
    if not isinstance(sample, ProgressSample):
        raise TypeError(f"{name} only supports progress samples")
    return sample


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


@register_infer("gvl")
class GVLRemote(OpenAIChatInfer):
    """GVL's shuffled-frame protocol, transported through chat completions."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def predict(self, sample):
        sample = _progress_sample(sample, "GVL")
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
        content.extend(vision_content([frames[i] for i in order]))
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
        return _prediction(sample, values, self.config, raw)


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


@register_infer("roboreward")
class RoboRewardRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def predict(self, sample):
        sample = _progress_sample(sample, "RoboReward")
        prompt = (
            "You are a robot trajectory evaluator. Assess the robot's task progress by examining the video "
            "frames and assign exactly one discrete score. 1 - No Success: no goal-relevant change. "
            "2 - Minimal Progress: movement toward the goal without meaningful completion. "
            "3 - Partial Completion: clear progress but a major requirement remains unmet. "
            "4 - Near Completion: correct region and intent with one minor unmet requirement. "
            "5 - Perfect Completion: the final state satisfies all requirements. Judge the final state without "
            f"a time limit. Task: {sample.trajectory.task}"
        )
        content = _progress_content(prompt, sample.trajectory.frames)
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
        return _prediction(sample, [(score - 1) / 4] * len(sample.trajectory.frames), self.config, raw)


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
        sample = _progress_sample(sample, "RoboDopamine")
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
        return _prediction(sample, values, self.config, raw_responses)


@register_infer("topreward")
class TopRewardRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

    @staticmethod
    def _true_logprob(response: dict[str, Any]) -> float:
        try:
            positions = response["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RemoteError("TOPReward response is missing choices[0].logprobs.content") from exc
        for position in positions:
            candidates = [position, *(position.get("top_logprobs") or [])]
            matches = [
                candidate
                for candidate in candidates
                if isinstance(candidate, dict) and str(candidate.get("token", "")).strip().lower() == "true"
            ]
            if matches:
                try:
                    return max(float(candidate["logprob"]) for candidate in matches)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RemoteError("TOPReward True candidate has no numeric logprob") from exc
        raise RemoteError("TOPReward top_logprobs did not contain True or ' True'")

    def predict(self, sample):
        sample = _progress_sample(sample, "TOPReward")
        frames = list(sample.trajectory.frames)
        num_frames = len(frames)
        configured_samples = self.config.options.get("num_prefix_samples", 15)
        if not isinstance(configured_samples, int) or isinstance(configured_samples, bool) or configured_samples < 1:
            raise ValueError("TOPReward options.num_prefix_samples must be a positive integer")
        if num_frames > 2:
            count = min(configured_samples, num_frames)
            prefix_lengths = sorted({int(value) for value in np.linspace(1, num_frames, count, dtype=int)})
        else:
            prefix_lengths = [num_frames]

        rewards: list[float] = []
        raw_responses = []
        task = sample.trajectory.task or "Complete the task."
        prompt = (
            "The supplied frames show a robot manipulation trajectory that completes the following task: "
            f"{task} Decide whether the preceding statement is True or False. Answer with exactly True or False."
        )
        for length in prefix_lengths:
            messages = [{"role": "user", "content": _progress_content(prompt, frames[:length])}]
            last_error = None
            response = None
            for parse_attempt in range(self.config.max_retries + 1):
                response = self._completion(
                    messages,
                    {
                        "temperature": 0,
                        "max_tokens": 1,
                        "logprobs": True,
                        "top_logprobs": 20,
                    },
                )
                try:
                    rewards.append(self._true_logprob(response))
                    break
                except RemoteError as exc:
                    last_error = exc
                    if parse_attempt >= self.config.max_retries:
                        raise RemoteError(f"TOPReward could not obtain True logprob: {last_error}") from exc
            assert response is not None
            raw_responses.append({"prefix_length": length, "true_logprob": rewards[-1], "response": response})

        reward_array = np.asarray(rewards, dtype=float)
        if len(reward_array) == 1 or float(reward_array.max()) == float(reward_array.min()):
            normalized = np.ones_like(reward_array)
        else:
            normalized = (reward_array - reward_array.min()) / (reward_array.max() - reward_array.min())
        values = np.interp(
            np.arange(1, num_frames + 1, dtype=float),
            np.asarray(prefix_lengths, dtype=float),
            normalized,
        )
        return _prediction(sample, np.clip(values, 0.0, 1.0).tolist(), self.config, raw_responses)


@register_infer("vlac")
class VLACRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def predict(self, sample):
        sample = _progress_sample(sample, "VLAC")
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
        parsed, raw = self._chat([{"role": "user", "content": _progress_content(prompt, frames)}], schema)
        values = [float(value) for value in parsed["progress"]]
        if max(values) > 1.0:
            values = [value / 100.0 for value in values]
        if any(not 0 <= value <= 1 for value in values):
            raise RemoteError(f"VLAC returned values outside [0,1]: {values}")
        if len(values) < len(frames):
            values.extend([values[-1]] * (len(frames) - len(values)))
        elif len(values) > len(frames):
            values = values[: len(frames)]
        return _prediction(sample, values, self.config, raw)


class RewardHeadProgressRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}
    baseline_name = "reward model"

    def predict(self, sample):
        sample = _progress_sample(sample, self.baseline_name)
        frames = list(sample.trajectory.frames)
        prompt = (
            f"Run the {self.baseline_name} progress head for the ordered robot trajectory. "
            f"Task: {sample.trajectory.task}. Return exactly one normalized progress value per frame."
        )
        schema = progress_schema(len(frames))
        parsed, raw = self._chat([{"role": "user", "content": _progress_content(prompt, frames)}], schema)
        return _prediction(sample, parsed["progress"], self.config, raw)


@register_infer("rbm")
class RBMRemote(RewardHeadProgressRemote):
    baseline_name = "RBM"


@register_infer("rewind")
class ReWiNDRemote(RewardHeadProgressRemote):
    baseline_name = "ReWiND"


def create_infer(config: InferConfig):
    infer_cls = INFERS.get(config.name)
    if config.transport and infer_cls.transport != config.transport:
        raise ValueError(f"Infer '{config.name}' uses transport '{infer_cls.transport}', not '{config.transport}'")
    return infer_cls(config)
