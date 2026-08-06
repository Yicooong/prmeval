from __future__ import annotations

import hashlib
import random

from ..core.config import BaselineConfig
from ..core.registry import BASELINES, register_baseline
from ..core.schemas import PreferencePrediction, PreferenceSample, ProgressPrediction, ProgressSample
from .base import RemoteError
from .images import vision_content
from .openai import OpenAIChatBaseline, PREFERENCE_SCHEMA
from .specialized import SpecializedBaseline


def _prediction(sample, values, config, raw):
    return ProgressPrediction(
        sample_id=sample.sample_id,
        progress=[float(v) for v in values],
        model=config.model_id,
        model_version=config.model_version,
        raw_response=raw,
    )


@register_baseline("progress_test")
class ProgressTestRemote(OpenAIChatBaseline):
    """Generic OpenAI-compatible VLM used to smoke-test progress inference."""

    capabilities = {"progress"}

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


@register_baseline("gvl")
class GVLRemote(OpenAIChatBaseline):
    """GVL's shuffled-frame protocol, transported through chat completions."""
    capabilities = {"progress"}

    def predict(self, sample):
        if not isinstance(sample, ProgressSample):
            return super().predict(sample)
        frames = list(sample.trajectory.frames)
        order = list(range(len(frames)))
        random.Random(int(hashlib.sha256(sample.sample_id.encode()).hexdigest()[:16], 16)).shuffle(order)
        task = sample.trajectory.task
        prompt = (
            f"You are an expert roboticist tasked to predict task completion percentages for frames of a robot "
            f"for the task of {task}. Percentages are between 0 and 100, where 100 is full completion. "
            "The query frames are in random order; judge each frame independently. The first image below is the "
            "initial robot scene and has 0 percent completion. Return one normalized progress value in [0,1] "
            "for every numbered query frame, in the presented order."
        )
        content = [{"type": "text", "text": prompt}, {"type": "text", "text": "Initial robot scene:"}]
        content.extend(vision_content(frames[:1], "Initial frame"))
        content.append({"type": "text", "text": "Randomly ordered query frames:"})
        content.extend(vision_content([frames[i] for i in order]))
        schema = {
            "name": "gvl_progress_prediction", "strict": True,
            "schema": {"type": "object", "properties": {"progress": {
                "type": "array", "minItems": len(frames), "maxItems": len(frames),
                "items": {"type": "number", "minimum": 0, "maximum": 1}}},
                "required": ["progress"], "additionalProperties": False},
        }
        parsed, raw = self._chat([{"role": "user", "content": content}], schema)
        shuffled = [float(v) for v in parsed["progress"]]
        if len(shuffled) != len(frames):
            raise RemoteError(f"GVL returned {len(shuffled)} values for {len(frames)} frames")
        values = [0.0] * len(frames)
        for presented, original in enumerate(order):
            values[original] = shuffled[presented]
        return _prediction(sample, values, self.config, raw)


@register_baseline("rlvlmf")
class RLVLMFRemote(OpenAIChatBaseline):
    capabilities = {"preference"}

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
            sample_id=sample.sample_id, chosen_probability=chosen_probability, preference=preference,
            model=self.config.model_id, model_version=self.config.model_version, raw_response=raw,
        )


@register_baseline("roboreward")
class RoboRewardRemote(OpenAIChatBaseline):
    capabilities = {"progress"}

    def predict(self, sample):
        if not isinstance(sample, ProgressSample):
            raise TypeError("RoboReward only supports progress samples")
        prompt = (
            "Given the task, assign a discrete end-of-episode progress score: 1 no goal-relevant change; "
            "2 minimal progress; 3 partial completion with a major unmet requirement; 4 near completion with "
            "one minor unmet requirement; 5 perfect completion. Judge the final state without a time limit. "
            f"Task: {sample.trajectory.task}."
        )
        content = [{"type": "text", "text": prompt}]
        content.extend(vision_content(sample.trajectory.frames))
        schema = {"name": "roboreward_score", "strict": True, "schema": {
            "type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
            "required": ["score"], "additionalProperties": False}}
        parsed, raw = self._chat([{"role": "user", "content": content}], schema)
        score = int(parsed["score"])
        return _prediction(sample, [(score - 1) / 4] * len(sample.trajectory.frames), self.config, raw)


@register_baseline("robodopamine")
class RoboDopamineRemote(OpenAIChatBaseline):
    capabilities = {"progress"}

    def predict(self, sample):
        if not isinstance(sample, ProgressSample):
            raise TypeError("RoboDopamine only supports progress samples")
        frames = list(sample.trajectory.frames)
        values, raw_responses = [0.0], []
        schema = {"name": "relative_progress", "strict": True, "schema": {
            "type": "object", "properties": {"relative_change_percent": {
                "type": "number", "minimum": -100, "maximum": 100}},
            "required": ["relative_change_percent"], "additionalProperties": False}}
        for before, after in zip(frames, frames[1:]):
            prompt = (
                "You are a rigorous robot task progress evaluator. Compare BEFORE and AFTER for the task "
                f"'{sample.trajectory.task}'. Return a signed relative change percentage: positive when AFTER "
                "moves closer to completion, negative for regression, and zero when unchanged or ambiguous. "
                "Scale improvement relative to what remained at BEFORE and regression relative to progress "
                "already achieved at BEFORE."
            )
            content = [{"type": "text", "text": prompt}]
            content.extend(vision_content([before], "BEFORE"))
            content.extend(vision_content([after], "AFTER"))
            parsed, raw = self._chat([{"role": "user", "content": content}], schema)
            change = float(parsed["relative_change_percent"]) / 100
            previous = values[-1]
            current = previous + (1 - previous) * change if change >= 0 else previous + previous * change
            values.append(max(0.0, min(1.0, current)))
            raw_responses.append(raw)
        return _prediction(sample, values, self.config, raw_responses)


for _name in ("rbm", "rewind"):
    register_baseline(_name)(type(f"{_name.title()}Remote", (SpecializedBaseline,), {}))

register_baseline("topreward")(type(
    "ToprewardRemote", (SpecializedBaseline,),
    {"capabilities": {"progress"}, "progress_prediction_type": "instruction_likelihood"},
))
register_baseline("vlac")(type("VlacRemote", (SpecializedBaseline,), {"capabilities": {"progress"}}))


def create_baseline(config: BaselineConfig):
    baseline_cls = BASELINES.get(config.name)
    if config.transport and baseline_cls.transport != config.transport:
        raise ValueError(
            f"Baseline '{config.name}' uses transport '{baseline_cls.transport}', not '{config.transport}'"
        )
    return baseline_cls(config)
