from __future__ import annotations

from typing import Any

from ..core.schemas import EvaluationSample, PreferencePrediction, PreferenceSample, ProgressPrediction, ProgressSample
from .base import RemoteError, RemoteInfer, parse_json_content, vision_content


PROGRESS_SCHEMA = {
    "name": "progress_prediction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "progress": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}}
        },
        "required": ["progress"],
        "additionalProperties": False,
    },
}


def progress_schema(num_frames: int) -> dict[str, Any]:
    """Build a strict progress schema whose output length matches the input frames."""
    return {
        "name": "progress_prediction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "progress": {
                    "type": "array",
                    "minItems": num_frames,
                    "maxItems": num_frames,
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                }
            },
            "required": ["progress"],
            "additionalProperties": False,
        },
    }

PREFERENCE_SCHEMA = {
    "name": "preference_prediction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "preference": {"type": "string", "enum": ["A", "B", "tie"]},
            "probability_a": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["preference", "probability_a"],
        "additionalProperties": False,
    },
}


def _validate_structured_output(value: dict[str, Any], definition: dict[str, Any]) -> None:
    for field in definition.get("required", []):
        if field not in value:
            raise RemoteError(f"Structured output is missing required field '{field}'")
    if definition.get("additionalProperties") is False:
        unexpected = set(value) - set(definition.get("properties", {}))
        if unexpected:
            raise RemoteError(f"Structured output has unexpected fields: {sorted(unexpected)}")
    for field, field_schema in definition.get("properties", {}).items():
        if field not in value:
            continue
        item = value[field]
        expected = field_schema.get("type")
        if expected == "array":
            if not isinstance(item, list):
                raise RemoteError(f"Field '{field}' must be an array")
            if len(item) < field_schema.get("minItems", 0) or len(item) > field_schema.get("maxItems", float("inf")):
                raise RemoteError(f"Field '{field}' has invalid length {len(item)}")
            for element in item:
                _validate_number(field, element, field_schema.get("items", {}))
        elif expected in {"number", "integer"}:
            _validate_number(field, item, field_schema)
            if expected == "integer" and (not isinstance(item, int) or isinstance(item, bool)):
                raise RemoteError(f"Field '{field}' must be an integer")
        elif expected == "string":
            if not isinstance(item, str) or item not in field_schema.get("enum", [item]):
                raise RemoteError(f"Field '{field}' has invalid value {item!r}")


def _validate_number(field: str, value: Any, definition: dict[str, Any]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RemoteError(f"Field '{field}' must contain numbers")
    if value < definition.get("minimum", float("-inf")) or value > definition.get("maximum", float("inf")):
        raise RemoteError(f"Field '{field}' is outside the allowed range")


class OpenAIChatInfer(RemoteInfer):
    transport = "openai_chat"
    capabilities = {"progress", "preference"}

    def progress_prompt(self, sample: ProgressSample) -> str:
        task = sample.trajectory.task
        return (
            f"You are an expert roboticist. The task is: {task}. Predict task progress for every supplied frame. "
            "Return one float per frame between 0 and 1, where 0 is the starting state and 1 is complete. "
            "If the robot is not performing the task, return 0."
        )

    def preference_prompt(self, sample: PreferenceSample) -> str:
        task = sample.chosen_trajectory.task
        return (
            f"Compare trajectories A and B for the task: {task}. Decide which trajectory makes more progress. "
            "Return A, B, or tie and the probability that A is better."
        )

    def _chat(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        last_error = None
        for parse_attempt in range(self.config.max_retries + 1):
            response = self._post_json("/v1/chat/completions", payload)
            try:
                message = response["choices"][0]["message"]
                content = message.get("parsed", message.get("content"))
                parsed = parse_json_content(content)
                _validate_structured_output(parsed, schema["schema"])
                return parsed, response
            except (KeyError, IndexError, TypeError, RemoteError) as exc:
                last_error = exc
                if parse_attempt >= self.config.max_retries:
                    break
                payload["messages"] = [{
                    "role": "system",
                    "content": "The previous response was invalid. Return only an object matching the JSON schema.",
                }, *messages]
        raise RemoteError(f"Could not parse a schema-conforming response: {last_error}")

    def predict(self, sample: EvaluationSample):
        if isinstance(sample, ProgressSample):
            content = [{"type": "text", "text": self.progress_prompt(sample)}]
            content.extend(vision_content(sample.trajectory.frames))
            schema = progress_schema(len(sample.trajectory.frames))
            parsed, raw = self._chat([{"role": "user", "content": content}], schema)
            values = [float(v) for v in parsed["progress"]]
            if not values or any(not 0 <= v <= 1 for v in values):
                raise RemoteError(f"Invalid progress output: {values}")
            return ProgressPrediction(
                sample_id=sample.sample_id,
                progress=values,
                model=self.config.model_id,
                model_version=self.config.model_version,
                raw_response=raw,
            )
        if isinstance(sample, PreferenceSample):
            content = [{"type": "text", "text": self.preference_prompt(sample)}, {"type": "text", "text": "Trajectory A:"}]
            content.extend(vision_content(sample.chosen_trajectory.frames, "A frame"))
            content.append({"type": "text", "text": "Trajectory B:"})
            content.extend(vision_content(sample.rejected_trajectory.frames, "B frame"))
            parsed, raw = self._chat([{"role": "user", "content": content}], PREFERENCE_SCHEMA)
            label = {"A": "chosen", "B": "rejected", "tie": "tie"}[parsed["preference"]]
            return PreferencePrediction(
                sample_id=sample.sample_id,
                chosen_probability=float(parsed["probability_a"]),
                preference=label,
                model=self.config.model_id,
                model_version=self.config.model_version,
                raw_response=raw,
            )
        raise TypeError(f"Unsupported sample: {type(sample)!r}")
