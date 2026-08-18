from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..core.config import InferConfig
from .base import RemoteError, parse_json_content


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


def _validate_structured_output(value: dict[str, Any], definition: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise RemoteError("Structured output must be an object")
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
        _validate_value(field, value[field], field_schema)


def _validate_value(field: str, value: Any, definition: dict[str, Any]) -> None:
    expected = definition.get("type")
    if expected == "array":
        if not isinstance(value, list):
            raise RemoteError(f"Field '{field}' must be an array")
        if len(value) < definition.get("minItems", 0) or len(value) > definition.get("maxItems", float("inf")):
            raise RemoteError(f"Field '{field}' has invalid length {len(value)}")
        for index, element in enumerate(value):
            _validate_value(f"{field}[{index}]", element, definition.get("items", {}))
    elif expected == "object":
        if not isinstance(value, dict):
            raise RemoteError(f"Field '{field}' must be an object")
        _validate_structured_output(value, definition)
    elif expected in {"number", "integer"}:
        _validate_number(field, value, definition)
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise RemoteError(f"Field '{field}' must be an integer")
    elif expected == "string":
        if not isinstance(value, str) or value not in definition.get("enum", [value]):
            raise RemoteError(f"Field '{field}' has invalid value {value!r}")


def _validate_number(field: str, value: Any, definition: dict[str, Any]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RemoteError(f"Field '{field}' must contain numbers")
    if value < definition.get("minimum", float("-inf")) or value > definition.get("maximum", float("inf")):
        raise RemoteError(f"Field '{field}' is outside the allowed range")


class OpenAIChatClient:
    """Reusable OpenAI-compatible Chat Completions client."""

    def __init__(self, config: InferConfig):
        if not config.base_url:
            raise ValueError("OpenAI-compatible inference requires infer.base_url")
        if not config.model_id:
            raise ValueError("OpenAI-compatible inference requires infer.model_id")
        self.config = config
        self._attempts = 0

    def begin_prediction(self) -> None:
        self._attempts = 0

    def attempts(self) -> int:
        return self._attempts

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.config.base_url is not None
        base = self.config.base_url.rstrip("/")
        suffix = path.lstrip("/")
        if base.endswith("/v1") and suffix.startswith("v1/"):
            suffix = suffix[3:]
        url = f"{base}/{suffix}"
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._attempts += 1
            try:
                response = httpx.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code >= 400:
                    raise RemoteError(
                        f"HTTP {response.status_code}: {response.text[:300]}",
                        raw_response=response.text,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError, RemoteError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(min(30.0, (2**attempt) + random.random()))
        raise RemoteError(
            f"Remote request failed after {self.config.max_retries + 1} attempts: {last_error}",
            raw_response=getattr(last_error, "raw_response", None),
        )

    def completion(
        self,
        messages: list[dict[str, Any]],
        request_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        payload.update(request_options or {})
        return self._post_json("/v1/chat/completions", payload)

    def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        request_options: dict[str, Any] | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        options = {
            **(request_options or {}),
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        last_error = None
        last_response = None
        for parse_attempt in range(self.config.max_retries + 1):
            response = self.completion(messages, options)
            last_response = response
            try:
                message = response["choices"][0]["message"]
                content = next(
                    (
                        message.get(field)
                        for field in ("parsed", "content", "reasoning_content")
                        if message.get(field) is not None
                    ),
                    None,
                )
                parsed = parse_json_content(content)
                _validate_structured_output(parsed, schema["schema"])
                if validator:
                    validator(parsed)
                return parsed, response
            except (KeyError, IndexError, TypeError, RemoteError) as exc:
                last_error = exc
                if parse_attempt >= self.config.max_retries:
                    break
                messages = [
                    {
                        "role": "system",
                        "content": "The previous response was invalid. Return only an object matching the JSON schema.",
                    },
                    *messages,
                ]
        raise RemoteError(
            f"Could not parse a schema-conforming response: {last_error}",
            raw_response=last_response,
        )
