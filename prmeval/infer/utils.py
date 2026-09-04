from __future__ import annotations

from typing import Any

from .base import RemoteError


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


def validate_structured_output(value: dict[str, Any], definition: dict[str, Any]) -> None:
    """Validate the subset of JSON Schema used by built-in inference models."""
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
        if field in value:
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
        validate_structured_output(value, definition)
    elif expected in {"number", "integer"}:
        _validate_number(field, value, definition)
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            raise RemoteError(f"Field '{field}' must be an integer")
    elif expected == "string" and (not isinstance(value, str) or value not in definition.get("enum", [value])):
        raise RemoteError(f"Field '{field}' has invalid value {value!r}")


def _validate_number(field: str, value: Any, definition: dict[str, Any]) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RemoteError(f"Field '{field}' must contain numbers")
    if value < definition.get("minimum", float("-inf")) or value > definition.get("maximum", float("inf")):
        raise RemoteError(f"Field '{field}' is outside the allowed range")


def normalize_api_base_url(base_url: str) -> str:
    """Normalize an API URL to end exactly with ``/v1``."""
    normalized = base_url.strip().rstrip("/")
    if "/v1" in normalized:
        normalized = normalized.split("/v1", 1)[0]
    return f"{normalized}/v1"
