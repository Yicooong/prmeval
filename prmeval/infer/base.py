from __future__ import annotations

import base64
import io
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..core.config import InferConfig
from ..core.schemas import EvaluationSample, Prediction


class RemoteError(RuntimeError):
    """Remote inference failure with an optional backend response for diagnostics."""

    def __init__(self, message: str, *, raw_response: Any = None):
        super().__init__(message)
        self.raw_response = raw_response


def image_data_url(frame: Any, quality: int = 85) -> str:
    if isinstance(frame, str) and frame.startswith("data:image/"):
        return frame
    if isinstance(frame, (str, Path)):
        path = Path(frame)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Encoding numpy frames requires the 'data' extra (Pillow)") from exc
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def vision_content(frames: Any, prefix: str = "Frame") -> list[dict[str, Any]]:
    return [
        item
        for index, frame in enumerate(frames)
        for item in (
            {"type": "text", "text": f"{prefix} {index + 1}:"},
            {"type": "image_url", "image_url": {"url": image_data_url(frame), "detail": "low"}},
        )
    ]


class Infer(ABC):
    capabilities: ClassVar[set[str]] = set()

    def __init__(self, config: InferConfig):
        self.config = config

    def begin_prediction(self) -> None:
        return None

    def attempts(self) -> int:
        return 1

    @abstractmethod
    def predict(self, sample: EvaluationSample) -> Prediction:
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        return {
            "model": model_identity(self.config),
            "model_version": self.config.model_version,
        }


def model_identity(config: InferConfig) -> str:
    return config.model_id or config.model_path or config.name


def parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise RemoteError(f"Expected string or object response content, got {type(content)!r}")

    text = content.strip()
    try:
        value = json.loads(text)
        if not isinstance(value, dict):
            raise RemoteError(f"Expected a JSON object response, got {type(value)!r}")
        return value
    except json.JSONDecodeError as strict_error:
        # A few OpenAI-compatible servers occasionally wrap otherwise valid
        # structured output in a Markdown fence or emit one duplicated opening
        # brace. Recover only a complete JSON object; schema validation still
        # runs in OpenAIChatClient.chat after this function returns.
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()

        candidates = [text]
        malformed_prefix = re.match(r'^\{\s*"\s*\{(?=")', text)
        if malformed_prefix:
            candidates.append("{" + text[malformed_prefix.end() :])

        decoder = json.JSONDecoder()
        for candidate in candidates:
            for start, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    value, end = decoder.raw_decode(candidate, start)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and not candidate[end:].strip().strip("}"):
                    return value
        raise RemoteError(f"Response did not contain valid JSON: {content[:300]}") from strict_error
