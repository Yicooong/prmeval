from __future__ import annotations

import base64
import io
import json
import random
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import httpx
import numpy as np

from ..core.config import InferConfig
from ..core.schemas import EvaluationSample, Prediction


class RemoteError(RuntimeError):
    pass


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


class RemoteInfer(ABC):
    capabilities: ClassVar[set[str]] = set()
    transport: str

    def __init__(self, config: InferConfig):
        self.config = config
        self._local = threading.local()

    def begin_prediction(self) -> None:
        self._local.attempts = 0

    def attempts(self) -> int:
        return int(getattr(self._local, "attempts", 0))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.config.headers}
        key = self.config.api_key
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = self.config.base_url.rstrip("/")
        suffix = path.lstrip("/")
        if base.endswith("/v1") and suffix.startswith("v1/"):
            suffix = suffix[3:]
        url = f"{base}/{suffix}"
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self._local.attempts = self.attempts() + 1
            try:
                response = httpx.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.config.timeout_seconds,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RemoteError(f"retryable HTTP {response.status_code}: {response.text[:300]}")
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, ValueError, RemoteError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                delay = min(30.0, (2**attempt) + random.random())
                time.sleep(delay)
        raise RemoteError(f"Remote request failed after {self.config.max_retries + 1} attempts: {last_error}")

    @abstractmethod
    def predict(self, sample: EvaluationSample) -> Prediction:
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        return {
            "model": self.config.model_id,
            "model_version": self.config.model_version,
            "base_url": self.config.base_url,
            "transport": self.transport,
        }


def parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise RemoteError(f"Expected string or object response content, got {type(content)!r}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RemoteError(f"Response did not contain valid JSON: {content[:300]}") from exc
