from __future__ import annotations

import json
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ..core.config import BaselineConfig
from ..core.schemas import EvaluationSample, Prediction


class RemoteError(RuntimeError):
    pass


class RemoteBaseline(ABC):
    capabilities: set[str] = set()
    transport: str

    def __init__(self, config: BaselineConfig):
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
