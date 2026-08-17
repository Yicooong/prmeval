from __future__ import annotations

from typing import Any

import numpy as np

from ..base import vision_content
from ..model import ProgressModel, ProgressResult, RemoteContext
from ..openai import progress_schema


class ProgressTestModel(ProgressModel):
    """Generic OpenAI-compatible VLM for end-to-end remote pipeline tests."""

    supports_local = False
    supports_remote = True

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        raise RuntimeError("progress_test is remote-only")

    @staticmethod
    def _prompt(task_description: str, num_frames: int, options: dict[str, Any]) -> str:
        custom_prompt = options.get("prompt")
        if custom_prompt:
            return (
                str(custom_prompt)
                .replace("{task}", task_description)
                .replace("{num_frames}", str(num_frames))
            )
        return (
            "Estimate the robot's task progress in every image. "
            f"Task: {task_description}. "
            f"Return exactly {num_frames} progress values in image order. "
            "Each value must be between 0 and 1, where 0 means no task progress and 1 means the task is fully "
            "completed. Return the values in the progress field required by the response JSON schema."
        )

    @classmethod
    def remote_compute_progress(
        cls,
        frames_array: np.ndarray,
        task_description: str,
        reference_video_path: str | None,
        remote: RemoteContext,
        options: dict[str, Any],
    ) -> ProgressResult:
        del reference_video_path
        content: list[dict[str, Any]] = [
            {"type": "text", "text": cls._prompt(task_description, len(frames_array), options)}
        ]
        content.extend(vision_content(frames_array))
        parsed, raw = remote.chat(
            [{"role": "user", "content": content}],
            progress_schema(len(frames_array)),
        )
        return ProgressResult(
            values=np.asarray(parsed["progress"], dtype=float),
            raw_response=raw,
        )
