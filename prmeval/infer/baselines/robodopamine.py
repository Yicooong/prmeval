#!/usr/bin/env python3
"""Robo-Dopamine (GRM) baseline for progress prediction.

Reference: https://github.com/FlagOpen/Robo-Dopamine
Models:
  - GRM-3B: https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-3B
  - GRM-8B: https://huggingface.co/tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview
Supports single-view (same frames for all three camera inputs) and optional goal image.
When no goal/reference is provided, a blank placeholder image is used per upstream recommendation.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import image_data_url
from ..model import ProgressModel, ProgressResult, RemoteContext

logger = logging.getLogger(__name__)

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
""".strip()

# Known model IDs for config / docs
ROBODOPAMINE_GRM_3B = "tanhuajie2001/Robo-Dopamine-GRM-3B"
ROBODOPAMINE_GRM_8B = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview"


class RoboDopamine(ProgressModel):
    """Robo-Dopamine GRM baseline. Uses single-view frames for all three camera inputs.
    Supports single-view without goal image (blank placeholder used for REFERENCE END).
    """

    supports_remote = True

    def __init__(
        self,
        model_path: str = ROBODOPAMINE_GRM_3B,
        frame_interval: int = 1,
        batch_size: int = 1,
        eval_mode: str = "incremental",
    ):
        try:
            import cv2
            from .rbd_inference import GRMInference
        except ImportError as exc:
            raise RuntimeError(
                "RoboDopamine local inference requires OpenCV and its GRM model dependencies"
            ) from exc

        self.cv2 = cv2
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self._grm = GRMInference(model_path=model_path, max_image_num=8)

    def _make_blank_goal_image(self, out_path: Path, height: int = 224, width: int = 224) -> None:
        """Write a neutral gray placeholder image for 'no goal' single-view setting."""
        blank = np.full((height, width, 3), 128, dtype=np.uint8)
        self.cv2.imwrite(str(out_path), self.cv2.cvtColor(blank, self.cv2.COLOR_RGB2BGR))

    def _goal_image_path(
        self, tmpdir: Path, frames_dir: Path, num_frames: int, reference_video_path: Optional[str]
    ) -> Optional[str]:
        """Resolve goal image path: reference video last frame, or blank placeholder when none."""
        if reference_video_path and os.path.exists(reference_video_path):
            cap = self.cv2.VideoCapture(reference_video_path)
            if cap.isOpened():
                cap.set(
                    self.cv2.CAP_PROP_POS_FRAMES,
                    max(0, int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT)) - 1),
                )
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    goal_path = tmpdir / "goal_from_reference.png"
                    self.cv2.imwrite(str(goal_path), frame, [int(self.cv2.IMWRITE_PNG_COMPRESSION), 3])
                    return str(goal_path)
        # Single-view without goal: use blank placeholder per upstream recommendation
        blank_path = tmpdir / "blank_goal.png"
        self._make_blank_goal_image(blank_path)
        return str(blank_path)

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: Optional[str] = None,
    ) -> np.ndarray:
        if frames_array is None or frames_array.size == 0:
            return np.array([], dtype=np.float64)

        num_frames = frames_array.shape[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            frames_dir = tmpdir_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for i in range(num_frames):
                frame = frames_array[i]
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frame_bgr = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
                self.cv2.imwrite(
                    str(frames_dir / f"frame_{i:06d}.png"),
                    frame_bgr,
                    [int(self.cv2.IMWRITE_PNG_COMPRESSION), 3],
                )

            out_root = tmpdir_path / "out"
            out_root.mkdir(parents=True, exist_ok=True)
            goal_image = self._goal_image_path(
                tmpdir_path, frames_dir, num_frames, reference_video_path
            )
            # run_pipeline: single-view = same dir for all cams; no-goal = blank placeholder
            run_root = self._grm.run_pipeline(
                cam_high_path=str(frames_dir),
                cam_left_path=str(frames_dir),
                cam_right_path=str(frames_dir),
                out_root=str(out_root),
                task=task_description,
                frame_interval=self.frame_interval,
                batch_size=self.batch_size,
                goal_image=goal_image,
                eval_mode=self.eval_mode,
                visualize=False,
            )

            pred_path = Path(run_root) / "pred_vllm.json"
            with open(pred_path, "r", encoding="utf-8") as f:
                results = json.load(f)

        progress_list = [0.0]
        for item in results:
            p = item.get("progress", 0.0)
            if isinstance(p, str) and p == "Error":
                p = progress_list[-1] if progress_list else 0.0
            progress_list.append(float(p))

        progress_arr = np.clip(np.array(progress_list, dtype=np.float64), 0.0, 1.0)
        if len(progress_arr) < num_frames:
            progress_arr = np.pad(
                progress_arr,
                (0, num_frames - len(progress_arr)),
                mode="edge",
            )
        elif len(progress_arr) > num_frames:
            progress_arr = progress_arr[:num_frames]

        return progress_arr

    @staticmethod
    def _remote_content(task: str, reference_start, reference_end, before, after) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": ROBODOPAMINE_PROMPT.format(task=task)}]
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
            content.extend((
                {"type": "text", "text": f"{label}:"},
                {"type": "image_url", "image_url": {"url": image_data_url(frame), "detail": "low"}},
            ))
        return content

    @classmethod
    def remote_compute_progress(
        cls,
        frames_array: np.ndarray,
        task_description: str,
        reference_video_path: str | None,
        remote: RemoteContext,
        options: dict,
    ) -> ProgressResult:
        if len(frames_array) == 0:
            return ProgressResult(np.array([], dtype=np.float64), raw_response=[])
        mode = str(options.get("eval_mode", "incremental")).lower()
        if mode not in {"incremental", "forward", "backward"}:
            raise ValueError("RoboDopamine options.eval_mode must be incremental, forward, or backward")
        interval = options.get("frame_interval", 1)
        if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
            raise ValueError("RoboDopamine options.frame_interval must be a positive integer")
        indices = list(range(0, len(frames_array), interval))
        if indices[-1] != len(frames_array) - 1:
            indices.append(len(frames_array) - 1)

        reference_end = np.full((224, 224, 3), 128, dtype=np.uint8)
        schema = {
            "name": "robodopamine_change",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "relative_change_percent": {"type": "number", "minimum": -100, "maximum": 100}
                },
                "required": ["relative_change_percent"],
                "additionalProperties": False,
            },
        }
        compact_values = [0.0]
        raw_responses = []
        previous = 0.0
        for transition, after_index in enumerate(indices[1:]):
            if mode == "incremental":
                before = frames_array[indices[transition]]
            elif mode == "forward":
                before = frames_array[indices[0]]
            else:
                before = reference_end
            content = cls._remote_content(
                task_description,
                frames_array[0],
                reference_end,
                before,
                frames_array[after_index],
            )
            parsed, raw = remote.chat([{"role": "user", "content": content}], schema)
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
            raw_responses.append({
                "transition": transition,
                "before_index": None if mode == "backward" else indices[transition] if mode == "incremental" else 0,
                "after_index": after_index,
                "response": raw,
            })

        values = np.clip(np.asarray(compact_values, dtype=float), 0.0, 1.0).tolist()
        if len(values) < len(frames_array):
            values.extend([values[-1]] * (len(frames_array) - len(values)))
        elif len(values) > len(frames_array):
            values = values[: len(frames_array)]
        return ProgressResult(values, raw_response=raw_responses)
