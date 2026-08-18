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
import os
import tempfile
from pathlib import Path
from typing import ClassVar

import numpy as np

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer

# Known model IDs for config / docs
ROBODOPAMINE_GRM_3B = "tanhuajie2001/Robo-Dopamine-GRM-3B"
ROBODOPAMINE_GRM_8B = "tanhuajie2001/Robo-Dopamine-GRM-2.0-8B-Preview"


@register_infer("robodopamine")
class RoboDopamine(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}
    """Robo-Dopamine GRM baseline. Uses single-view frames for all three camera inputs.
    Supports single-view without goal image (blank placeholder used for REFERENCE END).
    """

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("robodopamine requires infer.model_path")
        options = config.options
        self._initialize(
            model_path=config.model_path,
            frame_interval=int(options.get("frame_interval", 1)),
            batch_size=int(options.get("micro_batch_size", 1)),
            eval_mode=str(options.get("eval_mode", "incremental")),
        )

    def _initialize(
        self,
        model_path: str = ROBODOPAMINE_GRM_3B,
        frame_interval: int = 1,
        batch_size: int = 1,
        eval_mode: str = "incremental",
    ):
        global cv2, GRMInference
        try:
            import cv2

            from .rbd_inference import GRMInference
        except ImportError as exc:
            raise RuntimeError("RoboDopamine requires OpenCV and its model dependencies") from exc

        self.model_path = model_path
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.eval_mode = eval_mode
        self._grm = GRMInference(model_path=model_path, max_image_num=8)

    def _make_blank_goal_image(self, out_path: Path, height: int = 224, width: int = 224) -> None:
        """Write a neutral gray placeholder image for 'no goal' single-view setting."""
        blank = np.full((height, width, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(out_path), cv2.cvtColor(blank, cv2.COLOR_RGB2BGR))

    def _goal_image_path(
        self, tmpdir: Path, frames_dir: Path, num_frames: int, reference_video_path: str | None
    ) -> str | None:
        """Resolve goal image path: reference video last frame, or blank placeholder when none."""
        if reference_video_path and os.path.exists(reference_video_path):
            cap = cv2.VideoCapture(reference_video_path)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    goal_path = tmpdir / "goal_from_reference.png"
                    cv2.imwrite(str(goal_path), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    return str(goal_path)
        # Single-view without goal: use blank placeholder per upstream recommendation
        blank_path = tmpdir / "blank_goal.png"
        self._make_blank_goal_image(blank_path)
        return str(blank_path)

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
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
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(frames_dir / f"frame_{i:06d}.png"),
                    frame_bgr,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
                )

            out_root = tmpdir_path / "out"
            out_root.mkdir(parents=True, exist_ok=True)
            goal_image = self._goal_image_path(tmpdir_path, frames_dir, num_frames, reference_video_path)
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
            with open(pred_path, encoding="utf-8") as f:
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

    def predict(self, sample: EvaluationSample) -> Prediction:
        if not isinstance(sample, ProgressSample):
            raise TypeError(f"{self.config.name} only supports progress samples")
        reference_path = sample.trajectory.metadata.get("reference_video_path")
        values = np.asarray(
            self.compute_progress(
                np.asarray(sample.trajectory.frames),
                sample.trajectory.task,
                str(reference_path) if reference_path else None,
            ),
            dtype=float,
        ).reshape(-1)
        expected = len(sample.trajectory.frames)
        if len(values) != expected:
            raise ValueError(f"Progress length mismatch: expected {expected}, got {len(values)}")
        if not np.isfinite(values).all():
            raise ValueError("Progress values must be finite")
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Progress values must be in [0, 1]")
        return ProgressPrediction(
            sample_id=sample.sample_id,
            progress=values.tolist(),
            model=self.config.model_id or self.config.model_path or self.config.name,
            model_version=self.config.model_version,
        )
