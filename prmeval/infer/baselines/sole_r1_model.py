

import logging
import sys

import numpy as np
from typing import List, ClassVar

from .sole_r1.utils import build_payload
from .sole_r1.main import InferenceServer

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer

logger = logging.getLogger(__name__)


@register_infer("sole_r1")
class SoleR1(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError(f"{config.name} requires infer.model_path")
        
        self.from_zero = bool(config.options.get("from_zero", False))
        self.temperature = float(config.options.get("temperature", 1.0))
        self.client = InferenceServer(config.model_path, config)


    def call(self, payload: dict) -> dict | None:
        return self.client.callback(payload)

    def compute_batch_progress(
        self,
        frames_array: List[np.ndarray],
        task_description: List[str],
        reference_video_path: List[str],
    ) -> list[List[float]]:
        """Compute progress prediction for a trajectory.

        Args:
            frames_array: List of frames 
            task_description: Task description text

        Returns:
            List of List of progress values (0-1) for each frame
        """


        payload = build_payload(
            tasks=task_description,
            front_frames=frames_array,
            wrist_frames=None,
            from_zero=self.from_zero,
            temperature=self.temperature,
        )
        response = self.call(payload)
        if response is None:
            logging.error("No response received from server (timeout or connection error).")
            sys.exit(1)

        if not response.get("success", False):
            logging.error("Server returned an error: %s", response.get("message", "<no message>"))
            sys.exit(1)

        data = response["data"]
        valid_answers = data["valid_answers"]


        return [(0.01*answers).tolist() for answers in valid_answers]

        
    
    def predict(self, samples: List[EvaluationSample]) -> List[Prediction]:
        result = []
        if not isinstance(samples[0], ProgressSample):
            raise TypeError(f"{self.config.name} only supports progress samples")
        reference_paths = [sample.trajectory.metadata.get("reference_video_path") for sample in samples]
        values = self.compute_batch_progress(
                    [np.asarray(sample.trajectory.frames) for sample in samples],
                    [sample.trajectory.task for sample in samples],
                    [str(reference_path) if reference_path else None for reference_path in reference_paths],
        )
          
        if len(values) != len(samples):
            raise ValueError("input length must same as output")
        for value, sample in zip(values, samples):
            expected = len(sample.trajectory.frames)
            if len(value) != expected:
                raise ValueError(f"Progress length mismatch: expected {expected}, got {len(value)}")
            if not np.isfinite(value).all():
                raise ValueError("Progress values must be finite")
            
            result.append(
                ProgressPrediction(
                sample_id=sample.sample_id,
                progress=value,
                model=self.config.model_id or self.config.model_path or self.config.name,
                model_version=self.config.model_version,
                )
            )
        return result 








