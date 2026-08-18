#!/usr/bin/env python3
"""
RBM and ReWiND model class for baseline evaluation.

This class provides a unified interface for loading RBM/ReWiND models from checkpoints
and computing progress predictions.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer

logger = logging.getLogger(__name__)


@register_infer("rewind")
@register_infer("rbm")
class RBMModel(Infer):
    """RBM/ReWiND model for baseline evaluation with unified compute methods."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError(f"{config.name} requires infer.model_path")
        self._initialize(config.model_path)

    def _initialize(self, model_path: str):
        """Initialize the RBM/ReWiND model wrapper.

        Args:
            model_path: Path to model checkpoint (HuggingFace repo ID or local path)
                           The config.yaml will be loaded from the checkpoint automatically
        """
        try:
            import torch
            from robometer.data.dataset_types import ProgressSample
            from robometer.data.datasets.helpers import create_trajectory_from_dict
            from robometer.evals.eval_server import forward_model
            from robometer.utils.save import load_model_from_hf
            from robometer.utils.setup_utils import setup_batch_collator
        except ImportError as exc:
            raise RuntimeError("RBM/ReWiND local inference requires torch and the robometer model package") from exc

        self.torch = torch
        self.ProgressSample = ProgressSample
        self.create_trajectory_from_dict = create_trajectory_from_dict
        self.forward_model = forward_model
        self.checkpoint_path = model_path

        # Automatically determine device (cuda:0 if available, else cpu)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load model, config, processor, and tokenizer using the helper function
        # This handles loading config.yaml from checkpoint and setting up everything
        logger.info(f"Loading model from checkpoint: {model_path}")
        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=model_path,
            device=device,
        )

        # Store loaded components
        self.exp_config = exp_config
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer

        # Optimize model for inference
        self.model.eval()  # Set to evaluation mode (disables dropout, batch norm updates, etc.)

        # Enable cuDNN benchmarking for faster inference (if using CUDA)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Determine if this is ReWiND or RBM
        self.is_rewind = "rewind" in exp_config.model.base_model_id.lower()
        logger.info(f"Model type: {'ReWiND' if self.is_rewind else 'RBM'}")

        # Create batch collator using the loaded config
        self.batch_collator = setup_batch_collator(
            processor=processor,
            tokenizer=tokenizer,
            cfg=exp_config,
            is_eval=True,
        )

        logger.info(f"Model loaded successfully on device: {self.device}")

    def compute_progress(
        self,
        frames_array: list | Any | np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> list[float]:
        """Compute progress prediction for a trajectory.

        Args:
            frames_array: Array of frames (can be list, tensor, or numpy array)
            task_description: Task description text

        Returns:
            List of progress values (0-1) for each frame
        """
        # Create a ProgressSample from the inputs
        traj_dict = {
            "frames": frames_array,
            "task": task_description,
            "num_frames": len(frames_array) if hasattr(frames_array, "__len__") else frames_array.shape[0],
        }

        trajectory = self.create_trajectory_from_dict(traj_dict)
        sample = self.ProgressSample(trajectory=trajectory)

        # Collate into batch
        batch_inputs = self.batch_collator([sample])

        # Extract progress_inputs from batch_inputs (batch_collator returns nested structure)
        progress_inputs = batch_inputs["progress_inputs"]

        # Move to device
        progress_inputs = {
            k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v for k, v in progress_inputs.items()
        }

        # Forward pass with inference mode for additional optimization
        with self.torch.inference_mode():  # Faster than torch.no_grad() for inference-only code
            model_output, _ = self.forward_model(self.model, progress_inputs, sample_type="progress")

        # Extract progress logits
        progress_logits = model_output.progress_logits
        if progress_logits is None:
            raise ValueError("No progress logits returned from model")

        # Handle different output formats
        if isinstance(progress_logits, dict):
            # RBM format: {"A": tensor, "B": None}
            progress_tensor = progress_logits.get("A")
        else:
            # Direct tensor
            progress_tensor = progress_logits

        if progress_tensor is None:
            raise ValueError("No progress logits in 'A' key")

        # Convert to list of floats
        progress_values = progress_tensor.squeeze().cpu().tolist()

        # Ensure we have the right length
        if isinstance(progress_values, float):
            progress_values = [progress_values]
        return progress_values

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
