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
from transformers import AutoProcessor, Qwen3VLModel
from qwen_vl_utils import process_vision_info
import torch
from .robometer.utils import load_model_from_hf, convert_frames_to_pil_images


logger = logging.getLogger(__name__)



@register_infer("rewind")
@register_infer("robometer")
class RBMModel(Infer):
    """RBM/ReWiND model for baseline evaluation with unified compute methods."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError(f"{config.name} requires infer.model_path")
        self.is_rewind = False
        self._initialize(config.model_path)
        self.max_new_tokens = int(config.options.get("max_new_tokens", 128))

    def _initialize(self, model_path: str):
        """Initialize the RBM/ReWiND model wrapper.

        Args:
            model_path: Path to model checkpoint (HuggingFace repo ID or local path)
                           The config.yaml will be loaded from the checkpoint automatically
        """

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

        logger.info(f"Model type: {'ReWiND' if self.is_rewind else 'RBM'}")


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
        if frames_array is None or frames_array.size == 0:
            return []
        frames_pil = convert_frames_to_pil_images(frames_array)

        prompt = f"The task for the robot is '{task_description}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
        # Build content list
        content_list = [{"type": "text", "text": prompt}]
        for pil in frames_pil:
            content_list.append({"type": "image", "image": pil})

        message = [
            {
                "role": "user",
                "content": content_list,
            }
        ]
        text = self.processor.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=False,
                    add_vision_id=True,
                    enable_thinking=False,
                    fps=1,
                )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
                [message],
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        # Split videos and metadata (video_inputs is list of (video, video_metadata) tuples)
        if video_inputs is not None:
            videos, video_metadatas = zip(*video_inputs, strict=True)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            videos = None
            video_metadatas = None

        inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                do_resize=False,  # qwen-vl-utils already resized
                **video_kwargs,
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

    

        # Faster than torch.no_grad() for inference-only code
        with self.torch.inference_mode(): 
            model_output, _ = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic
                )

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
