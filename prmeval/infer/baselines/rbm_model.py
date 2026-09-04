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
import torch
from qwen_vl_utils import process_vision_info

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer
from .robometer.utils import _resize_pil, convert_bins_to_continuous, convert_frames_to_pil_images, load_model_from_hf

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
        self.base_model_path = config.options.get("base_model_path", None)
        if self.base_model_path is None:
            raise ValueError(f"{config.name} requires base_model_path in options")
        self._initialize(config.model_path)
        self.max_new_tokens = int(config.options.get("max_new_tokens", 128))
        self.use_multi_image = bool(config.options.get("use_multi_image", True))
        self.use_per_frame_progress_token = bool(config.options.get("use_per_frame_progress_token", True))

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
            base_model_path=self.base_model_path,
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

    def _add_vision_content_to_list(
        self, content_list: list[dict], frames_or_video: list | str, content_extras: dict
    ) -> None:
        """
        Add vision content (images or video) to a content list.

        Args:
            content_list: List to append vision content to
            frames_or_video: Either list of PIL Images (if use_multi_image) or video file path (str)
            content_extras: Dictionary with additional content parameters
        """
        if self.use_multi_image:
            # Add each image as a separate entry
            for img in frames_or_video:
                content_list.append(
                    {
                        "type": "image",
                        "image": img,
                        **content_extras,
                    }
                )
                # Add per-frame progress token after each frame if enabled
                if self.use_per_frame_progress_token:
                    content_list.append({"type": "text", "text": "<|prog_token|>"})
        else:
            # Add video entry
            content_list.append(
                {
                    "type": "video",
                    "video": frames_or_video,
                    "sample_fps": 1.0,
                    **content_extras,
                }
            )

    def _prepare_frames_for_conversation(self, frames: list, prefix: str = "tmp") -> tuple[list | str, dict]:
        """
        Prepare frames for conversation based on use_multi_image flag.

        Args:
            frames: List of PIL Images
            prefix: Prefix for temporary video file (if needed)

        Returns:
            tuple: (video_field, content_extras)
                - video_field: Either list of PIL Images (if use_multi_image) or video file path (str)
                - content_extras: Dictionary with resized_height/width or empty dict
        """
        if self.use_multi_image:
            # # Use images directly - return list of PIL Images
            # if self.resized_height is not None and self.resized_width is not None:
            #     content_extras = {
            #         "resized_height": self.resized_height,
            #         "resized_width": self.resized_width,
            #     }
            # else:
            #     frames = [_resize_pil(frame) for frame in frames]
            #     content_extras = {}
            content_extras = {}
            return frames, content_extras
        elif "Qwen" in self.base_model_id or "Molmo" in self.base_model_id:
            # Qwen and Molmo accept list of PIL Images directly
            if self.resized_height is not None and self.resized_width is not None:
                content_extras = {
                    "resized_height": self.resized_height,
                    "resized_width": self.resized_width,
                }
            else:
                frames = [_resize_pil(frame) for frame in frames]
                content_extras = {}
            return frames, content_extras
        # elif "SmolVLM" in self.base_model_id:
        #     frames = [_resize_pil(frame) for frame in frames]
        #     # Convert to video file for SmolVLM
        #     unique_id = uuid.uuid4().hex
        #     tmp = Path(tempfile.gettempdir()) / f"{prefix}_{unique_id}.mp4"
        #     write_mp4(frames, tmp, fps=1)
        #     return str(tmp), {}
        else:
            # Default: return frames as-is
            return frames, {}

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

        video_field, content_extras = self._prepare_frames_for_conversation(frames_pil, prefix="tmp_progress")

        prompt = f"The task for the robot is '{task_description}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
        # Build content list
        content_list = [{"type": "text", "text": prompt}]
        self._add_vision_content_to_list(content_list, video_field, content_extras)

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
        with torch.inference_mode():
            model_output, _ = self.model(
                **inputs,
                sample_type="progress",
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
        batch_size = progress_tensor.shape[0]
        results = []

        for i in range(batch_size):
            # Extract progress values for this sample
            if progress_tensor.ndim == 2:
                if progress_tensor.shape[1] == 1:
                    # Single value per sample
                    progress_values = [float(progress_tensor[i, 0].item())]
                else:
                    # Multiple values per sample
                    progress_values = progress_tensor[i].cpu().tolist()
            elif progress_tensor.ndim == 3:
                # Multiple values per sample, discrete multiple bins, convert to continuous
                progress_values = convert_bins_to_continuous(progress_tensor[i]).cpu().tolist()
            else:
                # Unexpected shape
                raise ValueError(f"Unexpected progress_tensor shape: {progress_tensor.shape}")

            # Ensure we have the right length
            if isinstance(progress_values, float):
                progress_values = [progress_values]

            results.append(progress_values)

        return results[0]

    def compute_batch_progress(
        self,
        frames_array: list[np.ndarray],
        task_description: list[str],
        reference_video_path: list[str],
    ) -> list[list[float]]:
        """Compute progress prediction for a trajectory.

        Args:
            frames_array: List of frames
            task_description: Task description text

        Returns:
            List of List of progress values (0-1) for each frame
        """
        if frames_array is None:
            return [[]]

        all_messages = []

        for frame_array in frames_array:
            frames_pil = convert_frames_to_pil_images(frame_array)

            video_field, content_extras = self._prepare_frames_for_conversation(frames_pil, prefix="tmp_progress")

            prompt = f"The task for the robot is '{task_description}'. Given the trajectory video, predict the task progress at each frame, how far along the robot is towards completing the task, a float between 0 and 1, where 0 is the starting state and 1 is when the task is completed. If the robot is not performing the same task, predict 0 progress."
            # Build content list
            content_list = [{"type": "text", "text": prompt}]
            self._add_vision_content_to_list(content_list, video_field, content_extras)

            message = [
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
            all_messages.append(message)

        text = [
            self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=True,
                enable_thinking=False,
                fps=1,
            )
            for message in all_messages
        ]

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            all_messages,
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
            text=text,
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            do_resize=False,  # qwen-vl-utils already resized
            **video_kwargs,
        )

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Faster than torch.no_grad() for inference-only code
        with torch.inference_mode():
            model_output, _ = self.model(
                **inputs,
                sample_type="progress",
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
        batch_size = len(progress_tensor)
        results = []

        for i in range(batch_size):
            progress_values = convert_bins_to_continuous(progress_tensor[i]).cpu().tolist()
            # Ensure we have the right length
            if isinstance(progress_values, float):
                progress_values = [progress_values]

            results.append(progress_values)

        return results

    def predict(self, samples: list[EvaluationSample]) -> list[Prediction]:

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
        for value, sample in zip(values, samples, strict=False):
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
