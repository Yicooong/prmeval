#!/usr/bin/env python3
# ruff: noqa: E501
"""
RoboReward baseline for discrete end-of-episode progress reward prediction.

RoboReward predicts discrete scores (1-5) for task completion:
- 1: No success
- 2: Minimal progress
- 3: Partial completion
- 4: Near completion
- 5: Perfect completion

Based on: https://huggingface.co/teetone/RoboReward-8B
"""

import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import ClassVar, List

import numpy as np
from PIL import Image

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer

logger = logging.getLogger(__name__)


def convert_frames_to_pil_images(frames, frames_shape=None):
    """Convert frames to PIL images if they are numpy arrays or serialized bytes.

    Handles:
    - Bytes with shape: deserializes to numpy array then converts
    - Numpy arrays (TxHxWxC or HxWxC): converts each frame to PIL Image
    - List of numpy arrays: converts each to PIL Image
    - List of PIL Images: returns as-is
    - List of mixed types (strings, PIL Images, numpy arrays): converts appropriately
    """

    # If frames are serialized bytes, deserialize first
    if isinstance(frames, bytes):
        # Deserialize bytes to numpy array (TxHxWxC) using provided shape
        if frames_shape is not None:
            # Convert to tuple if it's a list
            if isinstance(frames_shape, list):
                frames_shape = tuple(frames_shape)
            try:
                frames = np.frombuffer(frames, dtype=np.uint8).reshape(frames_shape)
            except Exception as e:
                print(f"Warning: Failed to reshape with provided shape {frames_shape}: {e}")
                # Fall back to 1D array
                frames = np.frombuffer(frames, dtype=np.uint8)
        else:
            # No shape provided, try to infer
            frames = np.frombuffer(frames, dtype=np.uint8)

    # If frames are numpy array (TxHxWxC), convert to list of PIL images
    if isinstance(frames, np.ndarray):
        pil_images = []

        # Handle different array shapes
        if len(frames.shape) == 4:  # TxHxWxC
            for i in range(frames.shape[0]):  # Iterate over time dimension
                frame = frames[i]  # HxWxC
                # Ensure uint8 dtype
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                pil_images.append(pil_image)
        elif len(frames.shape) == 3:  # HxWxC (single frame)
            frame = frames
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(frame)
            pil_images.append(pil_image)
        else:
            raise ValueError(f"Unexpected frames shape {frames.shape}. Expected 3D (HxWxC) or 4D (TxHxWxC) array.")

        return pil_images

    # If frames are list, handle each element
    if isinstance(frames, list):
        pil_images = []
        for frame in frames:
            if isinstance(frame, str):
                # File path - open it
                pil_images.append(Image.open(frame))
            elif isinstance(frame, Image.Image):
                # Already PIL Image
                pil_images.append(frame)
            elif isinstance(frame, np.ndarray):
                # Numpy array - convert to PIL
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(frame)
                pil_images.append(pil_image)
            else:
                # Try to convert to numpy array first
                try:
                    frame_array = np.array(frame)
                    if frame_array.dtype != np.uint8:
                        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
                    pil_images.append(Image.fromarray(frame_array))
                except Exception as e:
                    print(f"Warning: Could not convert frame to PIL Image: {e}")
                    continue
        return pil_images

    raise ValueError(f"Unsupported frames type: {type(frames)}")


@register_infer("roboreward")
class RoboReward(Infer):
    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("roboreward requires infer.model_path")
        options = config.options
        self._initialize(
            model_path=config.model_path,
            max_new_tokens=int(options.get("max_new_tokens", 128)),
            use_unsloth=bool(options.get("use_unsloth", False)),
            use_prefix_samples=bool(options.get("use_prefix_samples", False))
        )

    capabilities: ClassVar[set[str]] = {"progress"}
    """RoboReward baseline for discrete end-of-episode progress reward prediction."""

    def _initialize(
        self,
        model_path: str = "teetone/RoboReward-8B",
        max_new_tokens: int = 128,
        use_unsloth: bool = False,
        use_prefix_samples: bool = False
    ):
        """
        Initialize RoboReward model.

        Args:
            model_path: HuggingFace model path (e.g., "teetone/RoboReward-8B" or "teetone/RoboReward-4B")
            max_new_tokens: Maximum number of tokens to generate
            use_unsloth: Whether to use unsloth for faster inference (default: True)
        """
        global torch, Qwen3VLForConditionalGeneration, AutoProcessor, process_vision_info, FastVisionModel
        global HAS_UNSLOTH
        try:
            import torch
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("RoboReward requires torch, transformers, and qwen-vl-utils") from exc
        try:
            from unsloth import FastVisionModel

            HAS_UNSLOTH = True
        except ImportError:
            HAS_UNSLOTH = False

        logger.info(f"Loading RoboReward model: {model_path}")

        # Use unsloth for faster inference if available and requested
        if use_unsloth and HAS_UNSLOTH:
            print("Using Unsloth for faster inference")
            # Load model with unsloth's FastVisionModel
            self.model, _ = FastVisionModel.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                full_finetuning=False,  # Inference only
            )
        else:
            # Standard loading
            if use_unsloth and not HAS_UNSLOTH:
                print("Warning: Unsloth requested but not available, using standard loading")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # Auto device placement is best practice
            )

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, do_sample_frames=False, fps=1
        )
        self.max_new_tokens = max_new_tokens
        self.model_path = model_path
        self.use_prefix_samples = use_prefix_samples

        print(f"RoboReward model loaded on device: {self.model.device}")

    def _build_prompt(self, task_description: str) -> str:
        """Build the prompt for RoboReward inference.

        Args:
            task_description: Task instruction text

        Returns:
            Formatted prompt string
        """
        prompt = f"""Given the task, assign a discrete progress score reward (1,2,3,4,5) for the robot in the video in the format: ANSWER: <score>
Rubric for end-of-episode progress (judge only the final state without time limits):
1 - No Success: Final state shows no goal-relevant change for the command.
2 - Minimal Progress: Final state shows a small but insufficient change toward the goal.
3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement.
4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement.
5 - Perfect Completion: Final state satisfies all requirements.

Task: {task_description}"""
        return prompt

    def _parse_score(self, output_text: str) -> int | None:
        """Parse discrete score (1-5) from model output.

        Args:
            output_text: Model output text

        Returns:
            Discrete score (1-5) or None if parsing fails
        """
        # Look for "ANSWER: <number>" pattern
        pattern = r"ANSWER:\s*(\d+)"
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 5:
                return score

        # Fallback: look for any single digit 1-5 in the text
        pattern = r"\b([1-5])\b"
        matches = re.findall(pattern, output_text)
        if matches:
            # Take the last occurrence (most likely the answer)
            score = int(matches[-1])
            if 1 <= score <= 5:
                return score

        return None

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> list[float | None]:
        """
        Compute progress prediction for a frame sequence using RoboReward baseline.

        RoboReward predicts a discrete score (1-5) for the end-of-episode state.
        Since the sampler already uses use_frame_steps to create progressively longer
        sequences, we just process the single sequence provided here.

        Args:
            frames_array: (N, H, W, 3) uint8 array from trajectory frames (already a subsequence)
            task_description: Task description text

        Returns:
            List of discrete scores (1.0-5.0) for each frame.
            All frames get the same discrete score (end-of-episode score for this subsequence).
        """
   
        if frames_array is None or frames_array.size == 0:
            return []

        # Convert frames to PIL Images
        frames_pil = convert_frames_to_pil_images(frames_array)

        logger.info(f"RoboReward: Converted {len(frames_pil)} frames to PIL Images")

        if not frames_pil:
            return []
        result = []

        num_frames = len(frames_pil)
        if num_frames == 1:
            # Duplicate the single frame to make it 2 frames
            frames_pil = [frames_pil[0], frames_pil[0]]
            num_frames = 2
    
        if self.use_prefix_samples and num_frames > 2:
            num_samples = num_frames
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted({int(x) for x in prefix_lengths})
        else:
            prefix_lengths = [num_frames]

        # Ensure at least 2 frames for video processing (qwen_vl_utils requires minimum 2 frames)
        

        # Build prompt
        prompt = self._build_prompt(task_description)

        # Create temporary directory for frame files
        # Use individual frame files instead of video to avoid torchcodec memory issues
        # According to qwen-vl-utils docs, we can pass frames as a list of file paths
        tmpdir = tempfile.mkdtemp()
        unique_id = uuid.uuid4().hex

        # Save frames as individual JPEG files (much smaller than video, avoids torchcodec overhead)
        frame_paths = []
        for i, frame_pil in enumerate(frames_pil):
            frame_path = Path(tmpdir) / f"roboreward_{unique_id}_frame_{i:04d}.jpg"
            # Save as JPEG with reasonable quality to reduce file size
            frame_pil.save(frame_path, "JPEG", quality=85, optimize=True)
            frame_paths.append(f"file://{frame_path}")

        logger.info(f"RoboReward: Saved {len(frame_paths)} frames as JPEG files in {tmpdir}")

        for length in prefix_lengths:

            # Build message with frames as list of file paths (following Qwen3-VL pattern)
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frame_paths[:length], "sample_fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

            # Process vision info (qwen-vl-utils handles resizing)
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

            # Process inputs (do_resize=False since qwen-vl-utils already resized)
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

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Deterministic
                )

            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=True)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Parse score
            output_text = output_texts[0]
            discrete_score = self._parse_score(output_text)
            logger.info(f"RoboReward: Discrete score: {discrete_score}")

            if discrete_score is None:
                print(f"[!] Failed to parse score from output: {output_text}")
                discrete_score = 1  # Default to minimum score if parsing fails

            # Return same discrete score for all frames in this subsequence
            # Use original num_frames from frames_array (before duplication)
            # original_num_frames = len(convert_frames_to_pil_images(frames_array))

            # because RoboReward returns a score between 1 and 5, we need to normalize it to 0-1
            # result = [float(discrete_score) / 4.0 - 0.25] * original_num_frames
            result.append(float(discrete_score) / 4.0 - 0.25)

        lengths = np.array(prefix_lengths, dtype=np.float64)
        values = np.array(result, dtype=np.float64)
        frame_indices = np.arange(1, num_frames + 1, dtype=np.float64)
        progress = np.interp(frame_indices, lengths, values)
        
        # Clean up temporary directory
        shutil.rmtree(tmpdir, ignore_errors=True)

        return np.clip(progress.astype(np.float64), 0.0, 1.0).tolist()

  
    def compute_progress_with_prefix(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> list[float | None]:
        """
        Compute progress prediction for a frame sequence using RoboReward baseline.

        RoboReward predicts a discrete score (1-5) for the end-of-episode state.
        Since the sampler already uses use_frame_steps to create progressively longer
        sequences, we just process the single sequence provided here.

        Args:
            frames_array: (N, H, W, 3) uint8 array from trajectory frames (already a subsequence)
            task_description: Task description text

        Returns:
            List of discrete scores (1.0-5.0) for each frame.
            All frames get the same discrete score (end-of-episode score for this subsequence).
        """
   
        if frames_array is None or frames_array.size == 0:
            return []

        # Convert frames to PIL Images
        frames_pil = convert_frames_to_pil_images(frames_array)

        logger.info(f"RoboReward: Converted {len(frames_pil)} frames to PIL Images")

        if not frames_pil:
            return []

        num_frames = len(frames_pil)
        if num_frames == 1:
            # Duplicate the single frame to make it 2 frames
            frames_pil = [frames_pil[0], frames_pil[0]]
            num_frames = 2
    
        if self.use_prefix_samples and num_frames > 2:
            num_samples = num_frames
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted({int(x) for x in prefix_lengths})
        else:
            prefix_lengths = [num_frames]

        # Ensure at least 2 frames for video processing (qwen_vl_utils requires minimum 2 frames)
        

        # Build prompt
        prompt = self._build_prompt(task_description)

        # Create temporary directory for frame files
        # Use individual frame files instead of video to avoid torchcodec memory issues
        # According to qwen-vl-utils docs, we can pass frames as a list of file paths
        tmpdir = tempfile.mkdtemp()
        unique_id = uuid.uuid4().hex

        # Save frames as individual JPEG files (much smaller than video, avoids torchcodec overhead)
        frame_paths = []
        for i, frame_pil in enumerate(frames_pil):
            frame_path = Path(tmpdir) / f"roboreward_{unique_id}_frame_{i:04d}.jpg"
            # Save as JPEG with reasonable quality to reduce file size
            frame_pil.save(frame_path, "JPEG", quality=85, optimize=True)
            frame_paths.append(f"file://{frame_path}")

        logger.info(f"RoboReward: Saved {len(frame_paths)} frames as JPEG files in {tmpdir}")

        all_message = []
        for length in prefix_lengths:

            # Build message with frames as list of file paths (following Qwen3-VL pattern)
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frame_paths[:length], "sample_fps": 1.0},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            all_message.append(message)

        # Apply chat template
        text = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in all_message
        ]

        # Process vision info (qwen-vl-utils handles resizing)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            all_message,
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

        # Process inputs (do_resize=False since qwen-vl-utils already resized)
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
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # Deterministic
            )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids, strict=True)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Parse score
        
        discrete_score = [self._parse_score(output_text) for output_text in output_texts]
        logger.info(f"RoboReward: Discrete score: {discrete_score}")

        discrete_score = [
            1 if x is None else x 
            for x in discrete_score
        ]

        # Return same discrete score for all frames in this subsequence
        # Use original num_frames from frames_array (before duplication)
        # original_num_frames = len(convert_frames_to_pil_images(frames_array))

        # because RoboReward returns a score between 1 and 5, we need to normalize it to 0-1
        # result = [float(discrete_score) / 4.0 - 0.25] * original_num_frames
        result = [float(_discrete_score) / 4.0 - 0.25 for _discrete_score in discrete_score]
        
        # Clean up temporary directory
        shutil.rmtree(tmpdir, ignore_errors=True)

        return result

  
    def predict(self, samples: List[EvaluationSample]) -> List[Prediction]:
        result = []
        for sample in samples:
            if not isinstance(sample, ProgressSample):
                raise TypeError(f"{self.config.name} only supports progress samples")
            reference_path = sample.trajectory.metadata.get("reference_video_path")
            values = np.asarray(
                self.compute_progress_with_prefix(
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
            result.append(
                ProgressPrediction(
                    sample_id=sample.sample_id,
                    progress=values.tolist(),
                    model=self.config.model_id or self.config.model_path or self.config.name,
                    model_version=self.config.model_version,
                )
            )
        return result
