#!/usr/bin/env python3
"""TOPReward baseline: token probabilities as zero-shot rewards for progress prediction.

Uses VLM log-likelihood of task completion (e.g. "True") conditioned on trajectory
prefixes to produce a dense progress curve. No task-specific training.

Reference: https://github.com/TOPReward/TOPReward
Paper: TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics (arXiv:2602.19313)
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, List

import numpy as np
from PIL import Image

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer

logger = logging.getLogger(__name__)


# Default image size used by TOPReward for video frames
_TOPREWARD_IMG_SIZE = 224


def _to_pil(frame: np.ndarray) -> Image.Image:
    """Convert a single frame (H,W,C) or (C,H,W) to PIL RGB."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        return Image.fromarray(frame, "L").convert("RGB")
    return Image.fromarray(frame[:, :, :3], "RGB").resize((_TOPREWARD_IMG_SIZE, _TOPREWARD_IMG_SIZE))


def _frames_array_to_pil_list(frames_array: np.ndarray) -> list[Image.Image]:
    """Convert (T,H,W,C) or (T,C,H,W) to list of PIL images."""
    T = frames_array.shape[0]
    out = []
    for t in range(T):
        frame = frames_array[t]
        out.append(_to_pil(frame))
    return out


@dataclass
class _InstructionRewardResult:
    reward: float
    reduction: str
    token_count: int
    prefix_lengths: list[int] | None = None
    prefix_rewards: list[float] | None = None
    normalized_prefix_rewards: list[float] | None = None


def _normalize_rewards(rewards: Sequence[float], method: str = "minmax") -> np.ndarray:
    """Normalize rewards to [0, 1] (minmax)."""
    rewards_arr = np.array(rewards, dtype=np.float64)
    if len(rewards_arr) == 0:
        return rewards_arr
    if len(rewards_arr) == 1:
        return np.array([1.0])
    if method == "minmax":
        r_min, r_max = rewards_arr.min(), rewards_arr.max()
        if r_max == r_min:
            return np.ones_like(rewards_arr)
        return (rewards_arr - r_min) / (r_max - r_min)
    raise ValueError(f"Unknown normalization method: {method}")


@register_infer("topreward")
class TopReward(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}
    """TOPReward baseline: instruction-conditioned log-likelihood progress from a video VLM."""

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("topreward requires infer.model_path")
        options = config.options
        self._initialize(
            model_path=config.model_path,
            max_frames=int(options.get("max_frames", 64)),
            use_prefix_samples=bool(options.get("use_prefix_samples", False)),
            reduction=str(options.get("reduction", "mean")),
            add_chat_template=bool(options.get("add_chat_template", True)),
            fps=float(options.get("fps", 2.0)),
        )

    def _initialize(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_frames: int = 64,
        use_prefix_samples: bool = False,
        reduction: str = "mean",
        add_chat_template: bool = True,
        fps: float = 2.0,
        **kwargs: Any,
    ):
        """
        Args:
            model_path: HuggingFace model ID (Qwen3-VL recommended).
            max_frames: Max frames per trajectory (sampled if longer).
            use_prefix_samples: prefix lengths to evaluate for progress curve.
            reduction: "mean" or "sum" over instruction tokens.
            add_chat_template: Whether to use chat template for instruction prompt.
            fps: Frames per second for video input to the VLM.
        """
        global torch, F, process_vision_info, AutoProcessor, Qwen3VLForConditionalGeneration
        try:
            import torch
            import torch.nn.functional as F
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("TopReward requires torch, transformers, and qwen-vl-utils") from exc

        self.model_path = model_path
        self.max_frames = max_frames
        self.use_prefix_samples = use_prefix_samples
        self.reduction = reduction
        self.add_chat_template = add_chat_template
        self.fps = fps
        logger.info(f"Loading TOPReward model: {model_path}")
        for attn_impl in ("flash_attention_2", "sdpa", "eager"):
            try:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map="auto",
                    attn_implementation=attn_impl,
                )
                logger.info(f"Loaded with attn_implementation={attn_impl}")
                break
            except Exception as e:
                if attn_impl == "eager":
                    raise
                logger.warning(f"attn_implementation={attn_impl} failed: {e}, trying next.")
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="left", trust_remote_code=True)

    def _compute_instruction_reward(
        self,
        frames: list[Image.Image],
        instruction: str,
        prefix_length: list[int]
    ) -> list[float]:
        """Compute log-likelihood reward for instruction given video (single trajectory)."""
        prompt_text = "The above video shows a robot manipulation trajectory that completes the following task: "
        eos_token = self.processor.tokenizer.eos_token


        full_texts = []
        templated_messages = []
        for p_length in prefix_length:
            if self.add_chat_template:
                instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is:"
                templated_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames[:p_length]},
                            {"type": "text", "text": f"{prompt_text}{instruction_suffix}"},
                        ],
                    }
                ]
                prompt_chat = self.processor.apply_chat_template(
                    templated_message, tokenize=False, add_generation_prompt=True
                )
                full_text = f"{prompt_chat}True"
                templated_messages.append(templated_message)
                full_texts.append(full_text)
                
            else:
                instruction_suffix = f"{instruction} Decide whether the above statement is True or not. The answer is: True"
                user_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": frames[:p_length]},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]
                prompt_chat = self.processor.apply_chat_template(user_message, tokenize=False, add_generation_prompt=False)
                if eos_token:
                    prompt_chat = prompt_chat.split(eos_token)[0]
                full_text = f"{prompt_chat}{instruction_suffix}"
                templated_messages.append(user_message)
                full_texts.append(full_text)

        image_inputs, video_inputs = process_vision_info(templated_messages)
            
            

        inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )


        inputs = inputs.to(self.model.device)
        labels = inputs["input_ids"].clone()
        prompt_length = inputs["input_ids"].shape[1] - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        # rewards = []

        # for i in range(token_log_probs.shape[0]):
        #     valid_log_probs = token_log_probs[i][mask[i]]

        #     if self.reduction == "sum":
        #         reward = valid_log_probs.sum().item()
        #     else:
        #         reward = valid_log_probs.mean().item()

        #     rewards.append(reward)

        # return rewards

        masked_log_probs = token_log_probs[mask]
        return masked_log_probs.tolist()

    def _compute_instruction_rewards_for_prefixes(
        self,
        frames: list[Image.Image],
        instruction: str,
    ) -> _InstructionRewardResult:
        """Compute rewards for trajectory prefixes; return normalized curve."""
        num_frames = len(frames)

        if self.use_prefix_samples and num_frames > 2:
            num_samples = num_frames
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted({int(x) for x in prefix_lengths})
        else:
            prefix_lengths = [num_frames]

       
        rewards = self._compute_instruction_reward(frames, instruction, prefix_lengths)
            
        normalized = _normalize_rewards(rewards).tolist()
        return _InstructionRewardResult(
            reward=rewards[-1],
            reduction=self.reduction,
            token_count=0,
            prefix_lengths=prefix_lengths,
            prefix_rewards=rewards,
            normalized_prefix_rewards=normalized,
        )

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        """
        Compute progress curve from trajectory frames using TOPReward (log-likelihood of task completion).

        Args:
            frames_array: (T, H, W, C) or (T, C, H, W), RGB.
            task_description: Instruction string.
            reference_video_path: Unused (TOPReward is zero-shot).

        Returns:
            progress: (T,) float array in [0, 1].
        """
        if frames_array is None or frames_array.size == 0:
            return np.array([], dtype=np.float64)

        num_frames = frames_array.shape[0]
        if num_frames > self.max_frames:
            indices = np.linspace(0, num_frames - 1, self.max_frames, dtype=int)
            frames_array = frames_array[indices]
            num_frames = frames_array.shape[0]

        frames = _frames_array_to_pil_list(frames_array)
        result = self._compute_instruction_rewards_for_prefixes(frames, task_description or "Complete the task.")

        prefix_lengths = result.prefix_lengths or []
        normalized = result.normalized_prefix_rewards or []
        if not prefix_lengths or not normalized:
            return np.zeros(num_frames, dtype=np.float64)

        # Map prefix_lengths (1-based frame counts) -> normalized reward; then interpolate to every frame
        lengths = np.array(prefix_lengths, dtype=np.float64)
        values = np.array(normalized, dtype=np.float64)
        frame_indices = np.arange(1, num_frames + 1, dtype=np.float64)
        progress = np.interp(frame_indices, lengths, values)
        return np.clip(progress.astype(np.float64), 0.0, 1.0)

    def predict(self, samples: List[EvaluationSample]) -> List[Prediction]:
        result = []
        for sample in samples:
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
            result.append(
                ProgressPrediction(
                    sample_id=sample.sample_id,
                    progress=values.tolist(),
                    model=self.config.model_id or self.config.model_path or self.config.name,
                    model_version=self.config.model_version,
                )
        )
        return result
