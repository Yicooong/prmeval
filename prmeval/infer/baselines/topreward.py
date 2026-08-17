#!/usr/bin/env python3
"""TOPReward baseline: token probabilities as zero-shot rewards for progress prediction.

Uses VLM log-likelihood of task completion (e.g. "True") conditioned on trajectory
prefixes to produce a dense progress curve. No task-specific training.

Reference: https://github.com/TOPReward/TOPReward
Paper: TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics (arXiv:2602.19313)
"""

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from ..base import RemoteError, vision_content
from ..model import ProgressModel, ProgressResult, RemoteContext

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


def _frames_array_to_pil_list(frames_array: np.ndarray) -> List[Image.Image]:
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
    prefix_lengths: Optional[List[int]] = None
    prefix_rewards: Optional[List[float]] = None
    normalized_prefix_rewards: Optional[List[float]] = None


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


class TopReward(ProgressModel):
    """TOPReward baseline: instruction-conditioned log-likelihood progress from a video VLM."""

    supports_remote = True

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_frames: int = 64,
        num_prefix_samples: int = 15,
        reduction: str = "mean",
        add_chat_template: bool = True,
        fps: float = 2.0,
        **kwargs: Any,
    ):
        """
        Args:
            model_path: HuggingFace model ID (Qwen3-VL recommended).
            max_frames: Max frames per trajectory (sampled if longer).
            num_prefix_samples: Number of prefix lengths to evaluate for progress curve.
            reduction: "mean" or "sum" over instruction tokens.
            add_chat_template: Whether to use chat template for instruction prompt.
            fps: Frames per second for video input to the VLM.
        """
        try:
            import torch
            import torch.nn.functional as functional
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "TOPReward local inference requires torch, transformers, and qwen-vl-utils"
            ) from exc

        self.torch = torch
        self.functional = functional
        self.process_vision_info = process_vision_info
        self.model_path = model_path
        self.max_frames = max_frames
        self.num_prefix_samples = num_prefix_samples
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
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def _compute_instruction_reward(
        self,
        frames: List[Image.Image],
        instruction: str,
    ) -> float:
        """Compute log-likelihood reward for instruction given video (single trajectory)."""
        prompt_text = (
            "The above video shows a robot manipulation trajectory that completes the following task: "
        )
        eos_token = self.processor.tokenizer.eos_token

        if self.add_chat_template:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True or not. The answer is:"
            )
            templated_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames, "fps": self.fps},
                        {"type": "text", "text": f"{prompt_text}{instruction_suffix}"},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                templated_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = f"{prompt_chat}True"
            image_inputs, video_inputs = self.process_vision_info(templated_messages)
        else:
            instruction_suffix = (
                f"{instruction} Decide whether the above statement is True or not. The answer is: True"
            )
            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames, "fps": self.fps},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt_chat = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=False
            )
            if eos_token:
                prompt_chat = prompt_chat.split(eos_token)[0]
            full_text = f"{prompt_chat}{instruction_suffix}"
            image_inputs, video_inputs = self.process_vision_info(user_messages)

        inputs = self.processor(
            text=[full_text],
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
        with self.torch.no_grad():
            outputs = self.model(**inputs, labels=labels)

        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = self.functional.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]
        reward = (
            masked_log_probs.sum().item()
            if self.reduction == "sum"
            else masked_log_probs.mean().item()
        )
        return reward

    def _compute_instruction_rewards_for_prefixes(
        self,
        frames: List[Image.Image],
        instruction: str,
    ) -> _InstructionRewardResult:
        """Compute rewards for trajectory prefixes; return normalized curve."""
        num_frames = len(frames)
        num_samples = min(self.num_prefix_samples, num_frames)
        if num_frames > 2:
            prefix_lengths = np.linspace(1, num_frames, num_samples, dtype=int)
            prefix_lengths = sorted(set(int(x) for x in prefix_lengths))
        else:
            prefix_lengths = [num_frames]

        rewards = []
        for length in prefix_lengths:
            prefix_frames = frames[:length]
            r = self._compute_instruction_reward(prefix_frames, instruction)
            rewards.append(r)
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
        reference_video_path: Optional[str] = None,
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
            raise ValueError(
                f"TOPReward received {num_frames} frames but max_frames={self.max_frames}; "
                "set sampling.max_frames so Stage 1 and Stage 2 remain aligned"
            )

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

    @staticmethod
    def _true_logprob(response: dict[str, Any]) -> float:
        try:
            positions = response["choices"][0]["logprobs"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RemoteError("TOPReward response is missing choices[0].logprobs.content") from exc
        for position in positions:
            candidates = [position, *(position.get("top_logprobs") or [])]
            matches = [
                candidate
                for candidate in candidates
                if isinstance(candidate, dict) and str(candidate.get("token", "")).strip().lower() == "true"
            ]
            if matches:
                try:
                    return max(float(candidate["logprob"]) for candidate in matches)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RemoteError("TOPReward True candidate has no numeric logprob") from exc
        raise RemoteError("TOPReward top_logprobs did not contain True or ' True'")

    @classmethod
    def remote_compute_progress(
        cls,
        frames_array: np.ndarray,
        task_description: str,
        reference_video_path: str | None,
        remote: RemoteContext,
        options: dict,
    ) -> ProgressResult:
        num_frames = len(frames_array)
        if num_frames == 0:
            return ProgressResult(np.array([], dtype=np.float64), raw_response=[])
        scoring = str(options.get("scoring", "remote_top_logprobs"))
        if scoring == "exact_logits":
            raise ValueError("TOPReward scoring=exact_logits is only available in local mode")
        configured_samples = options.get("num_prefix_samples", 15)
        if not isinstance(configured_samples, int) or isinstance(configured_samples, bool) or configured_samples < 1:
            raise ValueError("TOPReward options.num_prefix_samples must be a positive integer")
        if num_frames > 2:
            count = min(configured_samples, num_frames)
            prefix_lengths = sorted({int(value) for value in np.linspace(1, num_frames, count, dtype=int)})
        else:
            prefix_lengths = [num_frames]

        task = task_description or "Complete the task."
        prompt = (
            "The supplied frames show a robot manipulation trajectory that completes the following task: "
            f"{task} Decide whether the preceding statement is True or False. Answer with exactly True or False."
        )
        rewards: list[float] = []
        raw_responses = []
        for length in prefix_lengths:
            content = [{"type": "text", "text": prompt}]
            content.extend(vision_content(frames_array[:length]))
            messages = [{"role": "user", "content": content}]
            last_error = None
            response = None
            for parse_attempt in range(remote.config.max_retries + 1):
                response = remote.completion(
                    messages,
                    {"temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 20},
                )
                try:
                    rewards.append(cls._true_logprob(response))
                    break
                except RemoteError as exc:
                    last_error = exc
                    if parse_attempt >= remote.config.max_retries:
                        raise RemoteError(f"TOPReward could not obtain True logprob: {last_error}") from exc
            assert response is not None
            raw_responses.append({"prefix_length": length, "true_logprob": rewards[-1], "response": response})

        normalized = _normalize_rewards(rewards)
        values = np.interp(
            np.arange(1, num_frames + 1, dtype=float),
            np.asarray(prefix_lengths, dtype=float),
            normalized,
        )
        return ProgressResult(np.clip(values, 0.0, 1.0), raw_response=raw_responses)
