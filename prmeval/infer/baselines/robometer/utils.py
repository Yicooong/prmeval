import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Qwen3VLModel

from .configs import (
    ExperimentConfig,
    ModelConfig,
    PEFTConfig,
)
from .rbm import RBM

logger = logging.getLogger(__name__)


def convert_bins_to_continuous(bin_logits: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert discrete bins to continuous progress values in [0, 1] via weighted sum of bin centers."""
    num_bins = bin_logits.shape[-1]
    if (bin_logits.sum(dim=-1) == 1).all():
        bin_probs = bin_logits
    else:
        bin_probs = (
            torch.softmax(bin_logits, dim=-1)
            if isinstance(bin_logits, torch.Tensor)
            else np.softmax(bin_logits, axis=-1)
        )
    bin_centers = (
        torch.linspace(0.0, 1.0, num_bins, device=bin_logits.device, dtype=bin_logits.dtype)
        if isinstance(bin_logits, torch.Tensor)
        else np.linspace(0.0, 1.0, num_bins)
    )
    return (
        (bin_probs * bin_centers).sum(dim=-1)
        if isinstance(bin_logits, torch.Tensor)
        else (bin_probs * bin_centers).sum(axis=-1)
    )


MAX_IMAGE_SIDE = 480  # bigger side
MAX_IMAGE_PIXELS = 1024 * 1024  # safety cap (1.0 MP). raise to 1.5MP if stable


def _resize_pil(pil: Image.Image, max_side: int = MAX_IMAGE_SIDE, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    pil = pil.convert("RGB")
    w, h = pil.size

    # Scale down if max side too large
    scale_side = min(1.0, max_side / float(max(w, h)))

    # Scale down if too many pixels (area cap)
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0

    scale = min(scale_side, scale_area)

    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)

    return pil


@dataclass
class ModelOutput:
    pref_logits: torch.Tensor | None = None
    success_logits: torch.Tensor | None = None
    progress_logits: torch.Tensor | None = None

    hidden_states: torch.Tensor | None = None


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


def _get_checkpoint_safetensors_files(checkpoint_path: Path, prefer_model_shards: bool = False) -> list[Path]:
    """Return checkpoint safetensors files, optionally preferring full model shards over sidecar adapter files."""
    if prefer_model_shards:
        index_path = checkpoint_path / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)
            weight_files = sorted(set(index_data.get("weight_map", {}).values()))
            indexed_files = [checkpoint_path / name for name in weight_files if (checkpoint_path / name).exists()]
            if indexed_files:
                return indexed_files

        model_shards = sorted(checkpoint_path.glob("model*.safetensors"))
        if model_shards:
            return model_shards

    return sorted(checkpoint_path.glob("*.safetensors"))


def _verify_checkpoint_loading(cfg: ModelConfig, model: Any, before_weights: dict) -> None:
    """
    Verify that checkpoint weights were loaded correctly by comparing before/after weights.

    Args:
        cfg: Model configuration
        model: The model after loading checkpoint
        before_weights: Dictionary of weights before loading (keys: visual, progress_head, lm_embed_tokens, lm_layer)
    """

    if "Qwen2.5" in cfg.base_model_id:
        after_visual = model.model.visual.blocks[0].mlp.down_proj.weight
        after_progress_head = model.progress_head[0].weight
        after_lm_embed_tokens = model.model.language_model.embed_tokens.weight
        after_lm_layer = model.model.language_model.layers[0].mlp.up_proj.weight
    elif "Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        after_visual = model.model.visual.blocks[0].mlp.linear_fc1.weight
        after_progress_head = model.progress_head[0].weight
        after_lm_embed_tokens = model.model.language_model.embed_tokens.weight
        after_lm_layer = model.model.language_model.layers[0].mlp.up_proj.weight
    else:
        return

    before_visual = before_weights["visual"]
    before_progress_head = before_weights["progress_head"]
    before_lm_embed_tokens = before_weights["lm_embed_tokens"]
    before_lm_layer = before_weights["lm_layer"]

    logger.info(
        f"Before visual: {before_visual.shape}, {before_visual.sum()} | After visual: {after_visual.shape}, {after_visual.sum()}"
    )
    logger.info(
        f"Before progress head: {before_progress_head.shape}, {before_progress_head.sum()} | After progress head: {after_progress_head.shape}, {after_progress_head.sum()}"
    )
    logger.info(
        f"Before LM embed tokens: {before_lm_embed_tokens.shape}, {before_lm_embed_tokens.sum()} | After LM embed tokens: {after_lm_embed_tokens.shape}, {after_lm_embed_tokens.sum()}"
    )
    logger.info(
        f"Before LM layer: {before_lm_layer.shape}, {before_lm_layer.sum()} | After LM layer: {after_lm_layer.shape}, {after_lm_layer.sum()}"
    )

    # check that before and after are different
    if torch.allclose(before_visual, after_visual):
        logger.warning("Before and after visual are the same! Check if you loaded the pretrained model correctly")
    if torch.allclose(before_progress_head, after_progress_head):
        logger.warning(
            "Before and after progress head are the same! Check if you loaded the pretrained model correctly"
        )
    if torch.allclose(before_lm_embed_tokens, after_lm_embed_tokens):
        logger.warning(
            "Before and after LM embed tokens are the same! Check if you loaded the pretrained model correctly"
        )


def _load_checkpoint_weights_from_safetensors(
    model,
    checkpoint_path: str,
    cfg: ModelConfig,
    load_adapters: bool = True,
    prefer_model_shards: bool = False,
) -> None:
    """
    Load checkpoint weights from safetensors files in a checkpoint directory.
    Includes verification for PEFT adapters and progress_head.

    This is needed when using Unsloth, as we can't use from_pretrained on checkpoints.
    Instead, we load the base model with Unsloth first, then manually load the checkpoint weights.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to checkpoint directory containing safetensors files
        cfg: Model configuration for verification
        load_adapters: If False, skip loading adapter weights (assumes already loaded via PeftModel.from_pretrained)
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {checkpoint_path}")

    # Collect safetensors files. For self-contained checkpoints, prefer model shards and
    # ignore sidecar adapter/custom-head files that may duplicate the same weights.
    safetensors_files = _get_checkpoint_safetensors_files(
        checkpoint_path,
        prefer_model_shards=prefer_model_shards,
    )
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in checkpoint directory: {checkpoint_path}")

    logger.info(f"Loading checkpoint weights from {len(safetensors_files)} safetensors file(s) in {checkpoint_path}")

    # Capture before weights for verification (adapter and progress_head)
    before_weights = {}
    before_progress_head = model.progress_head[0].weight.clone()
    before_weights["progress_head"] = before_progress_head

    # Capture adapter weights before loading (if PEFT is enabled)
    adapter_keys_before = []
    sample_adapter_keys = []
    if cfg.use_peft:
        model_state_dict = model.state_dict()
        adapter_keys_before = [k for k in model_state_dict.keys() if "lora_A" in k or "lora_B" in k]
        if adapter_keys_before:
            # Sample a few adapter weights to verify they change
            sample_adapter_keys = adapter_keys_before[:3]  # Check first 3 adapters
            for key in sample_adapter_keys:
                before_weights[f"adapter_{key}"] = model_state_dict[key].clone()
            logger.info(f"Found {len(adapter_keys_before)} adapter parameters in model before loading checkpoint")
        else:
            logger.warning("No adapter parameters found in model - PEFT may not be applied correctly")

    # Load all safetensors files and merge into a single state dict
    checkpoint_state_dict = {}
    for safetensors_file in safetensors_files:
        logger.debug(f"Loading weights from {safetensors_file.name}")
        file_state_dict = load_file(str(safetensors_file), device="cuda")
        checkpoint_state_dict.update(file_state_dict)

    # Check what adapter keys are in checkpoint
    checkpoint_adapter_keys = [k for k in checkpoint_state_dict.keys() if "lora_A" in k or "lora_B" in k]
    logger.info(f"Found {len(checkpoint_adapter_keys)} adapter parameters in checkpoint")
    if checkpoint_adapter_keys:
        logger.debug(f"Sample checkpoint adapter keys: {checkpoint_adapter_keys[:5]}")

    # If load_adapters=False, filter out adapter keys (they were already loaded via PeftModel.from_pretrained)
    if not load_adapters:
        logger.info("Skipping adapter weights (already loaded via PeftModel.from_pretrained)")
        checkpoint_state_dict = {
            k: v for k, v in checkpoint_state_dict.items() if "lora_A" not in k and "lora_B" not in k
        }

    # Get model's expected state dict keys
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())

    # Log sample model keys to understand structure
    sample_model_keys = [k for k in list(model_keys)[:10]]
    logger.info(f"Sample model keys (first 10): {sample_model_keys}")

    # Log sample checkpoint keys to understand structure
    sample_ckpt_keys = [k for k in list(checkpoint_state_dict.keys())[:10]]
    logger.info(f"Sample checkpoint keys (first 10): {sample_ckpt_keys}")

    # Check if model has adapter keys and what structure they use
    model_adapter_keys = [k for k in model_keys if "lora_A" in k or "lora_B" in k]
    if model_adapter_keys:
        logger.info(f"Found {len(model_adapter_keys)} adapter keys in model")
        logger.debug(f"Sample model adapter keys: {model_adapter_keys[:3]}")

    # Remap checkpoint keys to match model structure
    # For PEFT models wrapped in RBM: checkpoint has "model.model." but model expects "model.base_model.model.model."
    # Try multiple strategies: direct match, map "model.model." -> "model.base_model.model.model.", etc.
    remapped_state_dict = {}
    remapped_count = 0
    direct_match_count = 0
    remap_strategies = {}

    for ckpt_key, ckpt_value in checkpoint_state_dict.items():
        if ckpt_key in model_keys:
            # Direct match - use as is
            remapped_state_dict[ckpt_key] = ckpt_value
            direct_match_count += 1
        else:
            # Try different remapping strategies
            potential_keys = []

            # Strategy 1 (PEFT): Map "model.model." -> "model.base_model.model.model." (for PEFT wrapped in RBM)
            # This handles the case where Unsloth saved the full model from model.model
            if ckpt_key.startswith("model.model."):
                # For PEFT: model.model.* -> model.base_model.model.model.*
                peft_key = ckpt_key.replace("model.model.", "model.base_model.model.model.", 1)
                potential_keys.append(peft_key)
                # Also try: model.model.* -> model.* (fallback)
                potential_keys.append(ckpt_key.replace("model.model.", "model.", 1))

            # Strategy 2: Remove "model." prefix entirely
            if ckpt_key.startswith("model."):
                potential_keys.append(ckpt_key.replace("model.", "", 1))

            # Strategy 3: Add "model." prefix if missing
            if not ckpt_key.startswith("model."):
                potential_keys.append(f"model.{ckpt_key}")

            # Strategy 4: Try adding "base_model." between "model." and the rest (for non-PEFT wrapped models)
            if ckpt_key.startswith("model.") and not ckpt_key.startswith("model.base_model."):
                parts = ckpt_key.split(".", 1)
                if len(parts) == 2:
                    potential_keys.append(f"model.base_model.{parts[1]}")

            # Try each potential key
            matched = False
            for potential_key in potential_keys:
                if potential_key in model_keys:
                    remapped_state_dict[potential_key] = ckpt_value
                    remapped_count += 1
                    strategy = f"{ckpt_key} -> {potential_key}"
                    remap_strategies[strategy] = remap_strategies.get(strategy, 0) + 1
                    if remapped_count <= 5:  # Log first 5 remappings
                        logger.debug(f"Remapped: {strategy}")
                    matched = True
                    break

            if not matched:
                # Key still doesn't match, will be in unexpected_keys
                remapped_state_dict[ckpt_key] = ckpt_value

    if remapped_count > 0:
        logger.info(
            f"Remapped {remapped_count} checkpoint keys to match model structure (direct matches: {direct_match_count})"
        )
        if remap_strategies:
            logger.debug(f"Remapping strategies used: {dict(list(remap_strategies.items())[:5])}")

    # Load remapped state dict into model with strict=False to handle missing keys
    try:
        missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    except Exception as e:
        logger.error("load_state_dict failed", exc_info=True)
        raise e

    # Filter missing keys - base model keys are expected for PEFT checkpoints
    base_model_missing = [
        k
        for k in missing_keys
        if any(
            pattern in k for pattern in ["visual.", "language_model.", "text_encoder.", "text_model.", "embed_tokens"]
        )
    ]
    other_missing = [k for k in missing_keys if k not in base_model_missing]

    if base_model_missing:
        logger.info(f"Missing base model keys (expected for PEFT checkpoints): {len(base_model_missing)} keys")
    if other_missing:
        logger.warning(f"Missing non-base model keys: {len(other_missing)} keys")
        logger.debug(
            f"Missing keys: {other_missing[:10]}..." if len(other_missing) > 10 else f"Missing keys: {other_missing}"
        )

    # Filter unexpected keys - adapter keys might be expected if structure differs slightly
    adapter_unexpected = [k for k in unexpected_keys if "lora_A" in k or "lora_B" in k]
    other_unexpected = [k for k in unexpected_keys if k not in adapter_unexpected]

    if adapter_unexpected:
        logger.warning(f"Unexpected adapter keys in checkpoint (not in model): {len(adapter_unexpected)} keys")
        logger.debug(
            f"Unexpected adapter keys: {adapter_unexpected[:10]}..."
            if len(adapter_unexpected) > 10
            else f"Unexpected adapter keys: {adapter_unexpected}"
        )
    if other_unexpected:
        logger.warning(f"Unexpected non-adapter keys: {len(other_unexpected)} keys")
        logger.debug(
            f"Unexpected keys: {other_unexpected[:10]}..."
            if len(other_unexpected) > 10
            else f"Unexpected keys: {other_unexpected}"
        )

    # Verify progress_head loaded correctly
    after_progress_head = model.progress_head[0].weight
    progress_head_loaded = not torch.allclose(before_progress_head, after_progress_head, atol=1e-6)

    logger.info(
        f"Progress head - Before: shape={before_progress_head.shape}, sum={before_progress_head.sum():.6f} | "
        f"After: shape={after_progress_head.shape}, sum={after_progress_head.sum():.6f} | "
        f"Loaded: {progress_head_loaded}"
    )

    if not progress_head_loaded:
        logger.error("Progress head weights did not change after loading checkpoint!")
        logger.error("This indicates the checkpoint weights were not loaded correctly.")

    # Verify adapter weights loaded correctly only when this helper loads them.
    # If load_adapters=False, PeftModel.from_pretrained has already loaded adapters
    # and this helper only restores custom RBM weights.
    adapter_loaded_correctly = True
    if cfg.use_peft and adapter_keys_before and load_adapters:
        model_state_dict_after = model.state_dict()
        adapter_keys_after = [k for k in model_state_dict_after.keys() if "lora_A" in k or "lora_B" in k]

        logger.info(f"Adapter keys - Before: {len(adapter_keys_before)} | After: {len(adapter_keys_after)}")

        # Check if sample adapter weights changed
        for key in sample_adapter_keys:
            if key in before_weights:
                before_adapter = before_weights[f"adapter_{key}"]
                if key in model_state_dict_after:
                    after_adapter = model_state_dict_after[key]
                    adapter_changed = not torch.allclose(before_adapter, after_adapter, atol=1e-6)
                    logger.info(
                        f"Adapter {key} - Before: shape={before_adapter.shape}, sum={before_adapter.sum():.6f} | "
                        f"After: shape={after_adapter.shape}, sum={after_adapter.sum():.6f} | "
                        f"Loaded: {adapter_changed}"
                    )
                    if not adapter_changed:
                        logger.warning(f"Adapter {key} weights did not change after loading checkpoint!")
                        adapter_loaded_correctly = False
                else:
                    logger.warning(f"Adapter key {key} not found in model after loading!")
                    adapter_loaded_correctly = False

        # Check how many adapter keys from checkpoint were actually loaded
        # Need to check both original and remapped keys (matching the remapping strategies above)
        loaded_adapter_keys = []
        for ckpt_key in checkpoint_adapter_keys:
            # Check if original key exists in model
            if ckpt_key in model_state_dict_after:
                loaded_adapter_keys.append(ckpt_key)
            else:
                # Try remapping strategies to find the actual key used in model
                potential_keys = []
                if ckpt_key.startswith("model.model."):
                    # Strategy 1: PEFT wrapped in RBM
                    potential_keys.append(ckpt_key.replace("model.model.", "model.base_model.model.model.", 1))
                    # Strategy 2: Fallback
                    potential_keys.append(ckpt_key.replace("model.model.", "model.", 1))

                # Check if any remapped key exists in model
                for remapped_key in potential_keys:
                    if remapped_key in model_state_dict_after:
                        loaded_adapter_keys.append(ckpt_key)  # Count original key as loaded
                        break
        logger.info(f"Loaded {len(loaded_adapter_keys)}/{len(checkpoint_adapter_keys)} adapter keys from checkpoint")

        if len(loaded_adapter_keys) == 0:
            logger.error("No adapter weights were loaded from checkpoint!")
            adapter_loaded_correctly = False
        elif len(loaded_adapter_keys) < len(checkpoint_adapter_keys) * 0.5:  # Less than 50% loaded
            logger.warning(
                f"Only {len(loaded_adapter_keys)}/{len(checkpoint_adapter_keys)} adapter keys loaded - may indicate structure mismatch"
            )
            adapter_loaded_correctly = False

    if not adapter_loaded_correctly:
        logger.error("Adapter weights did not load correctly!")
    logger.info(f"Successfully loaded checkpoint weights from {checkpoint_path}")


def _checkpoint_has_full_model_shards(checkpoint_path: str) -> bool:
    """Whether a checkpoint directory contains self-contained model safetensors shards."""
    path = Path(checkpoint_path)
    return (path / "model.safetensors.index.json").exists() or any(path.glob("model*.safetensors"))


def _setup_processor_and_tokenizer(cfg: ModelConfig) -> AutoProcessor:
    """
    Setup processor and tokenizer for the model.

    Args:
        cfg: Model configuration

    Returns:
        Processor
    """
    if "SmolVLM" in cfg.base_model_id:
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_id,
            trust_remote_code=cfg.trust_remote_code,
            padding_side="left",
            size={"longest_edge": 512},
            max_image_size={"longest_edge": 512},
            use_fast=True,
        )
        logger.info(f"SmolVLM Processor: {processor}")
    elif "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        processor = AutoProcessor.from_pretrained(
            cfg.base_model_id,
            trust_remote_code=cfg.trust_remote_code,
            do_sample_frames=False,  # disable frame sampling here since we do this in the data generator
            # padding_side="left",
            padding_side="right",
        )
        logger.info(f"Qwen Processor: {processor}")
    else:
        raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return processor


def _load_base_model_standard(
    cfg: ModelConfig,
    torch_dtype: torch.dtype,
    extra_kwargs: dict,
    bnb: BitsAndBytesConfig | None,
) -> Any:
    """
    Load base model using standard transformers loading.

    Args:
        cfg: Model configuration
        torch_dtype: Torch dtype to use
        extra_kwargs: Extra kwargs for model loading (e.g., attn_implementation)
        bnb: Optional BitsAndBytesConfig for quantization

    Returns:
        Base model
    """
    # Check if it's Molmo, Qwen3 or Qwen2/2.5
    is_molmo = "Molmo" in cfg.base_model_id
    is_qwen3 = "Qwen3" in cfg.base_model_id or "qwen3" in cfg.base_model_id.lower()

    # Select appropriate model classes based on version and model type
    if is_molmo:
        # Molmo2 uses AutoModelForImageTextToText with trust_remote_code
        base_model = AutoModelForImageTextToText.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
            **extra_kwargs,
            quantization_config=bnb,
        )
        # Extract the base model for RBM
        base_model = base_model.model
        logger.info("Using Molmo2 models")
    else:
        qwen_model_cls = Qwen3VLModel
        base_model = qwen_model_cls.from_pretrained(
            cfg.base_model_id, torch_dtype=torch_dtype, **extra_kwargs, quantization_config=bnb, device_map="auto"
        )
        logger.info("Using Qwen3 models")

    return base_model


def _add_special_tokens_and_resize(cfg: ModelConfig, processor: AutoProcessor, base_model: Any) -> None:
    """
    Add RBM special tokens and resize token embeddings if needed.

    Args:
        cfg: Model configuration
        processor: Processor with tokenizer
        base_model: Base model to resize embeddings for
    """
    # Add RBM special tokens if they don't exist
    special_tokens = [
        "<|split_token|>",
        "<|reward_token|>",
        "<|pref_token|>",
        "<|sim_token|>",
        "<|prog_token|>",  # Per-frame progress token
    ]
    logger.info(f"Before adding special tokens: {len(processor.tokenizer.get_vocab())}")
    num_added = 0
    for token in special_tokens:
        if token not in processor.tokenizer.get_vocab():
            added = processor.tokenizer.add_special_tokens({"additional_special_tokens": [token]})
            num_added += added

    logger.info(f"Added {num_added} special tokens")

    base_model.resize_token_embeddings(len(processor.tokenizer))
    logger.info(f"Resized token embeddings to {len(processor.tokenizer)}")


def setup_model_and_processor(
    cfg: ModelConfig, hf_model_id: str = "", peft_config: PEFTConfig = None
) -> tuple[AutoProcessor, RBM]:
    """
    Shared function to set up model, processor, and tokenizer for both training and evaluation.

    Args:
        cfg: Model configuration
        hf_model_id: Optional HuggingFace model ID to load from

    Note:
        When use_unsloth is enabled for Qwen models:
        - The model will be loaded using unsloth's FastVisionModel
        - Automatically uses optimized gradient checkpointing
        - If use_peft is enabled, applies unsloth's optimized PEFT configuration
        - Use unsloth/Qwen models for best performance (e.g., unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit)
    """

    # Convert string dtype to torch dtype (used across all model loading paths)
    torch_dtype = getattr(torch, cfg.torch_dtype, torch.bfloat16)
    logger.info(f"Using torch dtype: {torch_dtype}")

    # Check if unsloth should be used
    use_unsloth = cfg.use_unsloth and "Qwen" in cfg.base_model_id

    if use_unsloth:
        logger.info("Unsloth mode enabled for faster training")

    # If quantization is enabled, use bitsandbytes (unless using unsloth)
    if cfg.quantization and not use_unsloth:
        bnb = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        bnb = None

    try:
        import flash_attn  # noqa: F401

        logger.info("Flash Attention 2 CUDA is available")
        has_flash_attn = True
    except:
        logger.info("Flash Attention 2 CUDA is not available")
        has_flash_attn = False

    if has_flash_attn:
        extra_kwargs = {"attn_implementation": "flash_attention_2"}
    else:
        extra_kwargs = {"attn_implementation": "sdpa"}

    # Determine if we're loading from a checkpoint and inspect any PEFT artifacts/weights.
    loading_from_checkpoint = bool(hf_model_id)
    checkpoint_path_for_load: str | None = None
    checkpoint_peft_info = {
        "has_adapter_files": False,
        "has_adapter_weights": False,
        "target_module": None,
        "adapter_load_path": None,
    }
    if hf_model_id:
        checkpoint_path_for_load = hf_model_id
        if checkpoint_path_for_load:
            # checkpoint_peft_info = _inspect_checkpoint_for_peft(checkpoint_path_for_load)
            if cfg.use_peft:
                if checkpoint_peft_info["has_adapter_files"]:
                    logger.info(
                        f"Checkpoint contains standalone PEFT adapter files for target "
                        f"{checkpoint_peft_info['target_module']}"
                    )
                elif checkpoint_peft_info["has_adapter_weights"]:
                    logger.info(
                        f"Checkpoint stores PEFT weights inside safetensors for target "
                        f"{checkpoint_peft_info['target_module']}; will reconstruct PEFT before loading"
                    )

    # For checkpoint loading, attach PEFT after wrapping in RBM so nested adapter layouts are preserved.
    apply_peft_before_wrap = cfg.use_peft and not loading_from_checkpoint

    # Load processor and tokenizer
    if "SmolVLM" in cfg.base_model_id or "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
        if "SmolVLM" in cfg.base_model_id:
            processor = AutoProcessor.from_pretrained(
                cfg.base_model_id,
                trust_remote_code=cfg.trust_remote_code,
                padding_side="left",
                size={"longest_edge": 512},
                max_image_size={"longest_edge": 512},
                use_fast=True,
            )
            logger.info(f"SmolVLM Processor: {processor}")

            base_model = AutoModelForImageTextToText.from_pretrained(
                cfg.base_model_id,
                torch_dtype=torch_dtype,
                **extra_kwargs,
                quantization_config=bnb,
            )
            model_cls = RBM

        elif "Qwen" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
            base_model = _load_base_model_standard(cfg, torch_dtype, extra_kwargs, bnb)
            tokenizer = None  # Will be loaded with processor

            model_cls = RBM

            # Setup processor and tokenizer
            processor = _setup_processor_and_tokenizer(cfg)
            if tokenizer is None:
                tokenizer = processor.tokenizer

        else:
            raise ValueError(f"Invalid base model id: {cfg.base_model_id}")

        # CRITICAL: Ensure PEFT is applied to base_model BEFORE wrapping in RBM (when checkpoint has adapters or we're not loading)
        # if apply_peft_before_wrap and cfg.use_peft and not isinstance(base_model, PeftModel):
        #     logger.warning("PEFT is enabled but base_model is not a PeftModel. Applying PEFT now...")
        #     if peft_config is None:
        #         raise ValueError("PEFT is enabled but peft_config is None. Cannot apply PEFT without configuration.")

        #     # Apply PEFT to base_model
        #     from peft import LoraConfig, get_peft_model

        #     lora_config = LoraConfig(
        #         r=peft_config.r,
        #         lora_alpha=peft_config.lora_alpha,
        #         target_modules=peft_config.target_modules,
        #         lora_dropout=peft_config.lora_dropout,
        #         bias=peft_config.bias,
        #     )
        #     base_model = get_peft_model(base_model, lora_config)
        #     logger.info("Applied PEFT to base_model before wrapping in RBM")

        # Verify PEFT was applied when we expect it
        # if apply_peft_before_wrap and cfg.use_peft:
        #     if isinstance(base_model, PeftModel):
        #         logger.info("Confirmed: base_model is a PeftModel - ready to load adapter weights from checkpoint")
        #     else:
        #         logger.error("CRITICAL: PEFT is enabled but base_model is not a PeftModel after applying PEFT!")
        #         raise ValueError(
        #             "Failed to apply PEFT to base_model. Cannot load adapter weights without PeftModel structure."
        #         )

        # Add special tokens and resize embeddings
        _add_special_tokens_and_resize(cfg, processor, base_model)

        # Initialize RBM model wrapper with the pre-loaded base model
        logger.info("Initializing RBM model...")
        tokenizer = processor.tokenizer

        model = model_cls(
            config=base_model.config,
            processor=processor,
            tokenizer=tokenizer,
            base_model=base_model,
            base_model_id=cfg.base_model_id,
            model_config=cfg,  # Pass ModelConfig for RBM-specific settings
        )

        # Load checkpoint if provided
        if hf_model_id:
            repo_id, revision_to_load = hf_model_id, None
            checkpoint_path = checkpoint_path_for_load
            if checkpoint_path is None:
                checkpoint_path = hf_model_id
            if checkpoint_path is None:
                raise ValueError(f"Could not resolve checkpoint path: {hf_model_id}")

            if cfg.use_peft:
                checkpoint_target = checkpoint_peft_info["target_module"] or (
                    "visual" if peft_config and peft_config.peft_vision_encoder else "language_model"
                )
                has_adapter_files = checkpoint_peft_info["has_adapter_files"]
                has_adapter_weights = checkpoint_peft_info["has_adapter_weights"]
                adapter_load_path = checkpoint_peft_info["adapter_load_path"] or checkpoint_path

                # if (has_adapter_files or has_adapter_weights) and not model_has_peft(model):
                #     if peft_config is None:
                #         raise ValueError("PEFT is enabled but peft_config is None. Cannot reconstruct adapters.")
                #     logger.info(f"Attaching PEFT to target '{checkpoint_target}' before loading checkpoint")
                #     model = setup_peft_model(model, peft_config, target_module=checkpoint_target)

                if _checkpoint_has_full_model_shards(checkpoint_path):
                    logger.info(
                        "Checkpoint includes full model safetensors shards; "
                        "preferring embedded full weights over standalone adapter files"
                    )
                    _load_checkpoint_weights_from_safetensors(
                        model,
                        checkpoint_path,
                        cfg,
                        load_adapters=True,
                        prefer_model_shards=True,
                    )
                # elif has_adapter_files:
                #     logger.info(
                #         f"Loading PEFT adapters from {adapter_load_path} into RBM target '{checkpoint_target}'"
                #     )
                #     try:
                #         if checkpoint_target == "model":
                #             model.model = PeftModel.from_pretrained(model.model, adapter_load_path)
                #         elif checkpoint_target == "visual":
                #             model.model.visual = PeftModel.from_pretrained(model.model.visual, adapter_load_path)
                #         else:
                #             model.model.language_model = PeftModel.from_pretrained(
                #                 model.model.language_model, adapter_load_path
                #             )
                #         logger.info("Successfully loaded PEFT adapters using PeftModel.from_pretrained()")
                #         if not _load_custom_heads_from_safetensors(model, checkpoint_path):
                #             logger.info("No custom_heads.safetensors found; falling back to generic safetensors loader")
                #             _load_checkpoint_weights_from_safetensors(
                #                 model,
                #                 checkpoint_path,
                #                 cfg,
                #                 load_adapters=False,
                #                 prefer_model_shards=True,
                #             )
                #     except Exception as e:
                #         logger.warning(f"PeftModel.from_pretrained() failed: {e}")
                #         logger.info("Falling back to manual loading for all weights")
                #         _load_checkpoint_weights_from_safetensors(
                #             model,
                #             checkpoint_path,
                #             cfg,
                #             load_adapters=True,
                #             prefer_model_shards=True,
                #         )
                # elif has_adapter_weights:
                #     logger.info("Checkpoint has embedded PEFT weights but no standalone adapter files")
                #     _load_checkpoint_weights_from_safetensors(
                #         model,
                #         checkpoint_path,
                #         cfg,
                #         load_adapters=True,
                #         prefer_model_shards=True,
                #     )
                else:
                    logger.info("Checkpoint has no PEFT adapter weights - loading plain model weights")
                    _load_checkpoint_weights_from_safetensors(
                        model,
                        checkpoint_path,
                        cfg,
                        load_adapters=True,
                        prefer_model_shards=True,
                    )
            else:
                # For non-PEFT models, we can use from_pretrained as before
                # Capture before weights for verification
                before_weights = {}
                if "Qwen2.5" in cfg.base_model_id:
                    before_weights = {
                        "visual": model.model.visual.blocks[0].mlp.down_proj.weight,
                        "progress_head": model.progress_head[0].weight,
                        "lm_embed_tokens": model.model.language_model.embed_tokens.weight,
                        "lm_layer": model.model.language_model.layers[0].mlp.up_proj.weight,
                    }
                elif "Qwen3" in cfg.base_model_id or "Molmo" in cfg.base_model_id:
                    before_weights = {
                        "visual": model.model.visual.blocks[0].mlp.linear_fc1.weight,
                        "progress_head": model.progress_head[0].weight,
                        "lm_embed_tokens": model.model.language_model.embed_tokens.weight,
                        "lm_layer": model.model.language_model.layers[0].mlp.up_proj.weight,
                    }

                # Load the model from the evaluation path
                # model = model_cls.from_pretrained(
                #     repo_id,
                #     processor=processor,
                #     tokenizer=tokenizer,
                #     base_model=base_model,
                #     base_model_id=cfg.base_model_id,
                #     model_config=cfg,
                #     revision=revision_to_load,
                # )
                _load_checkpoint_weights_from_safetensors(
                    model,
                    checkpoint_path,
                    cfg,
                    load_adapters=False,
                    prefer_model_shards=True,
                )

                # Verify weights were loaded
                if before_weights:
                    _verify_checkpoint_loading(cfg, model, before_weights)

    # elif "rewind_transformer" in cfg.base_model_id or "rewind_scale_transformer" in cfg.base_model_id:
    # elif "rewind" in cfg.base_model_id:
    #     # Initialize new model with encoders
    #     # Pretrained image and text encoders
    #     image_encoder = AutoModel.from_pretrained("facebook/dinov2-base")
    #     text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
    #     processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    #     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

    #     if hf_model_id:
    #         repo_id, revision_to_load = parse_hf_model_id_and_revision(hf_model_id, model_name="ReWiND model")

    #         model = ReWiNDTransformer.from_pretrained(
    #             repo_id,
    #             processor=processor,
    #             image_encoder=image_encoder,
    #             text_encoder=text_encoder,
    #             tokenizer=tokenizer,
    #             revision=revision_to_load,
    #         )
    #     else:
    #         train_img = cfg.train_vision_encoder
    #         train_text = cfg.train_language_model

    #         for p in image_encoder.parameters():
    #             p.requires_grad = train_img

    #         for p in text_encoder.parameters():
    #             p.requires_grad = train_text

    #         logger.info("Initializing ReWiND model...")
    #         model = ReWiNDTransformer(
    #             config=cfg,
    #             processor=processor,
    #             tokenizer=tokenizer,
    #             image_encoder=image_encoder,
    #             text_encoder=text_encoder,
    #         )

    logger.info("Model architecture initialized")
    logger.info(f"Model architecture: {model}")

    # Configure which parts of the model to train based on config
    # IMPORTANT: When using PEFT (via Unsloth or standard), PEFT already handles freezing
    # base model parameters. We should NOT override requires_grad on base model params.
    peft_applied = cfg.use_peft and (cfg.use_unsloth or cfg.peft_vision_encoder)

    # Helper function to check if a parameter is part of the base model (vision/language)
    def is_base_model_param(name: str) -> bool:
        """Check if parameter belongs to base model (vision/language) that PEFT handles."""
        base_model_patterns = ["visual", "vision", "language_model", "text_encoder", "text_model", "image_encoder"]
        return any(pattern in name for pattern in base_model_patterns)

    # Helper function to check if a parameter is a prediction head
    def is_prediction_head(name: str) -> bool:
        """Check if parameter belongs to a prediction head."""
        head_patterns = ["progress_head", "success_head", "preference_head", "similarity_head"]
        return any(pattern in name for pattern in head_patterns)

    for name, param in model.named_parameters():
        # 1. Handle prediction heads - always controlled by their individual flags
        if is_prediction_head(name):
            if "progress_head" in name:
                param.requires_grad = cfg.train_progress_head
            elif "success_head" in name:
                param.requires_grad = cfg.train_success_head
            elif "preference_head" in name:
                param.requires_grad = cfg.train_preference_head
            elif "similarity_head" in name:
                param.requires_grad = False

        # 2. Handle base model parameters (vision/language) - skip if PEFT is applied
        elif is_base_model_param(name):
            if peft_applied:
                # PEFT handles freezing/unfreezing - don't override
                continue

            # Set requires_grad based on config flags
            if "visual" in name or "vision" in name or "vision_model" in name:
                param.requires_grad = cfg.train_vision_encoder
            elif "language_model" in name or "text_encoder" in name or "text_model" in name:
                param.requires_grad = cfg.train_language_model
            elif "image_encoder" in name:
                param.requires_grad = cfg.train_vision_encoder

        # 3. Handle special cases
        elif "lm_head" in name:
            # Language modeling head should not be trainable for RBM
            param.requires_grad = False

        # 4. All other parameters (custom RBM parameters like frame_pool_attn, video_proj, text_proj)
        # should always be trainable
        else:
            param.requires_grad = True

    logger.info("Training configuration:")
    logger.info(f"  - Vision encoder: {cfg.train_vision_encoder}")
    logger.info(f"  - Language model: {cfg.train_language_model}")
    logger.info(f"  - Progress head: {cfg.train_progress_head}")
    logger.info(f"  - Success head: {getattr(cfg, 'train_success_head', False)}")
    logger.info(f"  - Preference head: {cfg.train_preference_head}")

    # When use_peft, skip verbose param list here; it will be printed after PEFT in setup_peft_model
    if not cfg.use_peft:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name:60} | {param.shape} | RG: {param.requires_grad}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.4f}"
    )
    return tokenizer, processor, model


def load_model_from_hf(
    model_path: str,
    base_model_path: str,
    device: torch.device,
) -> tuple[ExperimentConfig | None, Any | None, Any | None, Any | None]:
    """
    Load reward model config and model from HuggingFace or local checkpoint.

    This mirrors the logic used by the training/eval scripts:
    - Resolve checkpoint path (supports HF Hub with @ notation)
    - Locate config.yaml locally (if model_path is a directory) or download from HF
    - Use custom YAML loader for ReWiND configs
    - Filter config keys to ExperimentConfig
    - Clear training/logging sections
    - Load model artifacts via setup_model_and_processor

    Args:
        model_path: HuggingFace model repository ID or local checkpoint path.
                   Supports @ notation for tags: username/model@tag-name
        device: Device to load model on
    Returns:
        Tuple of (exp_config, tokenizer, processor, reward_model)
    """
    # Resolve checkpoint path (handles HF Hub downloads with @ notation)
    resolved_path = model_path
    if resolved_path is None:
        raise ValueError(f"Could not resolve checkpoint path: {model_path}")

    config_path: str | None = None

    # Parse repo_id and revision (tag) from model_path if using @tag format
    # This is used for downloading config.yaml if needed
    if "@" in model_path:
        repo_id, revision = model_path.split("@", 1)
    else:
        repo_id, revision = model_path, None

    resolved_path = Path(resolved_path)

    if resolved_path.exists():
        # Local checkpoint: look for config.yaml
        candidate_paths = [
            resolved_path / "config.yaml",
            resolved_path.parent / "config.yaml",
        ]
        config_path = None
        for candidate in candidate_paths:
            if candidate.is_file():
                config_path = candidate
                break

        # If config.yaml not found locally, try to download it from Hub
        if config_path is None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub") from e

            # Check if this is a HuggingFace repo (not a local path)
            if "/" in repo_id and not repo_id.startswith("/"):
                logger.info(
                    f"config.yaml not found locally, downloading from HuggingFace Hub: {repo_id}@{revision or 'latest'}"
                )
                try:
                    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
                    logger.info(f"Downloaded config.yaml to: {config_path}")
                except Exception as e:
                    logger.warning(f"Could not download config.yaml from Hub: {e}")
                    raise ValueError(f"config.yaml not found locally and could not be downloaded from Hub: {e}") from e
            else:
                raise ValueError(f"config.yaml not found in checkpoint directory or parent directory: {resolved_path}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub") from e
        # Download config with revision if specified
        logger.info(f"Downloading config.yaml from HuggingFace Hub: {repo_id}@{revision or 'latest'}")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml", revision=revision)
        logger.info(f"Downloaded config.yaml to: {config_path}")

    with open(config_path) as f:
        yaml_text = f.read()

    class _ConfigSafeLoader(yaml.SafeLoader):
        pass

    _ConfigSafeLoader.add_constructor(
        "tag:yaml.org,2002:python/object:robometer.models.rewind_transformer.ReWINDTransformerConfig",
        lambda loader, node: loader.construct_mapping(node),
    )

    model_config_dict = yaml.load(yaml_text, Loader=_ConfigSafeLoader)

    valid_keys = {f.name for f in fields(ExperimentConfig)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_keys}

    exp_config = ExperimentConfig(**filtered_config)
    # Use resolved_path for loading the actual model
    # Import here to avoid circular dependency with setup_utils

    # Extract PEFT config from the loaded experiment config
    peft_config = exp_config.peft if hasattr(exp_config, "peft") and exp_config.model.use_peft else None
    exp_config.model.base_model_id = base_model_path  # Override base_model_id with provided base_model_path
    tokenizer, processor, reward_model = setup_model_and_processor(
        exp_config.model, str(resolved_path), peft_config=peft_config
    )
    reward_model = reward_model.to(device)
    reward_model.eval()

    return exp_config, tokenizer, processor, reward_model
