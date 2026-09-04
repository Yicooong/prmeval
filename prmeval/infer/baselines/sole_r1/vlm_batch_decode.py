# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import gc
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor
from trl.data_utils import maybe_apply_chat_template

from .utils import (
    assemble_output_batch,
    get_answer_from_completion,
    make_conversation_image,
)


def load_model(
    checkpoint_path: Path,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    max_tokens: int = 200,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
    base_model_path: str = "Qwen/Qwen3-VL-8B-Instruct",
):
    """Load a Qwen VLM model and preprocessor.

    Args:
        checkpoint_path (Path): Path to the model checkpoint.
        gpu_memory_utilization (float): GPU memory utilization for loading the model.
        enable_prefix_caching (bool): Whether to enable prefix caching.
        max_model_len (int): Maximum sequence length.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.
        top_k (int): Top-k sampling parameter.
        max_tokens (int): Maximum number of tokens to generate.
        min_pixels (int): Minimum number of pixels for image processing.
        max_pixels (int): Maximum number of pixels for image processing.

    There are two quirks to resolve.
    1. This function first edits the saved model files, then loads and saved the model
       and then loads it again.
    2. We probably don't have to return two preprocessors.

    Returns:
        llm: Loaded LLM model.
        processing_class: Preprocessor object with updated parameters.
        processor: Original preprocessor object.
        generation_config: Default Transformers generation keyword arguments.
    """

    # Load only Qwen 3 8B.
    assert "q3vl8b", "We're only supporting Qwen 3 8B for these experiments."

    # Load model and processor.
    processing_class = AutoProcessor.from_pretrained(base_model_path)
    processing_class.tokenizer.padding_side = "left"
    pad_token_id = processing_class.tokenizer.pad_token_id
    processing_class.pad_token_id = pad_token_id
    processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
    processing_class.image_processor.max_pixels = max_pixels
    processing_class.image_processor.min_pixels = min_pixels

    processor = AutoProcessor.from_pretrained(checkpoint_path)

    llm = AutoModelForMultimodalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map="auto",
    )
    llm.eval()

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_tokens,
    }
    return llm, processing_class, processor, generation_config


def vlm_batch_decode(
    processing_class,
    processor,
    llm,
    generation_config,
    video_count: int,
    video_step_count: list[int],
    vlm_images: list[list[Image.Image]],
    questions: list[str],
    max_prompt_length: int = 2048,
    from_zero: bool = False,
):
    """Batch decode VLM model across multiple videos and time steps.

    Args:
        processing_class: Preprocessor object for preparing inputs.
        processor: Original preprocessor object.
        llm: Loaded LLM model.
        generation_config: Keyword arguments passed to ``transformers.generate``.
        video_count: Number of videos to process.
        video_step_count: List of step counts for each video.
        vlm_images: List of lists of images for each video.
        questions: Question prompt to use for each time step.
        max_prompt_length: Maximum length of the prompt input.
        from_zero: If False, includes previous progress in the prompt.
    """
    max_video_step = max(video_step_count)

    # batch decoding
    text_output_list_batch = []
    text_input_list_batch = []
    answer_list_batch = []
    prev_answer_batch = [["0" for _ in range(video_count)]]

    # Subtract one because we don't predict progress for the first time step.
    for video_step in range(max_video_step - 1):
        gc.collect()

        current_batch_prompts = []
        current_batch_images = []
        current_indices = []

        for ep_idx in range(video_count):
            # Subtract one because we don't predict progress for the first time step.
            # We video_step_count == 3, then we have 2 images and 2 predictions to make.
            if video_step >= video_step_count[ep_idx] - 1:
                continue
            else:
                image = vlm_images[ep_idx][video_step]
                question = questions[ep_idx]

                prompt = make_conversation_image(question)

                inputs = [
                    {
                        "image": image,
                        "problem": question,
                        "image_name": "hello123",
                        "prompt": prompt,
                    }
                ]

                prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]

                images = [x["image"] for x in inputs]

                if not from_zero:
                    prev_answer = prev_answer_batch[video_step][ep_idx]
                    replace_start = "The task progress for the previous timestep is "
                    replace_end = "%. "
                    prompts_text[0] = (
                        prompts_text[0].split(replace_start)[0]
                        + replace_start
                        + prev_answer
                        + replace_end
                        + prompts_text[0].split(replace_start)[1].split(replace_end)[1]
                    )

                current_batch_prompts.append(prompts_text[0])
                current_batch_images.append(images[0])
                current_indices.append(ep_idx)

        processor_kwargs = {
            "text": current_batch_prompts,
            "images": current_batch_images,
            "return_tensors": "pt",
            "padding": True,
            "add_special_tokens": False,
        }
        if max_prompt_length is not None:
            processor_kwargs.update(truncation=True, max_length=max_prompt_length)

        model_inputs = processing_class(**processor_kwargs)
        model_device = next(llm.parameters()).device
        model_inputs = model_inputs.to(model_device)

        generate_kwargs = dict(generation_config)
        generate_kwargs["do_sample"] = generate_kwargs.get("temperature", 1.0) > 0
        if not generate_kwargs["do_sample"]:
            generate_kwargs.pop("temperature", None)
            generate_kwargs.pop("top_p", None)
            generate_kwargs.pop("top_k", None)

        with torch.inference_mode():
            outputs = llm.generate(**model_inputs, **generate_kwargs)

        if outputs.shape[0] != len(current_batch_prompts):
            raise RuntimeError(
                f"Outputs length {outputs.shape[0]} is not equal to inputs length {len(current_batch_prompts)}"
            )

        # Decoder-only Transformers models return prompt + completion. Decode only
        # the newly generated tokens to match the old vLLM behavior.
        completion_ids = outputs[:, model_inputs["input_ids"].shape[1] :]

        text_output = processor.batch_decode(completion_ids, skip_special_tokens=True)

        text_output_list_batch = [
            *text_output_list_batch,
            assemble_output_batch(text_output, current_indices, video_count),
        ]

        text_input_list_batch = [
            *text_input_list_batch,
            assemble_output_batch(current_batch_prompts, current_indices, video_count),
        ]
        current_answers = [get_answer_from_completion(text_output_i) for text_output_i in text_output]
        current_answer_batch = assemble_output_batch(
            current_answers,
            current_indices,
            video_count,
        )
        prev_answer_batch.append(current_answer_batch)
        answer_list_batch.append(current_answer_batch)

    return (
        text_input_list_batch,
        text_output_list_batch,
        answer_list_batch,
    )
