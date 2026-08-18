#!/usr/bin/env python3
# ruff: noqa: E501
"""Direct VLM baseline for preference comparison."""

import base64
import os
import random
import time
from io import BytesIO
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, PreferencePrediction, PreferenceSample
from ..base import Infer, model_identity

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@register_infer("rlvlmf")
class RLVLMF(Infer):
    capabilities: ClassVar[set[str]] = {"preference"}
    """RL-VLM-F baseline for preference comparison between trajectories.

    Uses vision-language models (Gemini or GPT-4V) to directly compare two trajectories
    and predict which one better achieves a given task goal.
    """

    def __init__(self, config: InferConfig):
        super().__init__(config)
        options = config.options
        self.api_key = config.api_key or options.get("api_key")
        self.model_name = str(options.get("model_name", config.model_id or "gemini-2.5-flash"))
        self._initialize(
            vlm_provider=str(options.get("provider", "gemini")).lower(),
            temperature=config.temperature,
        )

    def _initialize(self, vlm_provider: str = "gemini", temperature: float = 0.0):
        self.vlm_provider = vlm_provider
        self.temperature = temperature

        # Setup VLM
        self._setup_vlm()

    def _setup_vlm(self):
        """Initialize VLM provider."""
        if self.vlm_provider == "gemini":
            api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if not GEMINI_AVAILABLE:
                raise RuntimeError("RLVLMF Gemini provider requires google-generativeai")
            if not api_key:
                raise ValueError("Set GEMINI_API_KEY environment variable")

            genai.configure(api_key=api_key)
            # Original RL-VLM-F used 'gemini-pro-vision' (deprecated); we use a supported model
            # Using latest flash model for speed and cost-effectiveness
            self.model = genai.GenerativeModel(self.model_name)

        elif self.vlm_provider == "openai":
            api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not OPENAI_AVAILABLE:
                raise RuntimeError("RLVLMF OpenAI provider requires openai")
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY environment variable")

            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.vlm_provider}")

    def compute_preference(
        self, chosen_images: list[Image.Image], rejected_images: list[Image.Image], task_description: str = ""
    ) -> dict[str, Any]:
        """Compute preference prediction between trajectories using VLM."""
        start_time = time.time()

        # DEBUG: Print task that reaches VLM baseline
        print(f"VLM Baseline received task: '{task_description}'")

        # Select last frame from each trajectory
        chosen_frame = chosen_images[-1]
        rejected_frame = rejected_images[-1]

        # Randomize image order to avoid position bias
        # chosen_is_first = True means chosen goes to Image A, False means chosen goes to Image B
        chosen_is_first = random.choice([True, False])

        if chosen_is_first:
            image_a, image_b = chosen_frame, rejected_frame
        else:
            image_a, image_b = rejected_frame, chosen_frame

        # Query VLM with randomized images
        prompt = self._build_prompt(task_description)

        if self.vlm_provider == "gemini":
            preference, raw_response = self._query_gemini(prompt, image_a, image_b)
        else:
            preference, raw_response = self._query_openai(prompt, image_a, image_b)

        # Process result - adjust for randomized order
        # preference is "A"/"B"/"tie" based on VLM's choice of Image A vs Image B
        # We need to map this back to chosen vs rejected based on randomization
        if preference == "A":
            # VLM chose Image A
            vlm_chose_chosen = chosen_is_first  # If chosen is first, A=chosen, else A=rejected
        elif preference == "B":
            # VLM chose Image B
            vlm_chose_chosen = not chosen_is_first  # If chosen is first, B=rejected, else B=chosen
        else:  # preference == "tie"
            vlm_chose_chosen = None  # Tie case

        # Determine correctness
        if vlm_chose_chosen is True:
            is_correct = True  # VLM correctly chose the chosen trajectory
        elif vlm_chose_chosen is None:
            # For ties, treating as incorrect since we expect clear preferences
            is_correct = False
        else:  # vlm_chose_chosen is False
            is_correct = False  # VLM incorrectly chose the rejected trajectory

        # Convert preference to probability
        # "A" means chosen was preferred, "B" means rejected was preferred, "tie" is 0.5
        if preference == "A":
            prediction_prob = 1.0  # Chosen is preferred
        elif preference == "B":
            prediction_prob = 0.0  # Rejected is preferred
        else:  # tie
            prediction_prob = 0.5

        # preference_pred: 1 if chosen is selected, 0 otherwise
        preference_pred = 1.0 if vlm_chose_chosen is True else 0.0

        processing_time = time.time() - start_time

        # Build result with all metadata
        result = {
            "is_correct": is_correct,
            "vlm_preference": preference,
            "prediction_prob": prediction_prob,
            "preference_pred": preference_pred,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": task_description,
            "num_chosen_frames": len(chosen_images),
            "num_rejected_frames": len(rejected_images),
            "selected_frames": 1,
            "strategy": "last_frame",
            "success": True,
            "error": None,
            "vlm_response": raw_response[:500] + "...",
            "vlm_chose_chosen": vlm_chose_chosen,
            "chosen_is_first": chosen_is_first,
            "prompt": prompt[:200] + "...",
            "processing_time_seconds": processing_time,
        }

        return result

    def _build_prompt(self, task: str) -> str:
        """Build RL-VLM-F prompt - exact match to original paper."""
        base = """ Each frame comes from a robot trajectory. (Think causally and use image comparison to verify any confusion between the base of the robot and the end effector)
1. What is shown in the first image (Image A)?
2. What is shown in the second image (Image B)?
3. For this question here is the Goal Text: {goal_text}
Is the goal being better achieved in Image A or Image B?
Reply a single line of 0 if the goal is better achieved in Image A, or 1 if it is better achieved in Image B.
Reply -1 if the text is really unsure about either making any progress towards the goal or there is absolutely no difference. (only use this if there is no discernible signs of progress in either image or no discernible difference between the progress in the two images for the given task)
"""

        if task:
            goal_text = f"The goal is {task}."
        else:
            goal_text = "Which image shows better task execution?"

        return base.format(goal_text=goal_text)

    def _query_gemini(self, prompt: str, chosen: Image.Image, rejected: Image.Image) -> tuple[str, str]:
        """Query Gemini for preference."""
        query = ["Consider the following two images:\nImage 1:", chosen, "Image 2:", rejected, prompt]

        response = self.model.generate_content(
            query,
            generation_config=genai.types.GenerationConfig(temperature=self.temperature),
            safety_settings=[
                {"category": cat, "threshold": "BLOCK_NONE"}
                for cat in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS",
                ]
            ],
        )

        full_response = response.text
        # Fix parsing: strip whitespace first, then split and take last non-empty line
        result = full_response.strip()
        if "\n" in result:
            lines = [line.strip() for line in result.split("\n") if line.strip()]
            result = lines[-1] if lines else ""

        # Parse response
        if "-1" in result:
            return "tie", full_response
        elif "0" in result:
            return "A", full_response  # Image 1 (chosen)
        elif "1" in result:
            return "B", full_response  # Image 2 (rejected)
        else:
            return "tie", full_response

    def _query_openai(self, prompt: str, chosen: Image.Image, rejected: Image.Image) -> tuple[str, str]:
        """Query GPT-4V for preference."""

        def to_base64(img: Image.Image) -> str:
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode()

        content = [
            {"type": "text", "text": "Consider the following two images:\nImage 1:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{to_base64(chosen)}", "detail": "high"},
            },
            {"type": "text", "text": "Image 2:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{to_base64(rejected)}", "detail": "high"},
            },
            {"type": "text", "text": prompt},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=1000,
        )

        full_response = response.choices[0].message.content.strip()
        result = full_response.split("\n")[-1].strip()

        # Parse response
        if "-1" in result:
            return "tie", full_response
        elif "0" in result:
            return "A", full_response
        elif "1" in result:
            return "B", full_response
        else:
            return "tie", full_response

    def predict(self, sample: EvaluationSample) -> Prediction:
        if not isinstance(sample, PreferenceSample):
            raise TypeError(f"{self.config.name} only supports preference samples")

        def to_images(frames) -> list[Image.Image]:
            images = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    images.append(frame)
                    continue
                array = np.asarray(frame)
                if array.dtype != np.uint8:
                    array = np.clip(array, 0, 255).astype(np.uint8)
                images.append(Image.fromarray(array))
            return images

        result = self.compute_preference(
            to_images(sample.chosen_trajectory.frames),
            to_images(sample.rejected_trajectory.frames),
            sample.chosen_trajectory.task,
        )
        choice = result.get("vlm_chose_chosen")
        preference = "tie" if choice is None else ("chosen" if choice else "rejected")
        probability = 0.5 if choice is None else (1.0 if choice else 0.0)
        return PreferencePrediction(
            sample_id=sample.sample_id,
            chosen_probability=probability,
            preference=preference,
            model=model_identity(self.config),
            model_version=self.config.model_version,
            raw_response=result,
        )
