from __future__ import annotations

from typing import Any, ClassVar, List
import logging
import sys
import numpy as np
from openai import OpenAI
import time
import os

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..utils import normalize_api_base_url
from .sole_r1.constants import (
    API_RESPONSE_REPLACEMENTS,
    USER_PROMPT_TEMPLATE_GPT,
)
from .sole_r1.utils import count_images, encode_images, process_images, build_payload
from ..base import Infer
logger = logging.getLogger(__name__)


@register_infer("openai_compatible")
class RemoteModel(Infer):
    """Progress model served through an OpenAI-compatible Chat Completions API."""

    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.base_url:
            raise ValueError("OpenAI-compatible inference requires infer.base_url")
        if not config.model_id:
            raise ValueError("OpenAI-compatible inference requires infer.model_id")
        self.config
        self.model_id = config.model_id or os.getenv("MODEL_ID")
        self.base_url = normalize_api_base_url(
            config.base_url or os.getenv("BASE_URL")
        )
        self.api_key = config.api_key or os.getenv("API_KEY") or "EMPTY"
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        

    def gpt(
        self,
        task_description_i: str,
        frame_list: list[str],
        try_count_max: int = 3,
    ) -> tuple[list[int], list[str], list]:
        """Predict task progress for a single episode using GPT.

        Args:
            model: The GPT model client.
            task_description_i: Description of the task.
            frame_list: List of base64-encoded image strings.
            try_count_max: Maximum number of retry attempts for API calls.

        Returns:
            progress_list: List of predicted progress percentages.
            response_text_list: List of raw response texts from the model.
            prompt_list: List of prompts sent to the model.
        """
        model = self.client
        image_file_num_list_idx = list(range(1, len(frame_list)))

        current_progress = 0
        prompt_list = []
        progress_list = []
        response_text_list = []
        messages_content = []
        response_text = None
        for current_idx in range(len(image_file_num_list_idx)):
            try_count = 0
            while try_count < try_count_max:
                try:
                    base64_image_prev = frame_list[image_file_num_list_idx[current_idx] - 1]
                    base64_image_current = frame_list[image_file_num_list_idx[current_idx]]
                    messages_content = [
                        {
                            "type": "text",
                            "text": USER_PROMPT_TEMPLATE_GPT.format(
                                task_description=task_description_i,
                                prev_progress=current_progress,
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_prev}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image_current}"},
                        },
                    ]
                    #
                    response = model.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": messages_content}],
                    )
                    response_text = response.choices[0].message.content

                    for src, target in API_RESPONSE_REPLACEMENTS.items():
                        response_text = response_text.replace(src, target)

                    current_progress = int(
                        response_text.split("<answer>")[1].split("</answer>")[0].replace("%", "").strip()
                    )
                    progress_list.append(current_progress)
                    response_text_list.append(response_text)
                    prompt_list.append(messages_content[0])
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    try_count += 1
                    if "quota" in str(e).lower():
                        print("Token quota exceeded, sleeping for 60 seconds.")
                        time.sleep(60)
                    print(f"Response: {response_text}")

            if try_count >= try_count_max:
                response_text_list.append("")
                if len(progress_list) > 0:
                    current_progress = progress_list[-1]
                else:
                    current_progress = 0
                progress_list.append(current_progress)
                if len(messages_content) > 0:
                    prompt_list.append(messages_content[0])
                else:
                    prompt_list.append("")

           
            logger.debug("\n\n*******************************************************************************")
            logger.debug(prompt_list[-1])
            logger.debug("\n----------------- Response -----------------")
            logger.debug(response_text_list[-1])
            logger.debug(progress_list[-1])

        return progress_list, response_text_list, prompt_list

    def callback(self, payload: dict[str, Any]) -> np.ndarray:
        if "front_images" not in payload:
            return {"success": False, "message": "No 'front_images' in payload."}
        if "tasks" not in payload:
            return {"success": False, "message": "No 'task' in payload."}

        try:
            front_images = payload["front_images"]
            tasks = payload["tasks"]

            num_episodes, _ = count_images(front_images)
            processed_front_images = process_images(front_images)
            encoded_front_images = encode_images(processed_front_images)

            answer_list_list = []
            response_text_list_list = []
            prompt_list_list = []

            results = []
            for episode_idx in range(num_episodes):
                progress_list, response_text_list, prompt_list = self.gpt(
                    tasks[episode_idx],
                    encoded_front_images[episode_idx],
                )

                logger.debug("\n\n----------------------------------------")
                logger.debug(f"Episode {episode_idx}.")
                logger.debug(f"progress_list: {progress_list}")
                results.append((episode_idx, progress_list, response_text_list, prompt_list))

            # Sort results by episode_idx to maintain order
            results.sort(key=lambda x: x[0])

            for _, progress_list, response_text_list, prompt_list in results:
                answer_list_list.append([str(v) for v in progress_list])
                response_text_list_list.append(response_text_list)
                prompt_list_list.append(prompt_list)

            valid_answers = []
            for episode in answer_list_list:
                valid_answers_ = [0]  # Assume the first step is always 0.
                for ans in episode:
                    try:
                        progress = int(ans)
                    except (ValueError, TypeError):
                        progress = valid_answers_[-1]
                    valid_answers_.append(progress)
                valid_answers.append(np.array(valid_answers_, dtype=np.int32))

            return {
                "success": True,
                "data": {
                    "valid_answers": valid_answers,
                    "text_outputs": response_text_list_list,
                    "text_inputs": prompt_list_list,
                },
            }

        except Exception as e:
            return {"success": False, "message": str(e)}

    def compute_batch_progress(
        self,
        frames_array: List[np.ndarray],
        task_description: List[str],
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
        )
        response = self.callback(payload)
        if response is None:
            logging.error("No response received from server (timeout or connection error).")
            sys.exit(1)

        if not response.get("success", False):
            logging.error("Server returned an error: %s", response.get("message", "<no message>"))
            sys.exit(1)

        data = response["data"]
        valid_answers = data["valid_answers"]

        return [(0.01 * answers).tolist() for answers in valid_answers]

    def predict(self, samples: List[EvaluationSample]) -> List[Prediction]:
        result = []
        if not isinstance(samples[0], ProgressSample):
            raise TypeError(f"{self.config.name} only supports progress samples")
        values = self.compute_batch_progress(
            [np.asarray(sample.trajectory.frames) for sample in samples],
            [sample.trajectory.task for sample in samples]
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
