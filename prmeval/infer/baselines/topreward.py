from typing import Any, ClassVar

import numpy as np

from ...core.registry import register_infer
from ..base import RemoteError
from ..openai import OpenAIChatInfer
from .common import prediction, progress_content, progress_sample


@register_infer("topreward")
class TopRewardRemote(OpenAIChatInfer):
    capabilities: ClassVar[set[str]] = {"progress"}

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

    def predict(self, sample):
        sample = progress_sample(sample, "TOPReward")
        frames = list(sample.trajectory.frames)
        num_frames = len(frames)
        configured_samples = self.config.options.get("num_prefix_samples", 15)
        if not isinstance(configured_samples, int) or isinstance(configured_samples, bool) or configured_samples < 1:
            raise ValueError("TOPReward options.num_prefix_samples must be a positive integer")
        if num_frames > 2:
            count = min(configured_samples, num_frames)
            prefix_lengths = sorted({int(value) for value in np.linspace(1, num_frames, count, dtype=int)})
        else:
            prefix_lengths = [num_frames]

        rewards: list[float] = []
        raw_responses = []
        task = sample.trajectory.task or "Complete the task."
        prompt = (
            "The supplied frames show a robot manipulation trajectory that completes the following task: "
            f"{task} Decide whether the preceding statement is True or False. Answer with exactly True or False."
        )
        for length in prefix_lengths:
            messages = [{"role": "user", "content": progress_content(prompt, frames[:length])}]
            last_error = None
            response = None
            for parse_attempt in range(self.config.max_retries + 1):
                response = self._completion(
                    messages,
                    {
                        "temperature": 0,
                        "max_tokens": 1,
                        "logprobs": True,
                        "top_logprobs": 20,
                    },
                )
                try:
                    rewards.append(self._true_logprob(response))
                    break
                except RemoteError as exc:
                    last_error = exc
                    if parse_attempt >= self.config.max_retries:
                        raise RemoteError(f"TOPReward could not obtain True logprob: {last_error}") from exc
            assert response is not None
            raw_responses.append({"prefix_length": length, "true_logprob": rewards[-1], "response": response})

        reward_array = np.asarray(rewards, dtype=float)
        if len(reward_array) == 1 or float(reward_array.max()) == float(reward_array.min()):
            normalized = np.ones_like(reward_array)
        else:
            normalized = (reward_array - reward_array.min()) / (reward_array.max() - reward_array.min())
        values = np.interp(
            np.arange(1, num_frames + 1, dtype=float),
            np.asarray(prefix_lengths, dtype=float),
            normalized,
        )
        return prediction(sample, np.clip(values, 0.0, 1.0).tolist(), self.config, raw_responses)
