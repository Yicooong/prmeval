from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..core.config import InferConfig
from ..core.registry import register_infer
from ..core.schemas import (
    EvaluationSample,
    Prediction,
    PreferencePrediction,
    PreferenceSample,
    ProgressPrediction,
    ProgressSample,
)
from .base import Infer


@dataclass
class ProgressResult:
    """Optional rich return value for models that need to preserve raw backend output."""

    values: np.ndarray | list[float]
    raw_response: Any = None


@dataclass
class PreferenceResult:
    chosen_probability: float
    preference: str
    raw_response: Any = None


class RemoteContext:
    """Shared OpenAI-compatible client exposed to optional model remote methods."""

    def __init__(self, config: InferConfig):
        from .openai import OpenAIChatInfer

        self.config = config
        self._client = OpenAIChatInfer(config)

    def begin_prediction(self) -> None:
        self._client.begin_prediction()

    def attempts(self) -> int:
        return self._client.attempts()

    def completion(
        self,
        messages: list[dict[str, Any]],
        request_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._client._completion(messages, request_options)

    def chat(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        request_options: dict[str, Any] | None = None,
        validator: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._client._chat(messages, schema, request_options, validator)


class ProgressModel(ABC):
    """Local-first model API with optional batching and remote execution methods."""

    supports_local: ClassVar[bool] = True
    supports_local_batch: ClassVar[bool] = False
    supports_remote: ClassVar[bool] = False
    supports_remote_batch: ClassVar[bool] = False

    @classmethod
    def load_local(cls, model_path: str, options: dict[str, Any]) -> ProgressModel:
        return cls(model_path=model_path, **options)  # type: ignore[call-arg]

    @abstractmethod
    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray | list[float] | ProgressResult:
        raise NotImplementedError

    def compute_progress_batch(
        self,
        frames_list: list[np.ndarray],
        task_descriptions: list[str],
        reference_video_paths: list[str | None] | None = None,
    ) -> list[np.ndarray | list[float] | ProgressResult]:
        paths = reference_video_paths or [None] * len(frames_list)
        return [
            self.compute_progress(frames, task, path)
            for frames, task, path in zip(frames_list, task_descriptions, paths, strict=True)
        ]

    @classmethod
    def remote_compute_progress(
        cls,
        frames_array: np.ndarray,
        task_description: str,
        reference_video_path: str | None,
        remote: RemoteContext,
        options: dict[str, Any],
    ) -> np.ndarray | list[float] | ProgressResult:
        raise NotImplementedError(f"{cls.__name__} does not support remote inference")

    @classmethod
    def remote_compute_progress_batch(
        cls,
        frames_list: list[np.ndarray],
        task_descriptions: list[str],
        reference_video_paths: list[str | None],
        remote: RemoteContext,
        options: dict[str, Any],
    ) -> list[np.ndarray | list[float] | ProgressResult]:
        return [
            cls.remote_compute_progress(frames, task, path, remote, options)
            for frames, task, path in zip(
                frames_list,
                task_descriptions,
                reference_video_paths,
                strict=True,
            )
        ]


class PreferenceModel(ABC):
    """Local-first preference model API with an optional remote implementation."""

    supports_local: ClassVar[bool] = True
    supports_remote: ClassVar[bool] = False

    @classmethod
    def load_local(cls, model_path: str, options: dict[str, Any]) -> PreferenceModel:
        return cls(model_path=model_path, **options)  # type: ignore[call-arg]

    @abstractmethod
    def compute_preference(
        self,
        chosen_frames: Any,
        rejected_frames: Any,
        task_description: str = "",
    ) -> PreferenceResult | dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def remote_compute_preference(
        cls,
        chosen_frames: Any,
        rejected_frames: Any,
        task_description: str,
        remote: RemoteContext,
        options: dict[str, Any],
    ) -> PreferenceResult:
        raise NotImplementedError(f"{cls.__name__} does not support remote inference")

class ProgressModelInfer(Infer):
    """Bridge a registered ProgressModel to the Stage 2 EvaluationSample protocol."""

    capabilities: ClassVar[set[str]] = {"progress"}
    model_cls: ClassVar[type[ProgressModel]]

    @classmethod
    def supports_batch_mode(cls, mode: str) -> bool:
        return cls.model_cls.supports_local_batch if mode == "local" else cls.model_cls.supports_remote_batch

    @classmethod
    def thread_safe_mode(cls, mode: str) -> bool:
        return mode == "remote"

    def __init__(self, config: InferConfig):
        super().__init__(config)
        self.transport = str(config.transport)
        self.thread_safe = self.thread_safe_mode(str(config.mode))
        self.supports_batch = self.supports_batch_mode(str(config.mode))
        self.local_model: ProgressModel | None = None
        self.remote: RemoteContext | None = None
        if config.mode == "local":
            assert config.model_path is not None
            self.local_model = self.model_cls.load_local(config.model_path, config.options)
        else:
            self.remote = RemoteContext(config)

    def begin_prediction(self) -> None:
        if self.remote:
            self.remote.begin_prediction()

    def attempts(self) -> int:
        return self.remote.attempts() if self.remote else 1

    def model_info(self) -> dict[str, Any]:
        info = super().model_info()
        if self.config.mode == "local":
            info["model_path"] = self.config.model_path
        else:
            info["base_url"] = self.config.base_url
        info["batch_size"] = self.config.batch_size
        info["max_concurrency"] = self.config.max_concurrency
        return info

    @staticmethod
    def _reference_path(sample: ProgressSample) -> str | None:
        value = sample.trajectory.metadata.get("reference_video_path")
        return str(Path(value)) if value else None

    def _prediction(
        self,
        sample: ProgressSample,
        result: np.ndarray | list[float] | ProgressResult,
    ) -> ProgressPrediction:
        raw = result.raw_response if isinstance(result, ProgressResult) else None
        values = result.values if isinstance(result, ProgressResult) else result
        array = np.asarray(values, dtype=float).reshape(-1)
        expected = len(sample.trajectory.frames)
        if len(array) != expected:
            raise ValueError(f"Progress length mismatch: expected {expected}, got {len(array)}")
        if not np.isfinite(array).all():
            raise ValueError("Progress values must be finite")
        if ((array < 0) | (array > 1)).any():
            raise ValueError("Progress values must be in [0, 1]")
        assert self.config.model_id is not None
        return ProgressPrediction(
            sample_id=sample.sample_id,
            progress=array.tolist(),
            model=self.config.model_id,
            model_version=self.config.model_version,
            raw_response=raw,
        )

    def predict(self, sample: EvaluationSample) -> Prediction:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: list[EvaluationSample]) -> list[Prediction]:
        progress_samples = []
        for sample in samples:
            if not isinstance(sample, ProgressSample):
                raise TypeError(f"{self.config.name} only supports progress samples")
            progress_samples.append(sample)

        frames_list = [np.asarray(sample.trajectory.frames) for sample in progress_samples]
        tasks = [sample.trajectory.task for sample in progress_samples]
        paths = [self._reference_path(sample) for sample in progress_samples]

        if self.config.mode == "local":
            assert self.local_model is not None
            results = self.local_model.compute_progress_batch(frames_list, tasks, paths)
        else:
            assert self.remote is not None
            results = self.model_cls.remote_compute_progress_batch(
                frames_list,
                tasks,
                paths,
                self.remote,
                self.config.options,
            )

        if len(results) != len(progress_samples):
            raise ValueError(
                f"Batch prediction count mismatch: expected {len(progress_samples)}, got {len(results)}"
            )
        return [
            self._prediction(sample, result)
            for sample, result in zip(progress_samples, results, strict=True)
        ]


class PreferenceModelInfer(Infer):
    """Bridge a registered PreferenceModel to the Stage 2 preference protocol."""

    capabilities: ClassVar[set[str]] = {"preference"}
    model_cls: ClassVar[type[PreferenceModel]]

    @classmethod
    def thread_safe_mode(cls, mode: str) -> bool:
        return mode == "remote"

    def __init__(self, config: InferConfig):
        super().__init__(config)
        self.transport = str(config.transport)
        self.thread_safe = self.thread_safe_mode(str(config.mode))
        self.local_model: PreferenceModel | None = None
        self.remote: RemoteContext | None = None
        if config.mode == "local":
            assert config.model_path is not None
            self.local_model = self.model_cls.load_local(config.model_path, config.options)
        else:
            self.remote = RemoteContext(config)

    def begin_prediction(self) -> None:
        if self.remote:
            self.remote.begin_prediction()

    def attempts(self) -> int:
        return self.remote.attempts() if self.remote else 1

    def model_info(self) -> dict[str, Any]:
        info = super().model_info()
        info["model_path" if self.config.mode == "local" else "base_url"] = (
            self.config.model_path if self.config.mode == "local" else self.config.base_url
        )
        return info

    def predict(self, sample: EvaluationSample) -> Prediction:
        if not isinstance(sample, PreferenceSample):
            raise TypeError(f"{self.config.name} only supports preference samples")
        if self.config.mode == "local":
            assert self.local_model is not None
            result = self.local_model.compute_preference(
                sample.chosen_trajectory.frames,
                sample.rejected_trajectory.frames,
                sample.chosen_trajectory.task,
            )
            if isinstance(result, dict):
                probability = float(result.get("chosen_probability", result.get("prediction_prob", 0.5)))
                preference = result.get("preference")
                if preference not in {"chosen", "rejected", "tie"}:
                    preference = "chosen" if probability > 0.5 else "rejected" if probability < 0.5 else "tie"
                result = PreferenceResult(probability, preference, raw_response=result)
        else:
            assert self.remote is not None
            result = self.model_cls.remote_compute_preference(
                sample.chosen_trajectory.frames,
                sample.rejected_trajectory.frames,
                sample.chosen_trajectory.task,
                self.remote,
                self.config.options,
            )
        if result.preference not in {"chosen", "rejected", "tie"}:
            raise ValueError(f"Invalid preference label: {result.preference}")
        if not 0 <= result.chosen_probability <= 1:
            raise ValueError("chosen_probability must be in [0, 1]")
        assert self.config.model_id is not None
        return PreferencePrediction(
            sample_id=sample.sample_id,
            chosen_probability=result.chosen_probability,
            preference=result.preference,
            model=self.config.model_id,
            model_version=self.config.model_version,
            raw_response=result.raw_response,
        )


def register_progress_model(name: str):
    """Register one local-first model class as a regular Stage 2 infer implementation."""

    def decorator(model_cls: type[ProgressModel]) -> type[ProgressModel]:
        modes = set()
        if model_cls.supports_local:
            modes.add("local")
        if model_cls.supports_remote:
            modes.add("remote")

        adapter = type(
            f"{model_cls.__name__}Infer",
            (ProgressModelInfer,),
            {
                "__module__": model_cls.__module__,
                "model_cls": model_cls,
                "supported_modes": modes,
            },
        )
        register_infer(name)(adapter)
        return model_cls

    return decorator


def register_preference_model(name: str):
    """Register one local-first preference model as a Stage 2 infer implementation."""

    def decorator(model_cls: type[PreferenceModel]) -> type[PreferenceModel]:
        modes = set()
        if model_cls.supports_local:
            modes.add("local")
        if model_cls.supports_remote:
            modes.add("remote")
        adapter = type(
            f"{model_cls.__name__}Infer",
            (PreferenceModelInfer,),
            {
                "__module__": model_cls.__module__,
                "model_cls": model_cls,
                "supported_modes": modes,
            },
        )
        register_infer(name)(adapter)
        return model_cls

    return decorator
