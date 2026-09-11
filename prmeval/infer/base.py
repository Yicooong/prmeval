from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from prmeval.core.config import InferConfig
from prmeval.core.schemas import EvaluationSample, Prediction


class Infer(ABC):
    capabilities: ClassVar[set[str]] = set()

    def __init__(self, config: InferConfig):
        self.config = config

    def begin_prediction(self) -> None:
        return None

    def attempts(self) -> int:
        return 1

    @abstractmethod
    def predict(self, samples: list[EvaluationSample]) -> list[Prediction]:
        raise NotImplementedError

    def model_info(self) -> dict[str, Any]:
        return {
            "model": self.config.model_id or self.config.model_path or self.config.name,
        }
