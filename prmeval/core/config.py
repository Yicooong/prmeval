from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

TemporalTransform = Literal["pause", "slow", "fast", "rewind", "retry", "truncate", "skip"]


class TemporalRobustnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_frames: int = Field(default=16, ge=1)
    min_length_ratio: float = Field(default=0.7, ge=0.7, le=1.0)
    max_length_ratio: float = Field(default=1.7, ge=1.0, le=1.7)
    transforms: list[TemporalTransform] = Field(
        default_factory=lambda: ["pause", "slow", "fast", "rewind", "retry", "truncate", "skip"]
    )
    variants_per_transform: int = Field(default=3, ge=1)
    pause_extra_ratio_range: tuple[float, float] = (0.2, 0.7)
    slow_gamma_range: tuple[float, float] = (1.5, 3.0)
    fast_gamma_range: tuple[float, float] = (0.33, 0.67)
    peak_progress_range: tuple[float, float] = (0.6, 0.9)
    retreat_ratio_range: tuple[float, float] = (0.25, 0.6)
    rewind_extra_ratio_range: tuple[float, float] = (0.2, 0.7)
    retry_extra_ratio_range: tuple[float, float] = (0.2, 0.7)
    truncate_retained_ratio_range: tuple[float, float] = (0.7, 0.9)
    skip_removed_ratio_range: tuple[float, float] = (0.1, 0.3)

    @model_validator(mode="after")
    def validate_ranges(self) -> TemporalRobustnessConfig:
        unit_ranges = (
            "pause_extra_ratio_range",
            "peak_progress_range",
            "retreat_ratio_range",
            "rewind_extra_ratio_range",
            "retry_extra_ratio_range",
            "truncate_retained_ratio_range",
            "skip_removed_ratio_range",
        )
        for name in unit_ranges:
            lower, upper = getattr(self, name)
            if not 0 <= lower <= upper <= 1:
                raise ValueError(f"temporal_robustness.{name} must be an ordered range within [0, 1]")
        for name in ("slow_gamma_range", "fast_gamma_range"):
            lower, upper = getattr(self, name)
            if not 0 < lower <= upper:
                raise ValueError(f"temporal_robustness.{name} must be an ordered positive range")
        if not self.transforms:
            raise ValueError("temporal_robustness.transforms must not be empty")
        if len(self.transforms) != len(set(self.transforms)):
            raise ValueError("temporal_robustness.transforms must not contain duplicates")
        return self


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "rbm-1m-ood"
    adapter: str = "huggingface"
    paths: list[str] = Field(default_factory=list)
    max_trajectories: int | None = Field(default=None, ge=1)
    eval_types: list[str] = Field(default_factory=lambda: ["reward_alignment"])
    base_frames: int = Field(default=8, ge=1)
    pad_frames: bool = False
    progress_type: Literal["absolute_first_frame", "absolute_wrt_total_frames", "relative_first_frame"] = (
        "absolute_first_frame"
    )
    random_seed: int = 42
    num_examples_per_quality: int | None = Field(default=5, ge=1)
    num_partial_successes: int | None = Field(default=None, ge=1)
    max_tasks: int | None = Field(default=None, ge=1)
    comparisons_per_task: int | None = Field(default=None, ge=1)
    max_comparisons: int | None = Field(default=None, ge=1)
    trajectories_per_source: int | None = Field(default=None, ge=1)
    temporal_robustness: TemporalRobustnessConfig = Field(default_factory=TemporalRobustnessConfig)

    @model_validator(mode="after")
    def validate_temporal_robustness(self) -> SamplingConfig:
        if "synthetic_temporal_robustness" not in self.eval_types:
            return self
        temporal = self.temporal_robustness
        if temporal.max_frames < 6:
            raise ValueError("synthetic_temporal_robustness requires temporal_robustness.max_frames >= 6")
        if self.progress_type == "relative_first_frame":
            raise ValueError("synthetic_temporal_robustness requires an absolute progress_type")
        if self.pad_frames:
            raise ValueError("synthetic_temporal_robustness does not support pad_frames because frame counts vary")
        if self.base_frames < 5:
            raise ValueError("synthetic_temporal_robustness requires sampling.base_frames >= 5")
        if self.base_frames > temporal.max_frames:
            raise ValueError("sampling.base_frames must not exceed temporal_robustness.max_frames")
        increasing = {"pause", "rewind", "retry"}.intersection(temporal.transforms)
        upper = min(int(temporal.max_length_ratio * self.base_frames), temporal.max_frames)
        if increasing and upper <= self.base_frames:
            raise ValueError("pause, rewind, and retry require room to increase beyond sampling.base_frames")
        decreasing = {"truncate", "skip"}.intersection(temporal.transforms)
        lower = math.ceil(temporal.min_length_ratio * self.base_frames)
        if decreasing and lower >= self.base_frames:
            raise ValueError("truncate and skip require room to decrease below sampling.base_frames")
        return self


class InferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str | None = None
    api_key: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    model_version: str | None = None
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=5, ge=0)
    temperature: float = 0.0
    max_tokens: int = 1024
    batch_size: int = Field(default=1, ge=1)
    headers: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def resolve_environment_values(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        resolved = dict(data)
        for field in ("base_url", "api_key", "model_id", "model_path"):
            value = resolved.get(field)
            if not value:
                continue
            if value in os.environ:
                resolved[field] = os.environ[value]
            elif value.isidentifier() and value.isupper():
                raise ValueError(f"Environment variable {value!r} configured by infer.{field} is missing")
        return resolved


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    infer: InferConfig
    metrics: list[str] = Field(default_factory=list)
    mode: Literal["separate", "continue"] = "separate"
    output_dir: str = "evaluation_output"
    run_name: str | None = None
    resume: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))
