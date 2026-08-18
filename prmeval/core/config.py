from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = "rbm-1m-ood"
    adapter: str = "huggingface"
    paths: list[str] = Field(default_factory=list)
    max_trajectories: int | None = Field(default=None, ge=1)
    eval_types: list[str] = Field(default_factory=lambda: ["reward_alignment"])
    max_frames: int = Field(default=8, ge=1)
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
    output_dir: str = "evaluation_output"
    run_name: str | None = None
    resume: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))
