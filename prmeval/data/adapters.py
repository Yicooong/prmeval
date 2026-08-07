from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.config import DatasetConfig
from ..core.registry import DATASETS, register_dataset
from ..core.schemas import Trajectory
from .manifests import resolve_manifest


class DatasetAdapter(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def load(self) -> Iterable[Trajectory]:
        raise NotImplementedError


def _trajectory_from_mapping(item: dict, root: Path | None = None) -> Trajectory:
    missing = [key for key in ("id", "task", "frames") if item.get(key) is None]
    if missing:
        raise ValueError(f"Trajectory is missing required fields: {', '.join(missing)}")
    frames = item.get("frames")
    if isinstance(frames, str) and root and not os.path.isabs(frames):
        configured = Path(frames)
        frames = str(configured if configured.exists() else root / configured)
    return Trajectory(
        id=str(item.get("id")),
        task=str(item.get("task") or ""),
        frames=frames,
        data_source=str(item.get("data_source") or "unknown"),
        is_robot=bool(item.get("is_robot", False)),
        quality_label=item.get("quality_label"),
        partial_success=item.get("partial_success"),
        preference_group_id=item.get("preference_group_id"),
        preference_rank=item.get("preference_rank"),
        metadata=item.get("metadata") or {},
        num_frames_total=item.get("num_frames"),
    )


@register_dataset("jsonl")
class JsonLinesDatasetAdapter(DatasetAdapter):
    def load(self) -> Iterable[Trajectory]:
        root = Path(self.config.root or ".").resolve()
        paths = self.config.paths or [self.config.name]
        count = 0
        for configured_path in paths:
            path = Path(configured_path)
            if not path.is_absolute():
                path = root / path
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield _trajectory_from_mapping(json.loads(line), root)
                        count += 1
                        if self.config.max_trajectories and count >= self.config.max_trajectories:
                            return


@register_dataset("processed_cache")
class ProcessedCacheDatasetAdapter(DatasetAdapter):
    """Load the index-based caches produced by data.prepare."""

    def load(self) -> Iterable[Trajectory]:
        try:
            from datasets import Dataset
        except ImportError as exc:
            raise RuntimeError("processed_cache requires the 'data' extra (huggingface datasets)") from exc
        configured_root = self.config.root or os.environ.get("PRMEVAL_PROCESSED_DATASETS_PATH")
        if not configured_root:
            raise ValueError("dataset.root or PRMEVAL_PROCESSED_DATASETS_PATH is required")
        root = Path(configured_root)
        dataset_names = resolve_manifest(self.config.name, self.config.paths)
        count = 0
        for dataset_name in dataset_names:
            cache = root / dataset_name.replace("/", "_").replace(":", "_") / "processed_dataset"
            if not cache.exists():
                raise FileNotFoundError(f"Processed dataset cache not found: {cache}")
            dataset = Dataset.load_from_disk(str(cache), keep_in_memory=False)
            for item in dataset:
                yield _trajectory_from_mapping(dict(item), root)
                count += 1
                if self.config.max_trajectories and count >= self.config.max_trajectories:
                    return


def load_frames(frames: object) -> np.ndarray:
    if isinstance(frames, np.ndarray):
        return frames
    if isinstance(frames, str):
        path = Path(frames)
        if path.suffix.lower() == ".npz":
            with np.load(path) as archive:
                return np.asarray(archive["frames"])
        raise ValueError(f"Only .npz frame paths are supported by the core loader: {path}")
    if isinstance(frames, list):
        return np.asarray(frames)
    raise TypeError(f"Unsupported frames value: {type(frames)!r}")


def create_dataset(config: DatasetConfig) -> DatasetAdapter:
    return DATASETS.get(config.adapter)(config)
