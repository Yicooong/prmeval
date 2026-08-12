from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core.config import SamplingConfig
from ..core.registry import DATASETS, register_dataset
from ..core.schemas import Trajectory


class DatasetAdapter(ABC):
    def __init__(self, config: SamplingConfig):
        self.config = config

    @abstractmethod
    def load(self) -> Iterable[Trajectory]:
        raise NotImplementedError


def _trajectory_from_mapping(item: dict, base_dir: Path | None = None) -> Trajectory:
    frames = next(
        (item[key] for key in ("frames", "frames_video", "video", "frames_path") if item.get(key) is not None),
        None,
    )
    missing = [key for key in ("id", "task") if item.get(key) is None]
    if frames is None:
        missing.append("frames")
    if missing:
        raise ValueError(f"Trajectory is missing required fields: {', '.join(missing)}")
    if isinstance(frames, str) and base_dir:
        configured = Path(frames)
        if not configured.is_absolute():
            frames = str(base_dir / configured)
    elif isinstance(frames, dict) and frames.get("path") and base_dir:
        configured = Path(frames["path"])
        if not configured.is_absolute():
            frames = {**frames, "path": str(base_dir / configured)}
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
        if not self.config.paths:
            raise ValueError("sampling.paths must contain at least one JSONL path")
        count = 0
        for configured_path in self.config.paths:
            path = Path(configured_path).expanduser().resolve()
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield _trajectory_from_mapping(json.loads(line), path.parent)
                        count += 1
                        if self.config.max_trajectories and count >= self.config.max_trajectories:
                            return


@register_dataset("huggingface")
class HuggingfaceDatasetAdapter(DatasetAdapter):
    """Load one or more local Hugging Face datasets directly from disk."""

    def load(self) -> Iterable[Trajectory]:
        try:
            from datasets import Dataset, DatasetDict, Video, load_from_disk
        except ImportError as exc:
            raise RuntimeError("huggingface requires the 'data' extra (huggingface datasets)") from exc
        if not self.config.paths:
            raise ValueError("sampling.paths must contain at least one local Hugging Face dataset path")

        count = 0
        for configured_path in self.config.paths:
            dataset_path = Path(configured_path).expanduser().resolve()
            if not dataset_path.exists():
                raise FileNotFoundError(f"Local Hugging Face dataset not found: {dataset_path}")
            loaded = load_from_disk(str(dataset_path), keep_in_memory=False)
            datasets = loaded.values() if isinstance(loaded, DatasetDict) else [loaded]
            for dataset in datasets:
                if not isinstance(dataset, Dataset):
                    raise TypeError(f"Expected a Hugging Face Dataset, got {type(dataset)!r}")
                # Cast video columns to Video(decode=False) to avoid decoding videos when loading the dataset
                for column in ("frames", "frames_video", "video"):
                    if column in dataset.column_names and isinstance(dataset.features.get(column), Video):
                        try:
                            dataset = dataset.cast_column(column, Video(decode=False))
                        except (TypeError, ValueError):
                            pass
                for item in dataset:
                    yield _trajectory_from_mapping(dict(item), dataset_path)
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
        from .prepare import _video_frames

        return _video_frames(path)
    if isinstance(frames, list):
        return np.asarray(frames)
    if isinstance(frames, dict):
        from .prepare import _video_frames

        return _video_frames(frames)
    raise TypeError(f"Unsupported frames value: {type(frames)!r}")


def create_dataset(config: SamplingConfig) -> DatasetAdapter:
    return DATASETS.get(config.adapter)(config)
