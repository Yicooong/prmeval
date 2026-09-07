from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prmeval.core.config import SamplingConfig
from prmeval.core.schemas import Trajectory


def load_hf_trajectory_pool(config: SamplingConfig) -> list[Trajectory]:
    """Load explicitly successful trajectories from JSONL files or local Hugging Face Datasets."""
    if not config.paths:
        raise ValueError("sampling.paths must contain at least one JSONL file or local Hugging Face dataset path")

    def _is_successful_trajectory(trajectory: Trajectory) -> bool:
        if trajectory.quality_label not in (None, "successful"):
            return False
        if trajectory.partial_success is not None and not np.isclose(trajectory.partial_success, 1.0):
            return False
        return trajectory.quality_label == "successful" or trajectory.partial_success is not None

    def _trajectory_from_mapping(item: dict, base_dir: Path) -> Trajectory:
        frames = next(
            (item[key] for key in ("frames", "frames_video", "video", "frames_path") if item.get(key) is not None),
            None,
        )
        missing = [key for key in ("id", "task") if item.get(key) is None]
        if frames is None:
            missing.append("frames")
        if missing:
            raise ValueError(f"Trajectory is missing required fields: {', '.join(missing)}")
        if isinstance(frames, str):
            configured = Path(frames)
            if not configured.is_absolute():
                frames = str(base_dir / configured)
        elif isinstance(frames, dict) and frames.get("path"):
            configured = Path(frames["path"])
            if not configured.is_absolute():
                frames = {**frames, "path": str(base_dir / configured)}
        return Trajectory(
            id=str(item["id"]),
            task=str(item["task"]),
            frames=frames,
            data_source=str(item.get("data_source") or "unknown"),
            is_robot=bool(item.get("is_robot", False)),
            quality_label=item.get("quality_label"),
            partial_success=item.get("partial_success"),
            preference_group_id=item.get("preference_group_id"),
            preference_rank=item.get("preference_rank"),
            metadata=item.get("metadata") or {},
            frame_indices=item.get("frame_indices"),
            num_frames_total=item.get("num_frames_total", item.get("num_frames")),
            target_progress=item.get("target_progress"),
            is_simulation=bool(item.get("is_simulation", False)),
        )

    def _load_jsonl(path: Path):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc.msg}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
                yield item

    trajectories: list[Trajectory] = []
    for configured_path in config.paths:
        dataset_path = Path(configured_path).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Trajectory source not found: {dataset_path}")

        if dataset_path.is_file():
            if dataset_path.suffix.lower() != ".jsonl":
                raise ValueError(f"Trajectory file must use the .jsonl extension: {dataset_path}")
            for item in _load_jsonl(dataset_path):
                trajectory = _trajectory_from_mapping(item, dataset_path.parent)
                if not _is_successful_trajectory(trajectory):
                    continue
                trajectories.append(trajectory)
                if config.max_trajectories and len(trajectories) >= config.max_trajectories:
                    return trajectories
            continue

        try:
            from datasets import Dataset, DatasetDict, Video, load_from_disk
        except ImportError as exc:
            raise RuntimeError("Hugging Face dataset sampling requires the 'data' extra") from exc
        loaded = load_from_disk(str(dataset_path), keep_in_memory=False)
        datasets = loaded.values() if isinstance(loaded, DatasetDict) else [loaded]
        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise TypeError(f"Expected a Hugging Face Dataset, got {type(dataset)!r}")
            for column in ("frames", "frames_video", "video"):
                if column in dataset.column_names and isinstance(dataset.features.get(column), Video):
                    try:
                        dataset = dataset.cast_column(column, Video(decode=False))
                    except (TypeError, ValueError):
                        pass
            for item in dataset:
                trajectory = _trajectory_from_mapping(dict(item), dataset_path)
                if not _is_successful_trajectory(trajectory):
                    continue
                trajectories.append(trajectory)
                if config.max_trajectories and len(trajectories) >= config.max_trajectories:
                    return trajectories
    return trajectories
