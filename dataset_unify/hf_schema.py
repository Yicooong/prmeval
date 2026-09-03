"""Canonical Hugging Face Dataset schema produced by dataset_unify."""

from collections.abc import Iterable, Mapping
from typing import Any

import datasets
from datasets import Dataset

STANDARD_DATASET_FIELDS = (
    "id",
    "task",
    "data_source",
    "frames",
    "is_robot",
    "is_simulation",
    "quality_label",
    "partial_success",
    "target_progress",
)

STANDARD_DATASET_FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "task": datasets.Value("string"),
        "data_source": datasets.Value("string"),
        "frames": datasets.Value("string"),
        "is_robot": datasets.Value("bool"),
        "is_simulation": datasets.Value("bool"),
        "quality_label": datasets.Value("string"),
        "partial_success": datasets.Value("float32"),
        "target_progress": datasets.Sequence(datasets.Value("float32")),
    }
)


def normalize_standard_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Select and normalize the fields shared with PRMEval."""
    missing = [field for field in ("id", "task", "data_source", "frames", "is_robot") if entry.get(field) is None]
    if missing:
        raise ValueError(f"Dataset entry is missing required fields: {', '.join(missing)}")

    quality_label = entry.get("quality_label")
    if quality_label is not None:
        quality_label = {"success": "successful", "fail": "failure", "failed": "failure"}.get(
            str(quality_label).lower(), str(quality_label)
        )
    partial_success = entry.get("partial_success")
    if partial_success is not None:
        partial_success = float(partial_success)
        if not 0.0 <= partial_success <= 1.0:
            raise ValueError(f"partial_success must be in [0, 1], got {partial_success}")
    target_progress = entry.get("target_progress")
    if target_progress is not None:
        target_progress = [float(value) for value in target_progress]
        if any(not 0.0 <= value <= 1.0 for value in target_progress):
            raise ValueError("target_progress values must be in [0, 1]")
    return {
        "id": str(entry["id"]),
        "task": str(entry["task"]),
        "data_source": str(entry["data_source"]),
        "frames": str(entry["frames"]),
        "is_robot": bool(entry["is_robot"]),
        "is_simulation": bool(entry.get("is_simulation", False)),
        "quality_label": quality_label,
        "partial_success": partial_success,
        "target_progress": target_progress,
    }


def build_standard_dataset(entries: Iterable[Mapping[str, Any]]) -> Dataset:
    """Build a Dataset with one stable schema for both empty and non-empty inputs."""
    normalized = [normalize_standard_entry(entry) for entry in entries]
    data = {field: [entry[field] for entry in normalized] for field in STANDARD_DATASET_FIELDS}
    return Dataset.from_dict(data, features=STANDARD_DATASET_FEATURES)
