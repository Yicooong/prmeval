"""Load offline simulator rollouts for the shared trajectory converter."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

VIEW_KEYS = {
    "external_main": ("external_main_rgb", "main_rgb"),
    "main": ("main_rgb", "external_main_rgb"),
    "third_person": ("third_rgb", "external_alt_rgb"),
    "external_alt": ("external_alt_rgb", "third_rgb"),
    "wrist": ("wrist_rgb",),
    "left_wrist": ("left_wrist_rgb",),
    "right_wrist": ("right_wrist_rgb",),
}

OUTCOME_LABELS = {
    "clean_success": "successful",
    "matched_failure": "failure",
    "unmatched_controller_failure": "failure",
}


def _required(row: dict[str, Any], key: str, sample_id: str) -> Any:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Simulator rollout {sample_id} is missing required field {key!r}")
    return value


def _load_frames(frames_path: str, frame_key: str, frame_indices: tuple[int, ...]) -> np.ndarray:
    with np.load(frames_path) as payload:
        return np.asarray(payload[frame_key][list(frame_indices)])


def _load_rows(dataset_path: str, max_trajectories: int | None) -> list[dict[str, Any]]:
    path = Path(dataset_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Simulator rollout index does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
            if max_trajectories not in (None, -1) and len(rows) >= max_trajectories:
                break
    if not rows:
        raise ValueError(f"Simulator rollout index contains no trajectories: {path}")
    return rows


def _trajectory(row: dict[str, Any], dataset_name: str, view: str) -> dict[str, Any]:
    sample_id = str(_required(row, "sample_id", "<unknown>"))
    frames_path = Path(str(_required(row, "frames_path", sample_id))).expanduser().resolve()
    if not frames_path.is_file():
        raise FileNotFoundError(f"Simulator rollout {sample_id} frame archive does not exist: {frames_path}")
    if view not in VIEW_KEYS:
        raise ValueError(f"Unsupported simulator view {view!r}; choose one of {sorted(VIEW_KEYS)}")

    frame_indices = tuple(int(value) for value in _required(row, "frame_indices", sample_id))
    target_progress = [float(value) for value in _required(row, "simulator_progress", sample_id)]
    if len(frame_indices) != len(target_progress):
        raise ValueError(
            f"Simulator rollout {sample_id} has {len(frame_indices)} frame indices but "
            f"{len(target_progress)} progress values"
        )
    if any(index < 0 for index in frame_indices):
        raise ValueError(f"Simulator rollout {sample_id} frame indices must be non-negative")
    if not target_progress or any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in target_progress):
        raise ValueError(f"Simulator rollout {sample_id} progress must be finite and within [0, 1]")

    with np.load(frames_path) as payload:
        frame_key = next((candidate for candidate in VIEW_KEYS[view] if candidate in payload), None)
        if frame_key is None:
            raise KeyError(
                f"Simulator rollout {sample_id} has no {view!r} view in {frames_path}; "
                f"expected one of {VIEW_KEYS[view]}"
            )
        if frame_indices and max(frame_indices) >= len(payload[frame_key]):
            raise ValueError(
                f"Simulator rollout {sample_id} references frame {max(frame_indices)} "
                f"but {frame_key} only has {len(payload[frame_key])} frames"
            )

    benchmark = _required(row, "benchmark", sample_id)
    if not isinstance(benchmark, dict):
        raise ValueError(f"Simulator rollout {sample_id} benchmark must be an object")
    suite_id = str(_required(benchmark, "suite_id", sample_id))
    outcome = str(_required(benchmark, "outcome", sample_id))
    if outcome not in OUTCOME_LABELS:
        raise ValueError(
            f"Simulator rollout {sample_id} has unsupported outcome {outcome!r}; "
            f"expected one of {sorted(OUTCOME_LABELS)}"
        )

    return {
        "id": sample_id,
        "task": str(_required(row, "task", sample_id)),
        "data_source": f"{dataset_name}:{suite_id}",
        "frames": partial(_load_frames, str(frames_path), frame_key, frame_indices),
        "is_robot": True,
        "is_simulation": True,
        "quality_label": OUTCOME_LABELS[outcome],
        "partial_success": target_progress[-1],
        "target_progress": target_progress,
    }


def load_simulator_rollout_dataset(
    dataset_path: str,
    dataset_name: str = "simulator_rollout",
    view: str = "external_main",
    max_trajectories: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load a sim-prm-eval ``samples.jsonl`` index grouped by task."""

    task_data: dict[str, list[dict[str, Any]]] = {}
    for row in _load_rows(dataset_path, max_trajectories):
        trajectory = _trajectory(row, dataset_name, view)
        task_data.setdefault(trajectory["task"], []).append(trajectory)
    return task_data
