"""Loader for the locally exported RBM-1M-OOD evaluation dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


# These six directories are the OOD sources named by the root export_summary.json.
# Other sibling directories in the export are auxiliary variants and are not part of
# the canonical 571-episode RBM-1M-OOD split.
SOURCE_DIRECTORIES = {
    "usc_trossen": "usc_trossen",
    "mit_franka": "rfm_new_mit_franka_rfm",
    "utd_so101": "utd_so101_clean_policy_ranking_top",
    "usc_xarm": "usc_xarm_policy_ranking",
    "usc_franka": "usc_franka_policy_ranking",
    "usc_koch": "usc_koch_p_ranking_all",
}


class RBM1MOODFrameLoader:
    """Pickle-friendly lazy decoder for one exported episode video."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path

    def __call__(self) -> np.ndarray:
        # Keep metadata-only inspection usable in environments that do not have
        # the optional OpenCV video dependency installed.
        from dataset_unify.video_helpers import load_video_frames

        return load_video_frames(self.video_path)


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    try:
        return pq.read_table(path).to_pylist()
    except Exception as exc:
        raise ValueError(f"Failed to read RBM-1M-OOD metadata: {path}") from exc


def _parse_bool(value: Any, *, field: str, metadata_path: Path) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid {field} value {value!r} in {metadata_path}")


def _parse_optional_float(value: Any, *, metadata_path: Path) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid partial_success value {value!r} in {metadata_path}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"partial_success must be in [0, 1], got {parsed} in {metadata_path}")
    return parsed


def _resolve_video_path(root: Path, source_dir: Path, row: dict[str, Any], metadata_path: Path) -> Path:
    candidates: list[Path] = []
    file_name = row.get("file_name")
    video_file_name = row.get("video_file_name")
    if file_name:
        candidates.append(source_dir / str(file_name))
    if video_file_name:
        candidates.append(root / str(video_file_name))
        candidates.append(source_dir / Path(str(video_file_name)).name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates) or "<no video path fields>"
    raise FileNotFoundError(f"Episode video referenced by {metadata_path} was not found; tried: {rendered}")


def load_rbm_1m_ood_dataset(
    dataset_path: str,
    max_trajectories: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load the canonical six-source RBM-1M-OOD split from exported MP4 files.

    Expected layout::

        <dataset_path>/
          usc_trossen/metadata.parquet + video_*.mp4
          mit_franka/metadata.parquet + video_*.mp4
          utd_so101/metadata.parquet + video_*.mp4
          usc_xarm/metadata.parquet + video_*.mp4
          usc_franka/metadata.parquet + video_*.mp4
          usc_koch/metadata.parquet + video_*.mp4

    Supplemental sibling directories are intentionally ignored because they do not
    belong to the RBM-1M-OOD manifest.
    """

    root = Path(dataset_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"RBM-1M-OOD dataset directory not found: {root}")

    limit = None if max_trajectories is None or max_trajectories < 0 else int(max_trajectories)
    task_data: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    total = 0

    for directory_name, canonical_source in SOURCE_DIRECTORIES.items():
        if limit is not None and total >= limit:
            break
        source_dir = root / directory_name
        metadata_path = source_dir / "metadata.parquet"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing canonical RBM-1M-OOD source metadata: {metadata_path}. "
                f"Expected source directories: {', '.join(SOURCE_DIRECTORIES)}"
            )

        for row in _read_metadata(metadata_path):
            if limit is not None and total >= limit:
                break

            trajectory_id = str(row.get("id") or "").strip()
            task = str(row.get("task") or row.get("language_instruction") or "").strip()
            if not trajectory_id or not task:
                raise ValueError(f"Every row in {metadata_path} must contain non-empty id and task fields")
            if trajectory_id in seen_ids:
                raise ValueError(f"Duplicate RBM-1M-OOD trajectory id {trajectory_id!r} in {metadata_path}")

            video_path = _resolve_video_path(root, source_dir, row, metadata_path)
            trajectory = {
                "id": trajectory_id,
                "task": task,
                "frames": RBM1MOODFrameLoader(str(video_path)),
                "is_robot": _parse_bool(row.get("is_robot"), field="is_robot", metadata_path=metadata_path),
                "quality_label": row.get("quality_label") or None,
                "partial_success": _parse_optional_float(row.get("partial_success"), metadata_path=metadata_path),
                "data_source": canonical_source,
            }
            task_data[task].append(trajectory)
            seen_ids.add(trajectory_id)
            total += 1

    if not task_data:
        raise ValueError(f"No RBM-1M-OOD trajectories found in {root}")

    print(f"Loaded {total} RBM-1M-OOD trajectories across {len(task_data)} tasks")
    return dict(task_data)
