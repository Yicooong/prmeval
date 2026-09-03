"""Shared helpers for cross-stage records and their on-disk artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..sample import load_frames
from .schemas import (
    EvaluationRecord,
    EvaluationSample,
    FrameReference,
    PreferenceSample,
    ProgressSample,
    RecordInputItem,
    Trajectory,
    ValuePayload,
    jsonable,
)

SAMPLE_SCHEMA_VERSION = "bench.record.v1"
QUALITY_RANK = {"successful": 2.0, "suboptimal": 1.0, "failure": 0.0, "failed": 0.0}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_for_progress(sample: ProgressSample) -> ValuePayload:
    trajectory = sample.trajectory
    if sample.eval_type == "progress":
        return ValuePayload(kind="progress", values=trajectory.target_progress)
    if sample.eval_type == "policy_ranking":
        rank = trajectory.partial_success
        if rank is None:
            rank = trajectory.preference_rank
        if rank is None:
            rank = QUALITY_RANK.get(trajectory.quality_label or "")
        return ValuePayload(
            kind="rank",
            value=float(rank) if rank is not None else None,
            label=trajectory.quality_label,
        )
    if sample.eval_type == "confusion_matrix":
        return ValuePayload(
            kind="task_match",
            value=float(trajectory.metadata.get("lang_task") == trajectory.metadata.get("video_task")),
        )
    return ValuePayload(kind="progress", values=trajectory.target_progress)


def sample_to_record(sample: EvaluationSample, dataset_name: str) -> EvaluationRecord:
    """Normalize internal sampler output into the one cross-stage Record schema."""
    if isinstance(sample, ProgressSample):
        trajectory = sample.trajectory
        items = [
            RecordInputItem(
                role="trajectory",
                frames=trajectory.frames,
                frame_indices=trajectory.frame_indices or [],
                source_id=trajectory.id,
                data=trajectory.metadata,
            )
        ]
        target = _target_for_progress(sample)
        source_name = trajectory.data_source
        task = trajectory.task
    else:
        chosen = sample.chosen_trajectory
        rejected = sample.rejected_trajectory
        items = [
            RecordInputItem(
                role="chosen",
                frames=chosen.frames,
                frame_indices=chosen.frame_indices or [],
                source_id=chosen.id,
                data=chosen.metadata,
            ),
            RecordInputItem(
                role="rejected",
                frames=rejected.frames,
                frame_indices=rejected.frame_indices or [],
                source_id=rejected.id,
                data=rejected.metadata,
            ),
        ]
        target = ValuePayload(kind="preference", label="chosen")
        source_name = chosen.data_source
        task = chosen.task
    return EvaluationRecord(
        sample_id=sample.sample_id,
        evaluation={
            "type": sample.eval_type,
            "dataset": {"name": dataset_name, "source": source_name},
        },
        input={"task": task, "items": items},
        target=target,
    )


def strip_record_frames(record: EvaluationRecord) -> EvaluationRecord:
    """Return a JSON-safe record without runtime frame arrays or disk frame references."""
    # Keep string frame references and remove in-memory frame arrays.
    items = [
        item if isinstance(item.frames, str) else item.model_copy(update={"frames": []}) for item in record.input.items
    ]
    return record.model_copy(update={"input": record.input.model_copy(update={"items": items})})


def _materialize_item(item: RecordInputItem, sample_id: str, bundle_dir: Path) -> RecordInputItem:
    frames = load_frames(item.frames)
    if len(frames) == 0:
        raise ValueError(f"Sample {sample_id} contains an empty '{item.role}' input")
    frames_dir = bundle_dir / "sample_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"{sample_id}-{item.role}.npz"
    np.savez_compressed(path, frames=frames)
    reference = FrameReference(
        path=path.relative_to(bundle_dir).as_posix(),
        num_frames=len(frames),
        sha256=_file_sha256(path),
    )
    return item.model_copy(update={"frames": reference})


def _materialize_record(record: EvaluationRecord, bundle_dir: Path) -> EvaluationRecord:
    items = [_materialize_item(item, record.sample_id, bundle_dir) for item in record.input.items]
    return record.model_copy(update={"input": record.input.model_copy(update={"items": items})})


def write_sample_artifacts(samples: Iterable[EvaluationSample], path: Path, dataset_name: str = "unknown") -> dict:
    """Write Stage-1 records using the same schema later enriched by inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    sample_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            if sample.sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
            record = _materialize_record(sample_to_record(sample, dataset_name), path.parent)
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
            counts[record.evaluation.type] += 1
            sample_count += 1
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "samples": sample_count,
        "eval_types": dict(sorted(counts.items())),
        "path": str(path),
    }


def _hydrate_item(item: RecordInputItem, bundle_dir: Path, verify: bool) -> RecordInputItem:
    reference = FrameReference.model_validate(item.frames)
    path = Path(reference.path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Frame path must stay inside the sample bundle: {reference.path}")
    resolved = bundle_dir / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing sampled frames: {resolved}")
    if verify and _file_sha256(resolved) != reference.sha256:
        raise ValueError(f"Frame checksum mismatch: {resolved}")
    with np.load(resolved) as archive:
        if reference.key not in archive:
            raise ValueError(f"Frame key '{reference.key}' is missing in {resolved}")
        frames = np.asarray(archive[reference.key])
    if len(frames) != reference.num_frames:
        raise ValueError(f"Frame count mismatch in {resolved}: expected {reference.num_frames}, got {len(frames)}")
    if item.frame_indices and len(item.frame_indices) != len(frames):
        raise ValueError(f"frame_indices/frame mismatch in {resolved}: {len(item.frame_indices)} != {len(frames)}")
    return item.model_copy(update={"frames": frames})


def load_sample_artifacts(path: Path, verify: bool = True) -> list[EvaluationRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Sample artifact not found: {path}")
    records: list[EvaluationRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = EvaluationRecord.model_validate_json(line)
                if record.execution is not None:
                    raise ValueError("expected a sampled record without execution results")
                if record.sample_id in seen:
                    raise ValueError(f"duplicate sample_id '{record.sample_id}'")
                seen.add(record.sample_id)
                hydrated_items = [_hydrate_item(item, path.parent, verify) for item in record.input.items]
                if record.target and record.target.kind == "progress":
                    values = record.target.values or []
                    if len(hydrated_items) != 1 or len(values) != len(hydrated_items[0].frames):
                        raise ValueError("progress target length must equal the sampled frame count")
                records.append(record)
            except Exception as exc:
                raise ValueError(f"Invalid sample record at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"No sample records found in {path}")
    return records


def _runtime_item(item: RecordInputItem, bundle_dir: Path) -> RecordInputItem:
    if isinstance(item.frames, np.ndarray):
        return item
    return _hydrate_item(item, bundle_dir, verify=False)


def record_to_sample(record: EvaluationRecord, bundle_dir: Path) -> EvaluationSample:
    """Convert the disk protocol to existing infer-internal sample types."""
    common = {
        "task": record.input.task,
        "data_source": record.evaluation.dataset.source or record.evaluation.dataset.name,
    }
    if record.evaluation.type == "quality_preference":
        by_role = {item.role: _runtime_item(item, bundle_dir) for item in record.input.items}
        chosen = by_role["chosen"]
        rejected = by_role["rejected"]
        return PreferenceSample(
            sample_id=record.sample_id,
            eval_type=record.evaluation.type,
            chosen_trajectory=Trajectory(
                id=chosen.source_id or f"{record.sample_id}:chosen",
                frames=chosen.frames,
                frame_indices=chosen.frame_indices,
                metadata=chosen.data,
                **common,
            ),
            rejected_trajectory=Trajectory(
                id=rejected.source_id or f"{record.sample_id}:rejected",
                frames=rejected.frames,
                frame_indices=rejected.frame_indices,
                metadata=rejected.data,
                **common,
            ),
        )
    item = _runtime_item(record.input.items[0], bundle_dir)
    frame_target = record.target.values if record.target and record.target.kind == "progress" else None
    return ProgressSample(
        sample_id=record.sample_id,
        eval_type=record.evaluation.type,
        trajectory=Trajectory(
            id=item.source_id or record.sample_id,
            frames=item.frames,
            frame_indices=item.frame_indices,
            target_progress=frame_target,
            metadata=item.data,
            quality_label=record.target.label if record.target else None,
            partial_success=(record.target.value if record.target and record.target.kind == "rank" else None),
            **common,
        ),
    )


def validate_sample_artifacts(path: Path) -> dict:
    records = load_sample_artifacts(path, verify=True)
    counts = Counter(record.evaluation.type for record in records)
    frame_count = sum(
        sum(FrameReference.model_validate(item.frames).num_frames for item in record.input.items) for record in records
    )
    return {
        "valid": True,
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "samples": len(records),
        "frames": frame_count,
        "eval_types": dict(sorted(counts.items())),
        "path": str(path),
    }
