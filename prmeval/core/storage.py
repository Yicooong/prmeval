"""评估记录及关联帧文件的保存、加载与文件校验。"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from prmeval.utils import load_frames

from .schemas import EvaluationRecord, EvaluationSample, FrameReference, ProgressSample, Trajectory
from .utils import _file_sha256, jsonable, read_jsonl


def _load_item_frames(item: Trajectory, bundle_dir: Path, verify: bool) -> Trajectory:
    """从 NPZ 引用加载帧并返回输入项副本; 检查路径、键名和帧数, 按 verify 决定是否校验哈希。"""
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


def _ensure_item_frames_loaded(item: Trajectory, bundle_dir: Path) -> Trajectory:
    """确保输入项包含帧数组: 已有数组直接返回, 否则从文件加载, 不重复校验哈希。"""
    if isinstance(item.frames, np.ndarray):
        return item
    return _load_item_frames(item, bundle_dir, verify=False)


def load_sample_records(path: Path, verify: bool = True) -> list[EvaluationRecord]:
    """读取并校验采样 JSONL, 检查重复 ID、帧文件和进度标签长度。

    帧数组仅临时加载用于校验, 返回的记录仍保留文件引用; verify 控制哈希校验。
    """
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
                if record.sample.sample_id in seen:
                    raise ValueError(f"duplicate sample_id '{record.sample.sample_id}'")
                seen.add(record.sample.sample_id)
                hydrated_items = [
                    _load_item_frames(item, path.parent, verify)
                    for item in _sample_trajectories(record.sample).values()
                ]
                if isinstance(record.sample, ProgressSample) and record.sample.trajectory.target_progress is not None:
                    values = record.sample.trajectory.target_progress
                    if len(hydrated_items) != 1 or len(values) != len(hydrated_items[0].frames):
                        raise ValueError("progress target length must equal the sampled frame count")
                records.append(record)
            except Exception as exc:
                raise ValueError(f"Invalid sample record at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"No sample records found in {path}")
    return records


def save_samples_to_bundle(records: Iterable[EvaluationRecord], path: Path) -> dict:
    """保存采样记录及关联的 NPZ 帧文件和 JSONL, 并返回数量和评估类型统计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    sample_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for source in records:
            if source.execution is not None or source.prediction is not None:
                raise ValueError("Expected a sampled record without inference results")
            sample = source.sample
            if sample.sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
            record = _save_record_frames(source, path.parent)
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
            counts[record.eval_type] += 1
            sample_count += 1
    return {
        "samples": sample_count,
        "eval_types": dict(sorted(counts.items())),
        "path": str(path),
    }


def validate_sample_bundle(path: Path) -> dict:
    """完整校验采样记录及关联帧文件, 返回样本数、帧数等统计; 校验失败时抛出异常。"""
    records = load_sample_records(path, verify=True)
    counts = Counter(record.eval_type for record in records)
    frame_count = sum(
        sum(
            FrameReference.model_validate(item.frames).num_frames
            for item in _sample_trajectories(record.sample).values()
        )
        for record in records
    )
    return {
        "valid": True,
        "samples": len(records),
        "frames": frame_count,
        "eval_types": dict(sorted(counts.items())),
        "path": str(path),
    }


def write_metric_details_jsonl(path: Path, records: list[EvaluationRecord], metrics: dict) -> None:
    """将已计算的逐样本和逐组指标详情写入 JSONL; 本函数不计算指标。"""
    record_rows = {
        record.sample.sample_id: {
            "detail_type": "record",
            **jsonable(record),
            "metrics": {},
        }
        for record in records
    }
    group_rows = []
    for metric_name, result in metrics.items():
        for sample_id, detail in result.get("details", {}).items():
            if sample_id in record_rows:
                record_rows[sample_id]["metrics"][metric_name] = detail
        for group_id, detail in result.get("task_details", {}).items():
            group_rows.append(
                {
                    "detail_type": "group",
                    "metric": metric_name,
                    "group_id": group_id,
                    **detail,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in [*record_rows.values(), *group_rows]:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def _save_item_frames(item: Trajectory, sample_id: str, role: str, bundle_dir: Path) -> Trajectory:
    """将一个输入项的帧压缩保存到 sample_frames 目录, 返回以文件引用替代帧内容的输入项副本。"""
    frames = load_frames(item.frames)
    if len(frames) == 0:
        raise ValueError(f"Sample {sample_id} contains an empty '{role}' input")
    frames_dir = bundle_dir / "sample_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"{sample_id}-{role}.npz"
    np.savez_compressed(path, frames=frames)
    reference = FrameReference(
        path=path.relative_to(bundle_dir).as_posix(),
        num_frames=len(frames),
        sha256=_file_sha256(path),
    )
    return item.model_copy(update={"frames": reference})


def _sample_trajectories(sample: EvaluationSample) -> dict[str, Trajectory]:
    if isinstance(sample, ProgressSample):
        return {"trajectory": sample.trajectory}
    return {"chosen_trajectory": sample.chosen_trajectory, "rejected_trajectory": sample.rejected_trajectory}


def _map_record_trajectories(
    record: EvaluationRecord, transform: Callable[[str, Trajectory], Trajectory]
) -> EvaluationRecord:
    trajectories = {
        name: transform(name, trajectory) for name, trajectory in _sample_trajectories(record.sample).items()
    }
    return record.model_copy(update={"sample": record.sample.model_copy(update=trajectories)})


def _save_record_frames(record: EvaluationRecord, bundle_dir: Path) -> EvaluationRecord:
    """Save sampled frames while preserving every other trajectory field."""
    return _map_record_trajectories(
        record,
        lambda name, trajectory: _save_item_frames(
            trajectory, record.sample.sample_id, name.removesuffix("_trajectory"), bundle_dir
        ),
    )


def load_record_frames(record: EvaluationRecord, bundle_dir: Path) -> EvaluationRecord:
    """Hydrate a copy of the sample, preserving the original record's frame references."""
    return _map_record_trajectories(record, lambda name, trajectory: _ensure_item_frames_loaded(trajectory, bundle_dir))


def clear_non_string_frame_values(record: EvaluationRecord) -> EvaluationRecord:
    """Release frame arrays in retained results; metrics need only sample labels and metadata."""
    return _map_record_trajectories(
        record,
        lambda name, trajectory: (
            trajectory if isinstance(trajectory.frames, str) else trajectory.model_copy(update={"frames": []})
        ),
    )


def prepare_inference_outputs(predictions_path: Path, errors_path: Path, *, resume: bool) -> set[str]:
    """Prepare output files and validate successful checkpoint IDs when resuming."""
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        predictions_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")
        return set()
    records = [EvaluationRecord.model_validate(row) for row in read_jsonl(predictions_path)]
    if any(not record.execution or record.execution.status != "success" for record in records):
        raise ValueError(f"Prediction checkpoint contains a non-success record: {predictions_path}")
    completed = [record.sample.sample_id for record in records]
    if len(completed) != len(set(completed)):
        raise ValueError(f"Duplicate successful sample_id found in {predictions_path}")
    return set(completed)


def append_inference_records(records: Iterable[EvaluationRecord], predictions_path: Path, errors_path: Path) -> None:
    """Append each completed record to its success or error artifact."""
    grouped = {
        predictions_path: [],
        errors_path: [],
    }

    for record in records:
        target = errors_path if record.execution and record.execution.status == "error" else predictions_path
        grouped[target].append(record)

    for target, target_records in grouped.items():
        if not target_records:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(jsonable(record), ensure_ascii=False) + "\n" for record in target_records)
            handle.flush()
