from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from prmeval.utils import load_frames

from .utils import _file_sha256, jsonable


class FrameworkModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class Trajectory(FrameworkModel):
    id: str
    task: str
    frames: Any
    data_source: str = "unknown"
    is_robot: bool = False
    is_simulation: bool = False
    quality_label: str | None = None
    partial_success: float | None = None
    preference_group_id: str | None = None
    preference_rank: int | float | None = None
    target_progress: list[float] | None = None
    frame_indices: list[int] | None = None
    num_frames_total: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProgressSample(FrameworkModel):
    sample_id: str
    trajectory: Trajectory
    sample_type: Literal["progress"] = "progress"
    eval_type: str


class PreferenceSample(FrameworkModel):
    sample_id: str
    chosen_trajectory: Trajectory
    rejected_trajectory: Trajectory
    sample_type: Literal["preference"] = "preference"
    eval_type: str


EvaluationSample = ProgressSample | PreferenceSample


class FrameReference(FrameworkModel):
    """Portable reference to a materialized frame array in a sample bundle."""

    type: Literal["npz"] = "npz"
    path: str
    key: str = "frames"
    num_frames: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=64)


class ProgressPrediction(FrameworkModel):
    sample_id: str
    progress: list[Annotated[float, Field(ge=0, le=1)]] = Field(min_length=1)
    model: str
    model_version: str | None = None
    raw_response: Any = None


class PreferencePrediction(FrameworkModel):
    sample_id: str
    chosen_probability: float = Field(ge=0, le=1)
    preference: Literal["chosen", "rejected", "tie"]
    model: str
    model_version: str | None = None
    raw_response: Any = None


Prediction = ProgressPrediction | PreferencePrediction


class DatasetIdentity(FrameworkModel):
    """Dataset identity used for metric slicing, independent of loading details."""

    name: str = Field(description="Canonical dataset name, for example rbm-1m-ood")
    source: str | None = Field(default=None, description="Optional subset or original source name")


class EvaluationIdentity(FrameworkModel):
    type: str = Field(description="Evaluation type, for example progress or policy_ranking")
    dataset: DatasetIdentity = Field(description="Dataset dimensions associated with this sample")


class RecordInputItem(FrameworkModel):
    """One media item in a request; preference evaluation can contain chosen and rejected items."""

    role: str = Field(default="trajectory", description="Input role such as trajectory, chosen, or rejected")
    frames: Any = Field(description="FrameReference on disk; temporarily hydrated to an array at runtime")
    frame_indices: list[int] = Field(default_factory=list, description="Sampled indices in the source sequence")
    source_id: str | None = Field(default=None, description="Optional source ID for audit and debugging only")
    data: dict[str, Any] = Field(default_factory=dict, description="Non-core extensions for this input item")


class RecordInput(FrameworkModel):
    task: str = Field(description="Natural-language task supplied to the model")
    items: list[RecordInputItem] = Field(min_length=1, description="Input items used by this model request")


class ValuePayload(FrameworkModel):
    """Extensible target/prediction payload validated by each metric according to kind."""

    kind: str = Field(description="Payload kind such as progress, rank, or preference")
    values: list[float] | None = Field(default=None, description="Sequence values such as a progress curve")
    value: float | None = Field(default=None, description="Single numeric value such as a ground-truth rank")
    label: str | None = Field(default=None, description="Discrete label such as chosen or successful")
    probability: float | None = Field(default=None, ge=0, le=1, description="Optional probability prediction")


class InferIdentity(FrameworkModel):
    name: str = Field(description="Registered infer name in the evaluation framework")
    model: str = Field(description="Local checkpoint or remote model identity used for inference")
    version: str | None = Field(default=None, description="Optional model or deployment version")


class ExecutionInfo(FrameworkModel):
    status: Literal["success", "error"] = Field(description="Inference status for this sample")
    error: str | None = Field(default=None, description="Error summary when inference fails")
    raw_response: Any = Field(default=None, description="Unparsed backend response retained when inference fails")


class EvaluationRecord(FrameworkModel):
    """Unified sample/inference record consumed, but not mutated, by metrics."""

    # schema_version controls protocol compatibility; execution presence tracks inference state.
    schema_version: Literal["bench.record.v1"] = Field(
        default="bench.record.v1", description="Unified record protocol version"
    )
    sample_id: str = Field(description="Unique sample ID preserved across all three stages")

    # evaluation/input/target originate in Stage 1 and must be preserved by Stage 2.
    evaluation: EvaluationIdentity = Field(description="Evaluation type and dataset dimensions")
    input: RecordInput = Field(description="Model input and sampling information")
    target: ValuePayload | None = Field(default=None, description="Metric target; never sent to the remote model")

    # infer/prediction/execution are populated by Stage 2 and forbidden on sampled records.
    infer: InferIdentity | None = Field(default=None, description="Identity of the predicting infer/model")
    prediction: ValuePayload | None = Field(default=None, description="Model output normalized by its baseline")
    execution: ExecutionInfo | None = Field(default=None, description="Inference status and optional error response")

    @model_validator(mode="after")
    def validate_inference_state(self):
        """校验记录的推理状态: 采样记录不能有结果, 成功必须有预测, 失败必须有错误信息。"""
        if self.execution is None:
            if self.infer is not None or self.prediction is not None:
                raise ValueError("A sampled record cannot contain infer or prediction results")
            return self
        if self.infer is None:
            raise ValueError("An inferred record requires infer information")
        if self.execution.status == "success" and self.prediction is None:
            raise ValueError("A successful inferred record requires a prediction")
        if self.execution.status == "error":
            if not self.execution.error:
                raise ValueError("An error inferred record requires an error message")
            if self.prediction is not None:
                raise ValueError("An error inferred record cannot contain a prediction")
        return self


def _build_target_for_eval_type(sample: ProgressSample) -> ValuePayload:
    """按评估类型构建标准答案: 进度序列、策略排名或语言任务与视频任务的匹配值。"""
    QUALITY_RANK = {"successful": 2.0, "suboptimal": 1.0, "failure": 0.0, "failed": 0.0}
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
    """将单轨迹或偏好样本转换为跨阶段的统一记录, 填充输入、数据集身份和标准答案。"""
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
        target = _build_target_for_eval_type(sample)
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


def _load_item_frames(item: RecordInputItem, bundle_dir: Path, verify: bool) -> RecordInputItem:
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


def _ensure_item_frames_loaded(item: RecordInputItem, bundle_dir: Path) -> RecordInputItem:
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
                if record.sample_id in seen:
                    raise ValueError(f"duplicate sample_id '{record.sample_id}'")
                seen.add(record.sample_id)
                hydrated_items = [_load_item_frames(item, path.parent, verify) for item in record.input.items]
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


SAMPLE_SCHEMA_VERSION = "bench.record.v1"


def save_samples_to_bundle(samples: Iterable[EvaluationSample], path: Path, dataset_name: str = "unknown") -> dict:
    """将样本转换为统一记录, 保存关联的 NPZ 帧文件及 JSONL, 并返回数量和评估类型统计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    sample_count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            if sample.sample_id in seen:
                raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
            seen.add(sample.sample_id)
            record = _save_record_frames(sample_to_record(sample, dataset_name), path.parent)
            handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
            counts[record.evaluation.type] += 1
            sample_count += 1
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "samples": sample_count,
        "eval_types": dict(sorted(counts.items())),
        "path": str(path),
    }


def validate_sample_bundle(path: Path) -> dict:
    """完整校验采样记录及关联帧文件, 返回样本数、帧数等统计; 校验失败时抛出异常。"""
    records = load_sample_records(path, verify=True)
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


def write_metric_details_jsonl(path: Path, records: list[EvaluationRecord], metrics: dict) -> None:
    """将已计算的逐样本和逐组指标详情写入 JSONL; 本函数不计算指标。"""
    record_rows = {
        record.sample_id: {
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


def validate_prediction_for_sample(sample: EvaluationSample, prediction: Prediction) -> None:
    """检查预测与样本的 ID、类型是否一致, 并要求进度预测数量等于输入帧数。"""
    if prediction.sample_id != sample.sample_id:
        raise ValueError(f"Prediction sample_id mismatch: expected {sample.sample_id}, got {prediction.sample_id}")
    if sample.sample_type == "progress" and not isinstance(prediction, ProgressPrediction):
        raise TypeError(f"Progress sample requires ProgressPrediction, got {type(prediction).__name__}")
    if sample.sample_type == "preference" and not isinstance(prediction, PreferencePrediction):
        raise TypeError(f"Preference sample requires PreferencePrediction, got {type(prediction).__name__}")
    if isinstance(prediction, ProgressPrediction):
        expected = len(sample.trajectory.frames)
        actual = len(prediction.progress)
        if actual != expected:
            raise ValueError(f"Progress length mismatch: expected {expected}, got {actual}")


def _clear_non_string_frame_values(record: EvaluationRecord) -> EvaluationRecord:
    """返回记录副本, 保留字符串形式的 frames, 将数组、结构化引用等非字符串值替换为空列表。"""
    items = [
        item if isinstance(item.frames, str) else item.model_copy(update={"frames": []}) for item in record.input.items
    ]
    return record.model_copy(update={"input": record.input.model_copy(update={"items": items})})


def _save_item_frames(item: RecordInputItem, sample_id: str, bundle_dir: Path) -> RecordInputItem:
    """将一个输入项的帧压缩保存到 sample_frames 目录, 返回以文件引用替代帧内容的输入项副本。"""
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


def _save_record_frames(record: EvaluationRecord, bundle_dir: Path) -> EvaluationRecord:
    """保存记录中所有输入项的帧, 返回包含文件引用的记录副本。"""
    items = [_save_item_frames(item, record.sample_id, bundle_dir) for item in record.input.items]
    return record.model_copy(update={"input": record.input.model_copy(update={"items": items})})


def record_to_sample(record: EvaluationRecord, bundle_dir: Path) -> EvaluationSample:
    """将统一记录转换为推理所需的单轨迹或偏好样本, 按需加载帧并恢复转换所需的标签字段。"""
    common = {
        "task": record.input.task,
        "data_source": record.evaluation.dataset.source or record.evaluation.dataset.name,
    }
    if record.evaluation.type == "quality_preference":
        by_role = {item.role: _ensure_item_frames_loaded(item, bundle_dir) for item in record.input.items}
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
    item = _ensure_item_frames_loaded(record.input.items[0], bundle_dir)
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
