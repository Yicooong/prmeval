"""内部样本与统一记录的字段转换及预测匹配校验。"""

from __future__ import annotations

from typing import Any

from .config import InferConfig
from .schemas import (
    EvaluationRecord,
    EvaluationSample,
    Prediction,
    PreferencePrediction,
    PreferenceSample,
    ProgressPrediction,
    ProgressSample,
    RecordInputItem,
    Trajectory,
    ValuePayload,
)


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


def record_to_sample(record: EvaluationRecord) -> EvaluationSample:
    """将帧已加载的统一记录转换为单轨迹或偏好样本并恢复标签字段; 调用方负责提前加载帧。"""
    common = {
        "task": record.input.task,
        "data_source": record.evaluation.dataset.source or record.evaluation.dataset.name,
    }
    if record.evaluation.type == "quality_preference":
        by_role = {item.role: item for item in record.input.items}
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
    item = record.input.items[0]
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


def build_inference_record(
    source: EvaluationRecord,
    config: InferConfig,
    prediction: Prediction | None = None,
    error: str | None = None,
    error_response: Any = None,
) -> EvaluationRecord:
    """Attach normalized prediction or execution failure to a source record."""
    normalized = None
    model = config.model_id or config.model_path or config.name
    version = config.model_version
    if isinstance(prediction, ProgressPrediction):
        model = prediction.model
        version = prediction.model_version
        normalized = ValuePayload(
            kind="progress",
            values=prediction.progress,
        )
    elif isinstance(prediction, PreferencePrediction):
        model = prediction.model
        version = prediction.model_version
        normalized = ValuePayload(
            kind="preference",
            label=prediction.preference,
            probability=prediction.chosen_probability,
        )
    payload = source.model_dump()
    payload.update(
        {
            "infer": {
                "name": config.name,
                "model": model,
                "version": version,
            },
            "prediction": normalized,
            "execution": {
                "status": "error" if error else "success",
                "error": error,
                "raw_response": error_response,
            },
        }
    )
    return EvaluationRecord.model_validate(payload)


def clear_non_string_frame_values(record: EvaluationRecord) -> EvaluationRecord:
    """返回记录副本, 保留字符串形式的 frames, 将数组、结构化引用等非字符串值替换为空列表。"""
    items = [
        item if isinstance(item.frames, str) else item.model_copy(update={"frames": []}) for item in record.input.items
    ]
    return record.model_copy(update={"input": record.input.model_copy(update={"items": items})})
