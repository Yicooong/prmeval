from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np

from ..core.registry import METRICS, register_metric
from ..core.schemas import EvaluationRecord


class Metric(ABC):
    @abstractmethod
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        raise NotImplementedError


def _successful(record: EvaluationRecord) -> bool:
    return bool(record.execution and record.execution.status == "success")


def _pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        return 0.0
    return float(np.corrcoef(target, prediction)[0, 1])


def _kendall(a: list[float], b: list[float]) -> float:
    concordant = discordant = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            product = (a[i] - a[j]) * (b[i] - b[j])
            concordant += product > 0
            discordant += product < 0
    total = len(a) * (len(a) - 1) / 2
    return float((concordant - discordant) / total) if total else 0.0


def _slice_key(record: EvaluationRecord) -> str:
    infer = record.infer.name if record.infer else "unknown"
    return f"{record.evaluation.dataset.name}:{infer}"


@register_metric("reward_alignment")
class RewardAlignmentMetric(Metric):
    """Per-sample progress MSE/Pearson, then equal-weighted sample aggregation."""

    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        valid = [
            record for record in records
            if record.evaluation.type == "reward_alignment"
            and _successful(record)
            and record.target is not None and record.target.kind == "progress"
            and record.prediction is not None and record.prediction.kind == "progress"
        ]
        details: dict[str, Any] = {}
        by_slice: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for record in valid:
            target = np.asarray(record.target.values or [], dtype=float)
            prediction = np.asarray(record.prediction.values or [], dtype=float)
            if len(target) == 0 or len(target) != len(prediction):
                raise ValueError(
                    f"reward_alignment sample '{record.sample_id}' requires non-empty, equal-length "
                    "target.values and prediction.values"
                )
            if np.any((target < 0) | (target > 1)) or np.any((prediction < 0) | (prediction > 1)):
                raise ValueError(f"reward_alignment sample '{record.sample_id}' contains progress outside [0, 1]")
            mse = float(np.mean((prediction - target) ** 2))
            correlation = _pearson(target, prediction)
            details[record.sample_id] = {
                "mse": mse,
                "pearson": correlation,
                "dataset": record.evaluation.dataset.name,
                "infer": record.infer.name if record.infer else None,
            }
            by_slice[_slice_key(record)].append((mse, correlation))
        losses = [value[0] for pairs in by_slice.values() for value in pairs]
        correlations = [value[1] for pairs in by_slice.values() for value in pairs]
        slices = {
            key: {
                "mse": float(np.mean([value[0] for value in values])),
                "pearson": float(np.mean([value[1] for value in values])),
                "num_samples": len(values),
            }
            for key, values in sorted(by_slice.items())
        }
        mean_mse = float(np.mean(losses)) if losses else None
        return {
            "mse": mean_mse,
            "loss": mean_mse,
            "pearson": float(np.mean(correlations)) if correlations else None,
            "num_samples": len(losses),
            "slices": slices,
            "details": details,
        }


@register_metric("quality_preference")
class QualityPreferenceMetric(Metric):
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        predictions = [
            record.prediction
            for record in records
            if record.evaluation.type == "quality_preference"
            and _successful(record)
            and record.prediction is not None
            and record.prediction.kind == "preference"
        ]
        correct = sum(prediction.label == "chosen" for prediction in predictions)
        ties = sum(prediction.label == "tie" for prediction in predictions)
        return {
            "accuracy": correct / len(predictions) if predictions else None,
            "tie_rate": ties / len(predictions) if predictions else None,
            "num_comparisons": len(predictions),
        }


@register_metric("policy_ranking")
class PolicyRankingMetric(Metric):
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        by_task: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
        for record in records:
            if not (
                record.evaluation.type == "policy_ranking"
                and _successful(record)
                and record.target is not None and record.target.kind == "rank"
                and record.target.value is not None
                and record.prediction is not None and record.prediction.kind == "progress"
                and record.prediction.values
            ):
                continue
            curve = [float(value) for value in record.prediction.values]
            key = f"{_slice_key(record)}:{record.input.task_id or record.input.task}"
            by_task[key].append((
                float(record.target.value), curve[-1], float(np.mean(curve)), float(np.sum(curve))
            ))
        task_scores = {
            task: {
                "last": _kendall([x[0] for x in pairs], [x[1] for x in pairs]),
                "average": _kendall([x[0] for x in pairs], [x[2] for x in pairs]),
                "sum": _kendall([x[0] for x in pairs], [x[3] for x in pairs]),
            }
            for task, pairs in by_task.items() if len(pairs) > 1
        }
        means = {
            name: float(np.mean([scores[name] for scores in task_scores.values()])) if task_scores else None
            for name in ("last", "average", "sum")
        }
        return {
            "kendall": means["last"],
            "kendall_last": means["last"],
            "kendall_average": means["average"],
            "kendall_sum": means["sum"],
            "num_tasks": len(task_scores),
            "task_details": task_scores,
        }


@register_metric("confusion_matrix")
class ConfusionMatrixMetric(Metric):
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        valid = [
            record for record in records
            if record.evaluation.type == "confusion_matrix"
            and _successful(record)
            and record.target is not None and record.target.kind == "task_match"
            and record.prediction is not None and record.prediction.kind == "progress"
            and record.prediction.values
            and record.target.data.get("lang_task") is not None
            and record.target.data.get("video_task") is not None
        ]
        tasks = sorted(
            {str(record.target.data["lang_task"]) for record in valid}
            | {str(record.target.data["video_task"]) for record in valid}
        )
        index = {task: i for i, task in enumerate(tasks)}
        matrix = np.zeros((len(tasks), len(tasks)), dtype=float)
        counts = np.zeros_like(matrix, dtype=int)
        for record in valid:
            row = index[str(record.target.data["lang_task"])]
            column = index[str(record.target.data["video_task"])]
            matrix[row, column] += float(record.prediction.values[-1])
            counts[row, column] += 1
        matrix = np.divide(matrix, counts, out=np.zeros_like(matrix), where=counts != 0)
        trace = float(np.trace(matrix))
        off_diagonal = float(np.sum(matrix) - trace)
        n = len(tasks)
        average_diagonal = trace / n if n else 0.0
        average_off_diagonal = off_diagonal / (n * n - n) if n > 1 else 0.0
        return {
            "tasks": tasks,
            "confusion_matrix": matrix.tolist(),
            "count_matrix": counts.tolist(),
            "trace": trace,
            "off_diagonal_sum": off_diagonal,
            "trace_minus_offdiag": trace - off_diagonal,
            "avg_diagonal": average_diagonal,
            "avg_off_diagonal": average_off_diagonal,
            "normalized_trace_minus_offdiag": average_diagonal - average_off_diagonal,
            "num_samples": len(valid),
        }


def compute_metrics(records: list[EvaluationRecord], metric_names: list[str]) -> dict[str, Any]:
    return {name: METRICS.get(name)().compute(records) for name in metric_names}
