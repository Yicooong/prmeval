from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
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


@register_metric("progress")
class ProgressMetric(Metric):
    """Per-sample progress MSE/Pearson, then equal-weighted sample aggregation."""

    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        valid = [
            record
            for record in records
            if record.evaluation.type in {"progress", "reward_alignment"}
            and _successful(record)
            and record.target is not None
            and record.target.kind == "progress"
            and record.prediction is not None
            and record.prediction.kind == "progress"
        ]
        details: dict[str, Any] = {}
        by_slice: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for record in valid:
            target = np.asarray(record.target.values or [], dtype=float)
            prediction = np.asarray(record.prediction.values or [], dtype=float)
            if len(target) == 0 or len(target) != len(prediction):
                raise ValueError(
                    f"progress sample '{record.sample_id}' requires non-empty, equal-length "
                    "target.values and prediction.values"
                )
            if np.any((target < 0) | (target > 1)) or np.any((prediction < 0) | (prediction > 1)):
                raise ValueError(f"progress sample '{record.sample_id}' contains progress outside [0, 1]")
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


def _event_scores(stats: list[dict[str, Any]], prefix: str) -> dict[str, float | int | None]:
    true = sum(int(stat[f"{prefix}_true"]) for stat in stats)
    predicted = sum(int(stat[f"{prefix}_predicted"]) for stat in stats)
    true_positive = sum(int(stat[f"{prefix}_true_positive"]) for stat in stats)
    if true == 0:
        return {"precision": None, "recall": None, "f1": None, "num_events": 0}
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / true
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "num_events": true}


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_temporal(stats: list[dict[str, Any]]) -> dict[str, Any]:
    trend_total = sum(int(stat["trend_total"]) for stat in stats)
    trend_correct = sum(int(stat["trend_correct"]) for stat in stats)
    original_edges = sum(int(stat["original_edges"]) for stat in stats)
    original_violations = sum(int(stat["original_violations"]) for stat in stats)
    original_samples = [stat for stat in stats if stat["transform"] == "original"]
    shortcut = [stat for stat in stats if stat["shortcut_gap"] is not None]
    return {
        "mae": _mean_or_none([stat["mae"] for stat in stats]),
        "mse": _mean_or_none([stat["mse"] for stat in stats]),
        "pearson": _mean_or_none([stat["pearson"] for stat in stats]),
        "endpoint_mae": _mean_or_none([stat["endpoint_mae"] for stat in stats]),
        "trend_accuracy": trend_correct / trend_total if trend_total else None,
        "trend_edges": trend_total,
        "regression": _event_scores(stats, "regression"),
        "plateau": _event_scores(stats, "plateau"),
        "monotonicity_violation": (
            {
                "edge_rate": original_violations / original_edges if original_edges else None,
                "sample_rate": (
                    sum(stat["original_violations"] > 0 for stat in original_samples) / len(original_samples)
                    if original_samples
                    else None
                ),
                "num_edges": original_edges,
                "num_samples": len(original_samples),
            }
        ),
        "temporal_shortcut": {
            "state_mae": _mean_or_none([stat["mae"] for stat in shortcut]),
            "normalized_time_mae": _mean_or_none([stat["time_mae"] for stat in shortcut]),
            "gap": _mean_or_none([stat["shortcut_gap"] for stat in shortcut]),
            "num_samples": len(shortcut),
        },
        "num_samples": len(stats),
        "average_base_frames": _mean_or_none([float(stat["base_frames"]) for stat in stats]),
        "average_transformed_frames": _mean_or_none([float(stat["transformed_frames"]) for stat in stats]),
        "average_length_ratio": _mean_or_none([float(stat["length_ratio"]) for stat in stats]),
    }


@register_metric("progress_temporal_variation")
class ProgressTemporalVariationMetric(Metric):
    """State-alignment and temporal-shortcut metrics over synthetic frame mappings."""

    prediction_trend_tolerance = 0.05

    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        valid = [
            record
            for record in records
            if record.evaluation.type in {"progress_temporal_variation", "synthetic_temporal_robustness"}
            and _successful(record)
            and record.target is not None
            and record.target.kind == "progress"
            and record.prediction is not None
            and record.prediction.kind == "progress"
        ]
        stats: list[dict[str, Any]] = []
        details: dict[str, Any] = {}
        for record in valid:
            target = np.asarray(record.target.values or [], dtype=float)
            prediction = np.asarray(record.prediction.values or [], dtype=float)
            if len(target) == 0 or len(target) != len(prediction):
                raise ValueError(
                    f"progress_temporal_variation sample '{record.sample_id}' requires non-empty, equal-length "
                    "target.values and prediction.values"
                )
            if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
                raise ValueError(f"progress_temporal_variation sample '{record.sample_id}' contains non-finite values")
            if np.any((target < 0) | (target > 1)) or np.any((prediction < 0) | (prediction > 1)):
                raise ValueError(
                    f"progress_temporal_variation sample '{record.sample_id}' contains progress outside [0, 1]"
                )
            metadata = record.input.items[0].data.get("synthetic_temporal", {})
            transform = str(metadata.get("transform") or "unknown")
            target_delta = np.diff(target)
            prediction_delta = np.diff(prediction)
            target_trend = np.sign(target_delta).astype(int)
            prediction_trend = np.where(
                prediction_delta > self.prediction_trend_tolerance,
                1,
                np.where(prediction_delta < -self.prediction_trend_tolerance, -1, 0),
            )
            regression_true = target_trend == -1
            regression_predicted = prediction_trend == -1
            plateau_true = target_trend == 0
            plateau_predicted = prediction_trend == 0
            error = prediction - target
            time_target = np.linspace(0, 1, len(prediction))
            time_mae = float(np.mean(np.abs(prediction - time_target))) if transform != "original" else None
            stat = {
                "sample_id": record.sample_id,
                "slice": _slice_key(record),
                "transform": transform,
                "mae": float(np.mean(np.abs(error))),
                "mse": float(np.mean(error**2)),
                "pearson": _pearson(target, prediction),
                "endpoint_mae": float(abs(error[-1])),
                "trend_correct": int(np.sum(target_trend == prediction_trend)),
                "trend_total": len(target_delta),
                "regression_true": int(np.sum(regression_true)),
                "regression_predicted": int(np.sum(regression_predicted)),
                "regression_true_positive": int(np.sum(regression_true & regression_predicted)),
                "plateau_true": int(np.sum(plateau_true)),
                "plateau_predicted": int(np.sum(plateau_predicted)),
                "plateau_true_positive": int(np.sum(plateau_true & plateau_predicted)),
                "original_edges": len(target_delta) if transform == "original" else 0,
                "original_violations": (
                    int(np.sum(prediction_delta < -self.prediction_trend_tolerance)) if transform == "original" else 0
                ),
                "time_mae": time_mae,
                "shortcut_gap": float(np.mean(np.abs(error))) - time_mae if time_mae is not None else None,
                "base_frames": int(metadata.get("base_frames", len(target))),
                "transformed_frames": int(metadata.get("transformed_frames", len(target))),
                "length_ratio": float(metadata.get("length_ratio", 1.0)),
            }
            stats.append(stat)
            details[record.sample_id] = {
                key: stat[key]
                for key in (
                    "transform",
                    "mae",
                    "mse",
                    "pearson",
                    "endpoint_mae",
                    "time_mae",
                    "shortcut_gap",
                    "base_frames",
                    "transformed_frames",
                    "length_ratio",
                )
            }

        def grouped(key):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for stat in stats:
                groups[str(key(stat))].append(stat)
            return {name: _aggregate_temporal(values) for name, values in sorted(groups.items())}

        result = _aggregate_temporal(stats)
        result.update(
            {
                "slices": grouped(lambda stat: stat["slice"]),
                "transforms": grouped(lambda stat: stat["transform"]),
                "transform_slices": grouped(lambda stat: f"{stat['slice']}:{stat['transform']}"),
                "details": details,
            }
        )
        return result


@register_metric("quality_preference")
class QualityPreferenceMetric(Metric):
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        valid = [
            record
            for record in records
            if record.evaluation.type == "quality_preference"
            and _successful(record)
            and record.prediction is not None
            and record.prediction.kind == "preference"
        ]
        predictions = [record.prediction for record in valid]
        correct = sum(prediction.label == "chosen" for prediction in predictions)
        ties = sum(prediction.label == "tie" for prediction in predictions)
        return {
            "accuracy": correct / len(predictions) if predictions else None,
            "tie_rate": ties / len(predictions) if predictions else None,
            "num_comparisons": len(predictions),
            "details": {
                record.sample_id: {
                    "correct": record.prediction.label == "chosen",
                    "tie": record.prediction.label == "tie",
                    "preference": record.prediction.label,
                    "chosen_probability": record.prediction.probability,
                }
                for record in valid
            },
        }


@register_metric("policy_ranking")
class PolicyRankingMetric(Metric):
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        by_task: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)
        details: dict[str, Any] = {}
        for record in records:
            if not (
                record.evaluation.type == "policy_ranking"
                and _successful(record)
                and record.target is not None
                and record.target.kind == "rank"
                and record.target.value is not None
                and record.prediction is not None
                and record.prediction.kind == "progress"
                and record.prediction.values
            ):
                continue
            curve = [float(value) for value in record.prediction.values]
            key = f"{_slice_key(record)}:{record.input.task}"
            target_rank = float(record.target.value)
            last = curve[-1]
            average = float(np.mean(curve))
            total = float(np.sum(curve))
            by_task[key].append((record.sample_id, target_rank, last, average, total))
            details[record.sample_id] = {
                "group_id": key,
                "target_rank": target_rank,
                "last": last,
                "average": average,
                "sum": total,
            }
        task_scores = {
            task: {
                "sample_ids": [x[0] for x in pairs],
                "last": _kendall([x[1] for x in pairs], [x[2] for x in pairs]),
                "average": _kendall([x[1] for x in pairs], [x[3] for x in pairs]),
                "sum": _kendall([x[1] for x in pairs], [x[4] for x in pairs]),
            }
            for task, pairs in by_task.items()
            if len(pairs) > 1
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
            "details": details,
        }


@register_metric("confusion_matrix")
class ConfusionMatrixMetric(Metric):
    def compute(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        valid = [
            record
            for record in records
            if record.evaluation.type == "confusion_matrix"
            and _successful(record)
            and record.target is not None
            and record.target.kind == "task_match"
            and record.prediction is not None
            and record.prediction.kind == "progress"
            and record.prediction.values
            and record.input.items[0].data.get("lang_task") is not None
            and record.input.items[0].data.get("video_task") is not None
        ]
        tasks = sorted(
            {str(record.input.items[0].data["lang_task"]) for record in valid}
            | {str(record.input.items[0].data["video_task"]) for record in valid}
        )
        index = {task: i for i, task in enumerate(tasks)}
        matrix = np.zeros((len(tasks), len(tasks)), dtype=float)
        counts = np.zeros_like(matrix, dtype=int)
        for record in valid:
            row = index[str(record.input.items[0].data["lang_task"])]
            column = index[str(record.input.items[0].data["video_task"])]
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
            "details": {
                record.sample_id: {
                    "lang_task": record.input.items[0].data["lang_task"],
                    "video_task": record.input.items[0].data["video_task"],
                    "target_match": bool(record.target.value),
                    "predicted_progress": float(record.prediction.values[-1]),
                }
                for record in valid
            },
        }


def compute_metrics(records: list[EvaluationRecord], metric_names: Iterable[str]) -> dict[str, Any]:
    return {name: METRICS.get(name)().compute(records) for name in metric_names}
