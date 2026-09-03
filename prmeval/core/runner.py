from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar
from statistics import mean

from tqdm import tqdm

from ..infer.base import Infer
from ..metrics.builtins import compute_metrics
from ..sample import EvalSampler, create_samplers, load_hf_trajectory_pool
from .config import EvalConfig
from .registry import INFERS
from .schemas import (
    EvaluationRecord,
    PreferencePrediction,
    ProgressPrediction,
    ValuePayload,
    jsonable,
)
from .utils import (
    batched,
    iter_sampler_samples,
    load_sample_artifacts,
    read_jsonl,
    record_to_sample,
    sample_to_record,
    strip_record_frames,
    validate_prediction_for_sample,
    write_metric_details,
    write_sample_artifacts,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


class Evaluator:
    def __init__(self, config: EvalConfig, show_progress: bool = True):
        self.config = config
        self.show_progress = show_progress and sys.stderr.isatty()
        run_name = config.run_name or "default"
        self.output_dir = Path(config.output_dir) / run_name
        self.samples_path = self.output_dir / "samples.jsonl"
        self.predictions_path = self.output_dir / "predictions.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.metrics_path = self.output_dir / "metrics.json"
        self.metrics_detail_path = self.output_dir / "metrics_detail.jsonl"

    def _with_progress(
        self,
        iterable: Iterable[T],
        *,
        description: str,
        unit: str,
        total: int | None = None,
    ) -> Iterable[T]:
        if not self.show_progress:
            return iterable
        return tqdm(
            iterable,
            desc=description,
            unit=unit,
            total=total,
            file=sys.stderr,
            dynamic_ncols=True,
        )

    def _prepare_inference_outputs(self) -> set[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.config.resume:
            self.predictions_path.write_text("", encoding="utf-8")
            self.errors_path.write_text("", encoding="utf-8")
            return set()
        records = [EvaluationRecord.model_validate(row) for row in read_jsonl(self.predictions_path)]
        if any(not record.execution or record.execution.status != "success" for record in records):
            raise ValueError(f"Prediction checkpoint contains a non-success record: {self.predictions_path}")
        completed = [record.sample_id for record in records]
        if len(completed) != len(set(completed)):
            raise ValueError(f"Duplicate successful sample_id found in {self.predictions_path}")
        return set(completed)

    def _build_coverage_summary(self, *, total: int, executed: int, skipped: int) -> dict[str, int]:
        successful_ids = {row["sample_id"] for row in read_jsonl(self.predictions_path)}
        failed_ids = {
            row["sample_id"] for row in read_jsonl(self.errors_path) if row["sample_id"] not in successful_ids
        }
        return {
            "total": total,
            "successful": len(successful_ids),
            "failed": len(failed_ids),
            "executed": executed,
            "skipped": skipped,
        }

    def _build_inference_record(
        self,
        source: EvaluationRecord,
        prediction=None,
        error: str | None = None,
        error_response=None,
    ) -> EvaluationRecord:
        normalized = None
        model = self.config.infer.model_id or self.config.infer.model_path or self.config.infer.name
        version = self.config.infer.model_version
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
                    "name": self.config.infer.name,
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

    def _run_inference_batch(
        self,
        infer: Infer,
        sources: list[EvaluationRecord],
        samples: list,
    ) -> list[EvaluationRecord]:
        if not samples:
            return []
        try:
            infer.begin_prediction()
            predictions = infer.predict(samples)
            if not isinstance(predictions, list):
                raise TypeError(f"Infer.predict() must return a list, got {type(predictions).__name__}")
            expected_ids = [sample.sample_id for sample in samples]
            actual_ids = [prediction.sample_id for prediction in predictions]
            if len(predictions) != len(samples):
                raise ValueError(f"Batch prediction count mismatch: expected {len(samples)}, got {len(predictions)}")
            if len(actual_ids) != len(set(actual_ids)):
                raise ValueError("Batch predictions contain duplicate sample_id values")
            if set(actual_ids) != set(expected_ids):
                missing = sorted(set(expected_ids) - set(actual_ids))
                extra = sorted(set(actual_ids) - set(expected_ids))
                raise ValueError(f"Batch prediction sample_id mismatch: missing={missing}, extra={extra}")
            by_id = dict(zip(actual_ids, predictions, strict=True))
            records = []
            for source, sample in zip(sources, samples, strict=True):
                prediction = by_id[sample.sample_id]
                validate_prediction_for_sample(sample, prediction)
                records.append(self._build_inference_record(source, prediction=prediction))
            return records
        except Exception as exc:
            return [
                self._build_inference_record(
                    source,
                    error=f"{type(exc).__name__}: {exc}",
                    error_response=getattr(exc, "raw_response", None),
                )
                for source in sources
            ]

    def _append_inference_records(self, records: Iterable[EvaluationRecord]) -> None:
        grouped = {
            self.predictions_path: [],
            self.errors_path: [],
        }

        for record in records:
            target = (
                self.errors_path if record.execution and record.execution.status == "error" else self.predictions_path
            )
            grouped[target].append(record)

        for target, target_records in grouped.items():
            if not target_records:
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            with target.open("a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(jsonable(record), ensure_ascii=False) + "\n" for record in target_records)
                handle.flush()

    def _create_samplers(self) -> list[EvalSampler]:
        pool = load_hf_trajectory_pool(self.config.sampling)
        return create_samplers(self.config.sampling, pool=pool)

    def sample(self, samples_path: str | Path | None = None) -> dict:
        """Stage 1: load a Hugging Face Dataset, sample it, and write the portable sample protocol."""
        destination = Path(samples_path) if samples_path else self.samples_path
        logger.info("Stage 1/3 Sample started: %s", destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and self.config.resume:
            records = load_sample_artifacts(destination)
            counts = Counter(record.evaluation.type for record in records)
            source_ids = {
                item.source_id for record in records for item in record.input.items if item.source_id is not None
            }
            summary = {
                "schema_version": "bench.record.v1",
                "samples": len(records),
                "eval_types": dict(sorted(counts.items())),
                "trajectories": len(source_ids),
                "path": str(destination),
                "reused": True,
            }
            logger.info(
                "Stage 1/3 Sample completed: reused %d samples",
                summary["samples"],
            )
            return summary

        samplers = self._create_samplers()
        samples = list(
            self._with_progress(
                iter_sampler_samples(samplers),
                description="Stage 1/3 Generate samples",
                unit="sample",
            )
        )
        if not samples:
            raise ValueError(
                f"Sampling produced no samples for eval types: {', '.join(self.config.sampling.eval_types)}"
            )
        summary = write_sample_artifacts(
            self._with_progress(
                samples,
                description="Stage 1/3 Write samples",
                unit="sample",
                total=len(samples),
            ),
            destination,
            self.config.sampling.dataset_name,
        )
        trajectories_loaded = max((sampler.pool_size for sampler in samplers), default=0)
        summary.update({"trajectories": trajectories_loaded, "reused": False})
        logger.info(
            "Stage 1/3 Sample completed: %d trajectories, %d samples",
            trajectories_loaded,
            summary["samples"],
        )
        return summary

    def infer(
        self,
        samples_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
    ) -> dict:
        """Stage 2: load only sample artifacts, call the model, and write EvaluationRecords."""
        source = Path(samples_path) if samples_path else self.samples_path
        destination = Path(predictions_path) if predictions_path else self.predictions_path
        logger.info("Stage 2/3 Infer started: %s", source)
        if destination != self.predictions_path:
            self.predictions_path = destination
            self.errors_path = destination.with_name(f"{destination.stem}.errors.jsonl")
            self.output_dir = destination.parent
        all_records = load_sample_artifacts(source)
        infer_cls = INFERS.get(self.config.infer.name)
        eval_types = {record.evaluation.type for record in all_records}
        required = {"preference" if eval_type == "quality_preference" else "progress" for eval_type in eval_types}
        unsupported = required - infer_cls.capabilities
        if unsupported:
            raise ValueError(f"Infer '{self.config.infer.name}' does not support: {', '.join(sorted(unsupported))}")
        completed = self._prepare_inference_outputs()
        records_to_run = [record for record in all_records if record.sample_id not in completed]
        logger.info(
            "Stage 2/3 Infer workload: %d pending, %d skipped",
            len(records_to_run),
            len(completed),
        )
        new_records: list[EvaluationRecord] = []
        pending_records = self._with_progress(
            records_to_run,
            description=f"Stage 2/3 Infer (skipped={len(completed)})",
            unit="sample",
            total=len(records_to_run),
        )
        infer = infer_cls(self.config.infer) if records_to_run else None
        for source_batch in batched(pending_records, self.config.infer.batch_size):
            runtime_sources: list[EvaluationRecord] = []
            runtime_samples = []
            for source_record in source_batch:
                try:
                    runtime_samples.append(record_to_sample(source_record, source.parent))
                    runtime_sources.append(source_record)
                except Exception as exc:
                    record = self._build_inference_record(
                        source_record,
                        error=f"{type(exc).__name__}: {exc}",
                        error_response=getattr(exc, "raw_response", None),
                    )
                    new_records.append(record)
                    self._append_inference_records([record])
            inferred_records = self._run_inference_batch(infer, runtime_sources, runtime_samples)  # type: ignore[arg-type]
            new_records.extend(inferred_records)
            self._append_inference_records(inferred_records)
        summary = {
            "coverage": self._build_coverage_summary(
                total=len(all_records), executed=len(new_records), skipped=len(completed)
            ),
            "samples": str(source),
            "predictions": str(self.predictions_path),
            "errors": str(self.errors_path),
        }
        logger.info(
            "Stage 2/3 Infer completed: %d successful, %d failed, %d executed, %d skipped",
            summary["coverage"]["successful"],
            summary["coverage"]["failed"],
            summary["coverage"]["executed"],
            summary["coverage"]["skipped"],
        )
        return summary

    def _infer_continuously(self) -> dict:
        """Sample and infer in bounded batches without materializing Stage-1 artifacts."""
        logger.info("Stage 1/3 Sample started: in-memory continuous pipeline")
        samplers = create_samplers(self.config.sampling)
        infer_cls = INFERS.get(self.config.infer.name)
        eval_types = [sampler.eval_type for sampler in samplers]
        required = {"preference" if eval_type == "quality_preference" else "progress" for eval_type in eval_types}
        unsupported = required - infer_cls.capabilities
        if unsupported:
            raise ValueError(f"Infer '{self.config.infer.name}' does not support: {', '.join(sorted(unsupported))}")
        completed = self._prepare_inference_outputs()

        generated = 0
        skipped = 0
        executed = 0
        seen: set[str] = set()
        samples = self._with_progress(
            iter_sampler_samples(samplers),
            description=f"Stage 1-2/3 Sample and infer (skipped={len(completed)})",
            unit="sample",
            total=mean([sampler.pool_size for sampler in samplers])
        )
        infer = None
        for sample_batch in batched(samples, self.config.infer.batch_size):
            generated += len(sample_batch)
            runtime_samples = []
            runtime_records = []
            for sample in sample_batch:
                if sample.sample_id in seen:
                    raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
                seen.add(sample.sample_id)
                if sample.sample_id in completed:
                    skipped += 1
                    continue
                record = sample_to_record(sample, self.config.sampling.dataset_name)
                runtime_samples.append(sample)
                runtime_records.append(strip_record_frames(record))
            if runtime_samples:
                infer = infer or infer_cls(self.config.infer)
                records = self._run_inference_batch(infer, runtime_records, runtime_samples)
                executed += len(records)
                self._append_inference_records(records)
        if generated == 0:
            raise ValueError(
                f"Sampling produced no samples for eval types: {', '.join(self.config.sampling.eval_types)}"
            )
        summary = {
            "coverage": self._build_coverage_summary(total=generated, executed=executed, skipped=skipped),
            "samples": None,
            "predictions": str(self.predictions_path),
            "errors": str(self.errors_path),
            "execution": {
                "mode": "continue",
                "batch_size": self.config.infer.batch_size,
                "trajectories": max((sampler.pool_size for sampler in samplers), default=0),
                "samples": generated,
            },
        }
        logger.info(
            "Stage 2/3 Infer completed: %d successful, %d failed, %d executed, %d skipped",
            summary["coverage"]["successful"],
            summary["coverage"]["failed"],
            summary["coverage"]["executed"],
            summary["coverage"]["skipped"],
        )
        return summary

    def evaluate_metrics(
        self,
        predictions_path: str | Path | None = None,
        *,
        coverage: dict[str, int] | None = None,
    ) -> dict:
        """Stage 3: compute metrics from complete post-model EvaluationRecords only."""
        source = Path(predictions_path) if predictions_path else self.predictions_path
        logger.info("Stage 3/3 Metrics started: %s", source)
        records = [EvaluationRecord.model_validate(row) for row in read_jsonl(source)]
        if not records:
            raise ValueError(f"No successful EvaluationRecord rows found in {source}")
        if any(not record.execution or record.execution.status != "success" for record in records):
            raise ValueError(f"Metric input must contain only successful records: {source}")
        identities = [
            (record.evaluation.dataset.name, record.infer.name if record.infer else None, record.sample_id)
            for record in records
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Metric input contains duplicate dataset/infer/sample identities: {source}")
        metric_names = self.config.metrics or self.config.sampling.eval_types
        metrics = compute_metrics(
            records,
            self._with_progress(
                metric_names,
                description="Stage 3/3 Compute metrics",
                unit="metric",
                total=len(metric_names),
            ),
        )
        if coverage is None:
            successful_ids = {record.sample_id for record in records}
            failed = (
                len({row["sample_id"] for row in read_jsonl(self.errors_path)} - successful_ids)
                if source == self.predictions_path
                else 0
            )
            coverage = {
                "total": len(records) + failed,
                "successful": len(records),
                "failed": failed,
                "executed": 0,
                "skipped": 0,
            }
        summary_metrics = {
            name: {key: value for key, value in result.items() if key not in {"details", "task_details"}}
            for name, result in metrics.items()
        }
        summary = {
            "metrics": summary_metrics,
            "coverage": coverage,
            "predictions": str(source),
            "details": str(self.metrics_detail_path),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path.write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
        write_metric_details(self.metrics_detail_path, records, metrics)
        logger.info(
            "Stage 3/3 Metrics completed: %d metrics from %d predictions",
            len(metrics),
            len(records),
        )
        return summary

    def run(self) -> dict:
        """Convenience orchestration for stage 1 -> stage 2 -> stage 3."""
        if self.config.mode == "continue":
            inference = self._infer_continuously()
        else:
            self.sample()
            inference = self.infer()
        if inference["coverage"]["successful"] == 0:
            logger.info("Stage 3/3 Metrics skipped: no successful predictions")
            return {"metrics": {}, **inference}
        return self.evaluate_metrics(coverage=inference["coverage"])
