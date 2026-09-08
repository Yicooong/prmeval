from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from tqdm import tqdm

from prmeval.infer.base import Infer
from prmeval.infer.baselines import load_builtin_infer
from prmeval.metrics.builtins import compute_metrics
from prmeval.sample.samplers import EvalSampler
from prmeval.sample.utils import load_hf_trajectory_pool

from .config import EvalConfig, SamplingConfig
from .conversions import (
    build_inference_record,
    clear_non_string_frame_values,
    record_to_sample,
    sample_to_record,
    validate_prediction_for_sample,
)
from .registry import INFERS, SAMPLERS
from .schemas import (
    EvaluationRecord,
    EvaluationSample,
    Trajectory,
)
from .storage import (
    append_inference_records,
    load_record_frames,
    load_sample_records,
    prepare_inference_outputs,
    save_samples_to_bundle,
    write_metric_details_jsonl,
)
from .utils import batched, read_jsonl

logger = logging.getLogger(__name__)
T = TypeVar("T")


def create_samplers(
    config: SamplingConfig,
    pool: list[Trajectory] | None = None,
) -> list[EvalSampler]:
    return [SAMPLERS.get(eval_type)(config, config.dataset_name, pool=pool) for eval_type in config.eval_types]


class Evaluator:
    def __init__(self, config: EvalConfig, show_progress: bool = True):
        self.config = config
        self.show_progress = show_progress and sys.stderr.isatty()
        run_name = config.run_name or "default"
        self.output_dir = Path(config.output_dir) / run_name if config.output_dir is not None else None
        self.samples_path = self.output_dir / "samples.jsonl" if self.output_dir is not None else None
        self.predictions_path = self.output_dir / "predictions.jsonl" if self.output_dir is not None else None
        self.errors_path = self.output_dir / "errors.jsonl" if self.output_dir is not None else None
        self.metrics_path = self.output_dir / "metrics.json" if self.output_dir is not None else None
        self.metrics_detail_path = self.output_dir / "metrics_detail.jsonl" if self.output_dir is not None else None

        self.samplers: list[EvalSampler] | None = None
        self._successful_records: list[EvaluationRecord] | None = None
        self._inference_summary: dict | None = None
        self._stages_started = False

    def _iter_sampler_samples(self, samplers: Iterable[EvalSampler]) -> Iterator[EvaluationSample]:
        """Yield every sample produced by each sampler in order."""
        for sampler in samplers:
            yield from sampler.sample()

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

    def _reset_inference_state(self) -> None:
        self._successful_records = None
        self._inference_summary = None
        self._stages_started = True

    def _build_coverage_summary(
        self, *, successful_ids: set[str], failed_ids: set[str], total: int, executed: int, skipped: int
    ) -> dict[str, int]:
        return {
            "total": total,
            "successful": len(successful_ids),
            "failed": len(failed_ids - successful_ids),
            "executed": executed,
            "skipped": skipped,
        }

    def _run_inference_batch(
        self,
        infer: Infer,
        sources: list[EvaluationRecord],
        samples: list[EvaluationSample],
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
                records.append(build_inference_record(source, self.config.infer, prediction=prediction))
            return records
        except Exception as exc:
            return [
                build_inference_record(
                    source,
                    self.config.infer,
                    error=f"{type(exc).__name__}: {exc}",
                    error_response=getattr(exc, "raw_response", None),
                )
                for source in sources
            ]

    def _compute_record_metrics(self, records: list[EvaluationRecord], source: str | Path = "memory") -> dict:
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
        return compute_metrics(
            records,
            self._with_progress(
                metric_names,
                description="Stage 3/3 Compute metrics",
                unit="metric",
                total=len(metric_names),
            ),
        )


    def sample(self, samples_path: str | Path | None = None) -> dict:
        """Stage 1: prepare lazy samplers, optionally materializing a portable sample bundle.
            param samples_path: optional path to write a sample bundle; requires output_dir and save_samples=True 
        """
        self._reset_inference_state()
        self.samplers = None
        save_samples = self.output_dir is not None and self.config.save_samples
        if samples_path is not None and not save_samples:
            raise ValueError("samples_path output requires output_dir and save_samples=True")
        destination = (Path(samples_path) if samples_path is not None else self.samples_path) if save_samples else None
        logger.info("Stage 1/3 Sample started: %s", destination or "prepare samplers")
        pool = load_hf_trajectory_pool(self.config.sampling)
        self.samplers = create_samplers(self.config.sampling, pool=pool)
        trajectories_loaded = len(pool)
        if destination is None:
            logger.info(
                "Stage 1/3 Samplers prepared: %d trajectories; samples generated during infer", trajectories_loaded
            )
            return {
                "schema_version": "bench.record.v1",
                "samples": None,
                "eval_types": None,
                "trajectories": trajectories_loaded,
                "path": None,
                "reused": False,
            }
        # 如果不保存.npz,程序到这里已经结束了
        # 如果保存.npz,程序会继续执行到这里,将采样结果保存到指定路径
        summary = save_samples_to_bundle(
            self._with_progress(
                self._iter_sampler_samples(self.samplers),
                description="Stage 1/3 Generate and write samples",
                unit="sample",
            ),
            destination,
            self.config.sampling.dataset_name,
        )
        if summary["samples"] == 0:
            raise ValueError(
                f"Sampling produced no samples for eval types: {', '.join(self.config.sampling.eval_types)}"
            )
        self.samples_path = destination
        summary.update({"trajectories": trajectories_loaded, "reused": False})
        logger.info("Stage 1/3 Sample completed: %d trajectories, %d samples", trajectories_loaded, summary["samples"])
        return summary

    def infer(
        self,
        samples_path: str | Path | None = None,
        predictions_path: str | Path | None = None,
    ) -> tuple[dict, list[EvaluationRecord]]:
        """Stage 2: consume either a sample bundle or prepared samplers in bounded batches."""
        self._reset_inference_state()
        if predictions_path is not None and self.output_dir is None:
            raise ValueError("predictions_path requires output_dir")
        source = Path(samples_path) if samples_path is not None else None
        if source is None and self.samples_path is not None and self.samples_path.exists():
            source = self.samples_path
        inputs: Iterable[EvaluationRecord | EvaluationSample]
        total = None
        # | 来源      | `inputs` 中的元素                    | 是否调用 sampler |
        # | 文件      | `EvaluationRecord`，帧以文件引用保存  | 否              |
        # | samplers  | `ProgressSample / PreferenceSample` | 是，迭代时才生成 |
        if source is not None:
            inputs = load_sample_records(source)
            total = len(inputs)
            eval_types = {record.evaluation.type for record in inputs}
        elif self.samplers is not None:
            inputs = self._iter_sampler_samples(self.samplers)
            eval_types = {sampler.eval_type for sampler in self.samplers}
            total = sum(sampler.pool_size for sampler in self.samplers)
        else:
            raise ValueError("infer() requires samples_path or prepared samplers; call sample() first")

        load_builtin_infer(self.config.infer.name)
        infer_cls = INFERS.get(self.config.infer.name)
        required = {"preference" if eval_type == "quality_preference" else "progress" for eval_type in eval_types}
        unsupported = required - infer_cls.capabilities
        if unsupported:
            raise ValueError(f"Infer '{self.config.infer.name}' does not support: {', '.join(sorted(unsupported))}")
        if predictions_path is not None:
            destination = Path(predictions_path)
            if destination != self.predictions_path:
                self.predictions_path = destination
                self.errors_path = destination.with_name(f"{destination.stem}.errors.jsonl")

        checkpoint: dict[str, EvaluationRecord] = {}
        if self.output_dir is not None:
            prepare_inference_outputs(self.predictions_path, self.errors_path, resume=self.config.resume)
            for row in read_jsonl(self.predictions_path):
                record = EvaluationRecord.model_validate(row)
                checkpoint[record.sample_id] = record
        initially_skipped = sum(item.sample_id in checkpoint for item in inputs) if source is not None else 0
        pending = self._with_progress(
            inputs,
            description=f"Stage 2/3 Infer (skipped={initially_skipped})",
            unit="sample",
            total=total,
        )
        logger.info("Stage 2/3 Infer started: %s", source or "samplers")
        successful: dict[str, EvaluationRecord] = {}
        failed_ids: set[str] = set()
        seen: set[str] = set()
        input_order: list[str] = []
        executed = skipped = 0
        infer = None
        for source_batch in batched(pending, self.config.infer.batch_size):
            runtime_sources: list[EvaluationRecord] = []
            runtime_samples: list[EvaluationSample] = []
            batch_records: list[EvaluationRecord] = []
            # 准备当前batch的模型输入
            for item in source_batch:
                if item.sample_id in seen:
                    raise ValueError(f"Duplicate sample_id: {item.sample_id}")
                seen.add(item.sample_id)
                input_order.append(item.sample_id)
                if item.sample_id in checkpoint:
                    successful[item.sample_id] = checkpoint[item.sample_id]
                    skipped += 1
                    continue
                # 如果是文件输入，加载并转化为sample(带有np文件)
                if isinstance(item, EvaluationRecord):
                    source_record = item
                    try:
                        sample = record_to_sample(load_record_frames(source_record, source.parent))
                    except Exception as exc:
                        batch_records.append(
                            build_inference_record(
                                source_record,
                                self.config.infer,
                                error=f"{type(exc).__name__}: {exc}",
                                error_response=getattr(exc, "raw_response", None),
                            )
                        )
                        continue
                else:
                    sample = item
                    source_record = clear_non_string_frame_values(
                        sample_to_record(sample, self.config.sampling.dataset_name)
                    )
                runtime_sources.append(source_record)
                runtime_samples.append(sample)
                del sample
            if runtime_samples:
                if infer is None:
                    infer = infer_cls(self.config.infer)
                batch_records.extend(self._run_inference_batch(infer, runtime_sources, runtime_samples))
            executed += len(batch_records)
            for record in batch_records:
                if record.execution.status == "success":
                    successful[record.sample_id] = record
                else:
                    failed_ids.add(record.sample_id)
                    logger.warning("Inference failed for %s: %s", record.sample_id, record.execution.error)
            if self.output_dir is not None:
                append_inference_records(batch_records, self.predictions_path, self.errors_path)
            del runtime_samples, runtime_sources, source_batch, item
        if not seen:
            raise ValueError(
                f"Sampling produced no samples for eval types: {', '.join(self.config.sampling.eval_types)}"
            )
        summary = {
            "coverage": self._build_coverage_summary(
                successful_ids=set(successful),
                failed_ids=failed_ids,
                total=len(seen),
                executed=executed,
                skipped=skipped,
            ),
            "samples": str(source) if source is not None else None,
            "predictions": str(self.predictions_path) if self.predictions_path is not None else None,
            "errors": str(self.errors_path) if self.errors_path is not None else None,
        }
        records = [successful[sample_id] for sample_id in input_order if sample_id in successful]
        self._successful_records = records
        self._inference_summary = summary
        logger.info(
            "Stage 2/3 Infer completed: %d successful, %d failed, %d executed, %d skipped",
            summary["coverage"]["successful"],
            summary["coverage"]["failed"],
            summary["coverage"]["executed"],
            summary["coverage"]["skipped"],
        )
        return summary, records


    def evaluate_metrics(
        self,
        predictions_path: str | Path | None = None,
        *,
        records: list[EvaluationRecord] | None = None,
        coverage: dict[str, int] | None = None,
    ) -> dict:
        """Stage 3: compute complete metrics from explicit input or the latest inference result."""
        if predictions_path is not None and records is not None:
            raise ValueError("Provide either predictions_path or records, not both")
        source = Path(predictions_path) if predictions_path is not None else None
        if source is None and records is None:
            if self._successful_records is not None:
                records = self._successful_records
            elif not self._stages_started and self.predictions_path is not None:
                source = self.predictions_path
            else:
                raise ValueError("evaluate_metrics() requires predictions_path or records; call infer() first")
        from_inference = records is not None and records is self._successful_records
        if source is not None:
            if not source.is_file():
                raise FileNotFoundError(f"Prediction artifact not found: {source}")
            records = [EvaluationRecord.model_validate(row) for row in read_jsonl(source)]
        elif from_inference:
            source = self.predictions_path
            if coverage is None:
                coverage = self._inference_summary["coverage"]
        if coverage is None:
            successful_ids = {record.sample_id for record in records}
            failed = (
                len({row["sample_id"] for row in read_jsonl(self.errors_path)} - successful_ids)
                if source is not None and source == self.predictions_path and self.errors_path is not None
                else 0
            )
            coverage = {
                "total": len(records) + failed,
                "successful": len(records),
                "failed": failed,
                "executed": 0,
                "skipped": 0,
            }
        logger.info("Stage 3/3 Metrics started: %s", source or "memory")
        if not records and coverage["successful"] == 0 and coverage["failed"] > 0:
            metrics = {}
        else:
            metrics = self._compute_record_metrics(records, source or "memory")
        summary = {
            "metrics": metrics,
            "coverage": coverage,
            "predictions": str(source) if source is not None else None,
            "details": str(self.metrics_detail_path) if self.metrics_detail_path is not None else None,
        }
        if self.output_dir is not None:
            summary_metrics = {
                name: {key: value for key, value in result.items() if key not in {"details", "task_details"}}
                for name, result in metrics.items()
            }
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.write_text(
                json.dumps({**summary, "metrics": summary_metrics}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            write_metric_details_jsonl(self.metrics_detail_path, records, metrics)
        logger.info("Stage 3/3 Metrics completed: %d metrics from %d predictions", len(metrics), len(records))
        return summary

    def run(self) -> dict:
        """Convenience orchestration for stage 1 -> stage 2 -> stage 3."""
        self.sample()
        summary, records = self.infer()
        return self.evaluate_metrics(records=records, coverage=summary["coverage"])
