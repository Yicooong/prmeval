from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from tqdm import tqdm

from ..infer import create_infer
from ..metrics.builtins import compute_metrics
from ..sample.adapters import create_dataset
from ..sample.samplers import create_sampler
from .artifacts import load_sample_artifacts, record_to_sample, write_sample_artifacts
from .config import EvalConfig
from .schemas import (
    EvaluationRecord,
    PreferencePrediction,
    ProgressPrediction,
    RunManifest,
    ValuePayload,
    jsonable,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _fingerprint(config: EvalConfig) -> str:
    canonical = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sampling_fingerprint(config: EvalConfig) -> str:
    canonical = json.dumps(config.sampling.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _redacted_config(config: EvalConfig) -> dict:
    value = config.model_dump()
    infer = value.get("infer", {})
    if infer.get("api_key"):
        infer["api_key"] = "***"
    infer["headers"] = {
        key: "***" if any(secret in key.lower() for secret in ("authorization", "api-key", "token")) else item
        for key, item in infer.get("headers", {}).items()
    }
    return value


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Evaluator:
    def __init__(self, config: EvalConfig, show_progress: bool = False):
        self.config = config
        self.show_progress = show_progress and sys.stderr.isatty()
        self.fingerprint = _fingerprint(config)
        run_name = config.run_name or self.fingerprint[:12]
        self.output_dir = Path(config.output_dir) / run_name
        self.samples_path = self.output_dir / "samples.jsonl"
        self.sample_manifest_path = self.output_dir / "sample_manifest.json"
        self.predictions_path = self.output_dir / "predictions.jsonl"
        self.errors_path = self.output_dir / "errors.jsonl"
        self.inference_summary_path = self.output_dir / "inference_summary.json"
        self.manifest_path = self.output_dir / "run_manifest.json"

    def _progress(
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

    def _prepare(self, model_info: dict) -> set[str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = RunManifest.model_validate_json(self.manifest_path.read_text())
            if existing.fingerprint != self.fingerprint:
                raise RuntimeError(
                    f"Output directory contains a different run fingerprint: {existing.fingerprint}"
                )
            if existing.model_info.get("sample_artifact_sha256") != model_info.get("sample_artifact_sha256"):
                raise RuntimeError("Output directory contains predictions for a different sample artifact")
            existing.status = "running"
            existing.completed_at = None
            existing.summary = None
            self.manifest_path.write_text(json.dumps(jsonable(existing), indent=2), encoding="utf-8")
        else:
            manifest = RunManifest(
                run_id=self.output_dir.name,
                fingerprint=self.fingerprint,
                config=_redacted_config(self.config),
                environment={"python": sys.version, "platform": platform.platform()},
                model_info=model_info,
            )
            self.manifest_path.write_text(json.dumps(jsonable(manifest), indent=2), encoding="utf-8")
        if not self.config.resume:
            self.predictions_path.write_text("", encoding="utf-8")
            self.errors_path.write_text("", encoding="utf-8")
            return set()
        return {row["sample_id"] for row in _read_jsonl(self.predictions_path)}

    def _record_for(
        self,
        source: EvaluationRecord,
        prediction=None,
        error: str | None = None,
        latency: float | None = None,
        attempts: int = 1,
    ) -> EvaluationRecord:
        normalized = None
        model = self.config.infer.model_id
        version = self.config.infer.model_version
        if isinstance(prediction, ProgressPrediction):
            model = prediction.model
            version = prediction.model_version
            normalized = ValuePayload(
                kind="progress",
                values=prediction.progress,
                data={"raw_response": prediction.raw_response},
            )
        elif isinstance(prediction, PreferencePrediction):
            model = prediction.model
            version = prediction.model_version
            normalized = ValuePayload(
                kind="preference",
                label=prediction.preference,
                probability=prediction.chosen_probability,
                data={"raw_response": prediction.raw_response},
            )
        payload = source.model_dump()
        payload.update({
            "stage": "inferred",
            "infer": {
                "name": self.config.infer.name,
                "model": model,
                "version": version,
            },
            "prediction": normalized,
            "execution": {
                "status": "error" if error else "success",
                "error": error,
                "attempts": attempts,
                "latency_seconds": latency,
            },
        })
        return EvaluationRecord.model_validate(payload)

    def _predict(self, infer, source: EvaluationRecord, bundle_dir: Path):
        started = time.monotonic()
        infer.begin_prediction()
        try:
            sample = record_to_sample(source, bundle_dir)
            prediction = infer.predict(sample)
            if isinstance(prediction, ProgressPrediction):
                expected = len(sample.trajectory.frames)
                actual = len(prediction.progress)
                if actual != expected:
                    raise ValueError(f"Progress length mismatch: expected {expected}, got {actual}")
            return self._record_for(
                source, prediction=prediction, latency=time.monotonic() - started, attempts=infer.attempts()
            )
        except Exception as exc:
            return self._record_for(
                source, error=f"{type(exc).__name__}: {exc}", latency=time.monotonic() - started,
                attempts=max(1, infer.attempts()),
            )

    def _samples(self, trajectories) -> Iterable:
        for eval_type in self.config.sampling.eval_types:
            yield from create_sampler(
                eval_type, self.config.sampling, self.config.sampling.dataset_name
            ).sample(trajectories)

    def sample(self, samples_path: str | Path | None = None) -> dict:
        """Stage 1: adapt a dataset, sample it, and write the portable sample protocol."""
        destination = Path(samples_path) if samples_path else self.samples_path
        logger.info("Stage 1/3 Sample started: %s", destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest_path = (
            self.sample_manifest_path
            if destination == self.samples_path
            else destination.with_name(f"{destination.stem}.manifest.json")
        )
        fingerprint = _sampling_fingerprint(self.config)
        if manifest_path.exists() and destination.exists() and self.config.resume:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    "Sample output contains a different sampling fingerprint: "
                    f"{existing.get('fingerprint')}"
                )
            summary = {**existing["summary"], "reused": True}
            logger.info(
                "Stage 1/3 Sample completed: reused %d samples",
                summary["samples"],
            )
            return summary

        trajectories = list(
            self._progress(
                create_dataset(self.config.sampling).load(),
                description="Stage 1/3 Load trajectories",
                unit="trajectory",
            )
        )
        samples = list(
            self._progress(
                self._samples(trajectories),
                description="Stage 1/3 Generate samples",
                unit="sample",
            )
        )
        if not samples:
            raise ValueError(
                f"Sampling produced no samples for eval types: {', '.join(self.config.sampling.eval_types)}"
            )
        summary = write_sample_artifacts(
            self._progress(
                samples,
                description="Stage 1/3 Write samples",
                unit="sample",
                total=len(samples),
            ),
            destination,
            self.config.sampling.dataset_name,
        )
        summary.update({"trajectories": len(trajectories), "fingerprint": fingerprint, "reused": False})
        manifest = {
            "schema_version": "prmeval.sample-manifest.v2",
            "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sampling": self.config.sampling.model_dump(),
            "summary": summary,
        }
        manifest_path.write_text(
            json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Stage 1/3 Sample completed: %d trajectories, %d samples",
            len(trajectories),
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
            self.manifest_path = destination.with_name(f"{destination.stem}.manifest.json")
            self.inference_summary_path = destination.with_name(f"{destination.stem}.summary.json")
            self.output_dir = destination.parent
        all_records = load_sample_artifacts(source)
        infer = create_infer(self.config.infer)
        eval_types = {record.evaluation.type for record in all_records}
        required = {
            "preference" if eval_type == "quality_preference" else "progress"
            for eval_type in eval_types
        }
        unsupported = required - infer.capabilities
        if unsupported:
            raise ValueError(
                f"Infer '{self.config.infer.name}' does not support: {', '.join(sorted(unsupported))}"
            )
        model_info = {
            **infer.model_info(),
            "sample_artifact": str(source),
            "sample_artifact_sha256": _file_sha256(source),
        }
        completed = self._prepare(model_info)
        records_to_run = [record for record in all_records if record.sample_id not in completed]
        logger.info(
            "Stage 2/3 Infer workload: %d pending, %d skipped",
            len(records_to_run),
            len(completed),
        )
        new_records: list[EvaluationRecord] = []
        with ThreadPoolExecutor(max_workers=self.config.infer.max_concurrency) as executor:
            future_map = {
                executor.submit(self._predict, infer, record, source.parent): record
                for record in records_to_run
            }
            futures = self._progress(
                as_completed(future_map),
                description=f"Stage 2/3 Infer (skipped={len(completed)})",
                unit="sample",
                total=len(records_to_run),
            )
            for future in futures:
                record = future.result()
                new_records.append(record)
                target = (
                    self.errors_path
                    if record.execution and record.execution.status == "error"
                    else self.predictions_path
                )
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
                    handle.flush()
        records = [EvaluationRecord.model_validate(row) for row in _read_jsonl(self.predictions_path)]
        successful_ids = {record.sample_id for record in records}
        unresolved_errors = {
            row["sample_id"]: row for row in _read_jsonl(self.errors_path) if row["sample_id"] not in successful_ids
        }
        summary = {
            "coverage": {
                "successful": len(records),
                "failed": len(unresolved_errors),
                "new": len(new_records),
                "skipped": len(completed),
            },
            "fingerprint": self.fingerprint,
            "samples": str(source),
            "predictions": str(self.predictions_path),
        }
        self.inference_summary_path.write_text(
            json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "Stage 2/3 Infer completed: %d successful, %d failed, %d new, %d skipped",
            summary["coverage"]["successful"],
            summary["coverage"]["failed"],
            summary["coverage"]["new"],
            summary["coverage"]["skipped"],
        )
        return summary

    def evaluate_metrics(self, predictions_path: str | Path | None = None) -> dict:
        """Stage 3: compute metrics from complete post-model EvaluationRecords only."""
        source = Path(predictions_path) if predictions_path else self.predictions_path
        logger.info("Stage 3/3 Metrics started: %s", source)
        records = [EvaluationRecord.model_validate(row) for row in _read_jsonl(source)]
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
            self._progress(
                metric_names,
                description="Stage 3/3 Compute metrics",
                unit="metric",
                total=len(metric_names),
            ),
        )
        if self.inference_summary_path.exists() and source == self.predictions_path:
            inference = json.loads(self.inference_summary_path.read_text(encoding="utf-8"))
            coverage = inference["coverage"]
        else:
            coverage = {"successful": len(records), "failed": 0, "new": 0, "skipped": 0}
        summary = {
            "metrics": metrics,
            "coverage": coverage,
            "fingerprint": self.fingerprint,
            "predictions": str(source),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "all_metrics.json").write_text(
            json.dumps(jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_compat_results(records)
        if self.manifest_path.exists():
            manifest = RunManifest.model_validate_json(self.manifest_path.read_text())
            manifest.status = "completed"
            manifest.completed_at = datetime.now(timezone.utc).isoformat()
            manifest.summary = summary
            self.manifest_path.write_text(
                json.dumps(jsonable(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info(
            "Stage 3/3 Metrics completed: %d metrics from %d predictions",
            len(metrics),
            len(records),
        )
        return summary

    def run(self) -> dict:
        """Convenience orchestration for stage 1 -> stage 2 -> stage 3."""
        self.sample()
        inference = self.infer()
        if inference["coverage"]["successful"] == 0:
            logger.info("Stage 3/3 Metrics skipped: no successful predictions")
            return {"metrics": {}, **inference}
        return self.evaluate_metrics()

    def _write_compat_results(self, records: list[EvaluationRecord]) -> None:
        by_eval: dict[str, list[dict]] = {}
        for record in records:
            prediction = record.prediction
            item = record.input.items[0]
            row = {
                "id": record.source.id or record.sample_id,
                "task": record.input.task,
                "data_source": record.evaluation.dataset.source or record.evaluation.dataset.name,
                "quality_label": record.extensions.get("quality_label"),
                "partial_success": record.extensions.get("partial_success"),
                "metadata": item.data,
                "target_progress": record.target.values if record.target else None,
            }
            if prediction and prediction.kind == "progress":
                row["progress_pred"] = prediction.values
            if prediction and prediction.kind == "preference":
                row["prediction_prob"] = prediction.probability
                row["preference_pred"] = 1.0 if prediction.label == "chosen" else 0.0
            by_eval.setdefault(record.evaluation.type, []).append(row)
        for eval_type, rows in by_eval.items():
            directory = self.output_dir / eval_type
            directory.mkdir(exist_ok=True)
            (directory / f"{self.config.sampling.dataset_name}_results.json").write_text(
                json.dumps(jsonable(rows), indent=2, ensure_ascii=False), encoding="utf-8"
            )
