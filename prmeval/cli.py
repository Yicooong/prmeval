from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .core.config import EvalConfig
from .core.registry import INFERS, METRICS, SAMPLERS
from .core.runner import Evaluator
from .core.schemas import EvaluationRecord, jsonable
from .core.utils import validate_sample_artifacts
from .metrics.builtins import compute_metrics
from .sample import load_hf_trajectory_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prmeval-eval", description="Local and remote robot reward evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run an evaluation")
    run.add_argument("--config", required=True)
    run.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    sample = sub.add_parser("sample", help="Stage 1: sample a dataset into bench.record.v1")
    sample.add_argument("--config", required=True)
    sample.add_argument("--output", help="Optional samples.jsonl destination")
    sample.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    infer = sub.add_parser("infer", help="Stage 2: run a model on sampled data")
    infer.add_argument("--config", required=True)
    infer.add_argument("--samples", help="Optional samples.jsonl source")
    infer.add_argument("--output", help="Optional predictions.jsonl destination")
    infer.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    stage_metrics = sub.add_parser("metrics", help="Stage 3: compute configured metrics")
    stage_metrics.add_argument("--config", required=True)
    stage_metrics.add_argument("--predictions", help="Optional predictions.jsonl source")
    stage_metrics.add_argument("--no-progress", action="store_true", help="Disable terminal progress bars")
    sub.add_parser("list-infers")
    sub.add_parser("list-samplers")
    sub.add_parser("list-metrics")
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--config", required=True)
    validate_samples = sub.add_parser("validate-samples")
    validate_samples.add_argument("--samples", required=True)
    validate_predictions = sub.add_parser("validate-predictions")
    validate_predictions.add_argument("--predictions", required=True)
    metrics = sub.add_parser("compute-metrics", help="Compute metrics from post-model EvaluationRecord JSONL")
    metrics.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    metrics.add_argument("--metrics", nargs="+", help="Metric names; defaults to eval types present in the file")
    metrics.add_argument("--output", help="Optional output JSON path")
    return parser


def _evaluator(args: argparse.Namespace) -> Evaluator:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    return Evaluator(EvalConfig.from_yaml(args.config), show_progress=not args.no_progress)


def _load_records(path: Path) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(EvaluationRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid EvaluationRecord at {path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"No EvaluationRecord rows found in {path}")
    return records


def _metric_summary_for_stdout(summary: dict) -> dict:
    """Return metric aggregates without verbose per-sample or per-task details."""
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return summary
    return {
        **summary,
        "metrics": {
            name: (
                {key: value for key, value in result.items() if key not in {"details", "task_details"}}
                if isinstance(result, dict)
                else result
            )
            for name, result in metrics.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-infers":
        print("\n".join(INFERS.names()))
    elif args.command == "list-samplers":
        print("\n".join(SAMPLERS.names()))
    elif args.command == "list-metrics":
        print("\n".join(METRICS.names()))
    elif args.command == "validate-dataset":
        config = EvalConfig.from_yaml(args.config)
        trajectories = load_hf_trajectory_pool(config.sampling)
        print(json.dumps({"valid": True, "trajectories": len(trajectories)}, indent=2))
    elif args.command == "validate-samples":
        print(json.dumps(validate_sample_artifacts(Path(args.samples)), indent=2, ensure_ascii=False))
    elif args.command == "validate-predictions":
        source = Path(args.predictions)
        records = _load_records(source)
        identities = [
            (record.evaluation.dataset.name, record.infer.name if record.infer else None, record.sample_id)
            for record in records
        ]
        if len(identities) != len(set(identities)):
            raise ValueError(f"Duplicate dataset/infer/sample identity found in {source}")
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": "bench.record.v1",
                    "records": len(records),
                    "path": str(source),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.command == "compute-metrics":
        configured_source = Path(args.predictions)
        source = configured_source.resolve()
        records = _load_records(source)
        legacy_metric_names = {
            "reward_alignment": "progress",
            "synthetic_temporal_robustness": "progress_temporal_variation",
        }
        metric_names = args.metrics or sorted(
            {legacy_metric_names.get(record.evaluation.type, record.evaluation.type) for record in records}
        )
        payload = {
            "source": str(configured_source),
            "num_records": len(records),
            "metrics": compute_metrics(records, metric_names),
        }
        rendered = json.dumps(jsonable(payload), indent=2, ensure_ascii=False)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "run":
        summary = _evaluator(args).run()
        print(json.dumps(_metric_summary_for_stdout(summary), indent=2))
    elif args.command == "sample":
        summary = _evaluator(args).sample(args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "infer":
        summary = _evaluator(args).infer(args.samples, args.output)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    elif args.command == "metrics":
        summary = _evaluator(args).evaluate_metrics(args.predictions)
        print(json.dumps(_metric_summary_for_stdout(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
