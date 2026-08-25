from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
from pydantic import ValidationError

from prmeval.core.config import EvalConfig, InferConfig, SamplingConfig
from prmeval.core.registry import INFERS, register_infer
from prmeval.core.runner import Evaluator
from prmeval.core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from prmeval.infer.base import Infer


@register_infer("unit_progress")
class UnitProgress(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}
    loads = 0
    calls: ClassVar[list[str]] = []

    def __init__(self, config: InferConfig):
        super().__init__(config)
        type(self).loads += 1

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        del reference_video_path
        type(self).calls.append(task_description)
        if task_description == self.config.options.get("fail_task"):
            raise RuntimeError("intentional sample failure")
        return np.linspace(0.0, 1.0, len(frames_array))

    def predict(self, sample: EvaluationSample) -> Prediction:
        if not isinstance(sample, ProgressSample):
            raise TypeError(f"{self.config.name} only supports progress samples")
        reference_path = sample.trajectory.metadata.get("reference_video_path")
        values = np.asarray(
            self.compute_progress(
                np.asarray(sample.trajectory.frames),
                sample.trajectory.task,
                str(reference_path) if reference_path else None,
            ),
            dtype=float,
        ).reshape(-1)
        expected = len(sample.trajectory.frames)
        if len(values) != expected:
            raise ValueError(f"Progress length mismatch: expected {expected}, got {len(values)}")
        if not np.isfinite(values).all():
            raise ValueError("Progress values must be finite")
        if ((values < 0) | (values > 1)).any():
            raise ValueError("Progress values must be in [0, 1]")
        return ProgressPrediction(
            sample_id=sample.sample_id,
            progress=values.tolist(),
            model=self.config.model_id or self.config.model_path or self.config.name,
            model_version=self.config.model_version,
        )


class InferEntryPointTest(unittest.TestCase):
    def setUp(self):
        UnitProgress.loads = 0
        UnitProgress.calls = []

    def test_config_rejects_removed_execution_fields(self):
        for field, value in (
            ("mode", "local"),
            ("transport", "openai_chat"),
            ("batch_size", 2),
            ("max_concurrency", 2),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                InferConfig.model_validate({"name": "unit_progress", field: value})

    def test_registry_directly_constructs_concrete_infer(self):
        config = InferConfig(name="unit_progress", model_path="checkpoint")
        infer = INFERS.get(config.name)(config)
        self.assertIsInstance(infer, UnitProgress)
        self.assertEqual(infer.config, config)

    def test_runner_is_sequential_initializes_once_and_isolates_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "trajectories.jsonl"
            pixel = [[0, 0, 0], [0, 0, 0]]
            frames = [[pixel, pixel], [pixel, pixel], [pixel, pixel]]
            tasks = ["first", "fail", "third"]
            dataset_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": f"trajectory-{index}",
                            "task": task,
                            "frames": frames,
                            "quality_label": "successful",
                            "data_source": "fixture",
                        }
                    )
                    + "\n"
                    for index, task in enumerate(tasks)
                ),
                encoding="utf-8",
            )
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="fixture",
                    adapter="jsonl",
                    paths=[str(dataset_path)],
                    eval_types=["reward_alignment"],
                    base_frames=3,
                ),
                infer=InferConfig(
                    name="unit_progress",
                    model_path="checkpoint",
                    options={"fail_task": "fail"},
                ),
                output_dir=str(root / "output"),
                run_name="sequential",
                resume=True,
            )
            evaluator = Evaluator(config)
            evaluator.sample()
            first = evaluator.infer()

            self.assertEqual(UnitProgress.loads, 1)
            self.assertEqual(UnitProgress.calls, tasks)
            self.assertEqual(first["coverage"]["successful"], 2)
            self.assertEqual(first["coverage"]["failed"], 1)
            prediction_rows = [json.loads(line) for line in evaluator.predictions_path.read_text().splitlines()]
            self.assertEqual([row["input"]["task"] for row in prediction_rows], ["first", "third"])
            error_rows = [json.loads(line) for line in evaluator.errors_path.read_text().splitlines()]
            self.assertEqual([row["input"]["task"] for row in error_rows], ["fail"])

            UnitProgress.calls = []
            resumed = Evaluator(config).infer(samples_path=evaluator.samples_path)
            self.assertEqual(resumed["coverage"]["skipped"], 2)
            self.assertEqual(UnitProgress.calls, ["fail"])


if __name__ == "__main__":
    unittest.main()
