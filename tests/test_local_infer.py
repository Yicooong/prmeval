from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prmeval.core.config import EvalConfig, InferConfig, SamplingConfig
from prmeval.core.runner import Evaluator
from prmeval.core.schemas import PreferenceSample, ProgressSample, Trajectory
from prmeval.infer import (
    PreferenceModel,
    PreferenceResult,
    ProgressModel,
    create_infer,
    register_preference_model,
    register_progress_model,
)


@register_progress_model("unit_local_progress")
class UnitLocalProgress(ProgressModel):
    supports_local_batch = True
    loads = 0
    batch_calls = 0

    def __init__(self, model_path: str, offset: float = 0.0):
        type(self).loads += 1
        self.model_path = model_path
        self.offset = offset

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        return np.clip(np.linspace(0.0, 1.0, len(frames_array)) + self.offset, 0.0, 1.0)

    def compute_progress_batch(
        self,
        frames_list: list[np.ndarray],
        task_descriptions: list[str],
        reference_video_paths: list[str | None] | None = None,
    ) -> list[np.ndarray]:
        type(self).batch_calls += 1
        paths = reference_video_paths or [None] * len(frames_list)
        return [
            self.compute_progress(frames, task, path)
            for frames, task, path in zip(frames_list, task_descriptions, paths, strict=True)
        ]


@register_progress_model("unit_hybrid_progress")
class UnitHybridProgress(ProgressModel):
    supports_remote = True
    loads = 0
    remote_calls = 0

    def __init__(self, model_path: str):
        type(self).loads += 1

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        return np.zeros(len(frames_array))

    @classmethod
    def remote_compute_progress(cls, frames_array, task_description, reference_video_path, remote, options):
        cls.remote_calls += 1
        return np.full(len(frames_array), options.get("value", 0.5))


@register_preference_model("unit_remote_preference")
class UnitRemotePreference(PreferenceModel):
    supports_local = False
    supports_remote = True

    def compute_preference(self, chosen_frames, rejected_frames, task_description=""):
        raise AssertionError("remote mode must not call the local method")

    @classmethod
    def remote_compute_preference(cls, chosen_frames, rejected_frames, task_description, remote, options):
        return PreferenceResult(0.8, "chosen", raw_response={"source": "remote"})


class LocalInferTest(unittest.TestCase):
    def setUp(self):
        UnitLocalProgress.loads = 0
        UnitLocalProgress.batch_calls = 0
        UnitHybridProgress.loads = 0
        UnitHybridProgress.remote_calls = 0

    def test_local_config_defaults_and_validation(self):
        config = InferConfig(name="unit_local_progress", model_path="/models/test")
        self.assertEqual(config.mode, "local")
        self.assertEqual(config.transport, "local_huggingface")
        self.assertEqual(config.model_id, "/models/test")
        self.assertEqual(config.max_concurrency, 1)

        with self.assertRaisesRegex(ValueError, "model_path"):
            InferConfig(name="unit_local_progress", mode="local")
        with self.assertRaisesRegex(ValueError, "max_concurrency=1"):
            InferConfig(
                name="unit_local_progress",
                mode="local",
                model_path="/models/test",
                max_concurrency=2,
            )

    def test_remote_mode_does_not_load_local_model(self):
        config = InferConfig(
            name="unit_hybrid_progress",
            mode="remote",
            base_url="http://service/v1",
            model_id="remote-model",
            options={"value": 0.25},
        )
        infer = create_infer(config)
        sample = ProgressSample(
            sample_id="sample",
            eval_type="reward_alignment",
            trajectory=Trajectory(
                id="trajectory",
                task="task",
                frames=np.zeros((3, 2, 2, 3), dtype=np.uint8),
            ),
        )

        prediction = infer.predict(sample)

        self.assertEqual(UnitHybridProgress.loads, 0)
        self.assertEqual(UnitHybridProgress.remote_calls, 1)
        self.assertEqual(prediction.progress, [0.25, 0.25, 0.25])

    def test_remote_preference_model_adapter(self):
        infer = create_infer(InferConfig(
            name="unit_remote_preference",
            mode="remote",
            base_url="http://service/v1",
            model_id="preference-model",
        ))
        trajectory = Trajectory(
            id="trajectory",
            task="task",
            frames=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        )
        sample = PreferenceSample(
            sample_id="preference-sample",
            eval_type="quality_preference",
            chosen_trajectory=trajectory,
            rejected_trajectory=trajectory.model_copy(update={"id": "rejected"}),
        )
        prediction = infer.predict(sample)
        self.assertEqual(prediction.preference, "chosen")
        self.assertEqual(prediction.chosen_probability, 0.8)
        self.assertEqual(prediction.raw_response, {"source": "remote"})

    def test_runner_groups_local_samples_into_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "trajectories.jsonl"
            pixel = [[0, 0, 0], [0, 0, 0]]
            frames = [[pixel, pixel], [pixel, pixel], [pixel, pixel]]
            rows = [
                {
                    "id": f"trajectory-{index}",
                    "task": "complete the task",
                    "frames": frames,
                    "quality_label": "successful",
                    "data_source": "fixture",
                }
                for index in range(5)
            ]
            dataset_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="local-fixture",
                    adapter="jsonl",
                    paths=[str(dataset_path)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(
                    name="unit_local_progress",
                    model_path="/models/test",
                    batch_size=2,
                ),
                metrics=["reward_alignment"],
                output_dir=str(root / "output"),
                run_name="local-batch",
                resume=False,
            )

            summary = Evaluator(config).run()

            self.assertEqual(summary["coverage"]["successful"], 5)
            self.assertEqual(UnitLocalProgress.loads, 1)
            self.assertEqual(UnitLocalProgress.batch_calls, 3)
            self.assertEqual(summary["execution"], {
                "mode": "local",
                "batch_size": 2,
                "max_concurrency": 1,
            })
            self.assertEqual(summary["metrics"]["reward_alignment"]["loss"], 0.0)

    def test_batch_size_requires_native_batch_support(self):
        config = InferConfig(
            name="unit_hybrid_progress",
            model_path="/models/test",
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "native batching"):
            create_infer(config)
        self.assertEqual(UnitHybridProgress.loads, 0)


if __name__ == "__main__":
    unittest.main()
