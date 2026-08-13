from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np

from prmeval.core.artifacts import load_sample_artifacts, validate_sample_artifacts
from prmeval.core.config import EvalConfig, InferConfig, SamplingConfig
from prmeval.core.registry import Registry
from prmeval.core.runner import Evaluator
from prmeval.core.schemas import (
    EvaluationRecord,
    PreferencePrediction,
    ProgressPrediction,
    ProgressSample,
    Trajectory,
)
from prmeval.infer.adapters import create_infer
from prmeval.infer.openai import OpenAIChatInfer, PROGRESS_SCHEMA
from prmeval.infer.specialized import SpecializedInfer, SpecializedRequest, SpecializedResponse
from prmeval.metrics.builtins import compute_metrics
from prmeval.sample.prepare import _uniform_indices, _video_frames
from prmeval.sample.progress import compute_progress
from prmeval.sample.samplers import ConfusionMatrixSampler, QualityPreferenceSampler, RewardAlignmentSampler


PIXEL = "data:image/jpeg;base64,/9j/2Q=="
GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "rbm_1m_ood_micro.jsonl"
METRICS_SMOKE_FIXTURE = Path(__file__).parents[1] / "examples" / "stage_3_smoke" / "predictions.jsonl"


class _MockResponse:
    calls = 0

    def __init__(self, payload):
        type(self).calls += 1
        schema = payload["response_format"]["json_schema"]
        image_count = schema["schema"]["properties"]["progress"].get("minItems", sum(
            item.get("type") == "image_url" for message in payload["messages"]
            for item in message["content"] if isinstance(message["content"], list)
        ))
        self._body = {
            "id": "completion-1",
            "choices": [{"message": {"content": json.dumps({
                "progress": [i / max(1, image_count - 1) for i in range(image_count)]
            })}}],
        }
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _BodyResponse:
    status_code = 200
    text = ""

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class FrameworkTest(unittest.TestCase):
    def test_dataset_loading_fields_belong_to_sampling_config(self):
        config = EvalConfig(
            sampling=SamplingConfig(
                dataset_name="fixture",
                adapter="jsonl",
                paths=["trajectories.jsonl"],
                max_trajectories=2,
            ),
            infer=InferConfig(name="gvl", base_url="https://example.com", model_id="model"),
        )

        self.assertFalse(hasattr(config, "dataset"))
        self.assertEqual(config.sampling.dataset_name, "fixture")
        self.assertEqual(config.sampling.paths, ["trajectories.jsonl"])
        self.assertEqual(config.sampling.max_trajectories, 2)
        with self.assertRaises(ValueError):
            EvalConfig.model_validate({
                "dataset": {"name": "legacy", "adapter": "jsonl"},
                "infer": {"name": "gvl", "base_url": "https://example.com", "model_id": "model"},
            })

    def test_infer_config_resolves_environment_variables(self):
        config_text = """
sampling: {dataset_name: fixture, adapter: jsonl}
infer:
  name: remote
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(config_text, encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"BASE_URL": "https://example.com/v1", "OPENAI_API_KEY": "secret", "MODEL_ID": "model"},
            ):
                config = EvalConfig.from_yaml(path)
                self.assertEqual(config.infer.base_url, "https://example.com/v1")
                self.assertEqual(config.infer.api_key, "secret")
                self.assertEqual(config.infer.model_id, "model")
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(ValueError, "BASE_URL"):
                    EvalConfig.from_yaml(path)
                with self.assertRaisesRegex(ValueError, "MODEL_ID"):
                    InferConfig(name="remote", base_url="https://example.com/v1", model_id="MODEL_ID")

        explicit = InferConfig(
            name="remote", base_url="https://explicit.example/v1",
            api_key="explicit-key", model_id="explicit-model",
        )
        with patch.dict(
            "os.environ",
            {"BASE_URL": "https://env.example/v1", "OPENAI_API_KEY": "env-key", "MODEL_ID": "env-model"},
        ):
            self.assertEqual(explicit.base_url, "https://explicit.example/v1")
            self.assertEqual(explicit.api_key, "explicit-key")
            self.assertEqual(explicit.model_id, "explicit-model")

    def test_local_dataset_preparation_helpers(self):
        frames = np.arange(10 * 2 * 2 * 3, dtype=np.uint8).reshape(10, 2, 2, 3)
        self.assertEqual(_uniform_indices(10, 4).tolist(), [0, 3, 6, 9])
        np.testing.assert_array_equal(_video_frames(frames), frames)

    def test_registry_rejects_duplicates(self):
        registry = Registry("thing")
        registry.register("x")(object)
        with self.assertRaises(ValueError):
            registry.register("x")(object)

    def test_progress_modes(self):
        self.assertEqual(compute_progress(5, [0, 1, 2, 3, 4]), [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(compute_progress(5, [0, 4], "absolute_wrt_total_frames"), [0.2, 1.0])
        self.assertEqual(compute_progress(5, [0, 2, 4], "relative_first_frame"), [0.0, 0.5, 0.5])

    def test_full_sample_frame_targets_stay_aligned(self):
        trajectory = Trajectory(
            id="t", task="task", frames=[[[[0, 0, 0]]]] * 10,
            data_source="fixture", quality_label="successful",
        )
        config = SamplingConfig(max_frames=3, progress_type="absolute_first_frame")
        samples = list(RewardAlignmentSampler(config, "fixture").sample([trajectory]))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].trajectory.frame_indices, [0, 4, 9])
        self.assertEqual(samples[0].trajectory.target_progress, [0.0, 4 / 9, 1.0])

    def test_pair_and_confusion_sampler_cardinality(self):
        trajectories = [
            Trajectory(
                id=f"t-{quality}", task="task-a", frames=[[[[0, 0, 0]]]], data_source="source",
                quality_label=quality,
            )
            for quality in ("failure", "suboptimal", "successful")
        ]
        pairs = list(QualityPreferenceSampler(SamplingConfig(max_frames=1), "fixture").sample(trajectories))
        self.assertEqual(len(pairs), 3)
        trajectories.append(Trajectory(
            id="task-b", task="task-b", frames=[[[[0, 0, 0]]]], data_source="source",
            quality_label="successful",
        ))
        confusion = list(ConfusionMatrixSampler(
            SamplingConfig(max_frames=1, trajectories_per_source=2), "fixture"
        ).sample(trajectories))
        self.assertEqual(len(confusion), 4)
        self.assertEqual({s.trajectory.metadata["lang_task"] for s in confusion}, {"task-a", "task-b"})

    def test_specialized_contract(self):
        request = SpecializedRequest(
            model="rbm",
            request_id="sample",
            prediction_type="progress",
            task="task",
            trajectories=[{"id": "t", "frames": []}],
        )
        self.assertEqual(request.prediction_type, "progress")
        response = SpecializedResponse.model_validate({
            "id": "r",
            "model": "rbm",
            "model_version": "v1",
            "predictions": [{"trajectory_id": "t", "progress": [0.0, 1.0]}],
        })
        self.assertEqual(response.predictions[0].progress, [0.0, 1.0])
        with self.assertRaises(ValueError):
            SpecializedResponse.model_validate({
                "id": "r", "model": "rbm", "predictions": [{"progress": [-0.1, 1.0]}]
            })

    def test_specialized_adapter_contract(self):
        infer = SpecializedInfer(InferConfig(
            name="rbm", base_url="http://service/v1", model_id="rbm-model", max_retries=0
        ))
        sample = ProgressSample(
            sample_id="sample-id", eval_type="reward_alignment",
            trajectory=Trajectory(
                id="trajectory", task="task", frames=[PIXEL, PIXEL], data_source="fixture",
                target_progress=[0.0, 1.0],
            ),
        )
        captured = {}

        def mock_post(url, **kwargs):
            captured.update(url=url, payload=kwargs["json"])
            return _BodyResponse({
                "id": "sample-id", "model": "rbm-model", "model_version": "v1",
                "predictions": [{"trajectory_id": "trajectory", "progress": [0.0, 1.0]}],
            })

        with patch("httpx.post", side_effect=mock_post):
            prediction = infer.predict(sample)
        self.assertEqual(captured["url"], "http://service/v1/evaluations")
        self.assertEqual(captured["payload"]["prediction_type"], "progress")
        self.assertEqual(len(captured["payload"]["trajectories"][0]["frames"]), 2)
        self.assertEqual(prediction.progress, [0.0, 1.0])

    def test_metric_goldens(self):
        progress_records = [
            EvaluationRecord(
                sample_id=f"p{i}", eval_type="policy_ranking", dataset="fixture", infer="mock",
                status="success", task="same task", trajectory_id=f"t{i}", quality_label=quality,
                prediction=ProgressPrediction(sample_id=f"p{i}", progress=[score], model="mock"),
            )
            for i, (quality, score) in enumerate([
                ("failure", 0.1), ("suboptimal", 0.5), ("successful", 0.9)
            ])
        ]
        ranking = compute_metrics(progress_records, ["policy_ranking"])["policy_ranking"]
        self.assertEqual(ranking["kendall"], 1.0)
        confusion_records = []
        for index, (language, video, score) in enumerate([
            ("task-a", "task-a", 1.0), ("task-a", "task-b", 0.0),
            ("task-b", "task-a", 0.0), ("task-b", "task-b", 1.0),
        ]):
            confusion_records.append(EvaluationRecord(
                sample_id=f"c{index}", eval_type="confusion_matrix", dataset="fixture", infer="mock",
                status="success", task=language, trajectory_id=f"v-{video}",
                metadata={"lang_task": language, "video_task": video},
                prediction=ProgressPrediction(sample_id=f"c{index}", progress=[score], model="mock"),
            ))
        confusion = compute_metrics(confusion_records, ["confusion_matrix"])["confusion_matrix"]
        self.assertEqual(confusion["confusion_matrix"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(confusion["normalized_trace_minus_offdiag"], 1.0)
        preference_record = EvaluationRecord(
            sample_id="q", eval_type="quality_preference", dataset="fixture", infer="mock",
            status="success", task="task", prediction=PreferencePrediction(
                sample_id="q", chosen_probability=0.8, preference="chosen", model="mock"
            ),
        )
        preference = compute_metrics([preference_record], ["quality_preference"])["quality_preference"]
        self.assertEqual(preference["accuracy"], 1.0)

    def test_mixed_post_model_records_compute_independent_metrics(self):
        records = [
            EvaluationRecord.model_validate_json(line)
            for line in METRICS_SMOKE_FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics = compute_metrics(records, [
            "reward_alignment", "policy_ranking", "quality_preference", "confusion_matrix"
        ])
        self.assertAlmostEqual(metrics["reward_alignment"]["loss"], 1 / 120)
        self.assertEqual(metrics["reward_alignment"]["num_samples"], 2)
        self.assertEqual(metrics["policy_ranking"]["kendall"], 1.0)
        self.assertAlmostEqual(metrics["quality_preference"]["accuracy"], 2 / 3)
        self.assertEqual(metrics["confusion_matrix"]["confusion_matrix"], [[0.9, 0.1], [0.2, 0.8]])

    def test_reward_alignment_slices_dataset_and_infer_with_one_sample_id(self):
        records = []
        for dataset, infer, prediction in [
            ("rbm-1m-ood", "rbm", [0.0, 0.5, 1.0]),
            ("rbm-1m-ood", "sole-r1", [0.0, 0.4, 0.8]),
            ("rbm-1m-id", "rbm", [0.0, 0.45, 0.9]),
            ("rbm-1m-id", "sole-r1", [0.0, 0.3, 0.7]),
        ]:
            records.append(EvaluationRecord(
                stage="inferred",
                sample_id="shared-sample-id",
                evaluation={"type": "reward_alignment", "dataset": {"name": dataset}},
                input={"task": "task", "items": [{"frames": [], "frame_indices": [0, 1, 2]}]},
                target={"kind": "progress", "values": [0.0, 0.5, 1.0]},
                infer={"name": infer, "model": infer},
                prediction={"kind": "progress", "values": prediction},
                execution={"status": "success"},
            ))
        reward = compute_metrics(records, ["reward_alignment"])["reward_alignment"]
        self.assertEqual(set(reward["slices"]), {
            "rbm-1m-ood:rbm", "rbm-1m-ood:sole-r1",
            "rbm-1m-id:rbm", "rbm-1m-id:sole-r1",
        })
        self.assertEqual(reward["slices"]["rbm-1m-ood:rbm"]["mse"], 0.0)

    def test_openai_parse_retry_and_v1_url(self):
        infer = OpenAIChatInfer(InferConfig(
            name="test", base_url="http://service/v1", model_id="model", max_retries=1
        ))
        responses = [
            _BodyResponse({"choices": [{"message": {"content": "not json"}}]}),
            _BodyResponse({"choices": [{"message": {"content": '{"progress":[0.5]}'}}]}),
        ]
        urls = []

        def mock_post(url, **_kwargs):
            urls.append(url)
            return responses.pop(0)

        with patch("httpx.post", side_effect=mock_post):
            parsed, _ = infer._chat([{"role": "user", "content": "prompt"}], PROGRESS_SCHEMA)
        self.assertEqual(parsed, {"progress": [0.5]})
        self.assertEqual(urls, ["http://service/v1/chat/completions"] * 2)

    def test_progress_test_infer_uses_default_prompt_and_exact_schema(self):
        infer = create_infer(InferConfig(
            name="progress_test", transport="openai_chat", base_url="http://service/v1",
            model_id="test-vlm", max_retries=0,
        ))
        sample = ProgressSample(
            sample_id="progress-test-1",
            eval_type="reward_alignment",
            trajectory=Trajectory(
                id="trajectory-1", task="put the cup on the plate",
                frames=[PIXEL, PIXEL, PIXEL], data_source="fixture",
            ),
        )
        captured = {}

        def mock_post(url, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs["json"]
            return _BodyResponse({
                "choices": [{"message": {"content": '{"progress":[0.0,0.5,1.0]}'}}]
            })

        with patch("httpx.post", side_effect=mock_post):
            prediction = infer.predict(sample)

        self.assertEqual(prediction.progress, [0.0, 0.5, 1.0])
        self.assertEqual(captured["url"], "http://service/v1/chat/completions")
        prompt = captured["payload"]["messages"][0]["content"][0]["text"]
        self.assertIn("Return exactly 3 progress values", prompt)
        progress_definition = (
            captured["payload"]["response_format"]["json_schema"]["schema"]["properties"]["progress"]
        )
        self.assertEqual(progress_definition["minItems"], 3)
        self.assertEqual(progress_definition["maxItems"], 3)

    def test_runner_resume_and_metrics(self):
        _MockResponse.calls = 0
        def mock_post(_url, **kwargs):
            return _MockResponse(kwargs["json"])

        with patch("httpx.post", side_effect=mock_post):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = EvalConfig(
                    sampling=SamplingConfig(
                        dataset_name="rbm-1m-ood-micro", adapter="jsonl", paths=[str(GOLDEN_FIXTURE)],
                        eval_types=["reward_alignment"], max_frames=3,
                    ),
                    infer=InferConfig(
                        name="gvl",
                        base_url="http://mock-service",
                        model_id="mock-vlm",
                        max_retries=0,
                    ),
                    output_dir=str(root / "output"),
                    run_name="run",
                )
                first = Evaluator(config).run()
                self.assertEqual(first["coverage"]["successful"], 1)
                self.assertAlmostEqual(first["metrics"]["reward_alignment"]["loss"], 0.5)
                self.assertEqual(_MockResponse.calls, 1)
                second = Evaluator(config).run()
                self.assertEqual(second["coverage"]["skipped"], 1)
                self.assertEqual(_MockResponse.calls, 1)

                changed = config.model_copy(deep=True)
                changed.infer.model_id = "different-model"
                with self.assertRaises(RuntimeError):
                    Evaluator(changed).run()

    def test_three_stages_are_independently_runnable(self):
        _MockResponse.calls = 0
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="rbm-1m-ood-micro", adapter="jsonl", paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"], max_frames=3,
                ),
                infer=InferConfig(
                    name="gvl", base_url="http://mock-service", model_id="mock-vlm", max_retries=0,
                ),
                output_dir=tmp,
                run_name="stages",
            )
            evaluator = Evaluator(config)

            with patch("httpx.post") as post:
                sampled = evaluator.sample()
            post.assert_not_called()
            self.assertEqual(sampled["samples"], 1)
            self.assertEqual(validate_sample_artifacts(evaluator.samples_path)["frames"], 3)
            record = load_sample_artifacts(evaluator.samples_path)[0]
            self.assertEqual(record.stage, "sampled")
            self.assertEqual(record.evaluation.type, "reward_alignment")
            self.assertEqual(
                record.input.items[0].frames["num_frames"], len(record.target.values)
            )

            with patch("httpx.post", side_effect=lambda _url, **kwargs: _MockResponse(kwargs["json"])):
                inferred = evaluator.infer()
            self.assertEqual(inferred["coverage"]["successful"], 1)
            record = EvaluationRecord.model_validate_json(
                evaluator.predictions_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(record.schema_version, "bench.record.v1")
            self.assertEqual(record.stage, "inferred")
            self.assertEqual(record.prediction.kind, "progress")
            self.assertIsInstance(record.input.items[0].frames, dict)
            self.assertEqual(record.input.items[0].frames["type"], "npz")
            self.assertEqual(record.target.values, [0.0, 0.5, 1.0])

            with patch("httpx.post") as post:
                measured = evaluator.evaluate_metrics()
            post.assert_not_called()
            self.assertIn("reward_alignment", measured["metrics"])

    def test_sample_bundle_detects_modified_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="rbm-1m-ood-micro", adapter="jsonl", paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"], max_frames=3,
                ),
                infer=InferConfig(name="gvl", base_url="http://unused", model_id="unused"),
                output_dir=tmp,
                run_name="tamper",
            )
            evaluator = Evaluator(config)
            evaluator.sample()
            row = json.loads(evaluator.samples_path.read_text(encoding="utf-8").splitlines()[0])
            frame_path = evaluator.samples_path.parent / row["input"]["items"][0]["frames"]["path"]
            frame_path.write_bytes(frame_path.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_sample_artifacts(evaluator.samples_path)

    def test_failed_sample_is_retried_and_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="rbm-1m-ood-micro", adapter="jsonl", paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"], max_frames=3,
                ),
                infer=InferConfig(name="gvl", base_url="http://service", model_id="mock", max_retries=0),
                output_dir=tmp,
                run_name="recovery",
            )
            with patch("httpx.post", side_effect=httpx.ConnectError("offline")):
                failed = Evaluator(config).run()
            self.assertEqual(failed["coverage"]["failed"], 1)
            with patch("httpx.post", side_effect=lambda _url, **kwargs: _MockResponse(kwargs["json"])):
                recovered = Evaluator(config).run()
            self.assertEqual(recovered["coverage"]["successful"], 1)
            self.assertEqual(recovered["coverage"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
