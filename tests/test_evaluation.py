from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np

from prmeval.cli import build_parser
from prmeval.cli import main as cli_main
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
from prmeval.infer.base import RemoteError, parse_json_content
from prmeval.infer.openai import OpenAIChatClient, progress_schema
from prmeval.metrics.builtins import compute_metrics
from prmeval.sample.prepare import _uniform_indices, _video_frames
from prmeval.sample.progress import compute_progress
from prmeval.sample.samplers import ConfusionMatrixSampler, QualityPreferenceSampler, RewardAlignmentSampler

PIXEL = "data:image/jpeg;base64,/9j/2Q=="
GOLDEN_FIXTURE = Path(__file__).parent / "fixtures" / "rbm_1m_ood_micro.jsonl"
METRICS_SMOKE_FIXTURE = Path(__file__).parents[1] / "examples" / "stage_3_smoke" / "predictions.jsonl"


def _make_progress_sample(num_frames: int = 3, sample_id: str = "progress-sample") -> ProgressSample:
    return ProgressSample(
        sample_id=sample_id,
        eval_type="reward_alignment",
        trajectory=Trajectory(
            id="trajectory",
            task="put the cup on the plate",
            frames=[PIXEL] * num_frames,
            data_source="fixture",
        ),
    )


class _MockResponse:
    calls = 0

    def __init__(self, payload):
        type(self).calls += 1
        schema = payload["response_format"]["json_schema"]
        if schema["name"] == "gvl_progress_prediction":
            image_count = schema["schema"]["properties"]["frames"]["minItems"]
            result = {
                "frames": [
                    {
                        "frame_number": i + 1,
                        "frame_description": f"frame {i + 1}",
                        "task_completion_percentage": 100 * i / max(1, image_count - 1),
                    }
                    for i in range(image_count)
                ]
            }
        else:
            image_count = schema["schema"]["properties"]["progress"].get(
                "minItems",
                sum(
                    item.get("type") == "image_url"
                    for message in payload["messages"]
                    for item in message["content"]
                    if isinstance(message["content"], list)
                ),
            )
            result = {"progress": [i / max(1, image_count - 1) for i in range(image_count)]}
        self._body = {
            "id": "completion-1",
            "choices": [{"message": {"content": json.dumps(result)}}],
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
    def test_remote_json_parser_recovers_known_compatible_server_wrappers(self):
        expected = {"progress": [0.0, 0.5, 1.0]}
        variants = [
            '```json\n{"progress": [0.0, 0.5, 1.0]}\n```',
            '{{"progress": [0.0, 0.5, 1.0]}\n}',
            '{"{"progress": [0.0, 0.5, 1.0]}',
        ]
        for content in variants:
            with self.subTest(content=content):
                self.assertEqual(parse_json_content(content), expected)

    def test_remote_json_parser_does_not_accept_incomplete_json(self):
        with self.assertRaisesRegex(RemoteError, "valid JSON"):
            parse_json_content('{"progress": [0.0, 0.5')

    @patch("prmeval.infer.openai.httpx.post")
    def test_openai_chat_falls_back_from_null_parsed_to_content(self, post):
        post.return_value = _BodyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "parsed": None,
                            "content": '{"progress": [0.0, 0.5, 1.0]}',
                        }
                    }
                ]
            }
        )
        infer = OpenAIChatClient(
            InferConfig(
                name="remote",
                base_url="https://example.com/v1",
                model_id="model",
                max_retries=0,
            )
        )
        parsed, _ = infer.chat([{"role": "user", "content": "prompt"}], progress_schema(3))
        self.assertEqual(parsed["progress"], [0.0, 0.5, 1.0])

    @patch("prmeval.infer.openai.httpx.post")
    def test_schema_parse_error_retains_backend_response(self, post):
        response = {
            "id": "failed-completion",
            "choices": [{"message": {"content": '{"progress": [0.0'}}],
        }
        post.return_value = _BodyResponse(response)
        infer = OpenAIChatClient(
            InferConfig(
                name="remote",
                base_url="https://example.com/v1",
                model_id="model",
                max_retries=0,
            )
        )
        with self.assertRaises(RemoteError) as raised:
            infer.chat([{"role": "user", "content": "prompt"}], progress_schema(3))
        self.assertEqual(raised.exception.raw_response, response)

    def test_stage_commands_accept_no_progress(self):
        parser = build_parser()
        for command in ("run", "sample", "infer", "metrics"):
            enabled = parser.parse_args([command, "--config", "config.yaml"])
            disabled = parser.parse_args([command, "--config", "config.yaml", "--no-progress"])
            self.assertFalse(enabled.no_progress)
            self.assertTrue(disabled.no_progress)

    def test_cli_enables_progress_and_keeps_summary_on_stdout(self):
        config = MagicMock()
        evaluator = MagicMock()
        evaluator.run.return_value = {"metrics": {}, "coverage": {"successful": 0}}
        stdout = io.StringIO()
        with (
            patch("prmeval.cli.EvalConfig.from_yaml", return_value=config),
            patch("prmeval.cli.Evaluator", return_value=evaluator) as evaluator_class,
            patch("prmeval.cli.logging.basicConfig"),
            redirect_stdout(stdout),
        ):
            self.assertEqual(cli_main(["run", "--config", "config.yaml"]), 0)
        evaluator_class.assert_called_once_with(config, show_progress=True)
        self.assertEqual(json.loads(stdout.getvalue()), evaluator.run.return_value)

    def test_cli_metric_summary_omits_verbose_details(self):
        config = MagicMock()
        evaluator = MagicMock()
        evaluator.evaluate_metrics.return_value = {
            "metrics": {
                "reward_alignment": {
                    "loss": 0.25,
                    "num_samples": 2,
                    "slices": {"dataset:model": {"mse": 0.25, "num_samples": 2}},
                    "details": {"sample-1": {"mse": 0.5}},
                },
                "policy_ranking": {
                    "kendall": 1.0,
                    "num_tasks": 1,
                    "task_details": {"task-1": {"last": 1.0}},
                },
            },
            "coverage": {"successful": 2},
        }
        stdout = io.StringIO()
        with (
            patch("prmeval.cli.EvalConfig.from_yaml", return_value=config),
            patch("prmeval.cli.Evaluator", return_value=evaluator),
            patch("prmeval.cli.logging.basicConfig"),
            redirect_stdout(stdout),
        ):
            self.assertEqual(cli_main(["metrics", "--config", "config.yaml"]), 0)

        rendered = json.loads(stdout.getvalue())
        self.assertNotIn("details", rendered["metrics"]["reward_alignment"])
        self.assertNotIn("task_details", rendered["metrics"]["policy_ranking"])
        self.assertEqual(rendered["metrics"]["reward_alignment"]["loss"], 0.25)
        self.assertEqual(rendered["metrics"]["reward_alignment"]["slices"]["dataset:model"]["num_samples"], 2)

    def test_cli_no_progress_is_forwarded_to_evaluator(self):
        config = MagicMock()
        evaluator = MagicMock()
        evaluator.sample.return_value = {"samples": 1}
        with (
            patch("prmeval.cli.EvalConfig.from_yaml", return_value=config),
            patch("prmeval.cli.Evaluator", return_value=evaluator) as evaluator_class,
            patch("prmeval.cli.logging.basicConfig"),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli_main(["sample", "--config", "config.yaml", "--no-progress"]), 0)
        evaluator_class.assert_called_once_with(config, show_progress=False)

    def test_dataset_loading_fields_belong_to_sampling_config(self):
        config = EvalConfig(
            sampling=SamplingConfig(
                dataset_name="fixture",
                adapter="jsonl",
                paths=["trajectories.jsonl"],
                max_trajectories=2,
            ),
            infer=InferConfig(name="progress_test", base_url="https://example.com", model_id="model"),
        )

        self.assertFalse(hasattr(config, "dataset"))
        self.assertEqual(config.sampling.dataset_name, "fixture")
        self.assertEqual(config.sampling.paths, ["trajectories.jsonl"])
        self.assertEqual(config.sampling.max_trajectories, 2)
        with self.assertRaises(ValueError):
            EvalConfig.model_validate(
                {
                    "dataset": {"name": "legacy", "adapter": "jsonl"},
                    "infer": {"name": "gvl", "base_url": "https://example.com", "model_id": "model"},
                }
            )

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
            name="remote",
            base_url="https://explicit.example/v1",
            api_key="explicit-key",
            model_id="explicit-model",
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
            id="t",
            task="task",
            frames=[[[[0, 0, 0]]]] * 10,
            data_source="fixture",
            quality_label="successful",
        )
        config = SamplingConfig(max_frames=3, progress_type="absolute_first_frame")
        samples = list(RewardAlignmentSampler(config, "fixture").sample([trajectory]))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].trajectory.frame_indices, [0, 4, 9])
        self.assertEqual(samples[0].trajectory.target_progress, [0.0, 4 / 9, 1.0])

    def test_pair_and_confusion_sampler_cardinality(self):
        trajectories = [
            Trajectory(
                id=f"t-{quality}",
                task="task-a",
                frames=[[[[0, 0, 0]]]],
                data_source="source",
                quality_label=quality,
            )
            for quality in ("failure", "suboptimal", "successful")
        ]
        pairs = list(QualityPreferenceSampler(SamplingConfig(max_frames=1), "fixture").sample(trajectories))
        self.assertEqual(len(pairs), 3)
        trajectories.append(
            Trajectory(
                id="task-b",
                task="task-b",
                frames=[[[[0, 0, 0]]]],
                data_source="source",
                quality_label="successful",
            )
        )
        confusion = list(
            ConfusionMatrixSampler(SamplingConfig(max_frames=1, trajectories_per_source=2), "fixture").sample(
                trajectories
            )
        )
        self.assertEqual(len(confusion), 4)
        self.assertEqual({s.trajectory.metadata["lang_task"] for s in confusion}, {"task-a", "task-b"})

    def test_metric_goldens(self):
        progress_records = [
            EvaluationRecord(
                sample_id=f"p{i}",
                eval_type="policy_ranking",
                dataset="fixture",
                infer="mock",
                status="success",
                task="same task",
                trajectory_id=f"t{i}",
                quality_label=quality,
                prediction=ProgressPrediction(sample_id=f"p{i}", progress=[score], model="mock"),
            )
            for i, (quality, score) in enumerate([("failure", 0.1), ("suboptimal", 0.5), ("successful", 0.9)])
        ]
        ranking = compute_metrics(progress_records, ["policy_ranking"])["policy_ranking"]
        self.assertEqual(ranking["kendall"], 1.0)
        confusion_records = []
        for index, (language, video, score) in enumerate(
            [
                ("task-a", "task-a", 1.0),
                ("task-a", "task-b", 0.0),
                ("task-b", "task-a", 0.0),
                ("task-b", "task-b", 1.0),
            ]
        ):
            confusion_records.append(
                EvaluationRecord(
                    sample_id=f"c{index}",
                    eval_type="confusion_matrix",
                    dataset="fixture",
                    infer="mock",
                    status="success",
                    task=language,
                    trajectory_id=f"v-{video}",
                    metadata={"lang_task": language, "video_task": video},
                    prediction=ProgressPrediction(sample_id=f"c{index}", progress=[score], model="mock"),
                )
            )
        confusion = compute_metrics(confusion_records, ["confusion_matrix"])["confusion_matrix"]
        self.assertEqual(confusion["confusion_matrix"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(confusion["normalized_trace_minus_offdiag"], 1.0)
        preference_record = EvaluationRecord(
            sample_id="q",
            eval_type="quality_preference",
            dataset="fixture",
            infer="mock",
            status="success",
            task="task",
            prediction=PreferencePrediction(sample_id="q", chosen_probability=0.8, preference="chosen", model="mock"),
        )
        preference = compute_metrics([preference_record], ["quality_preference"])["quality_preference"]
        self.assertEqual(preference["accuracy"], 1.0)

    def test_mixed_post_model_records_compute_independent_metrics(self):
        records = [
            EvaluationRecord.model_validate_json(line)
            for line in METRICS_SMOKE_FIXTURE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics = compute_metrics(
            records, ["reward_alignment", "policy_ranking", "quality_preference", "confusion_matrix"]
        )
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
            records.append(
                EvaluationRecord(
                    stage="inferred",
                    sample_id="shared-sample-id",
                    evaluation={"type": "reward_alignment", "dataset": {"name": dataset}},
                    input={"task": "task", "items": [{"frames": [], "frame_indices": [0, 1, 2]}]},
                    target={"kind": "progress", "values": [0.0, 0.5, 1.0]},
                    infer={"name": infer, "model": infer},
                    prediction={"kind": "progress", "values": prediction},
                    execution={"status": "success"},
                )
            )
        reward = compute_metrics(records, ["reward_alignment"])["reward_alignment"]
        self.assertEqual(
            set(reward["slices"]),
            {
                "rbm-1m-ood:rbm",
                "rbm-1m-ood:sole-r1",
                "rbm-1m-id:rbm",
                "rbm-1m-id:sole-r1",
            },
        )
        self.assertEqual(reward["slices"]["rbm-1m-ood:rbm"]["mse"], 0.0)

    def test_openai_parse_retry_and_v1_url(self):
        infer = OpenAIChatClient(
            InferConfig(name="test", base_url="http://service/v1", model_id="model", max_retries=1)
        )
        responses = [
            _BodyResponse({"choices": [{"message": {"content": "not json"}}]}),
            _BodyResponse({"choices": [{"message": {"content": '{"progress":[0.5]}'}}]}),
        ]
        urls = []

        def mock_post(url, **_kwargs):
            urls.append(url)
            return responses.pop(0)

        with patch("httpx.post", side_effect=mock_post):
            parsed, _ = infer.chat([{"role": "user", "content": "prompt"}], progress_schema(1))
        self.assertEqual(parsed, {"progress": [0.5]})
        self.assertEqual(urls, ["http://service/v1/chat/completions"] * 2)

    def test_runner_resume_and_metrics(self):
        _MockResponse.calls = 0

        def mock_post(_url, **kwargs):
            return _MockResponse(kwargs["json"])

        with patch("httpx.post", side_effect=mock_post):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = EvalConfig(
                    sampling=SamplingConfig(
                        dataset_name="rbm-1m-ood-micro",
                        adapter="jsonl",
                        paths=[str(GOLDEN_FIXTURE)],
                        eval_types=["reward_alignment"],
                        max_frames=3,
                    ),
                    infer=InferConfig(
                        name="progress_test",
                        base_url="http://mock-service",
                        model_id="mock-vlm",
                        max_retries=0,
                    ),
                    output_dir=str(root / "output"),
                    run_name="run",
                )
                first = Evaluator(config).run()
                self.assertEqual(first["coverage"]["successful"], 1)
                self.assertAlmostEqual(first["metrics"]["reward_alignment"]["loss"], 0.0)
                self.assertEqual(_MockResponse.calls, 1)
                second = Evaluator(config).run()
                self.assertEqual(second["coverage"]["skipped"], 1)
                self.assertEqual(_MockResponse.calls, 1)

                changed = config.model_copy(deep=True)
                changed.infer.model_id = "different-model"
                with self.assertRaises(RuntimeError):
                    Evaluator(changed).run()

    def test_run_tracks_all_three_stages_in_interactive_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="fixture",
                    adapter="jsonl",
                    paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(
                    name="progress_test",
                    base_url="http://mock-service",
                    model_id="mock-vlm",
                    max_retries=0,
                ),
                output_dir=tmp,
                run_name="progress",
            )

            with (
                patch("prmeval.core.runner.sys.stderr") as stderr,
                patch("prmeval.core.runner.tqdm", side_effect=lambda iterable, **_kwargs: iterable) as progress,
                patch("httpx.post", side_effect=lambda _url, **kwargs: _MockResponse(kwargs["json"])),
            ):
                stderr.isatty.return_value = True
                summary = Evaluator(config, show_progress=True).run()

            self.assertEqual(summary["coverage"]["successful"], 1)
            calls = {call.kwargs["desc"]: call.kwargs for call in progress.call_args_list}
            self.assertEqual(calls["Stage 1/3 Write samples"]["total"], 1)
            self.assertEqual(calls["Stage 2/3 Infer (skipped=0)"]["total"], 1)
            self.assertEqual(calls["Stage 3/3 Compute metrics"]["total"], 1)
            self.assertIn("Stage 1/3 Load trajectories", calls)
            self.assertIn("Stage 1/3 Generate samples", calls)

            with (
                patch("prmeval.core.runner.sys.stderr") as stderr,
                patch("prmeval.core.runner.tqdm", side_effect=lambda iterable, **_kwargs: iterable) as progress,
                patch("httpx.post") as post,
            ):
                stderr.isatty.return_value = True
                resumed = Evaluator(config, show_progress=True).run()
            post.assert_not_called()
            self.assertEqual(resumed["coverage"]["skipped"], 1)
            infer_call = next(
                call for call in progress.call_args_list if call.kwargs["desc"] == "Stage 2/3 Infer (skipped=1)"
            )
            self.assertEqual(infer_call.kwargs["total"], 0)

    def test_python_api_is_quiet_by_default_and_non_tty_disables_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="fixture",
                    adapter="jsonl",
                    paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(name="progress_test", base_url="http://unused", model_id="unused"),
                output_dir=tmp,
                run_name="quiet",
            )
            with patch("prmeval.core.runner.tqdm") as progress:
                Evaluator(config).sample()
            progress.assert_not_called()

            config.run_name = "non-tty"
            with (
                patch("prmeval.core.runner.sys.stderr") as stderr,
                patch("prmeval.core.runner.tqdm") as progress,
                self.assertLogs("prmeval.core.runner", level="INFO") as logs,
            ):
                stderr.isatty.return_value = False
                Evaluator(config, show_progress=True).sample()
            progress.assert_not_called()
            self.assertTrue(any("Stage 1/3 Sample completed" in message for message in logs.output))

    def test_three_stages_are_independently_runnable(self):
        _MockResponse.calls = 0
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="rbm-1m-ood-micro",
                    adapter="jsonl",
                    paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(
                    name="progress_test",
                    base_url="http://mock-service",
                    model_id="mock-vlm",
                    max_retries=0,
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
            self.assertEqual(record.input.items[0].frames["num_frames"], len(record.target.values))

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
                    dataset_name="rbm-1m-ood-micro",
                    adapter="jsonl",
                    paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(name="progress_test", base_url="http://unused", model_id="unused"),
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
                    dataset_name="rbm-1m-ood-micro",
                    adapter="jsonl",
                    paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(name="progress_test", base_url="http://service", model_id="mock", max_retries=0),
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

    def test_failed_schema_response_is_written_to_errors_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = EvalConfig(
                sampling=SamplingConfig(
                    dataset_name="rbm-1m-ood-micro",
                    adapter="jsonl",
                    paths=[str(GOLDEN_FIXTURE)],
                    eval_types=["reward_alignment"],
                    max_frames=3,
                ),
                infer=InferConfig(
                    name="progress_test",
                    base_url="http://service",
                    model_id="mock",
                    max_retries=0,
                ),
                output_dir=tmp,
                run_name="raw-error-response",
            )
            evaluator = Evaluator(config)
            evaluator.sample()
            response = {
                "id": "failed-completion",
                "choices": [{"message": {"content": '{"progress": [0.0'}}],
            }
            with patch("httpx.post", return_value=_BodyResponse(response)):
                summary = evaluator.infer()

            self.assertEqual(summary["coverage"]["failed"], 1)
            error_record = EvaluationRecord.model_validate_json(
                evaluator.errors_path.read_text(encoding="utf-8").strip()
            )
            self.assertEqual(error_record.execution.raw_response, response)
            self.assertIn("Could not parse", error_record.execution.error)


if __name__ == "__main__":
    unittest.main()
