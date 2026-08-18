from __future__ import annotations

import unittest
from types import MethodType
from unittest.mock import Mock

import numpy as np

from prmeval.core.config import InferConfig
from prmeval.core.schemas import PreferenceSample, ProgressSample, Trajectory
from prmeval.infer.base import Infer, RemoteError
from prmeval.infer.baselines import (
    GVL,
    RLVLMF,
    VLAC,
    RBMModel,
    RoboDopamine,
    RoboReward,
    SoleR1,
    TopReward,
)
from prmeval.infer.baselines.common import interpolate_prefix_values, trajectory_prefix_lengths
from prmeval.infer.baselines.progress_test import ProgressTestModel

PROGRESS_BASELINES = (
    ("gvl", GVL),
    ("rbm", RBMModel),
    ("robodopamine", RoboDopamine),
    ("roboreward", RoboReward),
    ("sole_r1", SoleR1),
    ("topreward", TopReward),
    ("vlac", VLAC),
    ("progress_test", ProgressTestModel),
)


def progress_sample(num_frames: int = 3) -> ProgressSample:
    return ProgressSample(
        sample_id="sample",
        eval_type="reward_alignment",
        trajectory=Trajectory(
            id="trajectory",
            task="pick up the cube",
            frames=np.zeros((num_frames, 2, 2, 3), dtype=np.uint8),
            metadata={"reference_video_path": "/tmp/reference.mp4"},
        ),
    )


def preference_sample() -> PreferenceSample:
    chosen = Trajectory(
        id="chosen",
        task="pick up the cube",
        frames=np.zeros((2, 2, 2, 3), dtype=np.uint8),
    )
    rejected = chosen.model_copy(update={"id": "rejected"})
    return PreferenceSample(
        sample_id="preference",
        eval_type="quality_preference",
        chosen_trajectory=chosen,
        rejected_trajectory=rejected,
    )


class TrajectoryPrefixesTest(unittest.TestCase):
    def test_samples_cumulative_prefixes_and_interpolates_endpoints(self):
        lengths = trajectory_prefix_lengths(num_frames=4, num_prefix_samples=3)
        self.assertEqual(lengths, [1, 2, 4])
        np.testing.assert_allclose(
            interpolate_prefix_values(4, lengths, [0.0, 0.5, 1.0]),
            [0.0, 0.5, 0.75, 1.0],
        )


class ProgressPredictContractTest(unittest.TestCase):
    def test_each_progress_baseline_predict_calls_its_compute_progress(self):
        sample = progress_sample()
        for name, baseline_cls in PROGRESS_BASELINES:
            with self.subTest(name=name):
                model = baseline_cls.__new__(baseline_cls)
                Infer.__init__(model, InferConfig(name=name, model_id=f"{name}-model"))
                compute = Mock(return_value=np.array([0.0, 0.5, 1.0]))
                model.compute_progress = compute

                prediction = baseline_cls.predict(model, sample)

                args = compute.call_args.args
                np.testing.assert_array_equal(args[0], sample.trajectory.frames)
                self.assertEqual(args[1], "pick up the cube")
                self.assertEqual(args[2], "/tmp/reference.mp4")
                self.assertEqual(prediction.progress, [0.0, 0.5, 1.0])
                self.assertEqual(prediction.sample_id, "sample")
                self.assertEqual(prediction.model, f"{name}-model")

    def test_each_progress_baseline_validates_its_output(self):
        sample = progress_sample()
        invalid = (
            ([0.0, 1.0], "length mismatch"),
            ([0.0, float("nan"), 1.0], "finite"),
            ([0.0, float("inf"), 1.0], "finite"),
            ([-0.1, 0.5, 1.0], r"in \[0, 1\]"),
            ([0.0, 0.5, 1.1], r"in \[0, 1\]"),
        )
        for name, baseline_cls in PROGRESS_BASELINES:
            model = baseline_cls.__new__(baseline_cls)
            Infer.__init__(model, InferConfig(name=name, model_id="model"))
            for values, message in invalid:
                model.compute_progress = Mock(return_value=values)
                with (
                    self.subTest(name=name, values=values),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    baseline_cls.predict(model, sample)

    def test_progress_predict_rejects_preference_sample(self):
        model = GVL.__new__(GVL)
        Infer.__init__(model, InferConfig(name="gvl", model_id="model"))
        model.compute_progress = Mock()
        with self.assertRaisesRegex(TypeError, "progress samples"):
            GVL.predict(model, preference_sample())


class PreferencePredictContractTest(unittest.TestCase):
    def test_rlvlmf_predict_calls_compute_preference(self):
        model = RLVLMF.__new__(RLVLMF)
        Infer.__init__(model, InferConfig(name="rlvlmf", model_id="provider-model"))

        def compute(_self, chosen, rejected, task):
            self.assertEqual(len(chosen), 2)
            self.assertEqual(len(rejected), 2)
            self.assertEqual(task, "pick up the cube")
            return {"vlm_chose_chosen": True, "provider": "mock"}

        model.compute_preference = MethodType(compute, model)
        prediction = model.predict(preference_sample())
        self.assertEqual(prediction.preference, "chosen")
        self.assertEqual(prediction.chosen_probability, 1.0)
        self.assertEqual(prediction.model, "provider-model")
        self.assertEqual(prediction.raw_response["provider"], "mock")


class SoleR1ComputeTest(unittest.TestCase):
    def test_compute_progress_uses_instance_client_and_carries_progress(self):
        frames = np.zeros((3, 2, 2, 3), dtype=np.uint8)
        client = Mock()
        client.completion.side_effect = [
            {"choices": [{"message": {"content": "<answer>12.5%</answer>"}}]},
            {"choices": [{"message": {"content": "<answer>80%</answer>"}}]},
        ]
        model = SoleR1.__new__(SoleR1)
        model.client = client
        values = model.compute_progress(frames, "task")
        np.testing.assert_allclose(values, [0.0, 0.125, 0.8])
        second_prompt = client.completion.call_args_list[1].args[0][1]["content"][-1]["text"]
        self.assertIn("previous timestep is 12.5%", second_prompt)

    def test_malformed_response_retains_raw_response(self):
        response = {"choices": [{"message": {"content": "no numeric answer"}}]}
        model = SoleR1.__new__(SoleR1)
        model.client = Mock()
        model.client.completion.return_value = response
        with self.assertRaises(RemoteError) as raised:
            model.compute_progress(np.zeros((2, 2, 2, 3), dtype=np.uint8), "task")
        self.assertEqual(raised.exception.raw_response, response)


if __name__ == "__main__":
    unittest.main()
