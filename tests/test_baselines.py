from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import numpy as np

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


class _RemoteStub:
    def __init__(self, chats=None, completions=None):
        self.chats = list(chats or [])
        self.completions = list(completions or [])
        self.chat_calls = []
        self.completion_calls = []
        self.config = SimpleNamespace(max_retries=0)

    def chat(self, messages, schema, request_options=None, validator=None):
        self.chat_calls.append((messages, schema, request_options))
        parsed = self.chats.pop(0)
        if validator:
            validator(parsed)
        return parsed, {"parsed": parsed}

    def completion(self, messages, request_options=None):
        self.completion_calls.append((messages, request_options))
        return self.completions.pop(0)


class TrajectoryPrefixesTest(unittest.TestCase):
    def test_samples_cumulative_prefixes_and_interpolates_endpoints(self):
        lengths = trajectory_prefix_lengths(num_frames=4, num_prefix_samples=3)

        self.assertEqual(lengths, [1, 2, 4])
        self.assertEqual(trajectory_prefix_lengths(num_frames=2, num_prefix_samples=2), [1, 2])
        self.assertEqual(trajectory_prefix_lengths(num_frames=4, num_prefix_samples=1), [4])
        np.testing.assert_allclose(
            interpolate_prefix_values(4, lengths, [0.0, 0.5, 1.0]),
            [0.0, 0.5, 0.75, 1.0],
        )

    def test_roboreward_local_wrapper_scores_each_prefix(self):
        model = RoboReward.__new__(RoboReward)
        model.num_prefix_samples = 3
        observed_lengths = []

        def score_prefixes(frames_list, _tasks):
            observed_lengths.extend(len(frames) for frames in frames_list)
            endpoint_values = {1: 0.0, 2: 0.5, 3: 0.75, 4: 1.0}
            return [[endpoint_values[len(frames)]] * len(frames) for frames in frames_list]

        model._compute_progress_batch_without_prefixes = score_prefixes
        values = model.compute_progress(np.zeros((4, 2, 2, 3), dtype=np.uint8), "task")

        self.assertEqual(observed_lengths, [1, 2, 4])
        np.testing.assert_allclose(values, [0.0, 0.5, 0.75, 1.0])

        observed_lengths.clear()
        batched = model.compute_progress_batch(
            [np.zeros((4, 2, 2, 3)), np.zeros((3, 2, 2, 3)), np.zeros((0, 2, 2, 3))],
            ["first", "second", "empty"],
        )

        self.assertEqual(observed_lengths, [1, 2, 4, 1, 2, 3])
        np.testing.assert_allclose(batched[0], [0.0, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(batched[1], [0.0, 0.5, 0.75])


        self.assertEqual(batched[2], [])
class BaselineRemoteTest(unittest.TestCase):
    def setUp(self):
        self.frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)

    def test_roboreward_remote_scores_prefixes_and_interpolates(self):
        remote = _RemoteStub(chats=[{"score": 1}, {"score": 3}, {"score": 4}])
        result = RoboReward.remote_compute_progress(
            self.frames,
            "task",
            None,
            remote,
            {"num_prefix_samples": 3},
        )
        np.testing.assert_allclose(result.values, [0.0, 0.5, 0.625, 0.75])
        self.assertEqual([item["prefix_length"] for item in result.raw_response], [1, 2, 4])
        self.assertEqual([item["score"] for item in result.raw_response], [1, 3, 4])
        self.assertEqual(remote.chat_calls[0][1]["name"], "roboreward_score")
        self.assertEqual([len(call[0][0]["content"]) // 2 for call in remote.chat_calls], [1, 2, 4])

    def test_sole_r1_remote_carries_progress_across_stage_one_frames(self):
        frames = np.stack([np.full((2, 2, 3), value, dtype=np.uint8) for value in (0, 64, 128)])
        remote = _RemoteStub(
            completions=[
                {"choices": [{"message": {"content": "<think>closer</think><answer>12.5%</answer>"}}]},
                {"choices": [{"message": {"content": "<think>grasped</think><answer>80%</answer>"}}]},
            ]
        )

        result = SoleR1.remote_compute_progress(frames, "pick up the cube", None, remote, {})

        np.testing.assert_allclose(result.values, [0.0, 0.125, 0.8])
        self.assertEqual(len(remote.completion_calls), 2)
        second_prompt = remote.completion_calls[1][0][1]["content"][-1]["text"]
        self.assertIn("previous timestep is 12.5%", second_prompt)
        first_images = [
            item["image_url"]["url"]
            for item in remote.completion_calls[0][0][1]["content"]
            if item["type"] == "image_url"
        ]
        self.assertEqual(len(first_images), 3)
        self.assertEqual(first_images[0], first_images[1])
        self.assertNotEqual(first_images[1], first_images[2])
        self.assertEqual([item["frame_index"] for item in result.raw_response], [1, 2])

    def test_sole_r1_single_frame_does_not_call_remote(self):
        remote = _RemoteStub()
        result = SoleR1.remote_compute_progress(self.frames[:1], "task", None, remote, {})
        self.assertEqual(result.values.tolist(), [0.0])
        self.assertEqual(remote.completion_calls, [])
        self.assertEqual(result.raw_response, [])

    def test_sole_r1_rejects_empty_or_malformed_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            SoleR1.remote_compute_progress(self.frames[:0], "task", None, _RemoteStub(), {})

        malformed = _RemoteStub(
            completions=[
                {"choices": [{"message": {"content": "progress is probably ten percent"}}]},
            ]
        )
        with self.assertRaisesRegex(Exception, "numeric <answer>"):
            SoleR1.remote_compute_progress(self.frames[:2], "task", None, malformed, {})

    def test_sole_r1_parses_signed_numbers_without_clipping(self):
        self.assertEqual(SoleR1._parse_progress_percent("<answer>-2.5%</answer>"), -2.5)
        remote = _RemoteStub(
            completions=[
                {"choices": [{"message": {"content": "<answer>-2.5%</answer>"}}]},
            ]
        )
        with self.assertRaisesRegex(Exception, "outside PRMEval"):
            SoleR1.remote_compute_progress(self.frames[:2], "task", None, remote, {})

    def test_topreward_remote_prefix_logprobs(self):
        completions = []
        for reward in (-3.0, -2.0, -1.0):
            completions.append(
                {
                    "choices": [
                        {
                            "message": {"content": "False"},
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "False",
                                        "logprob": -0.1,
                                        "top_logprobs": [{"token": " True", "logprob": reward}],
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
        remote = _RemoteStub(completions=completions)
        result = TopReward.remote_compute_progress(
            self.frames,
            "task",
            None,
            remote,
            {"num_prefix_samples": 3},
        )
        np.testing.assert_allclose(result.values, [0.0, 0.5, 0.75, 1.0])
        self.assertEqual(len(remote.completion_calls), 3)
        self.assertTrue(all(call[1]["logprobs"] for call in remote.completion_calls))

    def test_robodopamine_remote_incremental_curve(self):
        remote = _RemoteStub(
            chats=[
                {"relative_change_percent": 50},
                {"relative_change_percent": -50},
            ]
        )
        result = RoboDopamine.remote_compute_progress(
            self.frames,
            "task",
            None,
            remote,
            {"eval_mode": "incremental", "frame_interval": 2},
        )
        self.assertEqual(result.values, [0.0, 0.5, 0.25, 0.25])
        self.assertEqual(len(remote.chat_calls), 2)

    def test_vlac_remote_normalizes_and_pads(self):
        remote = _RemoteStub(chats=[{"progress": [0, 50]}])
        result = VLAC.remote_compute_progress(self.frames, "task", None, remote, {})
        self.assertEqual(result.values, [0.0, 0.5, 0.5, 0.5])

    def test_rbm_remote_requires_exact_progress_curve(self):
        remote = _RemoteStub(chats=[{"progress": [0.0, 0.25, 0.5, 1.0]}])
        result = RBMModel.remote_compute_progress(self.frames, "task", None, remote, {})
        self.assertEqual(result.values, [0.0, 0.25, 0.5, 1.0])
        definition = remote.chat_calls[0][1]["schema"]["properties"]["progress"]
        self.assertEqual(definition["minItems"], 4)
        self.assertEqual(definition["maxItems"], 4)

    def test_gvl_remote_restores_shuffled_order(self):
        parsed = {
            "frames": [
                {
                    "frame_number": index + 1,
                    "frame_description": str(index),
                    "task_completion_percentage": percentage,
                }
                for index, percentage in enumerate((10, 40, 70, 100))
            ]
        }
        remote = _RemoteStub(chats=[parsed])
        result = GVL.remote_compute_progress(self.frames, "task", None, remote, {})
        self.assertEqual(sorted(result.values), [0.1, 0.4, 0.7, 1.0])

    def test_rlvlmf_remote_maps_image_order_to_chosen_probability(self):
        chosen = self.frames[:2]
        rejected = self.frames[2:]
        remote = _RemoteStub(chats=[{"preference": "A", "probability_a": 0.8}])
        result = RLVLMF.remote_compute_preference(chosen, rejected, "task", remote, {})
        seed = int(hashlib.sha256(b"task|2|2").hexdigest()[:16], 16)
        if seed % 2:
            self.assertEqual((result.preference, result.chosen_probability), ("chosen", 0.8))
        else:
            self.assertEqual(result.preference, "rejected")
            self.assertAlmostEqual(result.chosen_probability, 0.2)


if __name__ == "__main__":
    unittest.main()
