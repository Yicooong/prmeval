from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import numpy as np

from prmeval.infer.baselines import (
    GVL,
    RBMModel,
    RLVLMF,
    RoboDopamine,
    RoboReward,
    TopReward,
    VLAC,
)


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


class BaselineRemoteTest(unittest.TestCase):
    def setUp(self):
        self.frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)

    def test_roboreward_remote_score_mapping(self):
        remote = _RemoteStub(chats=[{"score": 4}])
        result = RoboReward.remote_compute_progress(self.frames, "task", None, remote, {})
        self.assertEqual(result.values.tolist(), [0.75] * 4)
        self.assertEqual(remote.chat_calls[0][1]["name"], "roboreward_score")

    def test_topreward_remote_prefix_logprobs(self):
        completions = []
        for reward in (-3.0, -2.0, -1.0):
            completions.append({
                "choices": [{
                    "message": {"content": "False"},
                    "logprobs": {
                        "content": [{
                            "token": "False",
                            "logprob": -0.1,
                            "top_logprobs": [{"token": " True", "logprob": reward}],
                        }]
                    },
                }]
            })
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
        remote = _RemoteStub(chats=[
            {"relative_change_percent": 50},
            {"relative_change_percent": -50},
        ])
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
        seed = int(hashlib.sha256("task|2|2".encode()).hexdigest()[:16], 16)
        if seed % 2:
            self.assertEqual((result.preference, result.chosen_probability), ("chosen", 0.8))
        else:
            self.assertEqual(result.preference, "rejected")
            self.assertAlmostEqual(result.chosen_probability, 0.2)


if __name__ == "__main__":
    unittest.main()
