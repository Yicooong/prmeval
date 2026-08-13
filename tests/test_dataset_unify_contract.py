from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_unify.hf_schema import STANDARD_DATASET_FIELDS, build_standard_dataset
from prmeval.core.config import SamplingConfig
from prmeval.sample.adapters import HuggingfaceDatasetAdapter
from prmeval.sample.samplers import RewardAlignmentSampler


class DatasetUnifyContractTest(unittest.TestCase):
    frame_height = 8
    frame_width = 8

    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is required for the video contract test")
        try:
            import datasets  # noqa: F401
        except ImportError:
            self.skipTest("datasets is required for the dataset contract test")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_video(self, path: Path, frames: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.frame_width}x{self.frame_height}",
            "-r",
            "4",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        subprocess.run(command, input=frames.tobytes(), check=True)

    def test_standard_builder_has_fixed_schema_and_preserves_none(self) -> None:
        dataset = build_standard_dataset([
            {
                "id": "trajectory-none",
                "task": "do something",
                "data_source": "contract",
                "frames": "batch_0000/trajectory.mp4",
                "is_robot": False,
                "quality_label": None,
                "partial_success": None,
                "ignored": "not part of the standard schema",
            }
        ])

        self.assertEqual(tuple(dataset.column_names), STANDARD_DATASET_FIELDS)
        self.assertIsNone(dataset[0]["quality_label"])
        self.assertIsNone(dataset[0]["partial_success"])

        empty = build_standard_dataset([])
        self.assertEqual(tuple(empty.column_names), STANDARD_DATASET_FIELDS)

    def test_local_video_reaches_huggingface_adapter_and_sampler(self) -> None:
        dataset_name = "contract_dataset"
        dataset_root = self.root / "unified" / dataset_name
        relative_video_path = Path("shard_0000") / "episode_000000" / "clip.mp4"
        frames = np.stack([
            np.full((self.frame_height, self.frame_width, 3), value, dtype=np.uint8)
            for value in (0, 80, 160, 240)
        ])
        self._write_video(dataset_root / relative_video_path, frames)

        unified = build_standard_dataset([
            {
                "id": "trajectory-1",
                "task": "move the object",
                "data_source": dataset_name,
                "frames": relative_video_path.as_posix(),
                "is_robot": True,
                "quality_label": "successful",
                "partial_success": 1.0,
            }
        ])
        unified.save_to_disk(str(dataset_root))

        adapter = HuggingfaceDatasetAdapter(
            SamplingConfig(
                dataset_name=dataset_name,
                adapter="huggingface",
                paths=[str(dataset_root)],
            )
        )
        trajectories = list(adapter.load())
        self.assertEqual(len(trajectories), 1)
        self.assertTrue(trajectories[0].is_robot)
        self.assertEqual(trajectories[0].data_source, dataset_name)
        self.assertEqual(trajectories[0].frames, str(dataset_root / relative_video_path))

        sampler = RewardAlignmentSampler(SamplingConfig(max_frames=2), dataset_name)
        samples = list(sampler.sample(trajectories))
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].trajectory.is_robot)
        self.assertEqual(samples[0].trajectory.frames.shape, (2, self.frame_height, self.frame_width, 3))
        self.assertEqual(len(samples[0].trajectory.target_progress), 2)


if __name__ == "__main__":
    unittest.main()
