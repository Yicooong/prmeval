from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_unify.dataset_loaders.rbm_1m_ood_loader import (
    RBM1MOODFrameLoader,
    SOURCE_DIRECTORIES,
    load_rbm_1m_ood_dataset,
)


class RBM1MOODLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_source(self, directory_name: str, index: int, **overrides) -> None:
        source_dir = self.root / directory_name
        source_dir.mkdir(parents=True)
        video_name = f"video_{index:06d}.mp4"
        (source_dir / video_name).write_bytes(b"synthetic video placeholder")
        row = {
            "id": f"{directory_name}-{index}",
            "file_name": video_name,
            "video_file_name": f"{directory_name}/{video_name}",
            "language_instruction": f"Task for {directory_name}",
            "quality_label": "success",
            "data_source": directory_name,
            "task": f"Task for {directory_name}",
            "is_robot": "true",
            "partial_success": "",
        }
        row.update(overrides)
        pq.write_table(pa.Table.from_pylist([row]), source_dir / "metadata.parquet")

    def test_loads_only_canonical_sources_and_normalizes_metadata_types(self) -> None:
        for index, directory_name in enumerate(SOURCE_DIRECTORIES):
            self._write_source(directory_name, index)
        self._write_source("utd_so101_human", 99, is_robot="false")

        trajectories = [
            trajectory
            for group in load_rbm_1m_ood_dataset(str(self.root)).values()
            for trajectory in group
        ]

        self.assertEqual(len(trajectories), len(SOURCE_DIRECTORIES))
        self.assertEqual({trajectory["data_source"] for trajectory in trajectories}, set(SOURCE_DIRECTORIES.values()))
        self.assertTrue(all(trajectory["is_robot"] for trajectory in trajectories))
        self.assertTrue(all(trajectory["partial_success"] is None for trajectory in trajectories))
        self.assertTrue(all(isinstance(trajectory["frames"], RBM1MOODFrameLoader) for trajectory in trajectories))
        self.assertNotIn("utd_so101_human-99", {trajectory["id"] for trajectory in trajectories})

    def test_max_trajectories_stops_before_requiring_later_sources(self) -> None:
        first_directory = next(iter(SOURCE_DIRECTORIES))
        self._write_source(first_directory, 0, partial_success="0.5")

        task_data = load_rbm_1m_ood_dataset(str(self.root), max_trajectories=1)
        trajectory = next(iter(task_data.values()))[0]

        self.assertEqual(trajectory["partial_success"], 0.5)

    def test_missing_canonical_source_is_reported(self) -> None:
        first_directory = next(iter(SOURCE_DIRECTORIES))
        self._write_source(first_directory, 0)

        with self.assertRaisesRegex(FileNotFoundError, "Missing canonical RBM-1M-OOD source metadata"):
            load_rbm_1m_ood_dataset(str(self.root))


if __name__ == "__main__":
    unittest.main()
