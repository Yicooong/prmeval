from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from prmeval.core.config import EvalConfig, InferConfig, SamplingConfig
from prmeval.core.runner import Evaluator
from prmeval.infer.mock_server import ContractHandler


class RemotePipelineTest(unittest.TestCase):
    def test_progress_test_connects_all_three_stages_over_http(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), ContractHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset_path = root / "trajectories.jsonl"
                pixel = [[0, 0, 0], [0, 0, 0]]
                frames = [[pixel, pixel], [pixel, pixel], [pixel, pixel]]
                dataset_path.write_text(
                    json.dumps({
                        "id": "remote-trajectory",
                        "task": "complete the remote test task",
                        "frames": frames,
                        "quality_label": "successful",
                        "data_source": "remote-test-fixture",
                    })
                    + "\n",
                    encoding="utf-8",
                )
                config = EvalConfig(
                    sampling=SamplingConfig(
                        dataset_name="remote-test-fixture",
                        adapter="jsonl",
                        paths=[str(dataset_path)],
                        eval_types=["reward_alignment"],
                        max_frames=3,
                        progress_type="absolute_first_frame",
                    ),
                    infer=InferConfig(
                        name="progress_test",
                        mode="remote",
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        model_id="contract-vlm",
                        max_retries=0,
                        max_concurrency=2,
                    ),
                    metrics=["reward_alignment"],
                    output_dir=str(root / "output"),
                    run_name="remote-pipeline",
                    resume=False,
                )

                with patch.dict(
                    "os.environ",
                    {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
                ):
                    summary = Evaluator(config).run()

                output = root / "output" / "remote-pipeline"
                error_path = output / "errors.jsonl"
                failure_detail = error_path.read_text(encoding="utf-8") if error_path.exists() else summary
                self.assertEqual(summary["coverage"]["successful"], 1, failure_detail)
                self.assertEqual(summary["coverage"]["failed"], 0, summary)
                self.assertTrue((output / "samples.jsonl").is_file())
                self.assertTrue((output / "predictions.jsonl").is_file())
                self.assertTrue((output / "all_metrics.json").is_file())
                self.assertEqual(summary["execution"], {
                    "mode": "remote",
                    "batch_size": 1,
                    "max_concurrency": 2,
                })
                self.assertEqual(summary["metrics"]["reward_alignment"]["loss"], 0.0)

                prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(prediction["stage"], "inferred")
                self.assertEqual(prediction["infer"]["name"], "progress_test")
                self.assertEqual(prediction["prediction"]["values"], [0.0, 0.5, 1.0])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
