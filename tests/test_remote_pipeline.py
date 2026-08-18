from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from prmeval.core.config import EvalConfig, InferConfig, SamplingConfig
from prmeval.core.runner import Evaluator


class ContractHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        definition = payload["response_format"]["json_schema"]["schema"]["properties"]["progress"]
        count = definition["minItems"]
        result = {"progress": [index / max(1, count - 1) for index in range(count)]}
        body = json.dumps(
            {
                "id": "mock-chat-completion",
                "choices": [{"message": {"content": json.dumps(result)}}],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


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
                    json.dumps(
                        {
                            "id": "remote-trajectory",
                            "task": "complete the remote test task",
                            "frames": frames,
                            "quality_label": "successful",
                            "data_source": "remote-test-fixture",
                        }
                    )
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
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        model_id="contract-vlm",
                        max_retries=0,
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
                self.assertNotIn("execution", summary)
                self.assertTrue((output / "samples.jsonl").is_file())
                self.assertTrue((output / "predictions.jsonl").is_file())
                self.assertTrue((output / "all_metrics.json").is_file())

                prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
                self.assertEqual(prediction["stage"], "inferred")
                self.assertEqual(prediction["infer"]["name"], "progress_test")
                self.assertEqual(prediction["prediction"]["values"], [0.0, 0.5, 1.0])
                self.assertEqual(prediction["prediction"]["data"]["raw_response"], None)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
