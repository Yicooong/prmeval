"""Deterministic contract server for OpenAI-compatible and legacy specialized infer tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .specialized import SpecializedRequest


class ContractHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = self.path.rstrip("/")
        if path not in {"/v1/chat/completions", "/v1/evaluations"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if path == "/v1/chat/completions":
            self._chat_completion(payload)
        else:
            self._specialized(payload)

    def _chat_completion(self, payload: dict) -> None:
        response_format = payload.get("response_format")
        if not response_format:
            body = {
                "id": "mock-topreward",
                "choices": [
                    {
                        "message": {"content": "True"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "True",
                                    "logprob": -0.1,
                                    "top_logprobs": [{"token": "True", "logprob": -0.1}],
                                }
                            ]
                        },
                    }
                ],
            }
            self._send_json(body)
            return
        schema = response_format["json_schema"]
        definition = schema["schema"]["properties"]
        if "frames" in definition:
            count = definition["frames"]["minItems"]
            result = {
                "frames": [
                    {
                        "frame_number": index + 1,
                        "frame_description": f"mock frame {index + 1}",
                        "task_completion_percentage": 100 * index / max(1, count - 1),
                    }
                    for index in range(count)
                ]
            }
        elif "score" in definition:
            result = {"score": 3}
        elif "relative_change_percent" in definition:
            result = {"relative_change_percent": 25}
        elif "preference" in definition:
            result = {"preference": "A", "probability_a": 0.75}
        else:
            progress = definition["progress"]
            count = progress.get("minItems") or self._image_count(payload)
            result = {"progress": [index / max(1, count - 1) for index in range(count)]}
        self._send_json(
            {
                "id": "mock-chat-completion",
                "choices": [{"message": {"content": json.dumps(result)}}],
            }
        )

    @staticmethod
    def _image_count(payload: dict) -> int:
        return sum(
            item.get("type") == "image_url"
            for message in payload.get("messages", [])
            for item in message.get("content", [])
            if isinstance(message.get("content"), list)
        )

    def _specialized(self, payload: dict) -> None:
        request = SpecializedRequest.model_validate(payload)
        if request.prediction_type in {"progress", "instruction_likelihood"}:
            count = len(request.trajectories[0].frames)
            values = [i / max(1, count - 1) for i in range(count)]
            prediction = {"trajectory_id": request.trajectories[0].id, "progress": values}
        else:
            prediction = {"preference_probability": 0.75}
        self._send_json(
            {
                "id": request.request_id,
                "model": request.model,
                "model_version": "mock-v1",
                "predictions": [prediction],
                "usage": {},
            }
        )

    def _send_json(self, value: dict) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), ContractHandler).serve_forever()


if __name__ == "__main__":
    main()
