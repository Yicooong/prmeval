"""Deterministic contract server for OpenAI-compatible infer tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ContractHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = self.path.rstrip("/")
        if path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self._chat_completion(payload)

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
