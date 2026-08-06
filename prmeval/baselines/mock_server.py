"""Deterministic contract server for specialized endpoint integration tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .specialized import SpecializedRequest


class ContractHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        if self.path.rstrip("/") != "/v1/evaluations":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = SpecializedRequest.model_validate_json(self.rfile.read(length))
        if request.prediction_type in {"progress", "instruction_likelihood"}:
            count = len(request.trajectories[0].frames)
            values = [i / max(1, count - 1) for i in range(count)]
            prediction = {"trajectory_id": request.trajectories[0].id, "progress": values}
        else:
            prediction = {"preference_probability": 0.75}
        body = json.dumps({
            "id": request.request_id,
            "model": request.model,
            "model_version": "mock-v1",
            "predictions": [prediction],
            "usage": {},
        }).encode()
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
