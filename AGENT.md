# Repository Guidelines

Local model and checkpoint loading is supported for inference implementations. Built-in infer baselines may either load models/checkpoints locally or use the OpenAI-compatible `POST /v1/chat/completions` transport. Do not revive the removed `specialized`/`v1/evaluations` transport.
