# Local-first baselines

This package contains the built-in local-first baseline implementations.

Each progress model implements local `compute_progress()` and optional OpenAI-compatible
`remote_compute_progress()`. RL-VLM-F uses the equivalent `PreferenceModel` interface. Remote methods reuse
`RemoteContext`, so API authentication, retries, image encoding, request counting, and structured-output validation
remain framework responsibilities.

| Registry name | Class | Local | Remote |
|---|---|---:|---:|
| `progress_test` | `ProgressTestModel` | no | yes |
| `gvl` | `GVL` | no | yes |
| `rbm` / `rewind` | `RBMModel` | yes | yes |
| `rlvlmf` | `RLVLMF` | no | yes |
| `robodopamine` | `RoboDopamine` | yes | yes |
| `roboreward` | `RoboReward` | yes | yes |
| `topreward` | `TopReward` | yes | approximate top-logprobs mode |
| `vlac` | `VLAC` | yes | yes |

`progress_test` is a generic OpenAI-compatible remote model intended for protocol and Stage 1 → Stage 2 → Stage 3
smoke tests. It requests one progress value per input frame and does not load a local checkpoint.

TOPReward `scoring: exact_logits` is local-only. Its remote method uses the existing `logprobs/top_logprobs`
approximation and rejects an explicit `scoring: exact_logits` configuration.
