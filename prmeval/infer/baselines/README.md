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
| `sole_r1` | `SoleR1` | no | yes |
| `topreward` | `TopReward` | yes | approximate top-logprobs mode |
| `vlac` | `VLAC` | yes | yes |

`progress_test` is a generic OpenAI-compatible remote model intended for protocol and Stage 1 → Stage 2 → Stage 3
smoke tests. It requests one progress value per input frame and does not load a local checkpoint.

`sole_r1` consumes the ordered frames produced by Stage 1. It fixes the first progress value at zero, then sends the
first, previous, and current observations through sequential Chat Completions requests while carrying the previous
percentage forward. It does not load, sample, combine, or visualize raw videos.

TOPReward `scoring: exact_logits` is local-only. Its remote method uses the existing `logprobs/top_logprobs`
approximation and rejects an explicit `scoring: exact_logits` configuration.
