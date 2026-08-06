# Metric smoke test

`predictions.jsonl` contains legacy flat `EvaluationRecord` rows and verifies in-memory migration.
For the canonical unified v1 format and reward-alignment-only debugging, use
`examples/reward_alignment_v1` instead.

Run all metrics found in the file:

```bash
prmeval-eval compute-metrics \
  --predictions examples/metrics_smoke/predictions.jsonl \
  --output examples/metrics_smoke/metrics.json
```

The expected headline values are:

- reward alignment loss: `0.00833333333333333`, Pearson: `1.0`;
- policy ranking Kendall: `1.0`;
- quality preference accuracy: `0.6666666666666666`, tie rate: `0.0`;
- confusion matrix: `[[0.9, 0.1], [0.2, 0.8]]`;
- normalized confusion diagonal-minus-off-diagonal: `0.7`.
