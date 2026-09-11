# Metric smoke test

`predictions.jsonl` contains current `EvaluationRecord` rows for metric smoke tests.
Each row contains the top-level `eval_type`, `sample`, the original `prediction` (including `sample_id` and `model`),
and `execution`. Metrics read labels directly from the sample trajectories.

Run all metrics found in the file:

```bash
prmeval compute-metrics \
  --predictions examples/stage_3_smoke/predictions.jsonl \
  --output examples/stage_3_smoke/metrics.json
```

The expected headline values are:

- progress loss: `0.00833333333333333`, Pearson: `1.0`;
- policy ranking Kendall: `1.0`;
- quality preference accuracy: `0.6666666666666666`, tie rate: `0.0`;
- confusion matrix: `[[0.9, 0.1], [0.2, 0.8]]`;
- normalized confusion diagonal-minus-off-diagonal: `0.7`.
