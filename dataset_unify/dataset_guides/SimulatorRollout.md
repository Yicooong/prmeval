# Simulator Rollout

This converter turns offline rollouts produced by `sim-prm-eval` into PRMEval's
standard local Hugging Face dataset. Simulator code only generates data;
sampling, PRM inference, and metrics remain in PRMEval.

## Source contract

`dataset.dataset_path` must point to a `samples.jsonl` file. Every row must
provide:

- `sample_id`, `task`, and an immutable `frames_path` NPZ archive;
- aligned `frame_indices` and `simulator_progress` arrays;
- `benchmark.suite_id` and `benchmark.outcome`;
- the configured camera array, such as `external_main_rgb`, in the NPZ archive.

Missing paths, views, progress values, or benchmark fields fail conversion.
The converter never substitutes another camera or progress definition.

## Convert

```bash
python -m dataset_unify.generate_hf_dataset \
  --config_path=dataset_unify/configs/data_gen_configs/simulator_rollout.yaml
```

The checked-in config converts simulator rollouts with the common
`external_main` view. Each output row sets
`is_simulation: true`; `target_progress` contains the simulator's frame-aligned
ground truth, while `partial_success` contains only the final value.

The current PRMEval `compact` data pool evaluates explicitly successful
trajectories. The converted dataset preserves both success and failure rows,
but Stage 1 currently selects 43 successful rows and filters 45 failure rows.

## Evaluate

Point an existing PRMEval config at the converted dataset and use the `progress`
sampler/metric:

```yaml
sampling:
  dataset_name: simulator_rollout
  paths:
    - /path/to/prmeval-datasets/simulator_rollout
  eval_types: [progress]
  base_frames: 16

metrics: [progress]
```

PRMEval reads `target_progress` from the HF row and samples frames and targets
with the same indices. No simulator-specific inference implementation is
required.
