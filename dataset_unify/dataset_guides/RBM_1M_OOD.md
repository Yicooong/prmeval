# RBM-1M-OOD

This loader converts the local RBM-1M-OOD LeRobot export into the seven-field
`dataset_unify` Hugging Face Dataset contract.

## Detected source layout

The export contains one `metadata.parquet` and a sequence of `video_*.mp4` files
per source. The canonical OOD split contains these six source directories:

| Source directory | Canonical `data_source` | Episodes in the scanned export |
|---|---|---:|
| `usc_trossen` | `usc_trossen` | 27 |
| `mit_franka` | `rfm_new_mit_franka_rfm` | 304 |
| `utd_so101` | `utd_so101_clean_policy_ranking_top` | 30 |
| `usc_xarm` | `usc_xarm_policy_ranking` | 36 |
| `usc_franka` | `usc_franka_policy_ranking` | 24 |
| `usc_koch` | `usc_koch_p_ranking_all` | 150 |

Together these match the root LeRobot metadata and export summary: 571 episodes,
18,261 frames, 41 tasks. Supplemental sibling folders such as human, paired,
clutter, and wrist-camera variants are intentionally excluded.

## Convert

Edit the output directory in `dataset_unify/configs/data_gen_configs/rbm_1m_ood.yaml`
if needed, then run:

```bash
python -m dataset_unify.generate_hf_dataset \
  --config_path=dataset_unify/configs/data_gen_configs/rbm_1m_ood.yaml
```

For a one-trajectory smoke conversion, append:

```bash
--output.max_trajectories=1 --output.num_workers=0
```

The loader lazily decodes each source MP4. The shared converter applies
`max_frames`, resizing, cropping, and output FPS settings, and writes portable
relative MP4 paths into the final Dataset.
