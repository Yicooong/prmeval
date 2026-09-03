# Synthetic temporal robustness example

当前 Stage 1 不直接读取本目录中的旧版 `trajectories.jsonl`。先通过 `dataset_unify` 将数据转换为本地 Hugging Face
Dataset，并更新 `configs/eval/synthetic_temporal_robustness.yaml` 的 `sampling.paths`，再运行：

```bash
prmeval sample --config configs/eval/synthetic_temporal_robustness.yaml
```

默认配置会产生一个 Original 样本，以及 Pause、Slow、Fast、Rewind、Retry、Truncate、Skip 各三个变体。
每个样本的帧数不超过 `temporal_robustness.max_frames: 8`，并保留变换后的原始帧索引与逐帧 progress target。
