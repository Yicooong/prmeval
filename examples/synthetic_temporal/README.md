# Synthetic temporal robustness example

该目录提供一条包含 10 帧的成功轨迹，用于运行自动时序变换示例：

```bash
python -m prmeval.cli sample --config configs/eval/synthetic_temporal_robustness.yaml
```

默认配置会产生一个 Original 样本，以及 Pause、Slow、Fast、Rewind、Retry、Truncate、Skip 各三个变体。
每个样本的帧数不超过 `temporal_robustness.max_frames: 8`，并保留变换后的原始帧索引与逐帧 progress target。
