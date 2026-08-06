# Stage 2 reward-alignment 输入冒烟数据

这个目录现在是 **Stage 2 的输入 bundle**，不是 Stage 2 的模拟输出。

- `samples.jsonl`：3 条 `bench.record.v1` Record，全部为 `stage: sampled`；
- `sample_frames/*.npz`：Stage 2 实际读取的帧数组；
- `samples.manifest.json`：Stage 1 生成该 bundle 时的采样清单。

`samples.jsonl` 中不包含 `baseline`、`prediction` 或 `execution`。推理后，这些字段才会出现在指定的 `predictions.jsonl` 中。

先验证 Stage 2 输入及 NPZ 校验和：

```bash
conda activate bench
python -m prmeval.cli validate-samples \
  --samples examples/stage2_reward_alignment_smoke/samples.jsonl
```

调用 `progress_test` baseline：

```bash
export OPENAI_API_KEY='your-key'
python -m prmeval.cli infer \
  --config configs/eval/progress_test_smoke.yaml \
  --samples examples/stage2_reward_alignment_smoke/samples.jsonl \
  --output evaluation_output/progress-test-smoke/predictions.jsonl
```

在运行前，需要把配置中的 `base_url` 和 `model` 改成真实的 OpenAI-compatible 服务地址和模型名。

这三组图像只是彩色像素组成的协议测试帧，用于验证 bundle 加载、Base64 图片构造、API 调用、structured output 和结果落盘，不用于评价模型语义能力。
