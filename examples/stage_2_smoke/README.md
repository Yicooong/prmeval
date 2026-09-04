# Stage 2 reward-alignment 输入冒烟数据

这个目录现在是 **Stage 2 的输入 bundle**，不是 Stage 2 的模拟输出。

- `samples.jsonl`：3 条 `bench.record.v1` sampled Record，全部尚无 `execution`；
- `sample_frames/*.npz`：Stage 2 实际读取的帧数组。

`samples.jsonl` 中不包含 `infer`、`prediction` 或 `execution`。推理后，这些字段才会出现在指定的 `predictions.jsonl` 中。

先验证 Stage 2 输入及 NPZ 校验和：

```bash
prmeval validate-samples \
  --samples examples/stage_2_smoke/samples.jsonl
```

调用 `openai_compatible` infer：

```bash
export API_KEY='your-key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
prmeval infer \
  --config configs/eval/openai_compatible_remote.yaml \
  --samples examples/stage_2_smoke/samples.jsonl \
  --output evaluation_output/openai-compatible-smoke/predictions.jsonl
```

配置加载时会从环境变量展开服务地址和模型 ID。

这三组图像只是彩色像素组成的协议测试帧，用于验证 bundle 加载、Base64 图片构造、API 调用、structured output 和结果落盘，不用于评价模型语义能力。
