# Stage 1 sampling 说明

当前 Stage 1 只接受由 `datasets.save_to_disk()` 保存的本地 Hugging Face Dataset，不直接读取 JSONL。
本目录中的 `trajectories.jsonl` 是旧版测试 fixture；如需继续使用，应先通过 `dataset_unify` 转成标准 Dataset。

将 `configs/eval/openai_compatible_remote.yaml` 中的 `sampling.paths` 改为转换后的 Dataset 目录，然后运行：

```bash
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
prmeval sample \
  --config configs/eval/openai_compatible_remote.yaml
prmeval validate-samples \
  --samples evaluation_output/openai-compatible-remote/samples.jsonl
```

Stage 1 不会发送模型请求。不过三个阶段共用同一个配置对象，因此仍需设置配置中引用的 `BASE_URL`、
`API_KEY` 和 `MODEL_ID`，以便配置校验通过。
