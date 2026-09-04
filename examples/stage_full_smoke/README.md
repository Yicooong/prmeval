# Hugging Face Dataset 全流程冒烟测试

这个测试固定验证以下组合：

- dataset source：本地 Hugging Face Dataset
- sampler/evaluation type：`progress`
- infer：`openai_compatible`
- metric：`progress`

配置模板为 `configs/eval/openai_compatible_remote.yaml`。先将其中的 `sampling.paths` 改为本地 Hugging Face Dataset
目录。模板限制最多读取一条 trajectory；实际请求批次数量取决于生成的样本数和 `infer.batch_size`。

```bash
export API_KEY='your-key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'

# Stage 1
prmeval sample \
  --config configs/eval/openai_compatible_remote.yaml
prmeval validate-samples \
  --samples evaluation_output/openai-compatible-remote/samples.jsonl

# Stage 2
prmeval infer \
  --config configs/eval/openai_compatible_remote.yaml
prmeval validate-predictions \
  --predictions evaluation_output/openai-compatible-remote/predictions.jsonl

# Stage 3
prmeval metrics \
  --config configs/eval/openai_compatible_remote.yaml
```

所有运行产物位于 `evaluation_output/openai-compatible-remote/`。该目录是中间产物，已由 `.gitignore` 忽略。

远程模型输出可能随服务和模型版本变化，因此该示例只用于验证三阶段协议、产物和指标流程，不提供固定的
预测值或 golden metric。
