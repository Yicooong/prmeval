# Hugging Face Dataset 全流程冒烟测试

这个测试固定验证以下组合：

- dataset source：本地 Hugging Face Dataset
- sampler/evaluation type：`progress`
- infer：`progress_test`
- metric：`progress`

配置文件为 `configs/eval/progress_test_remote.yaml`。测试只读取一条 trajectory，并只发起一次远程模型请求。

```bash
export OPENAI_API_KEY='your-key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'

# Stage 1
python -m prmeval.cli sample \
  --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli validate-samples \
  --samples evaluation_output/jsonl-progress-full-smoke/samples.jsonl

# Stage 2
python -m prmeval.cli infer \
  --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli validate-predictions \
  --predictions evaluation_output/jsonl-progress-full-smoke/predictions.jsonl

# Stage 3
python -m prmeval.cli metrics \
  --config configs/eval/progress_test_remote.yaml
```

所有运行产物位于 `evaluation_output/jsonl-progress-full-smoke/`。该目录是中间产物，已由 `.gitignore` 忽略。

当前实际冒烟结果为：Stage 1 生成一条包含三帧的 sampled Record；Stage 2 成功生成 `[0.0, 0.5, 1.0]`；Stage 3 得到 MSE `0.0`、Pearson `1.0`。远程模型输出可能随服务版本变化，因此后两项数值不是固定 golden，只要求协议有效且 metric 成功完成。
