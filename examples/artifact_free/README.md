# 无产物运行

复制已有评估 YAML，保留实际数据集和模型配置，删除旧的 `mode` 字段，设置：

```yaml
sampling:
  save_samples: false
output_dir: null
```

运行 `python -m prmeval.cli run --config your_config.yaml`，或通过
`Evaluator(EvalConfig.from_yaml("your_config.yaml")).run()` 获取结果。
也可在同一个 Evaluator 上依次调用 `sample()`、`infer()`、`evaluate_metrics()`。
`sample()` 只准备 samplers，`infer()` 按 batch 生成样本并返回 `(summary, successful_records)`。

[expected_metrics.json](expected_metrics.json) 是 Python API 对单条 progress 样本的示例返回值：
包含 `metrics`、`coverage` 以及为 null 的路径。标签与预测均为 `[0, 0.5, 1]`，
数据集名为 `demo`，infer 名为 `demo_infer`，样本 ID 为 `sample-1`。
这是说明用的预期结果，无产物运行不会创建文件。

Python 返回的 `metrics` 保留 `details` 和适用指标的 `task_details`；CLI 输出省略这些明细的摘要。
失败记录不计入指标，全部失败返回空 `metrics` 和失败 coverage。无输出目录时每次推理重新执行。
