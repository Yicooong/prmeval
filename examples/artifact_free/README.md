# 无产物运行

复制已有评估 YAML，保留实际数据集和模型配置，设置：

```yaml
mode: continue
output_dir: null
```

运行 `python -m prmeval.cli run --config your_config.yaml`，或通过
`Evaluator(EvalConfig.from_yaml("your_config.yaml")).run()` 获取 dict。

[expected_metrics.json](expected_metrics.json) 是单条 progress 样本的示例返回值：
标签与预测均为 `[0, 0.5, 1]`，数据集名 `demo`，infer 名 `demo_infer`，样本 ID 为 `sample-1`。
实际样本 ID、指标名和数值取决于配置。此文件是说明用的预期结果，无产物运行不会创建它。

policy_ranking 等分组指标还包含 `task_details`；失败记录不计入指标，全部失败返回 `{}`。
