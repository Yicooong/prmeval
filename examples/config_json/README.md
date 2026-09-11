# JSON 配置冒烟示例

从仓库根目录运行以下命令，无需调用模型即可验证 JSON 加载和采样器准备：

```bash
prmeval sample --config examples/config_json/config.json --no-progress
```

预期加载 1 条轨迹并准备 `progress` 采样器。`sampling.save_samples` 默认为 `false`，此步骤不写入样本文件。

此示例省略 `output_dir` 和 `task_name`，后续推理和指标的默认目录为：
`/tmp/prmeval_evaluation_output/json-config-smoke_openai_compatible_progress/`。

调用模型前需配置有效的服务地址、模型 ID 和 API Key。扩展参数放在 `infer.model_extra_config` 中。
