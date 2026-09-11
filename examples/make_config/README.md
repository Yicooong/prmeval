# 生成配置模板

从仓库根目录将默认模板输出到终端：

```bash
python -m prmeval.cli make-config
```

输出示例见 [default.yaml](default.yaml)。模板从 `EvalConfig` 自动生成，包含嵌套默认值。
CLI 直接调用 `EvalConfig.export_config()`。保存模板：

```bash
python -m prmeval.cli make-config --output configs/eval/demo.yaml
python -m prmeval.cli make-config --format json --output configs/eval/demo.json
```

默认导出 YAML，JSON 输出示例见 [default.json](default.json)。Python 可直接调用 `EvalConfig.export_config("json")`。

`infer.name` 没有默认值，以 `null` 占位，需先填写才能加载配置；评测前还需填写数据集路径和模型参数。`task_name: null` 会在加载时生成任务名。
已有文件默认不会被覆盖；明确需要覆盖时传入 `--force`。
