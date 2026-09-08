# 全流程运行产物说明

PRMEval 使用同一个 `bench.record.v1` `EvaluationRecord` 串联三个阶段。启用全部产物写入时的标准运行目录为：

```text
<output_dir>/<run_name-or-default>/
├── samples.jsonl
├── sample_frames/
│   └── <sample_id>-<role>.npz
├── predictions.jsonl
├── errors.jsonl
├── metrics.json
└── metrics_detail.jsonl
```

`save_samples: false`（默认）不落盘 Stage 1 产物，因此没有 `samples.jsonl` 和 `sample_frames/`。

## Stage 1：Sample

有输出目录且 `save_samples: true` 时，Stage 1 输出 `samples.jsonl` 和 `sample_frames/*.npz`。`samples.jsonl` 每行代表一次实际送入 Infer 的请求，
而不是一条原始 trajectory。每行都是尚无 `execution` 的 sampled `EvaluationRecord`，包含稳定的 `sample_id`、
评测与数据集身份、任务、NPZ 引用、采样索引和 Metric 所需的 `target`。

一个普通 progress Record 通常引用一个 trajectory NPZ；preference Record 可以分别引用 chosen 和 rejected
两个 NPZ。NPZ 路径相对于 `samples.jsonl` 所在目录。加载时会验证路径安全、文件 SHA-256、数组键、帧数，
以及 progress target 与帧数的长度关系。

`sample()` 不读取 `resume`：每次重新加载原始数据并准备 samplers，开启采样落盘时重新生成并覆盖样本文件。
若要直接使用已有样本，调用 `infer(samples_path=...)`，无需先调用 `sample()`。框架不生成 sample manifest，
也不比较采样配置指纹；数据、采样或模型配置变化后应使用新运行目录，避免复用不匹配的预测。

## Stage 2：Infer

Stage 2 读取样本文件，或在没有样本文件时消费已准备的 samplers，两者互斥。
文件来源保留 NPZ 引用，sampler 来源清除帧数组；有输出目录时将补全后的 Record 逐批追加到对应文件。Stage 1 文件不会被原地修改。

- `predictions.jsonl`：只保存成功 Record；增加 `infer`、`prediction` 和成功的 `execution`。
- `errors.jsonl`：保存失败 Record；`prediction` 为 null，`execution` 包含错误和可选原始响应。

断点续跑只把 `predictions.jsonl` 中已有的成功 `sample_id` 当作完成状态。只出现在 `errors.jsonl` 的样本会在
下次运行中重试。`predictions.jsonl` 中每个 `sample_id` 最多允许一条成功记录。

框架不再生成 run manifest、inference summary 或配置指纹。一个运行目录必须只用于一组固定的数据、采样配置
和模型配置；配置改变时由调用者切换 `run_name` 或清理旧产物。

## Stage 3：Metric

Stage 3 接收内存成功记录或读取成功的 `predictions.jsonl`，从每条 Record 的 `target` 和 `prediction` 计算指标。它不读取原始
Dataset、NPZ 或模型。

### `metrics.json`

保存覆盖率、聚合指标和 detail 文件位置。聚合结果不包含逐样本 `details` 或逐任务 `task_details`：

```json
{
  "coverage": {
    "total": 100,
    "successful": 92,
    "failed": 8,
    "executed": 30,
    "skipped": 70
  },
  "metrics": {
    "progress": {
      "mse": 0.02,
      "loss": 0.02,
      "pearson": 0.95,
      "num_samples": 92,
      "slices": {}
    }
  },
  "predictions": "evaluation_output/example/predictions.jsonl",
  "details": "evaluation_output/example/metrics_detail.jsonl"
}
```

### `metrics_detail.jsonl`

`detail_type: record` 行包含完整的成功 `EvaluationRecord`，并增加该 Record 的逐条 `metrics`。因此数据生命周期是：

```text
Stage 1 Record = input + target
Stage 2 Record = Stage 1 Record + infer + prediction + execution
Stage 3 detail = Stage 2 Record + metrics
```

必须联合多条 Record 才能定义的指标（例如 task 内 policy ranking Kendall）额外写为
`detail_type: group` 行，其中包含 group ID、参与的 `sample_ids` 和分组指标。

## 续跑规则

```text
completed_ids = predictions.jsonl 中的成功 sample_id

遍历 Stage 1 Record 或重新生成的稳定 sample：
    sample_id 在 completed_ids 中 -> 跳过
    否则 -> 推理并立即追加成功或错误 Record

汇总当前输入范围内的历史成功和本次成功 Record -> 重新生成 metrics.json 和 metrics_detail.jsonl
```

`resume` 只控制 Stage 2；`resume: false` 在推理开始时清空 predictions 和 errors。成功结果逐批追加，
因此进程中断后，已经完整写入 `predictions.jsonl` 的样本可以继续复用。

## 最小保留集合

- 重新推理：保留 `samples.jsonl` 和 `sample_frames/`。
- 重新计算指标：只需 `predictions.jsonl` 和匹配的配置。
- 审计完整运行：保留整个运行目录，特别是 `errors.jsonl` 和 `metrics_detail.jsonl`。

## 关闭产物输出

`output_dir: null` 时，不创建输出目录，也不写入采样帧、样本、预测、错误或指标文件。
输出路径属性均为 `None`；不会读取历史预测检查点，但允许显式读取样本或预测文件。
`save_samples: true` 不能绕过这个总开关。

Python 的 `run()` 和 `evaluate_metrics()` 始终返回 `metrics`、`coverage`、`predictions`、`details`。
`metrics` 包含完整逐样本和逐组明细；CLI 与磁盘 `metrics.json` 只保留聚合指标。
没有对应文件来源或产物时路径为 `None`。示例见 [无产物运行示例](../examples/artifact_free/README.md)。

部分失败时仅使用成功记录计算指标；全部失败返回空 `metrics` 和完整 coverage。
有输出目录时也会保存这份摘要与空 `metrics_detail.jsonl`。采样为空仍报错。
