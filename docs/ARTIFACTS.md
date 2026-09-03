# 全流程运行产物说明

PRMEval 使用同一个 `bench.record.v1` `EvaluationRecord` 串联三个阶段。分离模式的标准运行目录为：

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

`mode: continue` 不落盘 Stage 1 产物，因此没有 `samples.jsonl` 和 `sample_frames/`。

## Stage 1：Sample

Stage 1 输出 `samples.jsonl` 和 `sample_frames/*.npz`。`samples.jsonl` 每行代表一次实际送入 Infer 的请求，
而不是一条原始 trajectory。每行都是尚无 `execution` 的 sampled `EvaluationRecord`，包含稳定的 `sample_id`、
评测与数据集身份、任务、NPZ 引用、采样索引和 Metric 所需的 `target`。

一个普通 progress Record 通常引用一个 trajectory NPZ；preference Record 可以分别引用 chosen 和 rejected
两个 NPZ。NPZ 路径相对于 `samples.jsonl` 所在目录。加载时会验证路径安全、文件 SHA-256、数组键、帧数，
以及 progress target 与帧数的长度关系。

`resume: true` 且 `samples.jsonl` 已存在时，Stage 1 验证并直接复用该文件。框架不再生成 sample manifest，
也不比较采样配置指纹；采样配置变化后应使用新运行目录、删除旧样本，或设置 `resume: false`。

## Stage 2：Infer

Stage 2 读取 `samples.jsonl` 中的每条 sampled Record，加载它引用的 NPZ，执行 Infer，并把补全后的 Record
追加到对应文件。Stage 1 文件不会被原地修改。

- `predictions.jsonl`：只保存成功 Record；增加 `infer`、`prediction` 和成功的 `execution`。
- `errors.jsonl`：保存失败 Record；`prediction` 为 null，`execution` 包含错误和可选原始响应。

断点续跑只把 `predictions.jsonl` 中已有的成功 `sample_id` 当作完成状态。只出现在 `errors.jsonl` 的样本会在
下次运行中重试。`predictions.jsonl` 中每个 `sample_id` 最多允许一条成功记录。

框架不再生成 run manifest、inference summary 或配置指纹。一个运行目录必须只用于一组固定的数据、采样配置
和模型配置；配置改变时由调用者切换 `run_name` 或清理旧产物。

## Stage 3：Metric

Stage 3 只读取成功的 `predictions.jsonl`，从每条 Record 的 `target` 和 `prediction` 计算指标。它不读取原始
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

读取全部成功 Record -> 重新生成 metrics.json 和 metrics_detail.jsonl
```

`resume: false` 会重新生成 Stage 1 产物，并在 Stage 2 开始时清空 predictions 和 errors。成功结果逐批追加，
因此进程中断后，已经完整写入 `predictions.jsonl` 的样本可以继续复用。

## 最小保留集合

- 重新推理：保留 `samples.jsonl` 和 `sample_frames/`。
- 重新计算指标：只需 `predictions.jsonl` 和匹配的配置。
- 审计完整运行：保留整个运行目录，特别是 `errors.jsonl` 和 `metrics_detail.jsonl`。
