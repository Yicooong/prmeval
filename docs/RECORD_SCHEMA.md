# EvaluationRecord 数据结构说明

## 1. 设计目标

`EvaluationRecord` 是 Stage 1 与 Stage 2 共用的统一数据结构，也是 Stage 3 的输入。新增 dataset 或 baseline 时不修改顶层结构；新增 metric 时，由 Metric 校验它需要的 `target.kind` 和 `prediction.kind`。

顶层结构如下：

```json
{
  "schema_version": "bench.record.v1",
  "stage": "sampled | inferred",
  "sample_id": "...",
  "evaluation": {},
  "input": {},
  "target": {},
  "baseline": null,
  "prediction": null,
  "execution": null,
  "source": {},
  "extensions": {}
}
```

## 2. 字段分类

| 分类 | 字段 | 作用 |
|---|---|---|
| 协议字段 | `schema_version`, `stage` | 格式兼容与单条记录生命周期 |
| 主键字段 | `sample_id` | 跨阶段对齐、去重和结果定位 |
| 评测路由 | `evaluation` | 指定 eval type 和数据集切片 |
| 模型输入 | `input` | 保存任务、帧引用和抽样信息 |
| 指标真值 | `target` | 保存 progress、rank、preference 等真值 |
| 模型身份 | `baseline` | 保存 baseline、远程模型和版本 |
| 模型输出 | `prediction` | 保存 adapter 归一化后的预测 |
| 执行审计 | `execution` | 保存成功/失败、重试次数和耗时 |
| 来源审计 | `source` | 保存可选的原始数据 ID |
| 扩展字段 | `extensions` | 保存不影响核心协议的附加信息 |

## 3. 顶层字段

### `schema_version`

必填，固定为 `bench.record.v1`。它不是指标，只用于识别数据协议版本。未来字段发生不兼容变化时应发布 v2，而不是让旧文件静默失效。

### `stage`

必填，可选值：

```text
sampled   Stage 1 已完成，等待模型推理
inferred  Stage 2 已执行，结果可能成功或失败
```

不增加 `metric_completed`。Metric 是多记录聚合过程，完成状态由 Metric 输出和 RunManifest 表示。

### `sample_id`

必填，是唯一跨阶段主键。Stage 2 必须原样保留 Stage 1 的值。

同一个 sample 分别由多个 baseline 推理时，可以在不同 run 中保持相同 `sample_id`。合并结果后的联合身份为：

```text
(evaluation.dataset.name, baseline.name, sample_id)
```

## 4. `evaluation`

```json
{
  "type": "reward_alignment",
  "dataset": {
    "name": "rbm-1m-ood",
    "split": "test",
    "source": "optional-subset"
  }
}
```

| 字段 | 必填 | 说明 |
|---|---:|---|
| `type` | 是 | 评测任务，如 `reward_alignment`、`policy_ranking` |
| `dataset.name` | 是 | 统一数据集名称，也是 Metric 切片维度 |
| `dataset.split` | 否 | test、validation 等划分 |
| `dataset.source` | 否 | 聚合数据集中的子来源 |

新增 dataset 只需要 DatasetAdapter 输出新的 `dataset.name/source`，不修改 Record Schema。

## 5. `input`

```json
{
  "task": "open the drawer",
  "task_id": "open the drawer",
  "items": [{
    "role": "trajectory",
    "frames": {
      "type": "npz",
      "path": "sample_frames/sample-id-trajectory.npz",
      "key": "frames",
      "num_frames": 3,
      "sha256": "..."
    },
    "frame_indices": [0, 15, 31],
    "num_frames_total": 32,
    "source_id": "optional-original-id",
    "data": {}
  }]
}
```

`task` 和 `items` 必填。Reward alignment 通常只有一个 `role: trajectory`；偏好评测可以包含 `chosen` 和 `rejected` 两项。

`source_id` 是原始数据审计信息，不是主键。Stage 3 不需要读取 `.npz` 文件，但完整 Record 会保留帧引用，便于追踪输入。

## 6. `target` 与 `prediction`

两者共用 `ValuePayload`：

```json
{
  "kind": "progress",
  "values": [0.0, 0.5, 1.0],
  "value": null,
  "label": null,
  "probability": null,
  "data": {}
}
```

| kind | target 典型字段 | prediction 典型字段 | 使用指标 |
|---|---|---|---|
| `progress` | `values` | `values` | MSE、Pearson、ranking reducer |
| `rank` | `value`, `label` | — | Kendall 真值 |
| `preference` | `label=chosen` | `label`, `probability` | Preference accuracy |
| `task_match` | `value`, `data` | — | Confusion matrix 真值 |

Metric 插件负责更严格的语义校验。例如 reward alignment 要求：

```text
target.kind == progress
prediction.kind == progress
target.values 非空
len(target.values) == len(prediction.values)
所有 progress 位于 [0, 1]
```

## 7. `baseline` 与 `execution`

成功推理记录示例：

```json
{
  "baseline": {
    "name": "rbm",
    "model": "robometer-4b",
    "version": "v1"
  },
  "execution": {
    "status": "success",
    "attempts": 1,
    "latency_seconds": 0.83,
    "error": null
  }
}
```

新增 baseline 只需要 adapter 填写上述字段并产生统一 `prediction`，不修改 Record 顶层结构。

## 8. 阶段必填规则

| 字段 | sampled | inferred success | inferred error |
|---|---:|---:|---:|
| `schema_version` | 必填 | 必填 | 必填 |
| `stage` | `sampled` | `inferred` | `inferred` |
| `sample_id` | 必填 | 必填且不变 | 必填且不变 |
| `evaluation` | 必填 | 原样保留 | 原样保留 |
| `input` | 必填 | 原样保留 | 原样保留 |
| `target` | 按 eval type 要求 | 原样保留 | 原样保留 |
| `baseline` | 必须为空 | 必填 | 建议保留 |
| `prediction` | 必须为空 | 必填 | 通常为空 |
| `execution` | 必须为空 | 必填、status=success | 必填、status=error |
| `execution.error` | — | 为空 | 必填 |

## 9. 完整 reward alignment 推理记录

```json
{
  "schema_version": "bench.record.v1",
  "stage": "inferred",
  "sample_id": "align-001",
  "evaluation": {
    "type": "reward_alignment",
    "dataset": {"name": "rbm-1m-ood", "split": "test"}
  },
  "input": {
    "task": "open the drawer",
    "task_id": "open the drawer",
    "items": [{
      "role": "trajectory",
      "frames": {"type": "npz", "path": "sample_frames/align-001-trajectory.npz", "key": "frames", "num_frames": 3, "sha256": "..."},
      "frame_indices": [0, 15, 31],
      "num_frames_total": 32,
      "source_id": "raw-trajectory-123",
      "data": {}
    }]
  },
  "target": {"kind": "progress", "values": [0.0, 0.5, 1.0]},
  "baseline": {"name": "rbm", "model": "robometer-4b", "version": "v1"},
  "prediction": {"kind": "progress", "values": [0.0, 0.4, 0.8]},
  "execution": {"status": "success", "attempts": 1, "latency_seconds": 0.83, "error": null},
  "source": {"id": "raw-trajectory-123", "data": {}},
  "extensions": {}
}
```

可运行示例位于 `examples/reward_alignment_v1/`。
