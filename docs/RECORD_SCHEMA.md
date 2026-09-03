# EvaluationRecord 数据结构说明

`EvaluationRecord` 是三个阶段共用的统一记录。Stage 1 写入输入与 target，Stage 2 在其副本上补充 infer、
prediction 和 execution，Stage 3 读取 target 与 prediction 计算指标。

## 顶层结构

```json
{
  "schema_version": "bench.record.v1",
  "sample_id": "stable-sample-id",
  "evaluation": {
    "type": "progress",
    "dataset": {"name": "dataset", "source": "optional-source"}
  },
  "input": {
    "task": "open the drawer",
    "items": []
  },
  "target": null,
  "infer": null,
  "prediction": null,
  "execution": null
}
```

不再保存显式 `stage`：`execution == null` 表示 sampled Record，`execution` 存在表示 Infer 已执行。
文件位置进一步明确状态：`samples.jsonl` 是 sampled，`predictions.jsonl` 是成功结果，`errors.jsonl` 是失败结果。

## 核心字段

| 字段 | 必需 | 含义 |
|---|---:|---|
| `schema_version` | 是 | 当前固定为 `bench.record.v1` |
| `sample_id` | 是 | 一次具体 Infer 请求的稳定主键 |
| `evaluation.type` | 是 | 选择 Metric，例如 `progress` 或 `policy_ranking` |
| `evaluation.dataset.name` | 是 | 数据集身份及 Metric 切片维度 |
| `evaluation.dataset.source` | 否 | 数据集内的来源或子集 |
| `input.task` | 是 | 提供给模型的任务文本，也是任务分组键 |
| `input.items` | 是 | 一个或多个 trajectory/chosen/rejected 输入 |
| `target` | 否 | Stage 1 写入的指标真值，不发送给模型 |
| `infer` | 否 | Stage 2 写入的 Infer 和模型身份 |
| `prediction` | 否 | Stage 2 写入的标准预测 |
| `execution` | 否 | Stage 2 写入的成功或失败状态 |

`sample_id` 不等于 `input.items[].source_id`。一条原始 trajectory 可以派生多个不同采样请求，这些请求共享
source ID，但拥有不同 sample ID。

## input item

```json
{
  "role": "trajectory",
  "frames": {
    "type": "npz",
    "path": "sample_frames/id-trajectory.npz",
    "key": "frames",
    "num_frames": 8,
    "sha256": "..."
  },
  "frame_indices": [0, 4, 8, 12, 16, 20, 24, 29],
  "source_id": "trajectory-001",
  "data": {}
}
```

`role` 区分 trajectory、chosen 和 rejected。`frame_indices` 保留抽帧和时序变换来源；`source_id` 用于定位
原始轨迹；Metric 专用元数据放在 `data`，例如 synthetic temporal 参数或 confusion 的语言/视频任务。

## target 与 prediction

二者使用相同的紧凑 payload，根据 `kind` 解释字段：

```json
{"kind": "progress", "values": [0.0, 0.5, 1.0]}
{"kind": "rank", "value": 1.0, "label": "successful"}
{"kind": "preference", "label": "chosen", "probability": 0.8}
{"kind": "task_match", "value": 1.0}
```

`evaluation.type` 与 `kind` 不能合并：多个评测可以使用相同的 progress payload，但计算不同指标。

## infer

```json
{
  "name": "progress_test",
  "model": "remote-model-or-checkpoint",
  "version": null
}
```

`name` 是注册的实现，`model` 是实际 checkpoint 或服务模型，含义不同。

## execution

成功：

```json
{"status": "success", "error": null, "raw_response": null}
```

失败：

```json
{
  "status": "error",
  "error": "TimeoutError: request timed out",
  "raw_response": null
}
```

错误响应只保存在 `execution.raw_response`。成功 prediction 不重复保存 raw response。

## 状态约束

- `execution == null` 时，`infer` 和 `prediction` 必须同时为空。
- execution 存在时必须有 `infer`。
- `execution.status == success` 时必须有 `prediction`。
- `execution.status == error` 时必须有错误消息，且不能有 `prediction`。

协议不再包含 `source`、`extensions`、`task_id`、`dataset.split`、`num_frames_total`、`attempts` 或
`latency_seconds`。旧扁平 EvaluationRecord 不再自动迁移。
