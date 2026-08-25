# 全流程运行产物说明

本文说明执行 `python -m prmeval.cli run` 后产生的全部文件、各阶段的读写关系、关键字段、断点续跑行为，以及如何移动和验证产物。评测记录的完整字段约束见 [EvaluationRecord 数据结构](RECORD_SCHEMA.md)。

## 产物根目录

默认产物目录由配置中的 `output_dir` 和 `run_name` 决定：

```text
<output_dir>/<run_name>/
```

如果没有设置 `run_name`，目录名使用完整评测配置指纹的前 12 位。默认配置等价于：

```yaml
output_dir: evaluation_output
run_name: null
resume: true
```

一次成功完成三个阶段的运行通常得到：

```text
evaluation_output/<run_name>/
├── samples.jsonl
├── sample_frames/
│   └── <sample_id>-<role>.npz
├── sample_manifest.json
├── predictions.jsonl
├── errors.jsonl                         # 可能不存在或为空
├── inference_summary.json
├── run_manifest.json
├── all_metrics.json
└── <eval_type>/
    └── <dataset_name>_results.json
```

`errors.jsonl`、`predictions.jsonl` 和 Stage 3 产物是否存在取决于实际结果。例如全部推理成功时可能没有 `errors.jsonl`；全部推理失败时没有可供 Stage 3 使用的成功预测，因此不会生成指标产物。

## 阶段与产物关系

| 阶段 | 读取 | 写入 | 用途 |
|---|---|---|---|
| Stage 1：Sample | 原始本地 Dataset；续跑时读取已有 `samples.jsonl` 和 `sample_manifest.json` | `samples.jsonl`、`sample_frames/`、`sample_manifest.json` | 固化模型输入、指标真值及采样配置 |
| Stage 2：Infer | `samples.jsonl`、`sample_frames/`；续跑时读取已有 `predictions.jsonl` 和 `run_manifest.json` | `predictions.jsonl`、`errors.jsonl`、`inference_summary.json`、`run_manifest.json` | 保存逐样本模型输出、错误和运行身份 |
| Stage 3：Metrics | `predictions.jsonl`、`inference_summary.json`、`run_manifest.json` | `all_metrics.json`、`run_manifest.json`、`<eval_type>/*_results.json` | 聚合指标并完成运行清单 |

终端中的进度条和日志不是运行产物：它们写入 stderr。CLI 返回的最终 JSON 摘要写入 stdout，只有显式重定向时才会保存，例如：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml > summary.json
```

## Stage 1 产物

### `samples.jsonl`

每行是一条 `schema_version: bench.record.v1`、`stage: sampled` 的 `EvaluationRecord`。它包含：

- 跨阶段不变的 `sample_id`；
- `evaluation` 中的评测类型和数据集身份；
- 提供给模型的 `input.task` 和一个或多个 `input.items`；
- 只供指标使用、不会发给模型的 `target`；
- 原始轨迹标识和扩展元数据。

图像帧不会嵌入 JSONL。`input.items[].frames` 是指向 `sample_frames/` 的相对引用：

```json
{
  "type": "npz",
  "path": "sample_frames/8ee2d2-trajectory.npz",
  "key": "frames",
  "num_frames": 3,
  "sha256": "<64-character-sha256>"
}
```

Progress 类样本满足以下对应关系：

```text
frames.num_frames = len(frame_indices) = len(target.values)
```

Preference 类样本通常包含 `chosen` 和 `rejected` 两个 item，各自拥有独立的 NPZ 文件。

### `sample_frames/*.npz`

每个文件保存一个样本输入项抽取后的帧数组，默认数组键为 `frames`。文件名由 `sample_id` 和输入角色组成，例如：

```text
<sample_id>-trajectory.npz
<sample_id>-chosen.npz
<sample_id>-rejected.npz
```

加载 Stage 1 产物时会验证：

- JSONL 中的路径必须保持在产物目录内；
- NPZ 文件存在且 SHA-256 与引用一致；
- 数组键和帧数量正确；
- progress 真值长度与帧数一致。
- synthetic temporal 样本允许 `frame_indices` 重复或非单调；索引、帧和 progress target 必须逐项对齐。

因此复制或归档 Stage 1 产物时，必须整体保留 `samples.jsonl` 与 `sample_frames/` 的相对目录结构。

### `sample_manifest.json`

记录采样协议版本、sampling 配置指纹、创建时间、完整 sampling 配置和 Stage 1 汇总：

```json
{
  "schema_version": "prmeval.sample-manifest.v2",
  "fingerprint": "<sampling-config-sha256>",
  "created_at": "2026-08-13T00:00:00+00:00",
  "sampling": {
    "dataset_name": "rbm-1m-ood",
    "adapter": "jsonl",
    "eval_types": ["reward_alignment"],
    "base_frames": 8
  },
  "summary": {
    "schema_version": "bench.record.v1",
    "samples": 10,
    "eval_types": {"reward_alignment": 10},
    "trajectories": 10,
    "fingerprint": "<sampling-config-sha256>",
    "reused": false,
    "path": "evaluation_output/example/samples.jsonl"
  }
}
```

示例省略了部分带默认值的 sampling 字段，实际文件会保存完整配置。

## Stage 2 产物

### `predictions.jsonl`

只保存推理成功的记录。每行仍是同一个 `EvaluationRecord`，并保持 Stage 1 的 `sample_id`、`evaluation`、`input` 和 `target` 不变，同时增加：

```json
{
  "stage": "inferred",
  "infer": {
    "name": "progress_test",
    "model": "your-model",
    "version": null
  },
  "prediction": {
    "kind": "progress",
    "values": [0.0, 0.5, 1.0],
    "data": {"raw_response": null}
  },
  "execution": {
    "status": "success",
    "attempts": 1,
    "latency_seconds": 1.25,
    "error": null
  }
}
```

Runner 按 `samples.jsonl` 的待执行记录顺序逐样本调用 `infer.predict()`；写入顺序与本次输入顺序一致。跨阶段关联仍应使用 `sample_id`，不要把行号当作协议主键。

### `errors.jsonl`

保存推理失败的完整 `EvaluationRecord`。失败记录的 `execution.status` 为 `error`，并包含异常类型、错误消息、请求次数和耗时；`prediction` 为 `null`。如果远程服务已经返回结果、但响应解析或 schema 校验失败，完整服务响应会保存在 `execution.raw_response`；连接阶段失败、没有响应时该字段为 `null`。

在 `resume: true` 下，失败样本会在下次运行中重试。历史错误行不会被删除，因此同一个 `sample_id` 可能出现多次；一旦该样本成功，其成功记录会写入 `predictions.jsonl`，并且不再计入当前未解决失败数。审计历史失败时读取 `errors.jsonl`，判断当前状态时以 `predictions.jsonl` 和 `inference_summary.json` 为准。

### `inference_summary.json`

保存 Stage 2 的覆盖率和输入输出定位：

```json
{
  "coverage": {
    "successful": 9,
    "failed": 1,
    "new": 3,
    "skipped": 7
  },
  "fingerprint": "<full-config-sha256>",
  "samples": "evaluation_output/example/samples.jsonl",
  "predictions": "evaluation_output/example/predictions.jsonl"
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `successful` | `predictions.jsonl` 中累计成功样本数 |
| `failed` | 尚未被成功结果覆盖的失败样本数 |
| `new` | 本次实际执行的样本数，包含成功和失败 |
| `skipped` | 本次因已有成功结果而跳过的样本数 |

### `run_manifest.json`

Stage 2 首次启动时创建运行清单，用来阻止不兼容产物写入同一目录。主要字段包括：

- `run_id`：产物目录名；
- `fingerprint`：完整 `EvalConfig` 的 SHA-256；
- `status`：Stage 2 期间为 `running`，Stage 3 成功完成后为 `completed`；
- `config`：运行配置快照，其中 API key 和敏感 header 值会脱敏；
- `environment`：Python 和平台信息；
- `model_info`：infer/model 信息、采样文件路径及其 SHA-256；
- `summary`：Stage 3 完成后写入的最终指标汇总；
- `created_at`、`completed_at`：创建和完成时间。

如果完整配置指纹或 `samples.jsonl` 文件 SHA-256 与已有清单不一致，运行会拒绝继续向该目录混写。

## Stage 3 产物

### `all_metrics.json`

这是标准的最终指标汇总，包含按配置计算的全部指标、推理覆盖率、配置指纹和预测来源：

```json
{
  "metrics": {
    "reward_alignment": {
      "mse": 0.02,
      "loss": 0.02,
      "pearson": 0.95,
      "num_samples": 9,
      "slices": {},
      "details": {}
    }
  },
  "coverage": {
    "successful": 9,
    "failed": 1,
    "new": 3,
    "skipped": 7
  },
  "fingerprint": "<full-config-sha256>",
  "predictions": "evaluation_output/example/predictions.jsonl"
}
```

不同 eval type 的指标字段不同，具体定义见 [三阶段评测流程](PIPELINE.md#stage-3指标计算)。当使用默认 `predictions.jsonl` 时，coverage 继承自 `inference_summary.json`。
`run` 和 `metrics` 命令在终端只打印汇总指标，不打印逐样本的 `details` 或逐任务的 `task_details`；这些详细结果仍完整保存在 `all_metrics.json` 中。

### `<eval_type>/<dataset_name>_results.json`

Stage 3 还会为每种评测写出兼容旧消费端的扁平结果。例如：

```text
reward_alignment/rbm-1m-ood_results.json
policy_ranking/rbm-1m-ood_results.json
quality_preference/rbm-1m-ood_results.json
confusion_matrix/rbm-1m-ood_results.json
```

文件是 JSON 数组，包含原始 ID、任务、数据源、标签、元数据、真值和预测。Progress 预测使用 `progress_pred`；preference 预测使用 `prediction_prob` 和 `preference_pred`。新工具应优先消费标准的 `predictions.jsonl` 与 `all_metrics.json`，兼容结果主要用于已有分析脚本。

Stage 3 完成后还会更新 `run_manifest.json`：将状态设为 `completed`，填写 `completed_at`，并把 `all_metrics.json` 的内容写入 `summary`。

## 自定义输入输出路径

单独运行阶段时可以覆盖默认路径：

```bash
python -m prmeval.cli sample \
  --config configs/eval/progress_test_remote.yaml \
  --output /tmp/my_samples.jsonl

python -m prmeval.cli infer \
  --config configs/eval/progress_test_remote.yaml \
  --samples /tmp/my_samples.jsonl \
  --output /tmp/my_predictions.jsonl

python -m prmeval.cli metrics \
  --config configs/eval/progress_test_remote.yaml \
  --predictions /tmp/my_predictions.jsonl
```

自定义 Stage 1 输出会在同一目录生成：

```text
my_samples.jsonl
my_samples.manifest.json
sample_frames/
```

自定义 Stage 2 输出会在同一目录生成：

```text
my_predictions.jsonl
my_predictions.errors.jsonl
my_predictions.summary.json
my_predictions.manifest.json
```

单独执行 `metrics --predictions` 时，`all_metrics.json` 和兼容结果仍写入配置的 `<output_dir>/<run_name>/`，不会自动写到传入预测文件旁边。此时如果预测文件不是 Evaluator 的默认路径，coverage 会按读到的成功记录重新构造，`failed`、`new` 和 `skipped` 均为 `0`。

## 断点续跑与产物生命周期

当 `resume: true` 时：

1. Stage 1 在 `samples.jsonl` 和采样清单同时存在且 sampling 指纹一致时直接复用；
2. Stage 2 从 `predictions.jsonl` 收集已成功的 `sample_id` 并跳过；
3. 仅失败或尚未执行的样本进入本轮推理；
4. Stage 3 始终根据当前全部成功预测重新计算指标。

当 `resume: false` 时，Stage 1 重新生成采样产物，Stage 2 在通过运行身份检查后清空默认预测和错误文件，再执行全部样本。建议不同实验使用不同 `run_name`，不要依靠关闭 resume 在同一目录中混用不同配置。

若 Stage 2 没有任何成功预测，`run` 会跳过 Stage 3；此时没有新的 `all_metrics.json`，运行清单状态也不会更新为 `completed`。

## 验证、移动与清理

验证 Stage 1 bundle：

```bash
python -m prmeval.cli validate-samples \
  --samples evaluation_output/<run_name>/samples.jsonl
```

验证成功预测的 schema 和联合身份唯一性：

```bash
python -m prmeval.cli validate-predictions \
  --predictions evaluation_output/<run_name>/predictions.jsonl
```

移动或归档时建议整体复制 `<output_dir>/<run_name>/`。如果只保留可复算指标的最小集合：

- 重新推理需要 `samples.jsonl`、`sample_frames/` 和对应采样清单；
- 重新计算指标只需要 `predictions.jsonl` 和匹配的配置；
- 审计一次完整运行应保留整个 run 目录，尤其是两个 manifest 和 `errors.jsonl`。

运行目录可能包含模型原始响应、任务文本、数据集元数据、服务地址和错误消息，不应提交到 Git 或发布到不可信位置。框架会脱敏清单中的常见认证字段。成功的 progress prediction 不保存远程原始响应；错误记录和 preference baseline 的扩展结果仍可能包含服务响应或模型元数据。
