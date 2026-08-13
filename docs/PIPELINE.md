# 三阶段评测流程

PRMEval 将一次评测拆成三个可独立运行和验证的阶段。Stage 1 与 Stage 2 通过 `bench.record.v1` JSONL 协议连接，Stage 3 只读取推理成功的记录。

```text
本地 Dataset
    │
    ▼
Stage 1: sample
    │  samples.jsonl + sample_frames/*.npz
    ▼
EvaluationRecord(stage="sampled")
    │
    ▼
Stage 2: infer
    │  predictions.jsonl / errors.jsonl
    ▼
EvaluationRecord(stage="inferred")
    │
    ▼
Stage 3: metrics
    │
    ▼
all_metrics.json + 分数据集结果
```

完整字段、必填规则和 JSON 示例见 [EvaluationRecord 数据结构](RECORD_SCHEMA.md)。

## Stage 1：数据采样

Stage 1 负责：

- 通过 Dataset Adapter 将不同数据源转换为统一内部轨迹；
- 按 eval type 选择轨迹和图像帧；
- 构造指标真值 `target`；
- 将帧保存为 NPZ，在 Record 中只保留 `FrameReference`；
- 写入 `stage: sampled` 的 `samples.jsonl`。

内部流程为：

```text
DatasetAdapter
    -> Trajectory
    -> EvalSampler
    -> ProgressSample / PreferenceSample
    -> EvaluationRecord(stage="sampled")
    -> samples.jsonl + sample_frames/*.npz
```

当前 v1 不包含 prefix sampling。一条 `sample_id` 对应一次模型请求和一条完整预测曲线。

对于 `reward_alignment`，采样器选择可用的成功轨迹，按 `max_frames` 均匀抽帧，并使用相同帧索引构造 `target.progress`。默认的 `absolute_first_frame` 定义为：

```text
progress = (frame_index - first_index) / (total_frames - first_index - 1)
```

因此第一帧为 `0`，最后一帧为 `1`。每条记录必须满足：

```text
NPZ 帧数量 = frame_indices 数量 = target.progress 数量
```

图片不会直接写入 JSONL。移动采样产物时，必须整体移动 `samples.jsonl` 和 `sample_frames/`，以保留相对路径及 SHA-256 校验关系。

运行并验证 Stage 1：

```bash
python -m prmeval.cli sample --config configs/eval/test_stage.yaml
python -m prmeval.cli validate-samples \
  --samples evaluation_output/jsonl-progress-full-smoke/samples.jsonl
```

## Stage 2：远程推理

Stage 2 只接收 `EvaluationRecord(stage="sampled")`。它加载 NPZ 帧，调用 infer adapter，并在原记录上补充：

```text
infer
prediction
execution
stage = inferred
```

远程请求只使用 `input.task`、`input.items[].frames` 和 infer adapter 的 prompt/请求参数，`target` 不会发送给模型。`sample_id`、`evaluation`、`input` 和 `target` 在推理过程中不得改变。

标准化后的结果示例：

```json
{
  "infer": {"name": "progress_test", "model": "your-model"},
  "prediction": {"kind": "progress", "values": [0.0, 0.5, 1.0]},
  "execution": {"status": "success", "attempts": 1}
}
```

成功结果写入 `predictions.jsonl`，失败结果写入 `errors.jsonl`。progress 输出数量与输入帧数不一致时，该样本会被记录为失败。

### Infer 协议

`progress_test` 用于联调通用 OpenAI-compatible 模型，只支持 progress 样本。它调用 `POST /v1/chat/completions`，以 Base64 多图片 `image_url` 发送帧，并通过 JSON Schema 约束返回等长的 progress 数组：

```json
{"progress": [0.0, 0.5, 1.0]}
```

GVL、RL-VLM-F、RoboReward 和 RoboDopamine 同样使用 OpenAI-compatible `/v1/chat/completions`。RBM/ReWiND、TOPReward 和 VLAC 使用专用的 `POST /v1/evaluations` 协议，其 request/response schema 位于 `prmeval.infer.specialized`。

服务地址、认证、模型、并发和 prompt 配置见 [配置文件说明](CONFIGURATION.md#infer)。

可以启动本地 contract mock 或查看已注册 infer：

```bash
python -m prmeval.infer.mock_server --port 8765
python -m prmeval.cli list-infers
```

运行并验证 Stage 2：

```bash
python -m prmeval.cli infer --config configs/eval/test_stage.yaml
python -m prmeval.cli validate-predictions \
  --predictions evaluation_output/jsonl-progress-full-smoke/predictions.jsonl
```

## Stage 3：指标计算

Stage 3 只读取满足以下条件的记录：

```text
stage = inferred
execution.status = success
```

它不读取原始 dataset、不加载 NPZ，也不调用模型。Metric 不修改 `EvaluationRecord.stage`；指标完成状态记录在 `all_metrics.json` 和 `run_manifest.json` 中。

当前内置评测包括：

| 评测 | 输入 | 指标 |
|---|---|---|
| `reward_alignment` | target progress 与 prediction progress | MSE、Pearson |
| `policy_ranking` | 任务内质量排序与预测终态 progress | Kendall |
| `quality_preference` | chosen/rejected 轨迹偏好 | Accuracy |
| `confusion_matrix` | 语言任务与视频任务匹配结果 | 混淆矩阵 |

`reward_alignment` 对每条样本分别计算 MSE 和 Pearson，再对样本等权平均，并按 `evaluation.dataset.name` 和 `infer.name` 切片。

Policy ranking 使用 `target(kind=rank).value`、`prediction(kind=progress).values` 和 `input.task_id`，按 dataset、infer 和 task 分组后计算 Kendall。

运行 Stage 3：

```bash
python -m prmeval.cli metrics --config configs/eval/test_stage.yaml
```

也可以不调用模型，直接从已有预测重新计算指标：

```bash
python -m prmeval.cli compute-metrics \
  --predictions examples/stage_3_smoke/predictions.jsonl \
  --metrics reward_alignment \
  --output /tmp/prmeval-metrics.json
```

## 跨阶段标识

`sample_id` 是跨阶段主键：

```text
Stage 1 sample_id -> Stage 2 sample_id -> Stage 3 明细 sample_id
```

单个 run 只对应一个 infer，因此 `sample_id` 在该 run 的 predictions 文件中唯一。合并多个 run 时使用联合身份：

```text
(dataset.name, infer.name, sample_id)
```

原始轨迹编号可以放在 `source.id`，用于审计和定位，但不参与核心去重与 reward alignment 分组。

## 运行产物与断点续跑

各文件的完整字段、生成条件、自定义输出命名、移动与验证方式见 [全流程运行产物说明](ARTIFACTS.md)。

一次完整运行通常生成：

```text
evaluation_output/<run_name>/
├── samples.jsonl
├── sample_frames/
├── sample_manifest.json
├── predictions.jsonl
├── errors.jsonl
├── inference_summary.json
├── run_manifest.json
├── all_metrics.json
└── reward_alignment/
    └── <dataset>_results.json
```

当 `resume: true` 时：

- Stage 1 使用 sampling 指纹判断是否可以复用已有 samples；
- Stage 2 跳过已经成功的 `sample_id`；
- 失败样本可以在下次运行时重试；
- 配置指纹不一致时会拒绝向同一输出目录混写。

运行产物默认位于 `evaluation_output/`，该目录不应提交到 Git。

## 连续执行

连续运行三个阶段：

```bash
python -m prmeval.cli run --config configs/eval/test_stage.yaml
```

CLI 默认向 stderr 输出阶段日志，并在交互式终端中展示各阶段进度：Stage 1 统计读取轨迹、生成样本和写入样本，Stage 2 统计已完成的推理样本，Stage 3 统计已计算的指标。断点续跑时，Stage 2 会同时报告待处理和已跳过的样本数。使用 `--no-progress` 可以关闭动态进度条；普通阶段日志不受影响。非交互式 stderr（例如 CI 或输出重定向）会自动禁用动态条，避免产生重复控制字符。

进度和日志使用 stderr，最终 JSON 摘要使用 stdout。例如下面的命令只将摘要写入文件：

```bash
python -m prmeval.cli run --config configs/eval/test_stage.yaml > summary.json
```

也可以通过 Python API 调用：

```python
from prmeval import EvalConfig, Evaluator

config = EvalConfig.from_yaml("configs/eval/test_stage.yaml")
evaluator = Evaluator(config)

sample_summary = evaluator.sample()
infer_summary = evaluator.infer()
metric_summary = evaluator.evaluate_metrics()
```

连续执行：

```python
summary = Evaluator(config).run()
```

Python API 默认不显示进度。如需在交互式 Python 终端中开启，可以显式传入：

```python
summary = Evaluator(config, show_progress=True).run()
```
