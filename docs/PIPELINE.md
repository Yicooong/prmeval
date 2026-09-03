# 三阶段评测流程

PRMEval 将一次评测拆成三个可独立运行和验证的阶段。Stage 1 与 Stage 2 通过 `bench.record.v1` JSONL 协议连接，Stage 3 只读取推理成功的记录。

```text
本地 Dataset
    │
    ▼
Stage 1: sample
    │  samples.jsonl + sample_frames/*.npz
    ▼
EvaluationRecord
    │
    ▼
Stage 2: infer
    │  predictions.jsonl / errors.jsonl
    ▼
EvaluationRecord(execution.status="success|error")
    │
    ▼
Stage 3: metrics
    │
    ▼
metrics.json + metrics_detail.jsonl
```

完整字段、必填规则和 JSON 示例见 [EvaluationRecord 数据结构](RECORD_SCHEMA.md)。

## Stage 1：数据采样

Stage 1 负责：

- 通过 `EvalSampler.pool` 从本地 Hugging Face Dataset 生成统一内部轨迹；
- 按 eval type 选择轨迹和图像帧；
- 构造指标真值 `target`；
- 将帧保存为 NPZ，在 Record 中只保留 `FrameReference`；
- 写入尚无 `execution` 的 sampled Record 到 `samples.jsonl`。

内部流程为：

```text
EvalSampler.pool (local Hugging Face Dataset)
    -> Trajectory
    -> EvalSampler.sample()
    -> ProgressSample / PreferenceSample
    -> EvaluationRecord
    -> samples.jsonl + sample_frames/*.npz
```

当前 v1 不包含 prefix sampling。一条 `sample_id` 对应一次模型请求和一条完整预测曲线。

对于 `progress`，采样器选择可用的成功轨迹，按 `base_frames` 均匀抽帧，并使用相同帧索引构造 `target.progress`。默认的 `absolute_first_frame` 定义为：

```text
progress = (frame_index - first_index) / (total_frames - first_index - 1)
```

因此第一帧为 `0`，最后一帧为 `1`。每条记录必须满足：

```text
NPZ 帧数量 = frame_indices 数量 = target.progress 数量
```

`progress_temporal_variation` 先从成功轨迹得到一条基准采样序列，再通过索引映射派生停滞、变速、
回退、重试、截断和跳帧样本。每个合成帧的 target 都直接查找其原始帧 progress；重复索引复制 target，
反向索引产生下降 target，不会按新视频的时间位置重新标注。变换类型、参数、基准/最终帧数和长度比例保存在
input item 的 `synthetic_temporal` metadata 中。默认长度限制为基准帧数的 70%～170%，并继续受
`sampling.temporal_robustness.max_frames` 硬上限约束；其中 `sampling.base_frames` 始终表示变换前的采样数量。

图片不会直接写入 JSONL。移动采样产物时，必须整体移动 `samples.jsonl` 和 `sample_frames/`，以保留相对路径及 SHA-256 校验关系。

运行并验证 Stage 1：

```bash
python -m prmeval.cli sample --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli validate-samples \
  --samples evaluation_output/jsonl-progress-full-smoke/samples.jsonl
```

## Stage 2：单入口模型推理

Stage 2 只接收尚无 `execution` 的 sampled Record。它加载 NPZ 帧，通过 registry 直接构造具体 baseline，并按样本顺序调用统一的 `predict()`：

```text
Evaluator.infer()
    -> infer_cls = INFERS.get(config.infer.name)
    -> infer = infer_cls(config.infer)
    -> record_to_sample(record, bundle_dir)
    -> infer.predict(sample)
    -> ProgressPrediction / PreferencePrediction
    -> EvaluationRecord(execution.status="success|error")
```

只有一个抽象父类 `Infer`。框架不区分 local 和 remote，也不提供 adapter、公共 batch、线程池或执行模式分派。checkpoint 加载、provider SDK 和 HTTP 调用均由具体 baseline 自己处理。Runner 只理解 `capabilities` 与标准 Prediction。

Progress baseline 的 `predict()` 校验 `ProgressSample`，把 frames、task 和可选 reference path 传给本类 `compute_progress()`，再通过共享函数构造 `ProgressPrediction`。共享校验要求输出一维、与输入帧等长、有限并位于 `[0,1]`。RLVLMF 是当前唯一 preference baseline，其 `predict()` 调用 `compute_preference()` 并构造 `PreferencePrediction`。

标准化结果示例：

```json
{
  "infer": {"name": "progress_test", "model": "your-model"},
  "prediction": {"kind": "progress", "values": [0.0, 0.5, 1.0]},
  "execution": {"status": "success"}
}
```

成功结果写入 `predictions.jsonl`；失败结果写入 `errors.jsonl`。单个 sample 失败不会中断或污染其他样本。成功的 progress prediction 不保存远程 raw response；远程失败可通过 `RemoteError.raw_response` 写入错误记录。

`progress_test` 和 SOLE-R1 在模型实例内部组合 `OpenAIChatClient`，但对 Runner 只暴露普通 `predict()`/`compute_progress()`。模型内部确实需要的 prefix 或 tensor micro-batch 是私有实现细节，不参与 Runner 调度。

Stage 2 不会重新抽帧。普通采样的模型输入帧数由 Stage 1 的 `sampling.base_frames` 控制；时序鲁棒样本还会受 `sampling.temporal_robustness.max_frames` 的最终硬上限约束。这样模型输入、target 和 progress prediction 始终一一对应。接口与注册示例见 [Infer 模型接入](INFER_MODELS.md)，连接和模型字段见 [配置文件说明](CONFIGURATION.md#infer)。

查看已注册 infer：

```bash
python -m prmeval.cli list-infers
```

运行并验证 Stage 2：

```bash
python -m prmeval.cli infer --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli validate-predictions \
  --predictions evaluation_output/jsonl-progress-full-smoke/predictions.jsonl
```

## Stage 3：指标计算

Stage 3 只读取满足以下条件的记录：

```text
execution.status = success
```

它不读取原始 dataset、不加载 NPZ，也不调用模型。聚合结果写入 `metrics.json`；完整 Record 与逐条指标写入
`metrics_detail.jsonl`。需要联合多条 Record 的指标还会写入 `detail_type: group` 的分组明细。

当前内置评测包括：

| 评测 | 输入 | 指标 |
|---|---|---|
| `progress` | target progress 与 prediction progress | MSE、Pearson |
| `progress_temporal_variation` | 合成变化后的逐帧 progress 与 prediction | MAE、趋势、回退、平台、终点、单调性与时间捷径 |
| `policy_ranking` | 任务内质量排序与预测终态 progress | Kendall |
| `quality_preference` | chosen/rejected 轨迹偏好 | Accuracy |
| `confusion_matrix` | 语言任务与视频任务匹配结果 | 混淆矩阵 |

`progress` 对每条样本分别计算 MSE 和 Pearson，再对样本等权平均，并按 `evaluation.dataset.name` 和 `infer.name` 切片。

Policy ranking 使用 `target(kind=rank).value`、`prediction(kind=progress).values` 和 `input.task`，按 dataset、infer 和 task 分组后计算 Kendall。

运行 Stage 3：

```bash
python -m prmeval.cli metrics --config configs/eval/progress_test_remote.yaml
```

也可以不调用模型，直接从已有预测重新计算指标：

```bash
python -m prmeval.cli compute-metrics \
  --predictions examples/stage_3_smoke/predictions.jsonl \
  --metrics progress \
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

原始轨迹编号放在 `input.items[].source_id`，用于审计和定位，但不参与核心去重或指标分组。

## 运行产物与断点续跑

各文件的完整字段、生成条件、自定义输出命名、移动与验证方式见 [全流程运行产物说明](ARTIFACTS.md)。

一次完整运行通常生成：

```text
evaluation_output/<run_name>/
├── samples.jsonl
├── sample_frames/
├── predictions.jsonl
├── errors.jsonl
├── metrics.json
└── metrics_detail.jsonl
```

当 `resume: true` 时：

- Stage 1 验证并复用已有 `samples.jsonl`；
- Stage 2 跳过已经成功的 `sample_id`；
- 失败样本可以在下次运行时重试；
- Stage 3 根据当前全部成功 Record 重写两个指标文件。

框架不再保存或比较配置指纹。数据、采样配置或模型配置发生变化时，应使用新的 `run_name`，避免向同一目录混写。

运行产物默认位于 `evaluation_output/`，该目录不应提交到 Git。

## 连续执行

连续运行三个阶段：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml
```

顶层 `mode` 控制 `run` 如何连接三个阶段：

- `separate`（默认）依次生成并读取 `samples.jsonl` 与 `sample_frames/*.npz`，适合阶段解耦和移动产物；
- `continue` 从 sampler 迭代器按 `infer.batch_size` 取样本并直接推理，不生成 sample JSONL、sample manifest 或 NPZ。

连续模式的帧数组只存在于当前推理批次的内存中。写入 predictions/errors 前，Runner 会把
`input.items[].frames` 置为空列表；frame indices、总帧数、target、source 和 metadata 仍会保留。连续模式仍将预测、错误、
manifest 和 metrics 写到运行目录，`resume: true` 时重新生成稳定 sample ID 并跳过已有成功预测。单独执行
`sample`、`infer` 或 `metrics` 不受 `mode` 影响，始终使用磁盘阶段协议。

CLI 默认向 stderr 输出阶段日志，并在交互式终端中展示各阶段进度：Stage 1 统计读取轨迹、生成样本和写入样本，Stage 2 统计已完成的推理样本，Stage 3 统计已计算的指标。断点续跑时，Stage 2 会同时报告待处理和已跳过的样本数。使用 `--no-progress` 可以关闭动态进度条；普通阶段日志不受影响。非交互式 stderr（例如 CI 或输出重定向）会自动禁用动态条，避免产生重复控制字符。

进度和日志使用 stderr，最终 JSON 摘要使用 stdout。例如下面的命令只将摘要写入文件：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml > summary.json
```

也可以通过 Python API 调用：

```python
from prmeval import EvalConfig, Evaluator

config = EvalConfig.from_yaml("configs/eval/progress_test_remote.yaml")
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
