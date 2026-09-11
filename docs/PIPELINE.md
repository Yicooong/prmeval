# 三阶段评测流程

PRMEval 将一次评测拆成三个可独立运行和验证的阶段。阶段之间可通过内存传递，也可通过 `EvaluationRecord` 文件连接；Stage 3 只使用推理成功的记录。

```text
本地 Dataset
    │
    ▼
Stage 1: sample
    │  samplers 或 samples.jsonl + sample_frames/*.npz
    ▼
EvaluationRecord
    │
    ▼
Stage 2: infer
    │  成功记录；可选写入 predictions.jsonl / errors.jsonl
    ▼
EvaluationRecord(execution.status="success|error")
    │
    ▼
Stage 3: metrics
    │
    ▼
完整指标；可选写入 metrics.json + metrics_detail.jsonl
```

完整字段、必填规则和 JSON 示例见 [EvaluationRecord 数据结构](RECORD_SCHEMA.md)。

## Stage 1：数据采样

Stage 1 准备共享同一 trajectory pool 的 samplers；开启采样落盘时还会流式执行以下采样及写入步骤：

- 通过 `EvalSampler.pool` 从 JSONL 文件或本地 Hugging Face Dataset 生成统一内部轨迹；
- 按 eval type 选择轨迹和图像帧；
- 在 Sample 的轨迹中保存真值标签；
- 将帧保存为 NPZ，在 Record 中只保留 `FrameReference`；
- 写入尚无 `execution` 的 sampled Record 到 `samples.jsonl`。

内部流程为：

```text
EvalSampler.pool (JSONL / local Hugging Face Dataset)
    -> Trajectory
    -> EvalSampler.sample()
    -> ProgressSample / PreferenceSample
    -> EvaluationRecord(eval_type=sampler.eval_type, sample=sample)
    -> samples.jsonl + sample_frames/*.npz
```

未开启采样落盘时，实际抽帧和样本生成由 Stage 2 按 batch 消费 sampler 时触发，Stage 1 不提前统计样本数。

当前不包含 prefix sampling。一条 `sample_id` 对应一次模型请求和一条完整预测曲线。

对于 `progress`，采样器选择可用的成功轨迹，按 `base_frames` 均匀抽帧，并使用相同帧索引构造
`sample.trajectory.target_progress`。默认的 `absolute_first_frame` 定义为：

```text
progress = (frame_index - first_index) / (total_frames - first_index - 1)
```

因此第一帧为 `0`，最后一帧为 `1`。每条记录必须满足：

```text
NPZ 帧数量 = frame_indices 数量 = target_progress 数量
```

`progress_temporal_variation` 先从成功轨迹得到一条基准采样序列，再通过索引映射派生停滞、变速、
回退、重试、截断和跳帧样本。每个合成帧的 target 都直接查找其原始帧 progress；重复索引复制 target，
反向索引产生下降 target，不会按新视频的时间位置重新标注。变换类型、参数、基准/最终帧数和长度比例保存在
`sample.trajectory.metadata.synthetic_temporal` 中。默认长度限制为基准帧数的 70%～170%，并继续受
`sampling.temporal_robustness.max_frames` 硬上限约束；其中 `sampling.base_frames` 始终表示变换前的采样数量。

图片不会直接写入 JSONL。移动采样产物时，必须整体移动 `samples.jsonl` 和 `sample_frames/`，以保留相对路径及 SHA-256 校验关系。

运行并验证 Stage 1（先设置 `sampling.save_samples: true`、`output_dir: evaluation_output` 和 `task_name: openai-compatible-remote`）：

```bash
prmeval sample --config configs/eval/openai_compatible_remote.yaml
prmeval validate-samples \
  --samples evaluation_output/openai-compatible-remote/samples.jsonl
```

## Stage 2：单入口模型推理

Stage 2 优先读取显式样本文件，其次读取已有的当前/默认样本文件；只有没有文件输入时才消费 `self.samplers`。
文件来源只接受尚无 `execution` 的 sampled Record，加载 NPZ 帧后直接读取 record.sample，绝不再次采样。
两种来源共用 batch 推理逻辑，通过 registry 直接构造 baseline，并按 `infer.batch_size` 调用 `predict(samples)`：

```text
Evaluator.infer()
    -> infer_cls = INFERS.get(config.infer.name)
    -> infer = infer_cls(config.infer)
    -> runtime_record = load_record_frames(record, bundle_dir)
    -> runtime_record.sample
    -> infer.predict(samples)
    -> list[ProgressPrediction | PreferencePrediction]
    -> EvaluationRecord(execution.status="success|error")
```

只有一个抽象父类 `Infer`。框架不区分 local 和 remote，也不提供 adapter、线程池或 transport 分派。checkpoint
加载、provider SDK 和 HTTP 调用均由具体 baseline 自己处理。Runner 只理解 `capabilities`、批量 `predict()`
和标准 Prediction。

Progress baseline 的 `predict()` 接收样本列表，并为每个样本构造一个 `ProgressPrediction`。输出必须与输入数量一致，
`sample_id` 集合完全相同且不得重复；每条 progress 数组还必须与对应输入帧等长、有限并位于 `[0,1]`。
当前内置 baseline 都只声明 `progress` 能力；运行 `quality_preference` 需要自行接入声明 `preference` 能力的模型。

向采样记录补充的推理结果字段：

```json
{
  "prediction": {"sample_id": "sample-id", "model": "your-model", "progress": [0.0, 0.5, 1.0]},
  "execution": {"status": "success", "infer_name": "openai_compatible"}
}
```

有输出目录时，成功结果写入 `predictions.jsonl`；失败结果写入 `errors.jsonl`。一个批次抛出异常，或者返回数量/ID 不合法时，
该批次的所有样本都会记为失败，后续批次继续执行。成功的 progress prediction 不保存远程 raw response；远程失败可通过
异常的 `raw_response` 属性写入错误记录。

`openai_compatible` 在模型实例内部使用官方 OpenAI Python SDK。模型内部需要的 prefix 或 tensor micro-batch 是 baseline
私有实现细节，与 Runner 的 `infer.batch_size` 分组相互独立。

文件输入路径不会重新抽帧；sampler 输入路径在此阶段按需抽帧。普通采样的模型输入帧数由 Stage 1 的 `sampling.base_frames` 控制；时序鲁棒样本还会受 `sampling.temporal_robustness.max_frames` 的最终硬上限约束。这样模型输入、target 和 progress prediction 始终一一对应。接口与注册示例见 [Infer 模型接入](INFER_MODELS.md)，连接和模型字段见 [配置文件说明](CONFIGURATION.md#infer)。

查看已注册 infer：

```bash
prmeval list-infers
```

运行并验证 Stage 2：

```bash
prmeval infer --config configs/eval/openai_compatible_remote.yaml
prmeval validate-predictions \
  --predictions evaluation_output/openai-compatible-remote/predictions.jsonl
```

## Stage 3：指标计算

Stage 3 只读取满足以下条件的记录：

```text
execution.status = success
```

它不读取原始 dataset、不加载 NPZ，也不调用模型。有输出目录时，聚合结果写入 `metrics.json`；完整 Record 与逐条指标写入
`metrics_detail.jsonl`。需要联合多条 Record 的指标还会写入 `detail_type: group` 的分组明细。

当前内置评测包括：

| 评测 | 输入 | 指标 |
|---|---|---|
| `progress` | target progress 与 prediction progress | MSE、Pearson |
| `progress_temporal_variation` | 合成变化后的逐帧 progress 与 prediction | MAE、趋势、回退、平台、终点、单调性与时间捷径 |
| `policy_ranking` | 任务内质量排序与预测终态 progress | Kendall |
| `quality_preference` | chosen/rejected 轨迹偏好 | Accuracy |
| `confusion_matrix` | 语言任务与视频任务匹配结果 | 混淆矩阵 |

`progress` 对每条样本分别计算 MSE 和 Pearson，再对样本等权平均，并按 `sample.dataset_name` 和 `execution.infer_name` 切片。

Policy ranking 从 `sample.trajectory` 依次选择 partial_success、preference_rank 或质量标签等级作为真值，
使用 `prediction.progress` 和 `sample.trajectory.task`，按 dataset、infer 和 task 分组后计算 Kendall。

运行 Stage 3：

```bash
prmeval metrics --config configs/eval/openai_compatible_remote.yaml
```

也可以不调用模型，直接从已有预测重新计算指标：

```bash
prmeval compute-metrics \
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
(sample.dataset_name, execution.infer_name, sample.sample_id)
```

原始轨迹编号放在 `sample.trajectory.id`（偏好样本为两条轨迹各自的 id），用于审计和定位，但不参与核心去重或指标分组。

## 运行产物与断点续跑

各文件的完整字段、生成条件、自定义输出命名、移动与验证方式见 [全流程运行产物说明](ARTIFACTS.md)。

有输出目录且 `sampling.save_samples: true` 时，完整运行的产物结构为：

```text
evaluation_output/<task_name>/
├── samples.jsonl
├── sample_frames/
├── predictions.jsonl
├── errors.jsonl
├── metrics.json
└── metrics_detail.jsonl
```

`resume` 不控制 Stage 1：每次调用 `sample()` 都重新准备 samplers，开启采样落盘时重新生成并覆盖样本文件。
需要复用已有样本时直接调用 `infer(samples_path=...)`。当 `resume: true` 时：
- Stage 2 跳过已经成功的 `sample_id`；
- 失败样本可以在下次运行时重试；
- Stage 3 根据当前全部成功 Record 重写两个指标文件。

框架不再保存或比较配置指纹。数据、采样配置或模型配置发生变化时，应使用新的 `task_name`，避免向同一目录混写。

示例可将运行产物放在 `evaluation_output/`，该目录不应提交到 Git；默认产物根目录为 `/tmp/prmeval_evaluation_output`。

## 连续执行

连续运行三个阶段：

```bash
prmeval run --config configs/eval/openai_compatible_remote.yaml
```

`run()` 固定依次执行三个阶段，没有 mode 分支。`sampling.save_samples` 默认 `false`，
采样器生成的样本按 batch 直接推理；有输出目录时仍写入 predictions、errors 和 metrics。
设为 `sampling.save_samples: true` 后，Stage 1 保存样本 JSONL 和 NPZ，Stage 2 读取该文件，不再消费 sampler。
`output_dir: null` 关闭所有评估产物写入，但仍可读取显式输入文件。

sampler 输入的推理记录会清除帧数组，保留轨迹的采样索引、真值标签、ID 和 metadata；
文件输入的记录保留原始 NPZ 引用。成功记录在内存中保留，供全部推理完成后统一计算跨 batch 指标。
断点续跑只返回当前输入范围内的完整成功结果，全部命中时无需加载模型。

CLI 默认向 stderr 输出阶段日志，并在交互式终端中展示各阶段进度：Stage 1 准备采样器，启用采样落盘时显示样本生成和写入进度，Stage 2 显示样本消费进度，Stage 3 统计已计算的指标。断点续跑时，Stage 2 完成摘要会报告执行和跳过数量。使用 `--no-progress` 可以关闭动态进度条；普通阶段日志不受影响。非交互式 stderr（例如 CI 或输出重定向）会自动禁用动态条，避免产生重复控制字符。

进度和日志使用 stderr，最终 JSON 摘要使用 stdout。例如下面的命令只将摘要写入文件：

```bash
prmeval run --config configs/eval/openai_compatible_remote.yaml > summary.json
```

也可以通过 Python API 调用：

```python
from prmeval import EvalConfig, Evaluator

config = EvalConfig.from_yaml("configs/eval/openai_compatible_remote.yaml")
evaluator = Evaluator(config)

sample_summary = evaluator.sample()
infer_summary, successful_records = evaluator.infer()
metric_summary = evaluator.evaluate_metrics()
```

连续执行：

```python
summary = Evaluator(config).run()
```

Python API 默认在交互式 stderr 中显示进度；非交互环境会自动关闭。也可以显式控制：

```python
summary = Evaluator(config, show_progress=False).run()
```

`evaluate_metrics(predictions_path=None, *, records=None, coverage=None)` 可接收文件或记录列表，二者不可同时指定。
未指定输入时优先使用本实例最近一次推理结果；新实例可读取默认预测文件。没有输入则报错，不自动执行前置阶段。
再次调用 `sample()` 或 `infer()` 会清除旧推理状态；前置阶段失败后不能隐式复用旧指标输入。

`run()` 与 `evaluate_metrics()` 始终返回含 `metrics`、`coverage`、`predictions`、`details` 的字典。
Python 中 `metrics` 保留完整的 `details` 和 `task_details`；磁盘 metrics.json 和 CLI 只输出精简指标。
全部推理失败时返回空 `metrics` 和失败 coverage，有输出目录时同时写入摘要及空明细文件。

## 核心数据模块

- `prmeval/core/schemas.py`: 数据结构和记录自身的状态约束。
- `prmeval/core/conversions.py`: 预测匹配校验和推理记录组装。
- `prmeval/core/storage.py`: 记录和关联帧文件的保存、加载、文件校验及指标详情输出。

Record 在顶层保存 eval_type，并直接包含 Sample 和 Prediction。Sample 不再保存 eval_type。
不再进行样本字段转换或构造通用 target/prediction payload。
`load_record_frames(record, bundle_dir).sample` 获取包含帧数组的样本副本。
`load_sample_records()` 返回保留帧引用的记录，默认完整校验关联文件。
JSONL 格式已改为 eval_type/sample/prediction/execution；旧格式需重新生成或显式转换，NPZ 格式不变。

## 内存中的完整评估

```python
config = EvalConfig.from_yaml("your_config.yaml")  # YAML 中设置 output_dir: null
result = Evaluator(config).run()
progress_details = result["metrics"]["progress"]["details"]
```

不设置输出目录时，采样与推理仍按批次执行。成功 Record 去除帧数组后保留在内存中，
待全部推理完成，再统一校验并调用 `compute_metrics()`，因此跨批次的策略排名分组保持完整。
内存用量随成功记录的非帧数据增长。每次 `run()` 重新准备采样器并执行推理，不读取历史预测检查点；本次结果可供同实例继续调用指标阶段。

文件准备和推理记录追加写入由 storage 负责，预测校验和 Record 组装由 conversions 负责；runner 负责
运行编排及基于成功/失败 ID 的覆盖率统计，具体指标字段仍由 metrics 定义。
