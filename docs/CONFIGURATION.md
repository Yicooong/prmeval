# 配置文件说明

PRMEval 使用一个 YAML 文件描述采样、推理、指标和产物目录。下面示例调用 `openai_compatible` 完成 Stage 1 → Stage 2 → Stage 3：

```yaml
sampling:
  dataset_name: rbm-1m-ood
  paths: [/path/to/hf_datasets/rbm-1m-ood]
  max_trajectories: 1
  eval_types: [progress]
  base_frames: 3
  progress_type: absolute_first_frame

infer:
  name: openai_compatible
  base_url: BASE_URL
  api_key: API_KEY
  model_id: MODEL_ID
  timeout_seconds: 120
  max_retries: 0
  temperature: 0
  max_tokens: 4096
  options: {}

metrics: [progress]
save_samples: false
output_dir: evaluation_output
run_name: progress-full-smoke
resume: false
```

## 顶层配置

| 配置项 | 使用阶段 | 说明 |
|---|---|---|
| `sampling` | Stage 1 | JSONL/Hugging Face Dataset 路径、采样类型、轨迹与帧数限制 |
| `infer` | Stage 2 | baseline 名称、模型/连接信息和扩展参数 |
| `metrics` | Stage 3 | 需要计算的指标名称列表 |
| `save_samples` | Stage 1 | 默认 `false`；设为 `true` 且有输出目录时保存样本 JSONL 和帧 |
| `output_dir` | 全阶段 | 所有 run 的根目录，默认 `null`，表示不写入任何评估产物 |
| `run_name` | 全阶段 | 当前 run 的目录名称 |
| `resume` | Stage 2 | 是否复用已有成功预测并跳过已成功样本；不控制采样阶段 |

三个阶段共用一个 `EvalConfig`。只运行 Stage 1 时仍需保留 `infer` 块，但采样阶段不会构造模型。

## `sampling`

| 字段 | 说明 |
|---|---|
| `dataset_name` | 写入评测记录的数据集名称，也参与 sample ID 构造 |
| `paths` | 一个或多个轨迹 `.jsonl` 文件，或由 `datasets.save_to_disk()` 保存的本地 Dataset 目录 |
| `max_trajectories` | 最多读取的轨迹数 |
| `eval_types` | 需要构造的评测类型 |
| `base_frames` | 基准采样帧数；`progress` 等普通采样直接按此数量抽帧 |
| `progress_type` | progress 真值定义：`absolute_first_frame`、`absolute_wrt_total_frames` 或 `relative_first_frame` |
| `num_examples_per_quality` | `policy_ranking` 每个质量等级最多选择的轨迹数，默认 `5` |
| `num_partial_successes` | `policy_ranking` 使用连续完成度时，每个任务最多选择的轨迹数 |
| `max_tasks` | `policy_ranking` 最多评测的任务数 |
| `comparisons_per_task` | `quality_preference` 每个任务最多生成的比较对数 |
| `max_comparisons` | `quality_preference` 全局最多生成的比较对数 |
| `trajectories_per_source` | `confusion_matrix` 每个数据来源最多选择的轨迹数 |
| `temporal_robustness` | `progress_temporal_variation` 的最终帧数上限、变换类型、数量与参数范围 |

相对路径以运行命令时的当前目录为基准。路径识别和字段要求见 [JSONL 与本地 Hugging Face Dataset](DATASETS.md)。

### Synthetic temporal robustness

该评测从成功轨迹生成 Original、Pause、Slow、Fast、Rewind、Retry、Truncate 和 Skip。配置示例见
`configs/eval/synthetic_temporal_robustness.yaml`。核心配置如下：

```yaml
sampling:
  eval_types: [progress_temporal_variation]
  base_frames: 9                # 变换前的基准采样数量
  progress_type: absolute_first_frame
  temporal_robustness:
    random_seed: 42
    max_frames: 16              # 变换后的最终硬上限
    min_length_ratio: 0.7
    max_length_ratio: 1.7
    transforms: [pause, slow, fast, rewind, retry, truncate, skip]
    variants_per_transform: 3
```

`sampling.base_frames` 表示变换前的采样数量，`sampling.temporal_robustness.max_frames` 表示变换后的最终硬上限。合成序列长度始终位于
`ceil(min_length_ratio × base_frames)` 与
`min(floor(max_length_ratio × base_frames), max_frames)` 之间。默认最多减少 30%、最多增加 70%。Pause、Rewind
和 Retry 需要 `base_frames < max_frames`；该评测要求 `base_frames >= 5`、`temporal_robustness.max_frames >= 6`.

各变换的参数范围可通过 `pause_extra_ratio_range`、`slow_gamma_range`、`fast_gamma_range`、
`peak_progress_range`、`retreat_ratio_range`、`rewind_extra_ratio_range`、`retry_extra_ratio_range`、
`truncate_retained_ratio_range` 和 `skip_removed_ratio_range` 调整。所有随机结果由
`sampling.temporal_robustness.random_seed` 稳定决定。

## `infer`

| 字段 | 说明 |
|---|---|
| `name` | registry 中的 baseline 名称；通过 `list-infers` 查看 |
| `model_path` | checkpoint 路径或 Hugging Face ID；由需要本地 checkpoint 的 baseline 校验 |
| `base_url` | OpenAI-compatible 服务地址；由网络调用型 baseline 校验 |
| `api_key` | API Key 或保存 Key 的环境变量名 |
| `model_id` | 请求使用及记录到产物中的模型身份 |
| `model_version` | 可选模型版本 |
| `timeout_seconds` | 单次请求超时时间 |
| `max_retries` | 请求或响应解析失败后的最大重试次数 |
| `temperature` | 生成温度 |
| `max_tokens` | 单次响应的最大 token 数 |
| `batch_size` | Runner 每次从 sample 迭代器消费并调度的样本数，默认 `1` |
| `headers` | 附加 HTTP header |
| `options` | 传给具体 baseline 的扩展配置 |

除 `name` 外，`infer` 字段均有默认值或允许为空；但具体 baseline 可以在构造时要求额外字段。

框架不区分 local/remote，也不存在 transport 或 max_concurrency 分派。Runner 直接构造
`INFERS.get(name)` 返回的类，并按 `infer.batch_size` 将 `list[EvaluationSample]` 传给统一的 `predict()`。
模型使用 checkpoint、provider SDK 或 HTTP client 由自身实现决定。

配置示例：

```yaml
# checkpoint baseline
infer:
  name: topreward
  model_path: /models/topreward
  model_id: topreward-v1
  options:
    use_prefix_samples: True

# OpenAI-compatible baseline
infer:
  name: openai_compatible
  base_url: BASE_URL
  api_key: API_KEY
  model_id: MODEL_ID
  max_retries: 2
```

`robodopamine` 的模型内部 micro-batch 与运行策略放在 `options`，不属于 Runner 调度：

```yaml
infer:
  name: robodopamine
  model_path: /models/robo-dopamine
  options:
    micro_batch_size: 1
    eval_mode: incremental
    frame_interval: 1
```

推荐只在配置中保存环境变量名。`base_url`、`api_key`、`model_id` 和 `model_path` 中的全大写标识符会被解析为环境变量；变量缺失时配置初始化会报错。CLI 不会自动读取 `.env`。

各 baseline 的构造、批量 `predict` 与 Prediction 契约见 [Infer 模型接入](INFER_MODELS.md)。

## `metrics`

`metrics` 接收已注册指标名称列表，例如：

```yaml
metrics: [progress]
```

查看注册项：

```bash
prmeval list-samplers
prmeval list-infers
prmeval list-metrics
```

## 输出目录与续跑

`run()` 始终调用 `sample()` → `infer()` → `evaluate_metrics()`，不再接受顶层 `mode`。
迁移旧配置时删除 `mode`；需要可移植采样文件的旧分阶段运行应显式设置 `save_samples: true`。

| 配置 | 样本 JSONL / NPZ | 推理、错误、指标 |
|---|---|---|
| `output_dir: null` | 不写入 | 不写入 |
| 输出目录 + `save_samples: false`（默认） | 不写入 | 写入 |
| 输出目录 + `save_samples: true` | 写入 | 写入 |

产物写入 `<output_dir>/<run_name-or-default>/`。空字符串或纯空白目录会报错，使用 `null` 关闭写入。
显式输出路径不能绕过开关：`sample --output` 需要输出目录和 `save_samples: true`，
`infer --output` 需要输出目录。自定义预测文件的错误文件为同目录下 `<stem>.errors.jsonl`；
指标仍写入配置的运行目录。

`sample()` 每次都重新加载 trajectory pool 并创建 samplers；开启采样落盘时重新生成并覆盖样本文件，与 `resume` 无关。
`resume: true` 时，Stage 2 仅跳过当前输入中已有成功预测的 sample ID，失败样本重新执行；返回的成功记录和 coverage
包含当前输入范围内的历史成功结果与本次结果。`resume: false` 清空预测和错误输出并重新推理。

`save_samples` 只控制写入。推理输入依次选择显式 `samples_path`、已有的当前/默认样本文件、
准备好的 samplers，三者互斥；显式路径不存在会报错。关闭采样落盘不会删除或忽略已有样本文件。
框架不比较配置指纹，数据、采样或模型配置变化时应使用新运行目录。

## 不落盘的阶段执行

```yaml
output_dir: null
save_samples: false
```

`sample()` 只加载一次 trajectory pool 并准备 samplers，不提前生成样本。
其摘要中的 `samples` 和 `eval_types` 为 `null`，实际采样数量由 `infer()` 统计。
`infer.batch_size` 控制按批采样、加载帧和推理；采样落盘同样流式消费，不汇集全部帧。

同一个 Evaluator 可依次调用三个阶段，或直接调用 `run()`。无输出目录时不读取历史预测检查点，
每次推理重新执行；仍允许显式读取样本文件或预测文件。独立 CLI 进程无法共享 samplers 或推理结果，
因此跨进程执行后续阶段必须有文件输入。

`infer()` 返回 `(summary, successful_records)`；`run()` 和 `evaluate_metrics()` 始终返回
`metrics`、`coverage`、`predictions`、`details`。Python 返回完整指标明细，CLI 输出去掉逐条明细的摘要。
不落盘时 `details` 为 `null`；内存指标没有预测文件来源时 `predictions` 也为 `null`。

不要提交真实 API Key、私有服务地址、生成数据或 `evaluation_output/`。
