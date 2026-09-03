# 配置文件说明

PRMEval 使用一个 YAML 文件描述采样、推理、指标和产物目录。下面示例调用 `progress_test` 完成 Stage 1 → Stage 2 → Stage 3：

```yaml
sampling:
  dataset_name: rbm-1m-ood
  paths: [/path/to/hf_datasets/rbm-1m-ood]
  max_trajectories: 1
  eval_types: [progress]
  base_frames: 3
  progress_type: absolute_first_frame

infer:
  name: progress_test
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  timeout_seconds: 120
  max_retries: 0
  temperature: 0
  max_tokens: 4096
  options: {}

metrics: [progress]
mode: separate
output_dir: evaluation_output
run_name: progress-full-smoke
resume: false
```

## 顶层配置

| 配置项 | 使用阶段 | 说明 |
|---|---|---|
| `sampling` | Stage 1 | Hugging Face Dataset 路径、采样类型、轨迹与帧数限制 |
| `infer` | Stage 2 | baseline 名称、模型/连接信息和扩展参数 |
| `metrics` | Stage 3 | 需要计算的指标名称列表 |
| `mode` | `run` 编排 | `separate` 使用磁盘阶段产物；`continue` 在内存中连接采样与推理 |
| `output_dir` | 全阶段 | 所有 run 的根目录 |
| `run_name` | 全阶段 | 当前 run 的目录名称 |
| `resume` | Stage 1/2 | 是否复用兼容产物并跳过已成功样本 |

三个阶段共用一个 `EvalConfig`。只运行 Stage 1 时仍需保留 `infer` 块，但采样阶段不会构造模型。

## `sampling`

| 字段 | 说明 |
|---|---|
| `dataset_name` | 写入评测记录的数据集名称，也参与 sample ID 构造 |
| `paths` | 一个或多个由 `datasets.save_to_disk()` 保存的本地 Dataset 目录 |
| `max_trajectories` | 最多读取的轨迹数 |
| `eval_types` | 需要构造的评测类型 |
| `base_frames` | 基准采样帧数；`progress` 等普通采样直接按此数量抽帧 |
| `progress_type` | progress 真值的定义方式 |
| `temporal_robustness` | `progress_temporal_variation` 的最终帧数上限、变换类型、数量与参数范围 |

相对路径以运行命令时的当前目录为基准。目录结构与字段要求见 [本地 Hugging Face Dataset](DATASETS.md)。

### Synthetic temporal robustness

该评测从成功轨迹生成 Original、Pause、Slow、Fast、Rewind、Retry、Truncate 和 Skip。配置示例见
`configs/eval/synthetic_temporal_robustness.yaml`。核心配置如下：

```yaml
sampling:
  eval_types: [progress_temporal_variation]
  base_frames: 9                # 变换前的基准采样数量
  progress_type: absolute_first_frame
  temporal_robustness:
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
`truncate_retained_ratio_range` 和 `skip_removed_ratio_range` 调整。所有随机结果由 `random_seed` 稳定决定。

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

框架不区分 local/remote，也不存在 transport 或 max_concurrency 分派。Runner 直接构造
`INFERS.get(name)` 返回的类。未实现 `predict_batch()` 的 baseline 仍逐样本调用 `predict()`；实现该可选接口后，Runner
会把最多 `infer.batch_size` 个样本一次传入。模型使用 checkpoint、provider SDK 或 HTTP client 由自身实现决定。

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
  name: progress_test
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
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

各 baseline 的构造、`compute_progress`/`compute_preference` 与 `predict` 契约见 [Infer 模型接入](INFER_MODELS.md)。

## `metrics`

`metrics` 接收已注册指标名称列表，例如：

```yaml
metrics: [progress]
```

查看注册项：

```bash
python -m prmeval.cli list-samplers
python -m prmeval.cli list-infers
python -m prmeval.cli list-metrics
```

## 输出目录与续跑

产物写入 `<output_dir>/<run_name>/`；未设置 `run_name` 时使用 `default`。`resume: true` 时，Stage 1
验证并复用已有样本，Stage 2 跳过 `predictions.jsonl` 中已成功的 `sample_id`，失败样本会在下次运行时重试。
框架不比较配置指纹，因此数据、采样或模型配置改变时必须使用新的 `run_name`。详见
[三阶段评测流程](PIPELINE.md#运行产物与断点续跑)。

`mode: continue` 只影响 `run` 命令。它不生成 `samples.jsonl` 或 `sample_frames/*.npz`；
sample 迭代器没有独立 batch 配置，而是由 `infer.batch_size` 分批消费。独立运行 `sample`、`infer`、`metrics`
时始终使用可移植的磁盘阶段协议。连续模式仍写 predictions、errors 和最终指标，以便审计及按 sample ID 续跑。

不要提交真实 API Key、私有服务地址、生成数据或 `evaluation_output/`。
