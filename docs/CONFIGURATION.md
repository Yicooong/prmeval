# 配置文件说明

PRMEval 使用一个 YAML 文件描述采样、推理、指标和产物目录。下面示例调用 `progress_test` 完成 Stage 1 → Stage 2 → Stage 3：

```yaml
sampling:
  dataset_name: jsonl-full-smoke
  adapter: jsonl
  paths: [examples/stage_1_smoke/trajectories.jsonl]
  max_trajectories: 1
  eval_types: [reward_alignment]
  max_frames: 3
  pad_frames: false
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

metrics: [reward_alignment]
output_dir: evaluation_output
run_name: jsonl-progress-full-smoke
resume: false
```

## 顶层配置

| 配置项 | 使用阶段 | 说明 |
|---|---|---|
| `sampling` | Stage 1 | 数据源、adapter、采样类型、轨迹与帧数限制 |
| `infer` | Stage 2 | baseline 名称、模型/连接信息和扩展参数 |
| `metrics` | Stage 3 | 需要计算的指标名称列表 |
| `output_dir` | 全阶段 | 所有 run 的根目录 |
| `run_name` | 全阶段 | 当前 run 的目录名称 |
| `resume` | Stage 1/2 | 是否复用兼容产物并跳过已成功样本 |

三个阶段共用一个 `EvalConfig`。只运行 Stage 1 时仍需保留 `infer` 块，但采样阶段不会构造模型。

## `sampling`

| 字段 | 说明 |
|---|---|
| `dataset_name` | 写入评测记录的数据集名称，也参与 sample ID 构造 |
| `adapter` | 数据接入方式，当前支持 `jsonl` 和 `huggingface` |
| `paths` | 一个或多个本地数据路径 |
| `max_trajectories` | 最多读取的轨迹数 |
| `eval_types` | 需要构造的评测类型 |
| `max_frames` | 每个样本最多抽取的帧数 |
| `pad_frames` | 帧数不足时是否补齐 |
| `progress_type` | progress 真值的定义方式 |

相对路径以运行命令时的当前目录为基准。adapter 的目录结构与字段要求见 [本地数据格式与 Dataset Adapter](DATASETS.md)。

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
| `headers` | 附加 HTTP header |
| `options` | 传给具体 baseline 的扩展配置 |

框架不区分 local/remote，也不存在 `mode`、`transport`、`batch_size` 或 `max_concurrency`。Runner 直接构造 `INFERS.get(name)` 返回的类，并顺序逐样本调用 `predict()`。模型使用 checkpoint、provider SDK 或 HTTP client 由自身实现决定。

配置示例：

```yaml
# checkpoint baseline
infer:
  name: topreward
  model_path: /models/topreward
  model_id: topreward-v1
  options:
    num_prefix_samples: 15

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
metrics: [reward_alignment]
```

查看注册项：

```bash
python -m prmeval.cli list-datasets
python -m prmeval.cli list-samplers
python -m prmeval.cli list-infers
python -m prmeval.cli list-metrics
```

## 输出目录与续跑

产物写入 `<output_dir>/<run_name>/`。`resume: true` 时，Stage 1 根据 sampling 指纹复用样本；Stage 2 跳过已成功的 `sample_id`，失败样本会在下次运行时逐个重试。配置指纹不兼容时框架拒绝混写。详见 [三阶段评测流程](PIPELINE.md#运行产物与断点续跑)。

不要提交真实 API Key、私有服务地址、生成数据或 `evaluation_output/`。
