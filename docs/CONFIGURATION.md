# 配置文件说明

PRMEval 使用一个 YAML 文件描述采样、推理、指标和产物目录。完整示例见 [`configs/eval/test_stage.yaml`](../configs/eval/test_stage.yaml)。

## 完整示例

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
  transport: openai_chat
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  timeout_seconds: 120
  max_retries: 0
  max_concurrency: 1
  temperature: 0
  max_tokens: 4096

metrics: [reward_alignment]
output_dir: evaluation_output
run_name: jsonl-progress-full-smoke
resume: false
```

## 顶层配置

| 配置项 | 使用阶段 | 说明 |
|---|---|---|
| `sampling` | Stage 1 | 数据源、adapter、采样类型、轨迹与帧数限制 |
| `infer` | Stage 2 | infer adapter、远程服务、模型、并发与重试 |
| `metrics` | Stage 3 | 需要计算的指标名称列表 |
| `output_dir` | 全阶段 | 所有 run 的根目录 |
| `run_name` | 全阶段 | 当前 run 的目录名称 |
| `resume` | Stage 1/2 | 是否复用兼容产物并跳过已成功样本 |

三个阶段目前共用一个 `EvalConfig`。因此，即使只运行 Stage 1，配置也需要保留 `infer` 块，但采样阶段不会读取 API Key 或调用模型。

## `sampling`

常用字段：

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

相对 `paths` 以运行命令时的当前目录为基准。各 adapter 的目录结构与字段要求见 [本地数据格式与 Dataset Adapter](DATASETS.md)。

## `infer`

常用字段：

| 字段 | 说明 |
|---|---|
| `name` | 已注册的 infer 名称 |
| `transport` | 远程传输协议，例如 `openai_chat` |
| `base_url` | 服务地址或保存服务地址的环境变量名 |
| `api_key` | API Key 或保存 Key 的环境变量名；服务无需认证时可省略 |
| `model_id` | 模型 ID 或保存模型 ID 的环境变量名 |
| `timeout_seconds` | 单次请求超时时间 |
| `max_retries` | 请求失败后的最大重试次数 |
| `max_concurrency` | 最大并发请求数 |
| `temperature` | 生成温度 |
| `max_tokens` | 单次响应的最大 token 数 |
| `options` | 传给具体 infer adapter 的扩展配置 |

推荐只在配置中保存环境变量名：

```yaml
infer:
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
```

运行前设置对应值：

```bash
export OPENAI_API_KEY='your-api-key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
```

全大写标识符会被视为环境变量名。环境变量不存在时初始化配置会报错；字段也可以直接填写公开 URL 或模型 ID。CLI 不会自动读取 `.env`。

对于 reasoning 模型，`max_tokens` 同时覆盖思考过程和最终输出。如果响应只有 reasoning 而没有最终 JSON，可以适当提高该值。

`progress_test` 允许覆盖默认 prompt：

```yaml
infer:
  name: progress_test
  options:
    prompt: "Task: {task}. Return {num_frames} progress values in [0,1]."
```

## `metrics`

`metrics` 接收已注册指标名称的列表，例如：

```yaml
metrics:
  - reward_alignment
```

可通过以下命令查看当前注册项：

```bash
python -m prmeval.cli list-datasets
python -m prmeval.cli list-samplers
python -m prmeval.cli list-infers
python -m prmeval.cli list-metrics
```

## 输出目录与续跑

所有产物写入：

```text
<output_dir>/<run_name>/
```

当 `resume: true` 时，Stage 1 会根据 sampling 指纹判断已有样本能否复用，Stage 2 会跳过已成功的 `sample_id`。相同目录中的配置指纹不兼容时，框架会拒绝混写。完整产物结构见 [三阶段评测流程](PIPELINE.md#运行产物与断点续跑)。

不要在配置中提交真实 API Key、私有服务地址或内部模型 ID。
