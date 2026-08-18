# 配置文件说明

PRMEval 使用一个 YAML 文件描述采样、推理、指标和产物目录。下面的完整示例与
[`configs/eval/progress_test_remote.yaml`](../configs/eval/progress_test_remote.yaml) 保持一致，用于调用通用
OpenAI-compatible 远程模型并运行 Stage 1 → Stage 2 → Stage 3。

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
  mode: remote
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  timeout_seconds: 120
  max_retries: 0
  batch_size: 1
  max_concurrency: 4
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
| `infer` | Stage 2 | infer adapter、本地或远程模型、batch、并发与重试 |
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
| `name` | 已注册的 infer 名称；通过 `python -m prmeval.cli list-infers` 查看 |
| `mode` | `local` 或 `remote`；分别使用 `local_huggingface` 和 `openai_chat`，省略时根据 `model_path`/`base_url` 自动判断 |
| `model_path` | local 模式必填；Hugging Face ID 或本地 checkpoint 路径，也可以是环境变量名 |
| `base_url` | remote 模式必填；服务地址或保存服务地址的环境变量名 |
| `api_key` | API Key 或保存 Key 的环境变量名；服务无需认证时可省略 |
| `model_id` | 记录到产物中的模型身份；local 省略时使用 `model_path` |
| `timeout_seconds` | 单次请求超时时间 |
| `max_retries` | 请求失败后的最大重试次数 |
| `batch_size` | 每次 `infer.predict_batch()` 处理的样本数；默认 1 |
| `max_concurrency` | 同时运行的 batch 数；local 必须为 1，remote 默认 4 |
| `temperature` | 生成温度 |
| `max_tokens` | 单次响应的最大 token 数 |
| `options` | 传给具体 infer adapter 的扩展配置 |

内置模型统一使用 local-first 模型接口。`roboreward`、`robodopamine`、`topreward`、`vlac`、`rbm` 和
`rewind` 支持本地推理，并可选择远程模式；`gvl`、`rlvlmf`、`sole_r1` 和 `progress_test` 当前仅支持远程模式。
远程方法统一复用 OpenAI-compatible `POST /v1/chat/completions` 客户端。完整新增模型方法见
[本地优先模型接入](LOCAL_MODELS.md)。

`prmeval.infer.create_infer()` 是统一的模型构造入口。导入 `prmeval.infer` 时会加载
`prmeval/infer/baselines/__init__.py`，从而完成各模型的 registry 注册；项目不再提供
`prmeval.infer.adapters` 或 `specialized` transport。

各 progress infer 的远程调用行为如下：

| infer | 每个样本的调用方式 | 标准化输出 |
|---|---|---|
| `progress_test` | 完整有序轨迹单次请求，使用严格 JSON Schema | 与输入严格等长的 `[0,1]` progress curve |
| `gvl` | 按任务和帧数固定乱序的全部查询帧，单次请求 | 逐帧 0–100 百分比恢复原序后除以 100 |
| `roboreward` | 完整轨迹单次请求 | 1–5 分数映射为 `(score - 1) / 4` 并复制到全部帧 |
| `robodopamine` | 每个选中 transition 一次 8 图请求 | 按 incremental/forward/backward 公式累积 |
| `sole_r1` | 首帧固定为 0；随后逐帧发送首帧、上一帧和当前帧，并递推上一预测 | 解析 `<answer>` 百分比后除以 100 |
| `topreward` | 每个选中 trajectory prefix 一次请求 | True logprob 归一化并插值到全部帧 |
| `vlac` | 完整有序轨迹单次请求 | critic value 归一化，并按末值补齐或截断 |
| `rbm` / `rewind` | 完整有序轨迹单次请求 | 与输入严格等长的 `[0,1]` progress curve |

`robodopamine` 支持以下 `options`：

```yaml
infer:
  name: robodopamine
  mode: remote
  options:
    eval_mode: incremental  # incremental、forward 或 backward
    frame_interval: 1       # 正整数，默认逐帧 transition
```

`topreward` 可通过 `options.num_prefix_samples` 设置 prefix 数量，默认值为 15。对应的 OpenAI-compatible
服务必须支持请求参数 `logprobs: true`、`top_logprobs: 20`，并保证返回的生成 token 或候选 token 中包含
`True` 或带前导空格的 ` True`；否则该样本会严格失败并写入 `errors.jsonl`。

模型的帧数限制统一由 `sampling.max_frames` 控制。Infer 不会再次抽帧，以免 prediction、输入帧和 target
progress 失去对齐关系。

### Local 模式

```yaml
infer:
  name: my_progress_model
  mode: local
  model_path: /models/my-progress-model
  model_id: my-progress-model-v1
  batch_size: 4
  max_concurrency: 1
  options:
    dtype: bfloat16
```

local 模式只在 Stage 2 构造 infer 时加载一次模型。`batch_size > 1` 要求模型显式声明并实现原生 batch；
不支持 batch 的模型使用 `batch_size: 1`。单个本地模型实例不通过多线程并发访问，因此
`max_concurrency` 必须为 1。

安装通用 Hugging Face 与 Qwen-VL 可选依赖：

```bash
pip install -e '.[local-hf,local-qwen]'
```

模型模块必须懒加载 torch/transformers，使 remote-only 环境无需安装本地模型依赖也能导入 PRMEval。

### Remote 模式

存在 `base_url` 时可以自动识别为 remote；推荐显式写出 `mode: remote`。`progress_test` 的规范配置为：

```yaml
infer:
  name: progress_test
  mode: remote
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  batch_size: 1
  max_concurrency: 4
```

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

它只支持 progress 类型样本，不加载本地 checkpoint。每个样本发送一次包含任务文本和全部帧的请求，返回值
必须符合 JSON Schema，并且 `progress` 数组长度必须与输入帧数完全一致。可以直接运行完整示例：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml
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
