# PRMEval 机器人奖励模型评测框架

PRMEval 是一个面向机器人任务进度与偏好模型的远程评测框架。它负责读取本地数据集、构造统一评测样本、调用远程 baseline，并计算指标。

当前仓库只保留评测相关能力，不包含数据集上传、模型训练、FSDP、本地 checkpoint 加载或本地模型服务。通用远程模型通过 OpenAI-compatible API 调用；具有专用输出头的模型使用统一的 `/v1/evaluations` 协议。

## 核心设计

评测被拆成三个可以独立运行和验证的阶段：

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

三个阶段通过统一的 `bench.record.v1` 协议连接：

- Stage 1 创建 `stage: sampled` 的 Record；
- Stage 2 保留原始 `sample_id`、`input` 和 `target`，增加 `baseline`、`prediction`、`execution`；
- Stage 3 只读取推理成功的 Record，不加载图片，也不调用模型。

完整数据流见 [三阶段协议说明](docs/PIPELINE.md)，字段定义见 [EvaluationRecord 数据结构](docs/RECORD_SCHEMA.md)。

## 安装

推荐在项目目录中创建独立的虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

安装后的发行包、Python 模块和 CLI 名称分别为：

```text
发行包：prmeval
Python 模块：prmeval
评测 CLI：prmeval-eval
数据预处理 CLI：prmeval-data-preprocess
```

可以通过以下命令验证：

```bash
python -c "import prmeval; print(prmeval.__file__)"
python -m prmeval.cli --help
prmeval-eval --help
prmeval-data-preprocess --help
```

如需生成可视化，请安装可视化可选依赖：

```bash
pip install -e '.[viz]'
```

开发和测试依赖：

```bash
pip install -e '.[dev]'
```

默认安装包含远程请求、数据处理和视频解码所需依赖，支持 Python 3.10 及以上版本。

## 快速开始：完整冒烟测试

仓库提供了一个完整的端到端测试，固定使用：

```text
dataset adapter = jsonl
evaluation type = reward_alignment
baseline = progress_test
metric = reward_alignment
```

配置文件为 [full_smoke_jsonl.yaml](configs/eval/full_smoke_jsonl.yaml)。它只读取一条 trajectory，并发起一次远程模型请求。

先设置 OpenAI-compatible 服务的 API Key、地址和模型 ID：

```bash
export OPENAI_API_KEY='你的 API Key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
```

配置中的 `base_url`、`api_key` 和 `model_id` 可以直接填写环境变量名。初始化 `BaselineConfig` 时会读取并保存对应值；环境变量不存在时会直接报错。无需认证的服务可以省略 `api_key`。

依次执行三个阶段：

```bash
# Stage 1：从 JSONL 数据集采样
python -m prmeval.cli sample \
  --config configs/eval/test_stage.yaml

python -m prmeval.cli validate-samples \
  --samples evaluation_output/jsonl-progress-full-smoke/samples.jsonl

# Stage 2：调用 progress_test
python -m prmeval.cli infer \
  --config configs/eval/test_stage.yaml

python -m prmeval.cli validate-predictions \
  --predictions evaluation_output/jsonl-progress-full-smoke/predictions.jsonl

# Stage 3：计算 reward_alignment
python -m prmeval.cli metrics \
  --config configs/eval/test_stage.yaml
```

也可以使用便捷命令连续执行三阶段：

```bash
python -m prmeval.cli run \
  --config configs/eval/test_stage.yaml
```

详细说明和当前实测结果见 [完整冒烟测试说明](examples/stage_full_smoke/README.md)。测试帧只用于验证数据协议、图片编码、远程调用和结果落盘，不代表真实机器人场景。

## 配置文件

一个完整评测配置包含以下部分：

```yaml
dataset:
  name: jsonl-full-smoke
  adapter: jsonl
  root: .
  paths: [examples/stage_1_smoke/trajectories.jsonl]
  max_trajectories: 1

sampling:
  eval_types: [reward_alignment]
  max_frames: 3
  pad_frames: false
  progress_type: absolute_first_frame

baseline:
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

各部分职责如下：

| 配置块 | 作用阶段 | 说明 |
|---|---|---|
| `dataset` | Stage 1 | 数据来源、adapter、路径和加载数量 |
| `sampling` | Stage 1 | 评测类型、抽帧数量和 progress 定义 |
| `baseline` | Stage 2 | endpoint、模型、认证、并发和重试 |
| `metrics` | Stage 3 | 需要计算的指标 |
| `output_dir/run_name` | 全阶段 | 运行产物目录 |
| `resume` | Stage 1/2 | 是否复用同配置结果并跳过成功样本 |

目前三个阶段共用一个 `EvalConfig`，因此单独运行 Stage 1 时配置中仍需保留 `baseline` 块，但 Stage 1 不会读取 API Key，也不会调用模型。

## 本地数据格式与 Dataset Adapter

Dataset adapter 负责把不同的本地存储格式统一转换成内部 `Trajectory`。当前提供 `jsonl` 和 `processed_cache` 两种 adapter。

### JSONL adapter

适合冒烟测试、小型自定义数据集和已经具有 NPZ 帧的数据。推荐目录结构：

```text
my_dataset/
├── trajectories.jsonl
└── frames/
    ├── trajectory-001.npz
    └── trajectory-002.npz
```

`trajectories.jsonl` 每行表示一条完整轨迹：

```json
{"id":"trajectory-001","task":"pick up the red block","frames":"frames/trajectory-001.npz","data_source":"my-dataset","quality_label":"successful","partial_success":1.0}
```

必填字段为：

- `id`：轨迹 ID；
- `task`：自然语言任务；
- `frames`：NPZ 路径或内嵌帧数组。

NPZ 文件必须包含名为 `frames` 的数组，推荐格式为：

```text
shape = [T, H, W, C]
dtype = uint8
C = 3
```

需要注意，dataset source JSONL 与 Stage 1 输出的 `samples.jsonl` 不是同一种文件：

```text
trajectories.jsonl  --jsonl adapter-->  Trajectory
Trajectory          --sampler------->  samples.jsonl
samples.jsonl        --Stage 2------>   predictions.jsonl
```

### processed_cache adapter

适合较大的 Hugging Face 数据集和包含视频的正式评测。它读取由 `prmeval-data-preprocess` 生成的本地缓存：

```text
processed_datasets/
└── <cache_name>/
    ├── frames/
    │   └── *.npz
    ├── processed_dataset/
    │   ├── dataset_info.json
    │   ├── state.json
    │   └── data-*.arrow
    └── prepare_manifest.json
```

如果下载的数据还没有这种结构，复制并修改 [rbm_1m_ood_local.yaml](configs/data/rbm_1m_ood_local.yaml)，然后运行：

```bash
prmeval-data-preprocess \
  --config configs/data/rbm_1m_ood_local.yaml
```

预处理会：

1. 验证 `id` 和 `task`；
2. 读取 `frames`、`frames_video`、`video` 或 `frames_path`；
3. 解码视频并进行一次固定抽帧；
4. 保存压缩 NPZ；
5. 创建轻量 Hugging Face `processed_dataset` 索引；
6. 将失败轨迹记录到 `prepare_manifest.json`。

这个缓存是评测输入缓存，不包含模型 embedding，也不包含训练数据索引。

`rbm-1m-ood` 在 [manifests.py](prmeval/data/manifests.py) 中聚合了六个 OOD 子数据集。可以通过 `dataset.root` 指定缓存根目录，也可以设置：

```bash
export PRMEVAL_PROCESSED_DATASETS_PATH=/path/to/processed_datasets
```

## Stage 1：数据采样

Stage 1 的内部流程是：

```text
DatasetAdapter
    → Trajectory
    → EvalSampler
    → ProgressSample / PreferenceSample
    → EvaluationRecord(stage="sampled")
    → samples.jsonl + sample_frames/*.npz
```

对 `reward_alignment`，采样器会选择可用的成功轨迹、按 `max_frames` 均匀抽帧，并使用相同帧索引构造 `target.progress`。默认 `absolute_first_frame` 定义为：

```text
progress = (frame_index - first_index) / (total_frames - first_index - 1)
```

因此第一帧为 `0`，最后一帧为 `1`。

Stage 1 输出的每条 Record 必须满足：

```text
NPZ 帧数量 = frame_indices 数量 = target.progress 数量
```

图片不会直接写进 JSONL，而是保存为 `sample_frames/*.npz`。Record 中只保存相对路径、帧数和 SHA-256 校验和，所以移动数据时必须整体移动 `samples.jsonl` 和 `sample_frames/`。

## Stage 2：远程推理

Stage 2 只接收 `EvaluationRecord(stage="sampled")`。它加载 NPZ 帧，调用 baseline，然后生成 `stage: inferred` 的 Record。

`target` 不会发送给远程模型。发送内容只来自：

- `input.task`；
- `input.items[].frames`；
- baseline adapter 的 prompt 和请求参数。

模型输出经过 adapter 归一化后写入：

```json
{
  "baseline": {"name": "progress_test", "model": "your-model"},
  "prediction": {"kind": "progress", "values": [0.0, 0.5, 1.0]},
  "execution": {"status": "success", "attempts": 1}
}
```

成功结果写入 `predictions.jsonl`，失败结果写入 `errors.jsonl`。progress 输出数量与输入帧数不一致时，该样本会被记录为失败。

### 服务连接配置

推荐在配置中只保存环境变量名：

```yaml
base_url: BASE_URL
api_key: OPENAI_API_KEY
model_id: MODEL_ID
```

运行前设置：

```bash
export OPENAI_API_KEY='你的 API Key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
```

框架会自动发送：

```text
Authorization: Bearer <API Key>
```

字段也可以直接填写公开的 URL 或模型 ID；全大写标识符会被视为环境变量名。不要把真实 Key、私有服务地址或内部模型 ID 写入配置并提交到 Git。CLI 当前不会自动读取 `.env`；VS Code 的 `envFile` 配置可以读取项目根目录 `.env`。

### progress_test baseline

`progress_test` 是用于联调 Stage 2 的通用 OpenAI-compatible baseline，只支持 progress 样本。它调用：

```text
POST /v1/chat/completions
```

请求使用 Base64 多图片 `image_url` 和 JSON Schema structured output。默认 prompt 要求模型为每张图片按顺序返回一个 `[0,1]` 的 progress 值，Schema 同时约束输出数组长度等于输入帧数。

默认输出格式：

```json
{"progress": [0.0, 0.5, 1.0]}
```

可以覆盖默认 prompt：

```yaml
baseline:
  options:
    prompt: "Task: {task}. Return {num_frames} progress values in [0,1]."
```

对于 reasoning 模型，`max_tokens` 同时覆盖思考过程和最终 JSON。如果响应只有 `message.reasoning` 而 `message.content` 为 `null`，应提高：

```yaml
max_tokens: 4096
```

### 其他 baseline 协议

GVL、RL-VLM-F、RoboReward 和 RoboDopamine 使用 OpenAI-compatible `/v1/chat/completions`。

RBM/ReWiND、TOPReward 和 VLAC 使用专用的：

```text
POST /v1/evaluations
```

专用协议的 request/response schema 位于 `prmeval.baselines.specialized`。可以启动本地 contract mock：

```bash
python -m prmeval.baselines.mock_server --port 8765
```

查看当前注册的 baseline：

```bash
python -m prmeval.cli list-baselines
```

## Stage 3：指标计算

Stage 3 只读取：

```text
stage = inferred
execution.status = success
```

它不读取 dataset、不加载 NPZ，也不调用模型。

当前内置评测包括：

| 评测 | 输入 | 指标 |
|---|---|---|
| `reward_alignment` | target progress 与 prediction progress | MSE、Pearson |
| `policy_ranking` | 任务内质量排序与预测终态 progress | Kendall |
| `quality_preference` | chosen/rejected 轨迹偏好 | Accuracy |
| `confusion_matrix` | 语言任务与视频任务匹配结果 | 混淆矩阵 |

`reward_alignment` 对每条样本分别计算 MSE 和 Pearson，然后对样本等权平均，并按下面的维度切片：

```text
evaluation.dataset.name : baseline.name
```

不调用模型、直接重新计算指标：

```bash
python -m prmeval.cli compute-metrics \
  --predictions examples/stage_3_smoke/predictions.jsonl \
  --metrics reward_alignment \
  --output /tmp/prmeval-metrics.json
```

## 运行产物与断点续跑

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

- Stage 1 使用 dataset/sampling 指纹判断是否可以复用已有 samples；
- Stage 2 跳过已经成功的 `sample_id`；
- 失败样本仍可在下次运行中重试；
- 相同输出目录中的配置指纹不一致时会拒绝混写。

运行产物默认位于 `evaluation_output/`，该目录已加入 `.gitignore`。

## Python API

除了 CLI，也可以使用 Python API：

```python
from prmeval import EvalConfig, Evaluator

config = EvalConfig.from_yaml("configs/eval/full_smoke_jsonl.yaml")
evaluator = Evaluator(config)

sample_summary = evaluator.sample()
infer_summary = evaluator.infer()
metric_summary = evaluator.evaluate_metrics()
```

连续执行：

```python
summary = Evaluator(config).run()
```

## 调试建议

VS Code 中选择项目虚拟环境的 Python 解释器：

```text
.venv/bin/python
```

Windows 上对应路径为 `.venv\Scripts\python.exe`。

Stage 2 使用 `ThreadPoolExecutor`。即使 `max_concurrency: 1`，模型调用仍运行在工作线程中。建议在以下位置设置断点：

- `prmeval/evaluation/runner.py` 的 `_predict()`；
- `prmeval/baselines/adapters.py` 的 baseline `predict()`；
- `prmeval/baselines/openai.py` 的 `_chat()`；
- `prmeval/baselines/base.py` 的 `_post_json()`。

调试时建议：

```yaml
max_concurrency: 1
max_retries: 0
```

`ThreadPoolExecutor` 创建的是线程，不是子进程；VS Code 会在 `CALL STACK` 中显示类似 `ThreadPoolExecutor-0_0` 的工作线程。

## 查看注册项

```bash
python -m prmeval.cli list-datasets
python -m prmeval.cli list-samplers
python -m prmeval.cli list-baselines
python -m prmeval.cli list-metrics
```

当前 registry 允许独立扩展 dataset、sampler、baseline 和 metric。新增实现时注册一个新名称即可，不需要修改统一 Record 的顶层结构。

## 项目结构

```text
prmeval/
├── core/            配置、统一 Schema 和注册器
├── data/            dataset adapter、预处理、progress 与 sampler
├── baselines/       远程协议、prompt、parser 和 baseline adapter
├── evaluation/      三阶段编排、artifact 落盘和断点续跑
├── metrics/         内置指标与可选可视化
└── cli.py           命令行入口

configs/
├── data/            本地数据预处理配置
└── eval/            评测与冒烟配置

docs/                中文协议与数据结构说明
examples/            分阶段及端到端冒烟样例
tests/               单元测试、contract test 和 golden fixture
```

## 测试

在已激活的项目虚拟环境中运行现有测试：

```bash
PYTHONPATH=. python -m unittest tests.test_evaluation -v
```

安装开发依赖后也可以运行：

```bash
pytest -q
```
