# 配置文件说明

PRMEval 使用一个 YAML 或 JSON 文件描述采样、推理、指标和产物目录。下面示例调用 `openai_compatible` 完成 Stage 1 → Stage 2 → Stage 3：

```yaml
sampling:
  save_samples: false
  dataset_name: rbm-1m-ood
  paths: [/path/to/hf_datasets/rbm-1m-ood]
  max_trajectories: 1
  base_frames: 3
  progress_type: absolute_first_frame

infer:
  name: openai_compatible
  base_url: null  # 从环境变量 BASE_URL 读取
  api_key: null  # 从环境变量 API_KEY 读取
  model_id: null  # 从环境变量 MODEL_ID 读取
  model_extra_config: {}

eval_types: progress
output_dir: evaluation_output
task_name: progress-full-smoke
resume: false
```

## 生成 YAML 或 JSON 配置

直接从 `prmeval/core/config.py` 中的配置模型生成完整模板：

```bash
python -m prmeval.cli make-config
```

不传 `--output` 时将 YAML 输出到终端，也可以重定向到文件。指定路径时自动创建父目录：

```bash
python -m prmeval.cli make-config --output configs/eval/my_eval.yaml
python -m prmeval.cli make-config --format json --output configs/eval/my_eval.json
```

CLI 直接调用 `EvalConfig.export_config(format="yaml")`，按类定义递归导出字段、默认值和默认工厂，
不接受模型、数据集或评测类型参数。必填但没有默认值的字段（目前为 `infer.name`）输出为 `null`，
并在 YAML 顶部注释中列出；JSON 输出不含注释。填写这些字段后才能加载配置。
`task_name` 的声明默认值为 `null`，加载时自动生成名称。
格式默认是 YAML，使用 `--format json` 选择 JSON；不根据输出文件名推断格式。

Python 也可直接导出：

```python
from prmeval.core.config import EvalConfig

print(EvalConfig.export_config())  # 默认 YAML
print(EvalConfig.export_config("json"))  # JSON
```

运行评测前填写 `sampling.paths` 和所选模型需要的连接信息或 checkpoint；生成配置不加载模型或读取数据集。
若输出文件已存在，命令会报错；需要覆盖时增加 `--force`。

## 加载 YAML 和 JSON

CLI 的 `--config` 支持 `.yaml`、`.yml` 和 `.json`，两种格式采用相同的字段与校验规则。
Python 可以按格式加载，也可以自动识别后缀：

```python
from prmeval.core.config import EvalConfig

config = EvalConfig.from_yaml("config.yaml")
config = EvalConfig.from_json("config.json")
config = EvalConfig.from_file("config.json")
```

最小 JSON 示例（完整示例见 `examples/config_json/config.json`）：

```json
{
  "sampling": {"dataset_name": "demo", "paths": ["examples/stage_1_smoke/trajectories.jsonl"]},
  "infer": {"name": "openai_compatible", "model_extra_config": {}},
  "eval_types": "progress"
}
```

旧字段 `sampling.eval_types`、`metrics`、`run_name` 和 `infer.options` 已删除，不再接受。
将评测类型移至顶层 `eval_types`，把单元素列表改为字符串；`run_name` 改为 `task_name`，
`infer.options` 改为 `infer.model_extra_config`。多个评测类型需分别运行。
省略 `task_name` 或设为 `null` 时自动生成名称，显式名称保持原值。
`infer.model_version`、`timeout_seconds`、`max_retries`、`temperature`、`max_tokens` 和 `headers` 已从通用配置删除；请求参数由具体 baseline 管理，不能继续放在 `infer` 顶层。

原顶层 `save_samples` 已移到 `sampling.save_samples`，旧位置不再接受。

## 顶层配置

| 配置项 | 使用阶段 | 说明 |
|---|---|---|
| `sampling` | Stage 1 | JSONL/Hugging Face Dataset 路径、采样类型、轨迹与帧数限制 |
| `infer` | Stage 2 | baseline 名称、模型/连接信息和扩展参数 |
| `eval_types` | Stage 1、3 | 单个评测类型字符串，默认 `progress`；同时选择采样器和指标 |
| `output_dir` | 全阶段 | 产物根目录，默认 `/tmp/prmeval_evaluation_output`；显式设为 `null` 时不写产物 |
| `task_name` | 全阶段 | 任务目录名称，默认 `{sampling.dataset_name}_{infer.name}_{eval_types}` |
| `resume` | Stage 2 | 是否复用已有成功预测并跳过已成功样本；不控制采样阶段 |

三个阶段共用一个 `EvalConfig`。只运行 Stage 1 时仍需保留 `infer` 块，但采样阶段不会构造模型。

## `sampling`

| 字段 | 说明 |
|---|---|
| `dataset_name` | 写入评测记录的数据集名称，也参与 sample ID 构造 |
| `paths` | 一个或多个轨迹 `.jsonl` 文件，或由 `datasets.save_to_disk()` 保存的本地 Dataset 目录 |
| `save_samples` | 默认 `false`；设为 `true` 时保存样本 JSONL 和帧，并要求顶层 `output_dir` 非空 |
| `max_trajectories` | 最多读取的轨迹数 |
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
eval_types: progress_temporal_variation
sampling:
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
| `base_url` | OpenAI-compatible 服务地址；为 null 时读取环境变量 BASE_URL |
| `api_key` | 模型服务 API Key；为 null 时读取环境变量 API_KEY |
| `model_id` | 请求使用及记录到产物中的模型身份；为 null 时读取环境变量 MODEL_ID |
| `batch_size` | Runner 每次从 sample 迭代器消费并调度的样本数，默认 `1` |
| `model_extra_config` | 传给具体 baseline 的扩展配置 |

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
  model_extra_config:
    use_prefix_samples: True

# OpenAI-compatible baseline
infer:
  name: openai_compatible
  base_url: null  # 从环境变量 BASE_URL 读取
  api_key: null  # 从环境变量 API_KEY 读取
  model_id: null  # 从环境变量 MODEL_ID 读取
```

`robodopamine` 的模型内部 micro-batch 与运行策略放在 `model_extra_config`，不属于 Runner 调度：

```yaml
infer:
  name: robodopamine
  model_path: /models/robo-dopamine
  model_extra_config:
    micro_batch_size: 1
    eval_mode: incremental
    frame_interval: 1
```

不要在受版本控制的配置中保存真实密钥。`InferConfig` 在字段校验后，为省略或值为 `null` 的 `model_id`、`base_url`、`api_key` 分别读取环境变量 `MODEL_ID`、`BASE_URL`、`API_KEY`。环境变量不存在时保持 `None`。显式配置的值（包括空字符串）优先，不会把字符串内容当作环境变量名解析。CLI 不自动读取 `.env`，运行前需导出对应变量。`make-config` 导出声明的默认值，这三个字段仍为 `null`，不会写入环境变量中的值。

各 baseline 的构造、批量 `predict` 与 Prediction 契约见 [Infer 模型接入](INFER_MODELS.md)。

## `eval_types`

`eval_types` 接收单个已注册的评测类型名称，采样和指标计算使用同一名称。例如：

```yaml
eval_types: progress
```

查看注册项：

```bash
prmeval list-samplers
prmeval list-infers
prmeval list-metrics
```

## 输出目录与续跑

`run()` 始终调用 `sample()` → `infer()` → `evaluate_metrics()`，不再接受顶层 `mode`。
迁移旧配置时删除 `mode`；需要可移植采样文件的旧分阶段运行应显式设置 `sampling.save_samples: true`。

| 配置 | 样本 JSONL / NPZ | 推理、错误、指标 |
|---|---|---|
| `output_dir: null` + `sampling.save_samples: false` | 不写入 | 不写入 |
| 输出目录 + `sampling.save_samples: false`（默认） | 不写入 | 写入 |
| 输出目录 + `sampling.save_samples: true` | 写入 | 写入 |

产物写入 `<output_dir>/<task_name>/`。空字符串或纯空白目录会报错，使用 `null` 关闭写入。
显式输出路径不能绕过开关：`sample --output` 需要输出目录和 `sampling.save_samples: true`，
`infer --output` 需要输出目录。自定义预测文件的错误文件为同目录下 `<stem>.errors.jsonl`；
指标仍写入配置的运行目录。

`EvalConfig.validate_sample_output` 在配置初始化、YAML/JSON 加载时校验：
`sampling.save_samples: true` 与 `output_dir: null` 的组合会直接报错。
对于运行时传入的 `sample(samples_path=...)`，runner 将路径通过校验上下文传给同一个
`model_validator`；关闭采样保存时不允许指定输出路径，校验发生在加载数据和写入文件之前。

`sample()` 每次都重新加载 trajectory pool 并创建 samplers；开启采样落盘时重新生成并覆盖样本文件，与 `resume` 无关。
`resume: true` 时，Stage 2 仅跳过当前输入中已有成功预测的 sample ID，失败样本重新执行；返回的成功记录和 coverage
包含当前输入范围内的历史成功结果与本次结果。`resume: false` 清空预测和错误输出并重新推理。

`sampling.save_samples` 只控制写入。推理输入依次选择显式 `samples_path`、已有的当前/默认样本文件、
准备好的 samplers，三者互斥；显式路径不存在会报错。关闭采样落盘不会删除或忽略已有样本文件。
框架不比较配置指纹，数据、采样或模型配置变化时应使用新运行目录。

## 不落盘的阶段执行

```yaml
sampling:
  save_samples: false
output_dir: null
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
