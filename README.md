# PRMEval 机器人奖励模型评测框架

PRMEval 是一个面向机器人任务进度与偏好模型的评测框架。它读取本地数据集，构造统一评测样本，通过本地 Hugging Face 模型或远程服务推理，并计算评测指标。

![PRMEval 架构图](assert/arch.png)

## 核心流程

评测由三个可独立运行和验证的阶段组成：

1. **Sample**：读取数据集、抽帧并生成统一的 `EvaluationRecord`。
2. **Infer**：调用本地或远程模型并保存标准化预测结果。
3. **Metrics**：读取成功的预测记录并计算、聚合指标。

阶段职责和数据流见 [三阶段评测流程](docs/PIPELINE.md)，各文件的用途、内容及断点续跑行为见 [全流程运行产物说明](docs/ARTIFACTS.md)，字段定义见 [EvaluationRecord 数据结构](docs/RECORD_SCHEMA.md)。

## 安装

PRMEval 支持 Python 3.10 及以上版本。

```bash
pip install -e .
```

本地 Hugging Face/Qwen-VL 模型使用可选依赖：

```bash
pip install -e '.[local-hf,local-qwen]'
```

## 快速开始

仓库提供了调用通用远程模型 `progress_test` 的端到端冒烟配置
[`configs/eval/progress_test_remote.yaml`](configs/eval/progress_test_remote.yaml)。运行前设置 OpenAI-compatible
服务信息：

```bash
export OPENAI_API_KEY='your-api-key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
```

连续执行采样、推理和指标计算：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml
```

命令行会在交互式终端中使用 `tqdm` 分别展示 Sample、Infer 和 Metrics 三个阶段的进度，并输出阶段开始、完成及断点续跑跳过数量。动态进度写入 stderr，最终 JSON 摘要写入 stdout，因此可以安全地重定向结果：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml > summary.json
```

在 CI、管道等非交互环境中，动态进度条会自动关闭，阶段日志仍会保留。也可以手动关闭动态进度条：

```bash
python -m prmeval.cli run --config configs/eval/progress_test_remote.yaml --no-progress
```

也可以单独运行各阶段：

```bash
python -m prmeval.cli sample --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli infer --config configs/eval/progress_test_remote.yaml
python -m prmeval.cli metrics --config configs/eval/progress_test_remote.yaml
```
查看已注册组件：

```bash
python -m prmeval.cli list-datasets
python -m prmeval.cli list-samplers
python -m prmeval.cli list-infers
python -m prmeval.cli list-metrics
```

完整命令和验证方式见 [三阶段评测流程](docs/PIPELINE.md)，冒烟测试的输入与预期产物见 [完整冒烟测试说明](examples/stage_full_smoke/README.md)。

## 文档

- [配置文件说明](docs/CONFIGURATION.md)
- [本地优先模型接入](docs/LOCAL_MODELS.md)
- [三阶段评测流程](docs/PIPELINE.md)
- [全流程运行产物说明](docs/ARTIFACTS.md)
- [EvaluationRecord 数据结构](docs/RECORD_SCHEMA.md)
- [本地数据格式与 Dataset Adapter](docs/DATASETS.md)
- [原始数据集统一工具](dataset_unify/README.md)

## 推理代码结构

```text
prmeval/infer/
├── __init__.py          # 注册加载与 create_infer() 公共入口
├── base.py              # HTTP、认证、重试、请求计数和图片编码
├── model.py             # 本地优先 ProgressModel、remote context 与通用 adapter
├── openai.py            # Chat Completions 与 JSON Schema 校验
├── mock_server.py       # 本地 OpenAI-compatible contract 服务
└── baselines/
    ├── __init__.py      # built-in 模型注册
    ├── progress_test.py # 通用 OpenAI-compatible 远程冒烟模型
    ├── gvl.py
    ├── roboreward.py
    ├── robodopamine.py
    ├── topreward.py
    ├── vlac.py
    ├── rbm_model.py     # 同时注册 rbm 和 rewind
    ├── rlvlmf.py
    └── rbd_inference.py
```

新增本地 progress 模型时，在 `baselines/` 中创建与 registry 名称一致的文件，使用
`ProgressModel` 接口实现 `compute_progress()`，可选实现原生 `compute_progress_batch()` 和
`remote_compute_progress()`，然后在 `baselines/__init__.py` 中导入并调用 `register_progress_model(...)`。
详细接口见 [本地优先模型接入](docs/LOCAL_MODELS.md)。

`progress_test` 不加载本地 checkpoint，仅用于检查通用 OpenAI-compatible 远程协议以及三个 Stage 的产物
衔接；真实 baseline 的算法正确性仍应使用各模型自己的回归测试验证。

原始数据需要先通过独立的 [`dataset_unify`](dataset_unify/) 工具转换为本地标准 Hugging Face Dataset，再交给 PRMEval 读取。数据统一的字段、配置和新增数据集方法均以该工具自己的文档为准。

## 致谢与来源说明

本项目基于开源项目 [Robometer](https://github.com/robometer/robometer) 进行重构与扩展。

感谢 Robometer 项目的作者和贡献者开源其代码。本仓库中的部分实现来源于 Robometer，并在此基础上进行了代码结构重组、重构以及功能扩展，以适配本项目的具体需求。
