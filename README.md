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

构建可分发的 wheel 并安装：

```bash
python -m pip install build
python -m build --wheel
pip install dist/prmeval-*.whl
```

安装后会提供 `prmeval` 命令。

本地 Hugging Face/Qwen-VL 模型使用可选依赖：

```bash
pip install -e '.[local-hf,local-qwen]'
```

## 快速开始

仓库提供了调用通用远程模型 `progress_test` 的端到端冒烟配置
[`configs/eval/progress_test_remote.yaml`](configs/eval/progress_test_remote.yaml)。运行前设置 OpenAI-compatible
服务信息，并将配置中的 `sampling.paths` 改为由 `datasets.save_to_disk()` 保存的本地 Dataset 目录：

```bash
export OPENAI_API_KEY='your-api-key'
export BASE_URL='https://your-service.example.com/v1'
export MODEL_ID='your-model-id'
```

连续执行采样、推理和指标计算：

```bash
prmeval run --config configs/eval/progress_test_remote.yaml
```

命令行会在交互式终端中使用 `tqdm` 分别展示 Sample、Infer 和 Metrics 三个阶段的进度，并输出阶段开始、完成及断点续跑跳过数量。动态进度写入 stderr，最终 JSON 摘要写入 stdout，因此可以安全地重定向结果：

```bash
prmeval run --config configs/eval/progress_test_remote.yaml > summary.json
```

在 CI、管道等非交互环境中，动态进度条会自动关闭，阶段日志仍会保留。也可以手动关闭动态进度条：

```bash
prmeval run --config configs/eval/progress_test_remote.yaml --no-progress
```

也可以单独运行各阶段：

```bash
prmeval sample --config configs/eval/progress_test_remote.yaml
prmeval infer --config configs/eval/progress_test_remote.yaml
prmeval metrics --config configs/eval/progress_test_remote.yaml
```
查看已注册组件：

```bash
prmeval list-samplers
prmeval list-infers
prmeval list-metrics
```

完整命令和验证方式见 [三阶段评测流程](docs/PIPELINE.md)，冒烟测试的输入与预期产物见 [完整冒烟测试说明](examples/stage_full_smoke/README.md)。

## 文档

- [安装与 wheel 构建](docs/INSTALLATION.md)
- [配置文件说明](docs/CONFIGURATION.md)
- [Infer 模型接入](docs/INFER_MODELS.md)
- [三阶段评测流程](docs/PIPELINE.md)
- [全流程运行产物说明](docs/ARTIFACTS.md)
- [EvaluationRecord 数据结构](docs/RECORD_SCHEMA.md)
- [本地 Hugging Face Dataset](docs/DATASETS.md)
- [原始数据集统一工具](dataset_unify/README.md)

## 致谢与来源说明

本项目基于开源项目 [Robometer](https://github.com/robometer/robometer) 进行重构与扩展。

感谢 Robometer 项目的作者和贡献者开源其代码。本仓库中的部分实现来源于 Robometer，并在此基础上进行了代码结构重组、重构以及功能扩展，以适配本项目的具体需求。
