# 安装与 wheel 构建

PRMEval 支持 Python 3.10 及以上版本。安装后只提供 `prmeval` 命令行入口。

## 从源码安装

开发环境推荐使用 editable 安装。包含各内置本地模型所需的 Python 依赖：

```bash
python -m pip install -e '.[dev,sole-r1]'
prmeval --help
```

只安装项目声明的核心依赖：

```bash
python -m pip install .
```

需要本地 Hugging Face/Qwen-VL 模型时安装对应可选依赖：

```bash
python -m pip install '.[local-qwen]'
```

核心依赖包含 OpenCV（`opencv-python-headless`），用于读取帧和视频，无需桌面 GUI。
当前实现按选择加载 baseline，普通安装可使用 CLI 和远程推理；本地模型需要以下 extras：

| 功能 | 安装命令 | 额外依赖 |
|---|---|---|
| Hugging Face 基础模型运行 | `pip install '.[local-hf]'` | torch、transformers、accelerate、safetensors、huggingface-hub |
| Qwen-VL / RoboReward / TopReward / RoboMeter | `pip install '.[local-qwen]'` | 包含 local-hf，另加 qwen-vl-utils |
| Sole-R1 | `pip install '.[sole-r1]'` | 包含 local-qwen，另加 TRL |
| Sole-R1 辅助绘图 | `pip install '.[sole-r1,visualization]'` | 另加 matplotlib |

使用 wheel 时，将命令中的 `.` 替换为 wheel 路径，例如
`python -m pip install './dist/prmeval-0.1.0-py3-none-any.whl[sole-r1]'`。

模型权重需另行下载或通过配置指定。CUDA/PyTorch/torchvision 的版本需与运行机器匹配。
启用量化时需额外安装 `bitsandbytes`；选择 Unsloth 时需安装 `unsloth`；Flash Attention 属于可选加速，
需根据 CUDA 和 PyTorch 版本单独安装 `flash-attn`。这些平台相关扩展不随默认安装引入。
版本下限是依赖约束，不代表已验证所有版本组合；发布前还需在目标 Python 和模型环境测试。

`dataset_unify/` 是源码仓库中的独立工具，不随 PRMEval wheel 安装。其转换入口使用 `pyrallis`，
不同 loader 还会使用 `h5py`、`pandas`、`pyarrow`、`torchvision`、TensorFlow/TFDS 或 LeRobot 等。
运行转换工具时应按所选数据源单独准备环境，不能将 `pip install prmeval` 视为安装所有转换器依赖。

## 构建 wheel

在仓库根目录执行：

```bash
python -m pip install '.[dev]'
python -m build
python -m twine check dist/*
```

生成文件位于 `dist/`，例如：

```text
dist/prmeval-0.1.0-py3-none-any.whl
```

安装并检查命令：

```bash
python -m pip install dist/prmeval-*.whl
prmeval --help
prmeval list-infers
```

wheel 只收集 `prmeval/` 下的包文件，以及安装所必需的 `*.dist-info` 元数据。`dataset_unify/`、`tests/`、
`configs/`、`docs/` 和 `examples/` 不会进入 wheel。`sole_r1/preprocessor_config.json` 通过 package-data 配置随包安装。

`python -m build` 先构建源码包，再从源码包构建 wheel。建议在全新环境安装 wheel，离开源码目录后
运行 `prmeval --help` 和 `prmeval list-infers`，避免 editable 安装或当前目录掩盖漏打包的问题。
示例 YAML 和文档需从源码仓库获取，或自行编写配置。

## 发布前检查

- 运行 `pytest`、`ruff check .`、`ruff format --check .`。当前 `.gitignore` 忽略整个 `tests/`，
  发布前应整理测试及 fixtures 并纳入版本管理，确保其他环境可复现验证。
- 检查 wheel 和源码包内容，确认没有本地数据、凭据或私有配置。
- 仓库目前未提供 LICENSE；发布前应确定项目许可证并保留引用代码要求的版权和许可文件，再填写许可证元数据。
- 补充实际仓库与问题反馈 URL；README 中的相对图片和文档链接需要核验在 PyPI 上的展示。
- 在支持的 Python 版本上验证基础安装与各模型 extra；实际 GPU 推理需独立验证。

配置依据：[Python Packaging User Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)。

## 基本用法

先将示例配置中的 `sampling.paths` 改为轨迹 JSONL 文件或本地 Hugging Face Dataset 目录，再执行：

```bash
prmeval run --config configs/eval/openai_compatible_remote.yaml
prmeval sample --config configs/eval/openai_compatible_remote.yaml
prmeval infer --config configs/eval/openai_compatible_remote.yaml
prmeval metrics --config configs/eval/openai_compatible_remote.yaml
```

`prmeval` 不会自动读取 `.env`。配置使用环境变量名时，必须在运行命令前导出对应变量。
