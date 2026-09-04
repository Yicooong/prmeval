# 安装与 wheel 构建

PRMEval 支持 Python 3.10 及以上版本。安装后只提供 `prmeval` 命令行入口。

## 从源码安装

开发环境推荐使用 editable 安装。若需要加载全部内置本地模型，同时安装两个模型 extras：

```bash
python -m pip install -e '.[dev,local-hf,local-qwen]'
prmeval --help
```

只安装项目声明的核心依赖：

```bash
python -m pip install .
```

需要本地 Hugging Face/Qwen-VL 模型时安装对应可选依赖：

```bash
python -m pip install '.[local-hf,local-qwen]'
```

内置 baseline 会在 `prmeval.infer` 导入时完成注册，其中部分模块会直接导入本地模型依赖。若当前环境没有这些
依赖，请安装上述 extras。个别 baseline 还可能依赖自身的模型代码或第三方库，具体要求以导入或构造时报错为准。

## 构建 wheel

在仓库根目录执行：

```bash
python -m pip install build
python -m build --wheel
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

## 基本用法

先将示例配置中的 `sampling.paths` 改为轨迹 JSONL 文件或本地 Hugging Face Dataset 目录，再执行：

```bash
prmeval run --config configs/eval/openai_compatible_remote.yaml
prmeval sample --config configs/eval/openai_compatible_remote.yaml
prmeval infer --config configs/eval/openai_compatible_remote.yaml
prmeval metrics --config configs/eval/openai_compatible_remote.yaml
```

`prmeval` 不会自动读取 `.env`。配置使用环境变量名时，必须在运行命令前导出对应变量。
