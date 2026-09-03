# 本地 Hugging Face Dataset

PRMEval Stage 1 只读取由 `datasets.load_from_disk()` 保存的本地 Hugging Face Dataset，不再提供
JSONL 数据源或 Dataset Adapter。异构原始数据应先通过独立的 [`dataset_unify`](../dataset_unify/)
工具转换成统一 Dataset。

## 配置

`sampling.paths` 中的每一项都必须是一个完整的 Dataset 或 DatasetDict 目录：

```yaml
sampling:
  dataset_name: rbm-1m-ood
  paths: [/path/to/hf_datasets/rbm-1m-ood]
  eval_types: [progress]
```

`dataset_name` 用于评测记录和 sample ID，不参与磁盘路径拼接。相对路径以运行命令时的当前目录为基准。

## 数据池

`load_hf_trajectory_pool()` 依次读取 `paths`、将 Dataset 行标准化为 `Trajectory`，过滤掉失败、次优、
部分成功和完全未标注的轨迹，并返回一个遵守 `max_trajectories` 限制的 `list[Trajectory]`。成功轨迹定义为
`quality_label: successful`，或在质量标签缺失时 `partial_success: 1.0`：

```text
local Hugging Face Dataset
    -> EvalSampler.pool
    -> Trajectory
    -> EvalSampler.sample()
    -> ProgressSample / PreferenceSample
```

Runner 每次运行只加载一次 Dataset，并把同一个 pool 列表注入所有 sampler。不同 sampler 可以独立遍历、
分组和筛选列表，但不应修改共享列表本身。视频帧仍然保持路径引用，只在具体 sampler 抽帧时物化。

由于 pool 只包含成功轨迹，`quality_preference` 以及需要多个质量等级的 `policy_ranking` 不适用于该加载模式。

## 字段与帧加载

每行至少需要：

- `id`：轨迹 ID；
- `task`：自然语言任务；
- `frames`：内嵌数组、NPZ 路径或视频路径。

同时兼容 `frames_video`、`video`、`frames_path` 字段。相对帧路径以所属 Dataset 目录为基准解析。
视频列会尽量以 `Video(decode=False)` 读取，避免 Dataset 在遍历时提前解码；真正抽帧时再由公共帧加载工具
完成 NPZ 或视频物化。

模拟轨迹可设置 `is_simulation: true` 并提供与源视频帧数一致的逐帧 `target_progress`。这种轨迹在
progress 与 progress temporal variation 采样中直接复用所提供的目标进度（抽帧后按相同索引取值），
不会再按 `progress_type` 计算目标进度；长度不一致会在采样时直接报错。

由 `dataset_unify` 生成的标准 Dataset 可直接用于采样。MP4 解码优先使用 PyAV，不可用时回退到系统
`ffmpeg`/`ffprobe`。

## 完整数据边界

```text
异构原始数据
    -> dataset_unify
    -> 本地标准 Hugging Face Dataset
    -> EvalSampler.pool
    -> PRMEval Stage 1 / Stage 2 / Stage 3
```

PRMEval 不直接依赖原始数据集 loader，也不会上传数据。
