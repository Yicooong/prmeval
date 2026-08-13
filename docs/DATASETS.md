# 本地数据格式与 Dataset Adapter

Dataset adapter 将不同本地存储格式转换成内部统一的 `Trajectory`。当前提供 `jsonl` 和 `huggingface` 两种 adapter。

原始数据集的标准化不属于 PRMEval 评测流程。需要转换异构数据时，请使用独立的 [`dataset_unify`](../dataset_unify/) 工具；其标准字段、转换配置、校验命令和新增 loader 方法见 [`dataset_unify/README.md`](../dataset_unify/README.md)。

## JSONL adapter

JSONL adapter 适合冒烟测试、小型自定义数据集和已经具有 NPZ 帧的数据。推荐目录结构：

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

NPZ 文件必须包含名为 `frames` 的数组，推荐格式：

```text
shape = [T, H, W, C]
dtype = uint8
C = 3
```

JSONL 中的相对帧路径以该 JSONL 文件所在目录为基准。

评测配置示例：

```yaml
sampling:
  dataset_name: my-dataset
  adapter: jsonl
  paths: [my_dataset/trajectories.jsonl]
```

数据源 JSONL 和 Stage 1 生成的 `samples.jsonl` 使用不同协议：

```text
trajectories.jsonl  --jsonl adapter-->  Trajectory
Trajectory          --sampler------->  samples.jsonl
samples.jsonl        --Stage 2------>   predictions.jsonl
```

## Hugging Face adapter

Hugging Face adapter 适合较大的本地数据集和包含视频的正式评测。`paths` 中的每一项都必须是可由 `datasets.load_from_disk()` 读取的完整目录：

```yaml
sampling:
  dataset_name: rbm-1m-ood
  adapter: huggingface
  paths: [/path/to/hf_datasets/rbm-1m-ood]
```

目录中的 `frames` 可以是内嵌数组、NPZ 路径或视频路径，也兼容 `frames_video`、`video`、`frames_path` 字段。相对路径以各自的 Dataset 目录为基准解析；`dataset_name` 只用于评测记录和 sample ID，不用于拼接磁盘路径。

由 `dataset_unify` 生成的标准 Hugging Face Dataset 可以直接交给该 adapter。MP4 解码优先使用 PyAV，不可用时回退到系统 `ffmpeg`/`ffprobe`。

## 从原始数据到评测

完整的数据边界是：

```text
异构原始数据
    -> dataset_unify
    -> 本地标准 Hugging Face Dataset
    -> huggingface adapter
    -> PRMEval Stage 1 / Stage 2 / Stage 3
```

PRMEval 只读取转换后的本地产物，不依赖各原始数据集的 loader，也不会上传数据。
