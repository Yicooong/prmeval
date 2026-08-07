# Dataset Unify：数据集统一工具

`dataset_unify` 是与 PRMEval 评测代码隔离的数据转换工具。它只负责读取不同来源的原始数据集，统一轨迹字段和视频存储方式，并将结果保存为本地 Hugging Face Dataset。

```text
原始数据集
    -> dataset_unify
    -> 本地标准 Hugging Face Dataset
    -> prmeval-data-preprocess
    -> PRMEval DatasetAdapter
    -> 采样、推理和指标计算
```

`dataset_unify` 不负责模型训练、评测、远程推理或数据上传，也不会生成语言向量。转换过程不需要 `SentenceTransformer`，输出中没有 `lang_vector`。

## 标准输出格式

保存到本地的 Dataset 使用以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | `str` | 全局唯一的轨迹 ID |
| `task` | `str` | 自然语言任务描述 |
| `data_source` | `str` | 数据来源名称 |
| `frames` | `str` | 相对于最终 Dataset 根目录的本地 MP4 路径 |
| `is_robot` | `bool` | 是否为机器人轨迹 |
| `quality_label` | `str \| None` | 如 `successful`、`suboptimal`、`failure` |
| `partial_success` | `float \| None` | `[0, 1]` 范围内的连续完成度 |

标准 Dataset 不包含 `lang_vector`、模型 embedding 或训练索引。当前只支持视频模式；原始 loader 内部可以暂时使用帧数组、可调用加载器或源视频数据，但写入最终 Dataset 前必须转换为本地 MP4 和标准字段。

所有通用和数据集专用 loader 最终都通过 [`hf_schema.py`](hf_schema.py) 的 `build_standard_dataset()` 构造 Dataset。该函数会筛选固定的 7 个字段、保留真正的 `None`、验证 `partial_success` 范围，并将 `success/fail/failed` 规范为 `successful/failure`：

```python
dataset = build_standard_dataset(entries)
```

## 目录结构

```text
dataset_unify/
├── generate_hf_dataset.py       统一转换入口
├── hf_schema.py                 固定的 7 字段 Dataset schema 与构造器
├── helpers.py                   视频与轨迹标准化工具
├── validate_dataset.py          本地 Dataset 字段校验
├── visualize_dataset.py         本地 Dataset 可视化
├── dataset_loaders/             各原始数据集的 loader
├── configs/data_gen_configs/    各数据集的转换配置
└── dataset_guides/              数据集说明与接入指南
```

各 loader 只处理源数据差异。目前存在两种内部返回方式：

- 通用 loader 返回 `dict[str, list[dict]]`，再由主程序统一生成视频和 Dataset；
- 需要流式读取或特殊视频处理的 loader 直接返回 `datasets.Dataset`。

这只是转换层内部实现差异。两种路径最终都调用 `build_standard_dataset()`；空 Dataset 和非空 Dataset 也使用相同 feature schema。PRMEval 不会直接导入这些 loader。

## 快速开始

先复制一个已有配置并修改原始数据路径、数据集名称和输出目录：

```yaml
dataset:
  dataset_path: /path/to/raw_dataset
  dataset_name: mit_franka_p-rank_rfm

output:
  output_dir: /path/to/unified_datasets
  max_trajectories: -1
  max_frames: 32
  use_video: true
  fps: 10
  shortest_edge_size: 240
  center_crop: false
  num_workers: 4
```

从项目根目录运行：

```bash
python -m dataset_unify.generate_hf_dataset \
  --config_path=dataset_unify/configs/data_gen_configs/mit_franka_prank.yaml
```

转换结果只保存在本地：

```text
<output.output_dir>/
└── <dataset_name>/
    ├── dataset_info.json
    ├── state.json
    ├── data-*.arrow
    ├── batch_*/trajectory_*.mp4
    └── ...
```

不同 loader 的视频分片目录名称可能不同，但 `frames` 都以最终 Dataset 目录为根。例如 Dataset 位于 `/data/unified/demo/` 时，字段应为 `shard_0000/episode_000001/clip.mp4`，而不是 `demo/shard_0000/episode_000001/clip.mp4`。

当前只支持 `output.use_video: true`。图片序列模式尚未形成与 PRMEval 一致的落盘协议，配置为 `false` 时转换器会明确报错。

## 校验本地 Dataset

```bash
python -m dataset_unify.validate_dataset \
  /path/to/unified_datasets/<dataset_name>
```

校验器会检查严格的 7 字段 schema、主要字段类型、质量标签以及 `partial_success` 的取值范围；缺少字段或包含旧的额外字段都会校验失败。

## 交给 PRMEval 使用

PRMEval 当前通过 `processed_cache` adapter 读取 NPZ 帧缓存。因此，统一后的本地 Hugging Face Dataset 还需要执行一次评测预处理：

视频解码优先使用 PyAV；如果当前环境没有 PyAV，则自动使用系统 `ffmpeg` 和 `ffprobe`。预处理生成的 `Trajectory` 会继续保留 `is_robot`。

```yaml
# configs/data/my_local_dataset.yaml
output_dir: /path/to/processed_datasets
max_frames: 32

sources:
  - path: /path/to/unified_datasets/<dataset_name>
    cache_name: <dataset_name>
```

```bash
prmeval-data-preprocess --config configs/data/my_local_dataset.yaml
```

评测配置使用生成的缓存：

```yaml
dataset:
  name: <dataset_name>
  adapter: processed_cache
  root: /path/to/processed_datasets
```

也可以通过环境变量设置缓存根目录：

```bash
export PRMEVAL_PROCESSED_DATASETS_PATH=/path/to/processed_datasets
```

## 新增数据集

1. 在 `dataset_loaders/` 中创建 `{dataset_name}_loader.py`。
2. 将源字段转换为至少包含 `id`、`task`、`frames`、`data_source`、`is_robot` 的轨迹字典。
3. 在 `generate_hf_dataset.py` 中注册分发逻辑。
4. 在 `configs/data_gen_configs/` 中添加本地配置。
5. 执行转换和 `validate_dataset` 校验。

更具体的源数据布局说明见 [`dataset_guides/`](dataset_guides/)。

## 接口测试

端到端 contract test 会真实生成一个最小 MP4，并验证以下完整链路：

```text
MP4 + build_standard_dataset
    -> save_to_disk
    -> prmeval-data-preprocess
    -> ProcessedCacheDatasetAdapter
    -> Trajectory
    -> RewardAlignmentSampler
```

运行：

```bash
/mnt/shared-storage-user/liuyicong/miniconda3/envs/bench/bin/python \
  -m unittest tests.test_dataset_unify_contract -v
```
