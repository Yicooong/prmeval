# EvaluationRecord 数据结构说明

`EvaluationRecord` 保存评测类型 `eval_type`，并组合 `EvaluationSample`、可选的 `Prediction` 和运行信息 `ExecutionInfo`。
Stage 1 生成 sample，Stage 2 添加 prediction 或错误信息，Stage 3 直接读取 sample 中的真值和 prediction 计算指标。

## 顶层结构

```python
class EvaluationRecord(FrameworkModel):
    eval_type: str
    sample: EvaluationSample
    prediction: Prediction | None = None
    execution: ExecutionInfo | None = None
```

成功的进度记录示例（省略可选字段）：

```json
{
  "eval_type": "progress",
  "sample": {
    "sample_id": "stable-sample-id",
    "dataset_name": "dataset",
    "trajectory": {
      "id": "trajectory-001",
      "task": "open the drawer",
      "frames": [],
      "frame_indices": [0, 4, 8],
      "target_progress": [0.0, 0.5, 1.0]
    }
  },
  "prediction": {
    "sample_id": "stable-sample-id",
    "model": "actual-model",
    "progress": [0.0, 0.4, 0.9]
  },
  "execution": {
    "status": "success",
    "infer_name": "openai_compatible"
  }
}
```

上述 `frames: []` 表示结果中已清除帧数组。可重新推理的采样文件必须包含有效帧引用。
采样记录的 prediction、execution 均为空；`predictions.jsonl` 保存成功记录，`errors.jsonl` 保存失败记录。

## sample

`EvaluationSample` 为 `ProgressSample | PreferenceSample`。两者都必须包含：

| 字段 | 含义 |
|---|---|
| `sample_id` | 一次模型请求的稳定 ID |
| `dataset_name` | 评测数据集名称，用于指标切片 |

`eval_type` 是 Record 的必填字段，用于选择评测，例如 `progress`、`policy_ranking`。
Sample 不包含评测类型；同一 Sample 可以由不同评测的 Record 引用。

`ProgressSample` 包含一个 `trajectory`，用于进度、时序变化、策略排名和任务匹配评测。
`PreferenceSample` 包含 `chosen_trajectory` 和 `rejected_trajectory`，用于偏好评测。
JSON 通过这些不同的结构字段恢复样本类型，不增加类型标记字段。

Trajectory 保留原有完整结构：`id`、`task`、`frames`、`data_source`、`is_robot`、`is_simulation`、
`quality_label`、`partial_success`、`preference_group_id`、`preference_rank`、`target_progress`、
`frame_indices`、`num_frames_total` 和 `metadata`。文件读写仅替换 frames 的表达，不重建或合并标签。
偏好样本中的两条轨迹分别保留自己的任务、数据来源和标签。

`sample_id` 与轨迹的 `id` 含义不同：同一原始轨迹可以派生多个采样请求。
`dataset_name` 表示评测数据集，`trajectory.data_source` 表示轨迹原始来源。
内置采样器从配置填入 dataset_name；自定义采样器也必须设置该字段。

## 真值和预测

Record 不再单独保存 target。指标从 Sample 读取真值：

| 评测 | 真值来源 | 预测 |
|---|---|---|
| 进度、时序变化 | `trajectory.target_progress` | `ProgressPrediction.progress` |
| 策略排名 | 依次选择非空的 `partial_success`、`preference_rank`、质量标签等级 | 进度曲线 |
| 偏好 | chosen/rejected 配对关系 | preference、chosen_probability |
| 任务匹配 | `trajectory.metadata.lang_task` 与 `video_task` 是否相同 | 进度曲线末值 |

质量标签等级：`successful=2`、`suboptimal=1`、`failure/failed=0`。数值 0 是有效真值，不触发回退。
时序变换信息保存在 `trajectory.metadata.synthetic_temporal`。

`Prediction` 为 `ProgressPrediction | PreferencePrediction`，保留自身的 sample_id 和 model，可独立使用：

```json
{"sample_id": "s1", "model": "actual-model", "progress": [0.0, 0.5, 1.0]}
{"sample_id": "s2", "model": "actual-model", "preference": "chosen", "chosen_probability": 0.8}
```

progress 为非空的 `[0,1]` 数值列表。preference 可为 chosen、rejected、tie，chosen_probability 位于 `[0,1]`。
模型接入代码继续构造现有 Prediction，不再转换为通用 ValuePayload。

## execution

`infer_name` 为实际使用的注册实现名称。成功时，实际模型身份由 `prediction.model` 保存。
失败时没有 Prediction，通过可选的 `execution.model` 保留尝试调用的模型；框架写入失败记录时会填充它。

```json
{
  "status": "error",
  "infer_name": "openai_compatible",
  "model": "attempted-model",
  "error": "TimeoutError: request timed out",
  "raw_response": null
}
```

`raw_response` 仅用于失败诊断。成功的 execution 不得包含非空的 model、error 或 raw_response。

## 状态约束

- execution 为空时，prediction 必须为空。
- execution 存在时，必须有 status 和 infer_name。
- success 必须有 Prediction，其 sample_id 必须等于 sample.sample_id。
- ProgressSample 对应 ProgressPrediction，PreferenceSample 对应 PreferencePrediction。
- error 必须有非空错误信息，且不能有 Prediction。
- 推理时额外校验 progress 数量与实际输入帧数相等。

## 帧存储和加载

内存中的 `trajectory.frames` 可以是数组。采样落盘时，数组保存为 NPZ，frames 替换为：

```json
{
  "type": "npz",
  "path": "sample_frames/id-trajectory.npz",
  "key": "frames",
  "num_frames": 3,
  "sha256": "64-character-sha256"
}
```

路径相对于样本文件目录。读入时校验路径、哈希、帧数、采样索引和进度标签长度。
`load_record_frames(record, bundle_dir).sample` 获取包含数组的样本副本，原记录仍保留文件引用。
指标只读取标签、元数据和预测，不加载帧。

`save_samples_to_bundle(records, path)` 接收尚无推理结果的 EvaluationRecord，保留每条记录的 eval_type，
并使用各 Sample 自带的 dataset_name。同一文件可以包含不同评测类型的记录。
内置采样器仍返回纯 Sample；Runner 使用采样器的 eval_type 包装 Record，推理结果继承源 Record 的 eval_type。
不再需要 sample_to_record 或 record_to_sample：分别使用 `EvaluationRecord(eval_type="progress", sample=sample)` 和 `record.sample`。

## 格式变更

本次 JSONL 格式不兼容旧的 evaluation/input/target/infer 结构；旧文件需要重新生成或显式转换。
不提供自动迁移或旧字段别名。顶层仅允许 eval_type、sample、prediction、execution，也不包含 schema_version。
配置 YAML/JSON 格式和 NPZ 帧格式不变。

旧的 `sample.eval_type` 已移至顶层 `eval_type`，读取旧 JSONL 前需移动该字段；Sample 不接受旧字段别名。
