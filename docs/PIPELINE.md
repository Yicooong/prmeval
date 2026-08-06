# 三阶段统一数据协议

PRMEval 的采样和推理阶段共用 `bench.record.v1` JSONL 协议。Stage 1 创建记录，Stage 2 在同一条记录上补充模型结果，Stage 3 只读取推理完成的记录。

```text
DatasetAdapter -> Sampler -> EvaluationRecord(stage=sampled)
                                      |
                                      v
                                Baseline 推理
                                      |
                                      v
                           EvaluationRecord(stage=inferred)
                                      |
                                      v
                               Metric 结果 JSON
```

## 阶段职责

### Stage 1：sample

Stage 1 负责：

- 将不同数据源转换为统一内部轨迹；
- 按 eval type 采样任务和图像帧；
- 构造指标真值 `target`；
- 将图像保存为 `.npz`，在 Record 中只保留 `FrameReference`；
- 写入 `stage: sampled` 的 `samples.jsonl`。

当前 v1 不包含 prefix sampling。一条 `sample_id` 对应一次模型请求和一条完整预测曲线。

### Stage 2：infer

Stage 2 从 sampled Record 中提取 `input.task` 和图像帧发送给 baseline。`target` 不会出现在远程请求中。

模型响应经过 baseline adapter 归一化后，Stage 2 在原 Record 上补充：

```text
baseline
prediction
execution
stage = inferred
```

成功记录写入 `predictions.jsonl`，失败记录写入 `errors.jsonl`。`sample_id`、`evaluation`、`input` 和 `target` 不得改变。

### Stage 3：metric

Stage 3 读取 `stage: inferred` 且 `execution.status: success` 的记录。Metric 不修改 `EvaluationRecord.stage`，因为 MSE、Pearson、Kendall 等指标通常需要重新计算或跨多条样本聚合。

指标完成状态属于 `all_metrics.json` 和 `run_manifest.json`，不属于单条 EvaluationRecord。

## 唯一标识

`sample_id` 是唯一的跨阶段主键：

```text
Stage 1 sample_id -> Stage 2 sample_id -> Stage 3 明细 sample_id
```

单个 run 只对应一个 baseline，因此 `sample_id` 在该 run 的 predictions 文件中唯一。合并多个 run 时使用联合身份：

```text
(dataset.name, baseline.name, sample_id)
```

原始数据中的轨迹编号可以放在 `source.id`，但它只用于审计和定位，不参与核心去重与 reward alignment 分组。

## Reward alignment 数据流

Stage 1 产生：

```json
{
  "target": {"kind": "progress", "values": [0.0, 0.5, 1.0]}
}
```

Stage 2 补充：

```json
{
  "prediction": {"kind": "progress", "values": [0.0, 0.4, 0.8]}
}
```

Stage 3 直接比较两个等长数组，逐 sample 计算 MSE 和 Pearson，再对 sample 等权平均。结果按 `evaluation.dataset.name` 和 `baseline.name` 切片。

## Policy ranking 数据流

Policy ranking 使用：

```text
target(kind=rank).value
prediction(kind=progress).values
input.task_id
```

Kendall 按 dataset、baseline 和 task 分组后计算。

## 命令

```bash
prmeval-eval sample --config config.yaml
prmeval-eval validate-samples --samples evaluation_output/<run>/samples.jsonl

prmeval-eval infer --config config.yaml
prmeval-eval validate-predictions --predictions evaluation_output/<run>/predictions.jsonl

prmeval-eval metrics --config config.yaml
```

完整字段、必填规则和 JSON 示例见 [RECORD_SCHEMA.md](RECORD_SCHEMA.md)。
