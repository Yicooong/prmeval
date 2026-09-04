# Infer 模型接入

Stage 2 使用统一抽象类 `Infer`。框架不区分本地和远程执行；checkpoint、provider SDK 或 HTTP 服务均由具体
baseline 自己管理。Runner 根据 `infer.name` 从 `INFERS` registry 取出类，在有待处理样本时构造一次，并调用：

```python
predict(samples: list[EvaluationSample]) -> list[Prediction]
```

`infer.batch_size` 决定 Runner 每次传入的样本数量。没有独立的 `predict_batch()` 接口。

## Progress baseline

在 `prmeval/infer/baselines/my_model.py` 中定义：

```python
from typing import ClassVar

import numpy as np

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction, ProgressPrediction, ProgressSample
from ..base import Infer, model_identity


@register_infer("my_model")
class MyModel(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("my_model requires infer.model_path")

        # 重型可选依赖放在模型构造阶段导入。
        import torch
        from transformers import AutoModel

        dtype = config.options.get("dtype", "bfloat16")
        self.model = AutoModel.from_pretrained(
            config.model_path,
            torch_dtype=getattr(torch, dtype),
        )

    def predict(self, samples: list[EvaluationSample]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for sample in samples:
            if not isinstance(sample, ProgressSample):
                raise TypeError("my_model only supports progress samples")

            values = np.asarray(
                self._run_model(sample.trajectory.frames, sample.trajectory.task),
                dtype=float,
            ).reshape(-1)
            if len(values) != len(sample.trajectory.frames):
                raise ValueError("Progress length must match the input frame count")
            if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError("Progress values must be finite and within [0, 1]")

            predictions.append(
                ProgressPrediction(
                    sample_id=sample.sample_id,
                    progress=values.tolist(),
                    model=model_identity(self.config),
                    model_version=self.config.model_version,
                )
            )
        return predictions
```

随后在 `prmeval/infer/baselines/__init__.py` 中导入该类，使注册装饰器在导入 `prmeval.infer` 时执行。
内置 baseline 也遵循“一种模型一个模块、由 `baselines/__init__.py` 显式导入”的约定。

## 返回值契约

Runner 会校验每批返回结果：

- 必须返回 `list`；
- Prediction 数量必须与输入样本数量相同；
- `sample_id` 不得重复，并且必须与输入集合完全一致；
- 返回顺序可以不同，Runner 会按 `sample_id` 重新对应；
- 每条预测还必须符合相应 sample 类型，例如 progress 长度必须等于帧数。

一次 `predict()` 调用抛出异常，或整批返回值不符合契约时，该批所有样本写入 `errors.jsonl`，后续批次继续执行。
因此不支持真正批量推理的实现，也应在 `predict()` 内遍历输入列表并逐条产生结果；如需缩小故障范围，可把
`infer.batch_size` 设为 `1`。

## Preference baseline

Preference 模型同样继承 `Infer`，声明 `capabilities = {"preference"}`，接收 `PreferenceSample` 并返回
`PreferencePrediction`。当前仓库内置 baseline 都只声明 `progress` 能力；使用 `quality_preference` 前需要接入
自定义 preference baseline。

## 配置映射

每个 baseline 构造函数接收完整 `InferConfig`：

```yaml
infer:
  name: my_model
  model_path: /models/my-model
  model_id: my-model-v1
  model_version: v1
  batch_size: 8
  options:
    dtype: bfloat16
```

远程 OpenAI-compatible 模型使用相同结构：

```yaml
infer:
  name: openai_compatible
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  timeout_seconds: 120
  max_retries: 2
  batch_size: 1
```

框架不会根据这些字段选择执行模式。具体 baseline 负责校验自身需要的字段：checkpoint 模型通常要求
`model_path`，OpenAI-compatible 模型通常要求 `base_url` 和 `model_id`，provider 模型读取自己的 key 或 options。

## 生命周期与依赖

`begin_prediction()` 是每批预测前的可选钩子，默认不执行任何操作。`attempts()` 可供模型报告最近调用次数；
目前 Runner 的记录协议不保存该值。

Runner 不提供线程池或 local/remote transport 分派。模型内部的 tensor micro-batch、prefix 推理或请求策略都属于
baseline 私有实现，不应再增加一层 Runner adapter。

Torch、Transformers 以及模型专属库属于可选依赖，建议在 baseline 构造阶段导入。`openai_compatible`
baseline 使用官方 OpenAI Python SDK 调用兼容服务。远程响应解析失败时可通过 `RemoteError.raw_response`
将原始响应写入错误记录。
