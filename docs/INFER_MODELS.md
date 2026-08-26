# Infer 模型接入

Stage 2 只保留一个抽象父类 `Infer`。框架不区分 local 和 remote；checkpoint、provider 或 HTTP 服务由具体 baseline 自己决定。Runner 根据 `infer.name` 从 `INFERS` 取出类并构造一次。单样本入口是 `predict()`，需要后端批处理的 baseline 可以选择覆写 `predict_batch()`。

## Progress baseline

在 `prmeval/infer/baselines/my_model.py` 中定义：

```python
from typing import ClassVar

import numpy as np

from ...core.config import InferConfig
from ...core.registry import register_infer
from ...core.schemas import EvaluationSample, Prediction
from ..base import Infer, predict_progress


@register_infer("my_model")
class MyModel(Infer):
    capabilities: ClassVar[set[str]] = {"progress"}

    def __init__(self, config: InferConfig):
        super().__init__(config)
        if not config.model_path:
            raise ValueError("my_model requires infer.model_path")
        options = config.options

        # 重型可选依赖必须在构造时导入。
        import torch
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            config.model_path,
            torch_dtype=getattr(torch, options.get("dtype", "bfloat16")),
        )

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        return self._run_model(frames_array, task_description, reference_video_path)

    def predict(self, samples: List[EvaluationSample]) -> List[Prediction]:
        return predict_progress(self, sample)
```

`predict_progress()` 会完成输入类型校验，传递 frames、task 和可选 reference path，并构造 `ProgressPrediction`。共享输出校验要求：

- 返回值可转换为一维浮点数组；
- 数量与输入帧严格一致；
- 不含 NaN/Inf；
- 所有值位于 `[0,1]`。

随后在 `baselines/__init__.py` 中导入该类，使注册装饰器在导入 `prmeval.infer` 时执行。

## Preference baseline

Preference 模型同样直接继承 `Infer`，声明 `capabilities = {"preference"}`，并在自己的 `predict()` 中校验 `PreferenceSample`、调用算法函数并构造 `PreferencePrediction`。当前内置 preference baseline 是 `rlvlmf`，其算法入口为 `compute_preference()`。

## 配置映射

每个 baseline 构造函数接收完整 `InferConfig`：

```yaml
infer:
  name: my_model
  model_path: /models/my-model
  model_id: my-model-v1
  model_version: v1
  options:
    dtype: bfloat16
```

远程模型也使用相同结构，例如：

```yaml
infer:
  name: progress_test
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  timeout_seconds: 120
  max_retries: 2
```

框架不根据字段选择运行模式。具体 baseline 负责校验自己需要的字段：checkpoint 模型通常要求 `model_path`，OpenAI-compatible 模型通常要求 `base_url` 和 `model_id`，provider 模型读取自身的 API key/options。

## 可选批量入口

默认的 `Infer.predict_batch()` 会逐条调用 `predict()`。Runner 会识别 baseline 是否覆写了该方法：未覆写时继续逐条执行并隔离单样本错误；覆写后则按 `infer.batch_size` 一次传入一批：

```python
def predict_batch(self, samples: list[EvaluationSample]) -> list[Prediction]:
    return self._backend_batch(samples)
```

返回结果可以调整顺序，但必须与输入具有完全相同且不重复的 `sample_id`。整批调用抛出异常，或返回数量、ID
集合不合法时，该批所有样本记为失败，后续批次继续执行。sample 迭代器没有第二个 batch size。

## 调度与依赖约束

Runner 不提供线程池或 local/remote 执行分派。默认路径顺序逐样本执行 `infer.predict(sample)`；可选批量路径按配置分组并调用 `infer.predict_batch(samples)`。

Torch、Transformers、OpenCV、evo_vlac 等可选依赖不得阻塞 `import prmeval.infer`。应在具体模型构造时导入，或使用安全的可选依赖检查。

OpenAI-compatible 请求使用组合式 `OpenAIChatClient`；它不是 `Infer` 子类。成功的 progress prediction 只保存标准化进度值，远程原始响应仅在失败时通过 `RemoteError.raw_response` 进入错误记录。
