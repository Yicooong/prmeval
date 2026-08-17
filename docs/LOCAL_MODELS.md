# 本地优先模型接入

本地 progress 模型通过 `ProgressModel` 接入 Stage 2。模型文件负责本地计算，并可选择在同一个类中增加
remote 方法；通用 adapter 负责把 `EvaluationSample` 转换成数组并构造 `ProgressPrediction`。

## 最小本地模型

在 `prmeval/infer/baselines/my_model.py` 中定义：

```python
from typing import Any

import numpy as np

from ..model import ProgressModel, register_progress_model


@register_progress_model("my_model")
class MyModel(ProgressModel):
    def __init__(self, model_path: str, dtype: str = "bfloat16", **_: Any):
        # 本地依赖必须在加载模型时导入，不能放在模块顶层。
        import torch
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, dtype),
            device_map="auto",
        )

    def compute_progress(
        self,
        frames_array: np.ndarray,
        task_description: str = "",
        reference_video_path: str | None = None,
    ) -> np.ndarray:
        # 返回值必须与 frames_array 等长、有限且位于 [0, 1]。
        return self._run_model(frames_array, task_description)
```

并从 `prmeval/infer/baselines/__init__.py` 导入 `MyModel`，使装饰器在 `prmeval.infer` 导入时完成注册。

配置：

```yaml
infer:
  name: my_model
  mode: local
  model_path: /models/my-model
  model_id: my-model-v1
  batch_size: 1
  max_concurrency: 1
  options:
    dtype: bfloat16
```

`options` 会传给 `load_local()`，其默认实现等价于 `MyModel(model_path=model_path, **options)`。模型只在
`create_infer()` 时加载一次。

## 本地原生 batch

支持真正 tensor batch 的模型声明 `supports_local_batch = True` 并覆盖：

```python
class MyModel(ProgressModel):
    supports_local_batch = True

    def compute_progress_batch(
        self,
        frames_list: list[np.ndarray],
        task_descriptions: list[str],
        reference_video_paths: list[str | None] | None = None,
    ) -> list[np.ndarray]:
        # processor 应一次处理多条输入，并尽量只调用一次模型 forward/generate。
        return self._run_batched(frames_list, task_descriptions)
```

`supports_local_batch` 未开启时配置 `batch_size > 1` 会在模型加载前立即报错，避免把逐条循环误认为 GPU
batch。不同视频长度可以在模型内部按帧数分桶或拆成 micro-batch；Runner 只负责顶层 sample 分组。

## 可选远程方法

远程模式不会构造本地模型实例。将远程方法写成 classmethod，并复用 `RemoteContext` 的认证、重试、图片编码
和 OpenAI-compatible JSON 解析：

```python
from ..base import vision_content
from ..model import ProgressResult, RemoteContext


class MyModel(ProgressModel):
    supports_remote = True

    @classmethod
    def remote_compute_progress(
        cls,
        frames_array: np.ndarray,
        task_description: str,
        reference_video_path: str | None,
        remote: RemoteContext,
        options: dict,
    ) -> ProgressResult:
        schema = {
            "name": "progress_prediction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "progress": {
                        "type": "array",
                        "minItems": len(frames_array),
                        "maxItems": len(frames_array),
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    }
                },
                "required": ["progress"],
                "additionalProperties": False,
            },
        }
        content = [{"type": "text", "text": cls.build_prompt(task_description)}]
        content.extend(vision_content(frames_array))
        parsed, raw = remote.chat([{"role": "user", "content": content}], schema)
        return ProgressResult(parsed["progress"], raw_response=raw)
```

remote 配置：

```yaml
infer:
  name: my_model
  mode: remote
  base_url: BASE_URL
  api_key: OPENAI_API_KEY
  model_id: MODEL_ID
  batch_size: 1
  max_concurrency: 8
```

共享 prompt、输出解析和归一化建议保留为模型类的静态方法。本地与远程函数只分别实现 Hugging Face
forward/generate 和 HTTP 调用，不要在计算流程内部反复判断 mode。

## Runner 调度

Runner 先按 `batch_size` 将未完成记录分组，再使用最多 `max_concurrency` 个线程调用
`infer.predict_batch()`：

```text
local:  batch_size=4, max_concurrency=1
        [s1,s2,s3,s4] -> [s5,s6,s7,s8] -> [s9,s10]

remote: batch_size=1, max_concurrency=4
        s1, s2, s3, s4 同时请求；完成后继续处理后续样本
```

NPZ 加载错误只影响对应样本；一次模型 batch 整体抛错时，该 batch 内所有样本写入 `errors.jsonl`。减小
`batch_size` 后可通过 `resume: true` 重跑失败样本。Runner 不会自动在 CUDA OOM 后拆 batch 重试。

## 依赖与约束

- torch、transformers 和模型专用库必须懒加载。
- local 模式固定 `max_concurrency: 1`；多 GPU 应使用独立进程和独立模型实例，而不是多线程共享模型。
- Stage 2 不重新抽帧；模型必须对 Stage 1 的全部输入帧返回等长 progress。
- `reference_video_path` 可从 trajectory metadata 传入，但相对路径比绝对路径更利于移动 sample bundle。
- 完整 logits、teacher forcing 等能力应只在 local `compute_progress()` 中实现；能力不足的 remote 方法不能静默冒充等价算法。

## Built-in baselines

local-first 模型现已统一位于 `prmeval/infer/baselines/`，导入 `prmeval.infer` 时会自动完成注册。
`progress_test` 是仅远程的通用 OpenAI-compatible 测试模型，用于验证 Stage 1 → Stage 2 → Stage 3
完整流程；它不加载本地 checkpoint。可直接使用
[`configs/eval/progress_test_remote.yaml`](../configs/eval/progress_test_remote.yaml) 运行。

各模型 local/remote 支持情况及 TOPReward 精确 logits 限制见
[`baselines/README.md`](../prmeval/infer/baselines/README.md)。
