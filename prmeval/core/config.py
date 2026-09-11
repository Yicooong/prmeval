from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

TemporalTransform = Literal["pause", "slow", "fast", "rewind", "retry", "truncate", "skip"]


class ConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemporalRobustnessConfig(ConfigBase):
    random_seed: int = Field(
        default=42, description="采样随机种子。用于时序变换生成以及策略排名、偏好比较和混淆矩阵采样中的随机选择。"
    )
    max_frames: int = Field(
        default=16, ge=1, description="时序变换后单条序列的帧数硬上限。progress_temporal_variation 评测要求至少为 6。"
    )
    min_length_ratio: float = Field(
        default=0.7,
        ge=0.7,
        le=1.0,
        description="变换后帧数相对 sampling.base_frames 的最小比例。帧数下限按乘积向上取整。",
    )
    max_length_ratio: float = Field(
        default=1.7,
        ge=1.0,
        le=1.7,
        description="变换后帧数相对 sampling.base_frames 的最大比例。帧数上限按乘积向下取整并受 max_frames 限制。",
    )
    transforms: list[TemporalTransform] = Field(
        default_factory=lambda: ["pause", "slow", "fast", "rewind", "retry", "truncate", "skip"],
        description="需要生成的时序变换类型。列表不能为空且不能重复。除这些变换外还会保留原始序列。",
    )
    variants_per_transform: int = Field(
        default=1, ge=1, description="每条轨迹中每种时序变换生成的变体数量。原始序列不计入此数量。"
    )
    pause_extra_ratio_range: tuple[float, float] = Field(
        default=(0.2, 0.7),
        description="pause 变换新增重复帧数相对基准帧数的比例取值区间。最终新增帧数受序列长度上限限制。",
    )
    slow_gamma_range: tuple[float, float] = Field(
        default=(1.5, 3.0),
        description="slow 变换的时间映射指数 gamma 的随机取值区间。默认大于 1 的指数使前段进度变慢。",
    )
    fast_gamma_range: tuple[float, float] = Field(
        default=(0.33, 0.67),
        description="fast 变换的时间映射指数 gamma 的随机取值区间。默认小于 1 的指数使前段进度变快。",
    )
    peak_progress_range: tuple[float, float] = Field(
        default=(0.6, 0.9), description="rewind 和 retry 变换到达的峰值位置范围。以基准序列中的相对位置比例表示。"
    )
    retreat_ratio_range: tuple[float, float] = Field(
        default=(0.25, 0.6),
        description="rewind 和 retry 从峰值位置回退的比例取值区间。回退比例以已到达的峰值位置为基准。",
    )
    rewind_extra_ratio_range: tuple[float, float] = Field(
        default=(0.2, 0.7),
        description="rewind 变换新增帧数相对基准帧数的比例取值区间。最终新增帧数受序列长度上限限制。",
    )
    retry_extra_ratio_range: tuple[float, float] = Field(
        default=(0.2, 0.7), description="retry 变换新增帧数相对基准帧数的比例取值区间。最终新增帧数受序列长度上限限制。"
    )
    truncate_retained_ratio_range: tuple[float, float] = Field(
        default=(0.7, 0.9),
        description="truncate 变换保留的序列前缀帧数相对基准帧数的比例取值区间。保留帧数受序列长度下限限制。",
    )
    skip_removed_ratio_range: tuple[float, float] = Field(
        default=(0.1, 0.3),
        description="skip 变换从序列中间移除的帧数相对基准帧数的比例取值区间。保留首尾帧并满足序列长度下限。",
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> TemporalRobustnessConfig:
        unit_ranges = (
            "pause_extra_ratio_range",
            "peak_progress_range",
            "retreat_ratio_range",
            "rewind_extra_ratio_range",
            "retry_extra_ratio_range",
            "truncate_retained_ratio_range",
            "skip_removed_ratio_range",
        )
        for name in unit_ranges:
            lower, upper = getattr(self, name)
            if not 0 <= lower <= upper <= 1:
                raise ValueError(f"temporal_robustness.{name} must be an ordered range within [0, 1]")
        for name in ("slow_gamma_range", "fast_gamma_range"):
            lower, upper = getattr(self, name)
            if not 0 < lower <= upper:
                raise ValueError(f"temporal_robustness.{name} must be an ordered positive range")
        if not self.transforms:
            raise ValueError("temporal_robustness.transforms must not be empty")
        if len(self.transforms) != len(set(self.transforms)):
            raise ValueError("temporal_robustness.transforms must not contain duplicates")
        return self


class SamplingConfig(ConfigBase):
    dataset_name: str = Field(
        default="eval_dataset", description="评测记录中的数据集名称。参与样本 ID 和默认 task_name 的生成。"
    )
    paths: list[str] = Field(
        default_factory=list,
        description="输入 JSONL 文件或本地 Hugging Face Dataset 目录列表。相对路径以当前工作目录为基准。",
    )
    save_samples: bool = Field(
        default=False,
        description="是否在采样阶段保存样本 JSONL 和帧文件。默认不保存。启用时必须设置顶层 output_dir。",
    )
    max_trajectories: int | None = Field(
        default=None, ge=1, description="跨所有输入路径最多加载的有效轨迹总数。None 表示不限制。"
    )
    base_frames: int = Field(
        default=8,
        ge=1,
        description="普通评测每条轨迹最多采样的帧数。同时作为时序变换前的基准帧数。时序鲁棒性评测要求至少为 5。",
    )
    progress_type: Literal["absolute_first_frame", "absolute_wrt_total_frames", "relative_first_frame"] = Field(
        default="absolute_first_frame",
        description=(
            "进度标签计算方式。absolute_first_frame 以首个采样帧为零点计算进度。absolute_wrt_total_frames 按源帧序号"
            "和总帧数计算进度。relative_first_frame 输出相邻采样帧的进度增量。"
        ),
    )

    # eval_type 配置参数
    num_examples_per_quality: int | None = Field(
        default=5,
        ge=1,
        description="policy_ranking 中每个任务的每个质量等级或完成度分组最多采样的轨迹数。None 表示不限制每组数量。",
    )
    num_partial_successes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "policy_ranking 使用连续完成度时每个任务最多选择的轨迹总数。None 时改用 num_examples_per_quality 逐组限制。"
        ),
    )
    max_tasks: int | None = Field(
        default=None, ge=1, description="policy_ranking 最多评测的可比较任务数。None 表示不限制。"
    )
    comparisons_per_task: int | None = Field(
        default=None, ge=1, description="quality_preference 中每个任务最多生成的轨迹比较对数。None 表示不限制。"
    )
    max_comparisons: int | None = Field(
        default=None,
        ge=1,
        description="quality_preference 跨全部任务的轨迹比较对数上限。在每任务限制之后应用。None 表示不限制。",
    )
    trajectories_per_source: int | None = Field(
        default=None,
        ge=1,
        description="confusion_matrix 构造任务匹配组合前每个数据来源最多选择的轨迹数。None 表示不限制。",
    )
    temporal_robustness: TemporalRobustnessConfig = Field(
        default_factory=TemporalRobustnessConfig,
        description="时序鲁棒性评测的变换参数。其 random_seed 同时供其他随机采样器使用。",
    )


class InferConfig(ConfigBase):
    name: str = Field(description="推理模型在 INFERS 注册表中的名称。必填。例如 openai_compatible 或 robometer。")
    base_url: str | None = Field(
        default=None,
        description="模型服务的请求地址。为 None 时尝试读取环境变量 BASE_URL。变量不存在则保持 None。",
    )
    api_key: str | None = Field(
        default=None, description="模型服务访问密钥。为 None 时尝试读取环境变量 API_KEY。变量不存在则保持 None。"
    )
    model_id: str | None = Field(
        default=None,
        description="用于请求和推理记录的模型标识。为 None 时读取环境变量 MODEL_ID。变量不存在则保持 None。",
    )
    model_path: str | None = Field(
        default=None,
        description="本地模型权重路径或 Hugging Face 模型仓库 ID。用于需要加载权重的模型。具体要求由模型实现决定。",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="runner 每批交给 predict 的样本数。模型内部批大小可通过 model_extra_config 配置。",
    )
    model_extra_config: dict[str, Any] = Field(
        default_factory=dict, description="由具体推理模型读取的扩展参数。可包含权重加载选项和模型内部批处理设置。"
    )

    @model_validator(mode="after")
    def resolve_environment_defaults(self) -> InferConfig:
        if self.model_id is None:
            self.model_id = os.getenv("MODEL_ID")
        if self.base_url is None:
            self.base_url = os.getenv("BASE_URL")
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY")
        return self


class EvalConfig(ConfigBase):
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="采样阶段配置。包含数据集输入、采样数量、评测专用参数和样本保存开关。",
    )
    infer: InferConfig = Field(description="推理阶段的模型选择与参数配置。单独运行采样阶段也需提供 infer.name。")
    eval_types: str = Field(
        default="progress",
        min_length=1,
        pattern=r"\S",
        description="单次评测的类型名称。同时选择同名采样器和指标。例如 progress 或 progress_temporal_variation。",
    )
    output_dir: str | None = Field(
        default="/tmp/prmeval_evaluation_output",
        description=(
            "评测产物根目录。实际文件写入其 task_name 子目录。"
            "None 关闭全部产物写入并要求 sampling.save_samples 为 false。"
        ),
    )
    task_name: str | None = Field(
        default=None,
        min_length=1,
        pattern=r"\S",
        description="任务输出子目录名称。None 时自动生成为 {sampling.dataset_name}_{infer.name}_{eval_types}。",
    )
    resume: bool = Field(
        default=True,
        description="推理阶段是否复用已有成功预测并跳过对应样本。仅在保存产物时生效。采样阶段仍会重新执行。",
    )

    @model_validator(mode="after")
    def resolve_output(self) -> EvalConfig:
        if self.output_dir is not None and not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty; use null to disable artifact writes")
        if self.task_name is None:
            self.task_name = f"{self.sampling.dataset_name}_{self.infer.name}_{self.eval_types}"
        return self

    @model_validator(mode="after")
    def validate_sample_output(self, info: ValidationInfo) -> EvalConfig:
        if self.sampling.save_samples and self.output_dir is None:
            raise ValueError("sampling.save_samples=True requires output_dir")
        context = info.context if isinstance(info.context, dict) else {}
        if context.get("samples_path") is not None and not self.sampling.save_samples:
            raise ValueError("samples_path output requires output_dir and sampling.save_samples=True")
        return self

    @model_validator(mode="after")
    def validate_temporal_robustness(self) -> EvalConfig:
        if self.eval_types != "progress_temporal_variation":
            return self
        temporal = self.sampling.temporal_robustness
        if temporal.max_frames < 6:
            raise ValueError("progress_temporal_variation requires temporal_robustness.max_frames >= 6")
        if self.sampling.progress_type == "relative_first_frame":
            raise ValueError("progress_temporal_variation requires an absolute progress_type")
        if self.sampling.base_frames < 5:
            raise ValueError("progress_temporal_variation requires sampling.base_frames >= 5")
        if self.sampling.base_frames > temporal.max_frames:
            raise ValueError("sampling.base_frames must not exceed temporal_robustness.max_frames")
        increasing = {"pause", "rewind", "retry"}.intersection(temporal.transforms)
        upper = min(int(temporal.max_length_ratio * self.sampling.base_frames), temporal.max_frames)
        if increasing and upper <= self.sampling.base_frames:
            raise ValueError("pause, rewind, and retry require room to increase beyond sampling.base_frames")
        decreasing = {"truncate", "skip"}.intersection(temporal.transforms)
        lower = math.ceil(temporal.min_length_ratio * self.sampling.base_frames)
        if decreasing and lower >= self.sampling.base_frames:
            raise ValueError("truncate and skip require room to decrease below sampling.base_frames")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))

    @classmethod
    def from_json(cls, path: str | Path) -> EvalConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_file(cls, path: str | Path) -> EvalConfig:
        path = Path(path)
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        raise ValueError(f"Unsupported config file extension {path.suffix!r}; expected .yaml, .yml, or .json")

    @classmethod
    def export_config(cls, format: Literal["yaml", "json"] = "yaml") -> str:
        """Export declared defaults as YAML or JSON, using null for missing required values."""
        if format not in {"yaml", "json"}:
            raise ValueError(f"Unsupported export format {format!r}; expected 'yaml' or 'json'")
        required_fields: list[str] = []

        def defaults_for(model: type[BaseModel], prefix: str = "") -> dict[str, Any]:
            defaults: dict[str, Any] = {}
            for name, field in model.model_fields.items():
                field_path = f"{prefix}{name}"
                if field.is_required():
                    annotation = field.annotation
                    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                        value = defaults_for(annotation, prefix=f"{field_path}.")
                    else:
                        value = None
                        required_fields.append(field_path)
                else:
                    value = field.get_default(call_default_factory=True)
                    if isinstance(value, BaseModel):
                        value = value.model_dump(mode="json")
                defaults[name] = value
            return defaults

        template = defaults_for(cls)
        if format == "json":
            return json.dumps(template, ensure_ascii=False, indent=2) + "\n"
        header = f"# Generated from {cls.__name__} field defaults.\n"
        if required_fields:
            header += f"# Required fields to fill before loading: {', '.join(required_fields)}\n"
        return header + yaml.safe_dump(template, allow_unicode=True, sort_keys=False, default_flow_style=None)
