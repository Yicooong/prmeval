"""Registered dataset-specific conversion entry points.

Loader modules are imported only after their registration matches, so optional
dataset dependencies do not affect unrelated conversions.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module
from typing import Any

from datasets import Dataset

from dataset_unify.helpers import flatten_task_data
from dataset_unify.registry import DATASET_CONVERTERS, MatchMode, register_dataset_converter

ConverterResult = list[dict[str, Any]] | Dataset
KwargsFactory = Callable[[Any], dict[str, Any]]


def _import_function(module_name: str, function_name: str) -> Callable[..., Any]:
    module = import_module(module_name)
    return getattr(module, function_name)


def _disable_cuda_for_tensorflow() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _register_trajectory_converter(
    name: str,
    module_name: str,
    function_name: str,
    *,
    patterns: tuple[str, ...] | None = None,
    kwargs_factory: KwargsFactory | None = None,
) -> None:
    def converter(cfg: Any) -> ConverterResult:
        print(f"Loading {name} dataset from: {cfg.dataset.dataset_path}")
        loader = _import_function(module_name, function_name)
        kwargs = kwargs_factory(cfg) if kwargs_factory else {}
        return flatten_task_data(loader(cfg.dataset.dataset_path, **kwargs))

    DATASET_CONVERTERS.register(name, patterns=patterns)(converter)


def _register_direct_converter(
    name: str,
    module_name: str,
    function_name: str,
    *,
    patterns: tuple[str, ...] | None = None,
    match_mode: MatchMode = MatchMode.CONTAINS,
    kwargs_factory: KwargsFactory | None = None,
    include_num_workers: bool = True,
    prepare: Callable[[], None] | None = None,
) -> None:
    def converter(cfg: Any) -> ConverterResult:
        if prepare:
            prepare()
        print(f"Converting {name} dataset directly to HF from: {cfg.dataset.dataset_path}")
        conversion_function = _import_function(module_name, function_name)
        kwargs = {
            "dataset_path": cfg.dataset.dataset_path,
            "dataset_name": cfg.dataset.dataset_name,
            "output_dir": cfg.output.output_dir,
            "max_trajectories": cfg.output.max_trajectories,
            "max_frames": cfg.output.max_frames,
            "fps": cfg.output.fps,
        }
        if include_num_workers:
            kwargs["num_workers"] = cfg.output.num_workers
        if kwargs_factory:
            kwargs.update(kwargs_factory(cfg))
        return conversion_function(**kwargs)

    DATASET_CONVERTERS.register(name, patterns=patterns, match_mode=match_mode)(converter)


# Registrations intentionally retain the old dispatch order for overlapping names.
_register_trajectory_converter("libero", "dataset_unify.dataset_loaders.libero_loader", "load_libero_dataset")


@register_dataset_converter("agibotworld")
def convert_agibotworld(cfg: Any) -> ConverterResult:
    converter = _import_function(
        "dataset_unify.dataset_loaders.agibotworld_loader", "convert_agibotworld_streaming_to_hf"
    )
    return converter(
        dataset_name=cfg.dataset.dataset_path,
        output_dir=cfg.output.output_dir,
        dataset_label=cfg.dataset.dataset_name or "agibotworld",
        max_trajectories=cfg.output.max_trajectories,
        max_frames=cfg.output.max_frames,
        fps=cfg.output.fps,
        num_workers=cfg.output.num_workers,
    )


_register_trajectory_converter(
    "egodex",
    "dataset_unify.dataset_loaders.egodex_loader",
    "load_egodex_dataset",
    kwargs_factory=lambda cfg: {"max_trajectories": cfg.output.max_trajectories},
)
_register_direct_converter(
    "oxe",
    "dataset_unify.dataset_loaders.oxe_loader",
    "convert_oxe_dataset_to_hf",
    patterns=("oxe_",),
    match_mode=MatchMode.PREFIX,
    prepare=_disable_cuda_for_tensorflow,
)
_register_trajectory_converter(
    "robofail",
    "dataset_unify.dataset_loaders.robofail_loader",
    "load_robofail_dataset",
    kwargs_factory=lambda cfg: {"max_trajectories": cfg.output.max_trajectories},
)
_register_trajectory_converter(
    "metaworld",
    "dataset_unify.dataset_loaders.mw_collected_loader",
    "load_metaworld_dataset",
    kwargs_factory=lambda cfg: {"dataset_name": cfg.dataset.dataset_name},
)
_register_direct_converter("h2r", "dataset_unify.dataset_loaders.h2r_loader", "convert_h2r_dataset_to_hf")
_register_direct_converter(
    "fino-net",
    "dataset_unify.dataset_loaders.fino_net_loader",
    "convert_fino_net_dataset_to_hf",
    patterns=("fino_net", "fino-net"),
)
_register_direct_converter(
    "epic",
    "dataset_unify.dataset_loaders.epic_loader",
    "convert_epic_dataset_to_hf",
    kwargs_factory=lambda cfg: {
        "shortest_edge_size": cfg.output.shortest_edge_size,
        "center_crop": cfg.output.center_crop,
    },
)
_register_trajectory_converter(
    "roboarena", "dataset_unify.dataset_loaders.roboarena_loader", "load_roboarena_dataset"
)
_register_trajectory_converter("ph2d", "dataset_unify.dataset_loaders.ph2d_loader", "load_ph2d_dataset")
_register_direct_converter(
    "galaxea", "dataset_unify.dataset_loaders.galaxea_loader", "convert_galaxea_dataset_to_hf"
)
_register_direct_converter(
    "molmoact",
    "dataset_unify.dataset_loaders.molmoact_loader",
    "convert_molmoact_dataset_to_hf",
    include_num_workers=False,
)
_register_trajectory_converter(
    "auto_eval", "dataset_unify.dataset_loaders.autoeval_loader", "load_autoeval_dataset"
)
_register_trajectory_converter(
    "usc_xarm_policy_ranking",
    "dataset_unify.dataset_loaders.usc_xarm_policy_ranking_loader",
    "load_usc_xarm_policy_ranking_dataset",
    kwargs_factory=lambda cfg: {"max_trajectories": cfg.output.max_trajectories},
)
_register_trajectory_converter(
    "usc_franka_policy_ranking",
    "dataset_unify.dataset_loaders.usc_franka_policy_ranking_loader",
    "load_usc_franka_policy_ranking_dataset",
    kwargs_factory=lambda cfg: {"max_trajectories": cfg.output.max_trajectories},
)
_register_trajectory_converter(
    "utd_so101_policy_ranking",
    "dataset_unify.dataset_loaders.utd_so101_loader",
    "load_utd_so101_dataset",
    kwargs_factory=lambda cfg: {
        "max_trajectories": cfg.output.max_trajectories,
        "is_robot": True,
        "data_source": "utd_so101",
    },
)
_register_trajectory_converter(
    "utd_so101_human",
    "dataset_unify.dataset_loaders.utd_so101_loader",
    "load_utd_so101_dataset",
    kwargs_factory=lambda cfg: {
        "max_trajectories": cfg.output.max_trajectories,
        "is_robot": False,
        "data_source": "utd_so101_human",
    },
)
_register_direct_converter("soar", "dataset_unify.dataset_loaders.soar_loader", "convert_soar_dataset_to_hf")
_register_direct_converter(
    "mit_franka_p-rank",
    "dataset_unify.dataset_loaders.mit_franka_prank_loader",
    "convert_mit_franka_prank_dataset_to_hf",
)
_register_direct_converter(
    "rfm_new_mit_franka",
    "dataset_unify.dataset_loaders.new_mit_franka_loader",
    "convert_new_mit_franka_dataset_to_hf",
    kwargs_factory=lambda cfg: {"exclude_wrist_cam": cfg.dataset.exclude_wrist_cam},
)


def _clean_policy_ranking_kwargs(cfg: Any) -> dict[str, Any]:
    dataset_name = cfg.dataset.dataset_name.lower()
    if "wrist" in dataset_name:
        return {"view": "wrist"}
    if "top" in dataset_name:
        return {"view": "top"}
    raise ValueError(f"Dataset name must specify view (wrist or top): {cfg.dataset.dataset_name}")


_register_direct_converter(
    "utd_so101_clean_policy_ranking",
    "dataset_unify.dataset_loaders.utd_so101_clean_policy_ranking_loader",
    "convert_utd_so101_clean_policy_ranking_to_hf",
    kwargs_factory=_clean_policy_ranking_kwargs,
)


def _paired_trajectory_kwargs(cfg: Any) -> dict[str, Any]:
    dataset_name = cfg.dataset.dataset_name.lower()
    if "usc_koch_human_robot_paired_human" in dataset_name:
        return {"trajectory_type": "human"}
    if "usc_koch_human_robot_paired_robot" in dataset_name:
        return {"trajectory_type": "robot"}
    raise ValueError(
        "Dataset name must specify either 'usc_koch_human_robot_paired_human' or "
        f"'usc_koch_human_robot_paired_robot': {cfg.dataset.dataset_name}"
    )


_register_direct_converter(
    "usc_koch_human_robot_paired",
    "dataset_unify.dataset_loaders.usc_koch_human_robot_paired_loader",
    "convert_usc_koch_human_robot_paired_to_hf",
    kwargs_factory=_paired_trajectory_kwargs,
)
_register_direct_converter(
    "usc_koch_p_ranking",
    "dataset_unify.dataset_loaders.usc_koch_p_ranking_loader",
    "convert_usc_koch_p_ranking_to_hf",
)
_register_trajectory_converter("egocot", "dataset_unify.dataset_loaders.egocot_loader", "load_egocot_dataset")
_register_direct_converter(
    "humanoid_everyday",
    "dataset_unify.dataset_loaders.humanoid_everyday_loader",
    "convert_humanoid_everyday_dataset_to_hf",
)
_register_trajectory_converter("motif", "dataset_unify.dataset_loaders.motif_loader", "load_motif_dataset")
_register_trajectory_converter(
    "failsafe", "dataset_unify.dataset_loaders.failsafe_loader", "load_failsafe_dataset"
)
_register_trajectory_converter(
    "racer",
    "dataset_unify.dataset_loaders.racer_loader",
    "load_racer_dataset",
    kwargs_factory=lambda cfg: {"dataset_name": cfg.dataset.dataset_name},
)
_register_trajectory_converter(
    "hand_paired",
    "dataset_unify.dataset_loaders.hand_paired_loader",
    "load_hand_paired_dataset",
    kwargs_factory=lambda cfg: {"dataset_name": cfg.dataset.dataset_name},
)
_register_trajectory_converter(
    "roboreward",
    "dataset_unify.dataset_loaders.roboreward_loader",
    "load_roboreward_dataset",
    kwargs_factory=lambda cfg: {"dataset_name": cfg.dataset.dataset_name},
)
_register_trajectory_converter(
    "robofac",
    "dataset_unify.dataset_loaders.robofac_loader",
    "load_robofac_dataset",
    kwargs_factory=lambda cfg: {"max_trajectories": cfg.output.max_trajectories},
)
_register_trajectory_converter(
    "rbm-1m-ood",
    "dataset_unify.dataset_loaders.rbm_1m_ood_loader",
    "load_rbm_1m_ood_dataset",
    kwargs_factory=lambda cfg: {"max_trajectories": cfg.output.max_trajectories},
)


def get_dataset_converter(dataset_name: str) -> Callable[[Any], ConverterResult]:
    return DATASET_CONVERTERS.get(dataset_name)
