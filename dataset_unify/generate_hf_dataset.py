#!/usr/bin/env python3
"""Convert supported source datasets into a standard local Hugging Face Dataset."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide INFO/WARN/ERROR; only FATAL remains
import multiprocessing as mp

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Optional

from pyrallis import wrap
from tqdm import tqdm

from datasets import Dataset

# Dataset-specific loaders exchange plain dictionaries; PRMEval types are intentionally not imported here.
from dataset_unify.helpers import (
    create_hf_trajectory,
    create_output_directory,
    flatten_task_data,
)
from dataset_unify.hf_schema import build_standard_dataset

# make sure these come after importing torch. otherwise something breaks...
try:
    import absl.logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_trajectory_subdir_path(trajectory_idx: int, files_per_subdir: int = 1000) -> str:
    """
    Generate subdirectory path for a trajectory to avoid too many files per directory.

    Args:
        trajectory_idx: Index of the trajectory
        files_per_subdir: Maximum files per subdirectory (default: 1000)

    Returns:
        str: Subdirectory name like 'batch_0000'
    """
    subdir_index = trajectory_idx // files_per_subdir
    return f"batch_{subdir_index:04d}"


def save_dataset_locally(dataset: Dataset, output_dir: str, dataset_name: str) -> str:
    """Save a converted Hugging Face Dataset under its dataset-specific directory."""
    dataset_path = os.path.join(output_dir, dataset_name.lower())
    dataset.save_to_disk(dataset_path)
    print(f"Dataset saved locally to: {dataset_path}")
    return dataset_path


@dataclass
class DatasetConfig:
    """Config for dataset settings"""

    dataset_path: str = field(default="", metadata={"help": "Path to the dataset"})
    dataset_name: str = field(default=None, metadata={"help": "Name of the dataset (defaults to dataset_type)"})
    exclude_wrist_cam: bool = field(default=False, metadata={"help": "Exclude wrist camera views (MIT Franka only)"})


@dataclass
class OutputConfig:
    """Config for output settings"""

    output_dir: str = field(default="robometer_dataset", metadata={"help": "Output directory for the dataset"})
    max_trajectories: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of trajectories to process (None for all)"}
    )
    max_frames: int = field(
        default=64, metadata={"help": "Maximum number of frames per trajectory (-1 for no downsampling)"}
    )
    use_video: bool = field(default=True, metadata={"help": "Must be true; frame-image mode is not supported"})
    shortest_edge_size: Optional[int] = field(default=240, metadata={"help": "Shortest edge size for video resizing"})
    center_crop: bool = field(
        default=False,
        metadata={"help": "Center crop the video to the target size. Defaults to False, which means no cropping."},
    )
    fps: int = field(default=10, metadata={"help": "Frames per second for video creation"})
    num_workers: int = field(
        default=-1, metadata={"help": "Number of parallel workers for processing (-1 for auto, 0 for sequential)"}
    )


@dataclass
class GenerateConfig:
    """Main configuration for dataset generation"""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def process_single_trajectory(args):
    """
    Worker function to process a single trajectory.

    Args:
        args: Tuple containing trajectory metadata and conversion settings.

    Returns:
        Dict: Processed trajectory data or None if failed
    """
    trajectory_idx, trajectory, hf_creator_fn, output_dir, dataset_name, max_frames, use_video, fps = args

    try:
        # Create output directory for this trajectory with subdirectory structure
        subdir_name = get_trajectory_subdir_path(trajectory_idx)
        full_video_path = os.path.join(
            output_dir, dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
        )
        relative_video_path = os.path.join(subdir_name, f"trajectory_{trajectory_idx:04d}.mp4")
        os.makedirs(os.path.dirname(full_video_path), exist_ok=True)

        processed_trajectory = hf_creator_fn(
            traj_dict=trajectory,
            video_path=full_video_path,
            max_frames=max_frames,
            dataset_name=dataset_name,
            use_video=use_video,
            fps=fps,
        )

        if processed_trajectory is None:
            return None

        # Replace the full path with relative path in the processed trajectory
        if processed_trajectory and "frames" in processed_trajectory:
            processed_trajectory["frames"] = relative_video_path

        return processed_trajectory

    except Exception as e:
        print(f"❌ Error processing trajectory {trajectory_idx}: {e}")
        return None


def convert_dataset_to_hf_format(
    trajectories: list[dict],
    hf_creator_fn: Callable[..., Any],
    output_dir: str = "robometer_dataset",
    dataset_name: str = "",
    max_trajectories: int | None = None,
    max_frames: int = -1,
    use_video: bool = True,
    fps: int = 10,
    num_workers: int = -1,
) -> Dataset:
    """Convert a list of trajectories to HuggingFace format."""

    if not use_video:
        raise ValueError("output.use_video=false is not supported; dataset_unify currently writes MP4 videos only")

    print(f"Converting {dataset_name} dataset to HuggingFace format...")

    # Create output directory
    create_output_directory(output_dir)

    # Validate input
    if not trajectories:
        raise ValueError(f"No trajectories provided for {dataset_name} dataset.")

    print(f"Processing {len(trajectories)} trajectories")

    # Limit trajectories if specified
    if max_trajectories != -1:
        trajectories = trajectories[:max_trajectories]

    # Determine number of workers
    if num_workers == -1:
        num_workers = min(cpu_count(), len(trajectories))
    elif num_workers == 0:
        num_workers = 1  # Sequential processing

    print(f"Using {num_workers} worker(s) for parallel processing")

    # Process trajectories
    all_entries = []

    if num_workers == 1:
        for trajectory_idx, trajectory in enumerate(tqdm(trajectories, desc="Processing trajectories")):
            # Create output directory for this trajectory with subdirectory structure
            subdir_name = get_trajectory_subdir_path(trajectory_idx)
            trajectory_dir = os.path.join(
                output_dir, dataset_name.lower(), subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
            )
            os.makedirs(os.path.dirname(trajectory_dir), exist_ok=True)

            processed_trajectory = hf_creator_fn(
                traj_dict=trajectory,
                video_path=trajectory_dir,
                max_frames=max_frames,
                dataset_name=dataset_name,
                use_video=use_video,
                fps=fps,
            )
            if processed_trajectory is None:
                continue
            processed_trajectory["frames"] = os.path.join(
                subdir_name, f"trajectory_{trajectory_idx:04d}.mp4"
            )
            all_entries.append(processed_trajectory)
    else:
        # Parallel processing
        all_entries = []  # ensure defined if Pool raises before we filter results
        print(f"Preparing {len(trajectories)} trajectories for parallel processing...")

        # Prepare arguments for worker processes
        worker_args = []
        for trajectory_idx, trajectory in enumerate(trajectories):
            args = (
                trajectory_idx,
                trajectory,
                hf_creator_fn,
                output_dir,
                dataset_name,
                max_frames,
                use_video,
                fps,
            )
            worker_args.append(args)

        # Use spawn to avoid CUDA context issues from forking after TF import
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # Process trajectories in parallel
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_single_trajectory, worker_args),
                    total=len(worker_args),
                    desc="Processing trajectories",
                )
            )

        # Filter out failed trajectories (None results)
        all_entries = [result for result in results if result is not None]

        if len(all_entries) < len(trajectories):
            failed_count = len(trajectories) - len(all_entries)
            print(f"⚠️  {failed_count} trajectories failed to process and were skipped")

    # Create HuggingFace dataset with proper features
    print(f"Creating HuggingFace dataset with {len(all_entries)} entries...")

    dataset = build_standard_dataset(all_entries)

    print(f"{dataset_name} HuggingFace dataset created successfully!")
    print(f"Total entries: {len(all_entries)}")

    save_dataset_locally(dataset, output_dir, dataset_name)

    return dataset


@wrap()
def main(cfg: GenerateConfig):
    """Main function to convert any dataset to HuggingFace format."""

    if not cfg.output.use_video:
        raise ValueError("output.use_video=false is not supported; dataset_unify currently writes MP4 videos only")

    # Import the appropriate dataset loader and trajectory creator
    if "libero" in cfg.dataset.dataset_name:
        from dataset_unify.dataset_loaders.libero_loader import load_libero_dataset

        # Load the trajectories using the loader
        task_data = load_libero_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "agibotworld" in (cfg.dataset.dataset_name or "").lower():
        # Stream + convert directly inside the AgiBotWorld loader
        from dataset_unify.dataset_loaders.agibotworld_loader import (
            convert_agibotworld_streaming_to_hf,
        )

        dataset = convert_agibotworld_streaming_to_hf(
            dataset_name=cfg.dataset.dataset_path,
            output_dir=cfg.output.output_dir,
            dataset_label=cfg.dataset.dataset_name or "agibotworld",
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )
        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name or "agibotworld")
        print("Dataset conversion complete!")
        return

    elif "egodex" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.egodex_loader import load_egodex_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading EgoDex dataset from: {cfg.dataset.dataset_path}")
        task_data = load_egodex_dataset(
            cfg.dataset.dataset_path,
            cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif cfg.dataset.dataset_name.lower().startswith("oxe_"):
        # Treat OXE like AgiBotWorld: create videos and HF entries directly in the loader
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from dataset_unify.dataset_loaders.oxe_loader import convert_oxe_dataset_to_hf

        print(f"Converting OXE dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_oxe_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "robofail" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.robofail_loader import load_robofail_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading RoboFail dataset from: {cfg.dataset.dataset_path}")
        task_data = load_robofail_dataset(
            cfg.dataset.dataset_path,
            cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "metaworld" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.mw_collected_loader import load_metaworld_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading metaworld dataset from: {cfg.dataset.dataset_path}")
        task_data = load_metaworld_dataset(
            cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
        )
        trajectories = flatten_task_data(task_data)
    elif "h2r" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly inside the H2R loader (OXE-style)
        from dataset_unify.dataset_loaders.h2r_loader import convert_h2r_dataset_to_hf

        print(f"Converting H2R dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_h2r_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "fino_net" in cfg.dataset.dataset_name.lower() or "fino-net" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly inside the FinoNet loader (H2R/OXE-style)
        from dataset_unify.dataset_loaders.fino_net_loader import convert_fino_net_dataset_to_hf

        print(f"Converting FinoNet dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_fino_net_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "epic" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (H2R/OXE-style)
        from dataset_unify.dataset_loaders.epic_loader import convert_epic_dataset_to_hf

        print(f"Converting EPIC-KITCHENS dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_epic_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
            shortest_edge_size=cfg.output.shortest_edge_size,
            center_crop=cfg.output.center_crop,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "roboarena" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.roboarena_loader import load_roboarena_dataset

        # Load the trajectories using the loader with max_trajectories limit
        print(f"Loading RoboArena dataset from: {cfg.dataset.dataset_path}")
        task_data = load_roboarena_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "ph2d" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.ph2d_loader import load_ph2d_dataset

        print(f"Loading Ph2d dataset from: {cfg.dataset.dataset_path}")
        task_data = load_ph2d_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "galaxea" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (OXE-style, multi-dataset)
        from dataset_unify.dataset_loaders.galaxea_loader import convert_galaxea_dataset_to_hf

        rlds_datasets = getattr(cfg.dataset, "rlds_datasets", []) or []
        print(f"Converting Galaxea RLDS to HF from: {cfg.dataset.dataset_path} | datasets={rlds_datasets}")
        dataset = convert_galaxea_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "molmoact" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (LeRobot parquet)
        from dataset_unify.dataset_loaders.molmoact_loader import convert_molmoact_dataset_to_hf

        print(f"Converting MolmoAct dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_molmoact_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "auto_eval" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.autoeval_loader import load_autoeval_dataset

        print(f"Loading AutoEval dataset from: {cfg.dataset.dataset_path}")
        task_data = load_autoeval_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "usc_xarm_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.usc_xarm_policy_ranking_loader import (
            load_usc_xarm_policy_ranking_dataset,
        )

        print(f"Loading USC xArm Policy Ranking dataset from: {cfg.dataset.dataset_path}")
        task_data = load_usc_xarm_policy_ranking_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "usc_franka_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.usc_franka_policy_ranking_loader import (
            load_usc_franka_policy_ranking_dataset,
        )

        print(f"Loading USC Franka Policy Ranking dataset from: {cfg.dataset.dataset_path}")
        task_data = load_usc_franka_policy_ranking_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "utd_so101_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.utd_so101_loader import (
            load_utd_so101_dataset,
        )

        print(f"Loading UTD SO101 robot dataset from: {cfg.dataset.dataset_path}")
        task_data = load_utd_so101_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
            is_robot=True,
            data_source="utd_so101",
        )
        trajectories = flatten_task_data(task_data)
    elif "utd_so101_human" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.utd_so101_loader import (
            load_utd_so101_dataset,
        )

        print(f"Loading UTD SO101 human dataset from: {cfg.dataset.dataset_path}")
        task_data = load_utd_so101_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
            is_robot=False,
            data_source="utd_so101_human",
        )
        trajectories = flatten_task_data(task_data)
    elif "soar" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.soar_loader import convert_soar_dataset_to_hf

        print(f"Converting SOAR RLDS (local) to HF from: {cfg.dataset.dataset_path} ")
        dataset = convert_soar_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "mit_franka_p-rank" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.mit_franka_prank_loader import convert_mit_franka_prank_dataset_to_hf

        print(f"Converting MIT-Franka-Prank dataset to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_mit_franka_prank_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "rfm_new_mit_franka" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.new_mit_franka_loader import convert_new_mit_franka_dataset_to_hf

        print(f"Converting New MIT Franka dataset to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_new_mit_franka_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
            exclude_wrist_cam=cfg.dataset.exclude_wrist_cam,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "utd_so101_clean_policy_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.utd_so101_clean_policy_ranking_loader import (
            convert_utd_so101_clean_policy_ranking_to_hf,
        )

        # Determine view from dataset name
        if "wrist" in cfg.dataset.dataset_name.lower():
            view = "wrist"
        elif "top" in cfg.dataset.dataset_name.lower():
            view = "top"
        else:
            raise ValueError(f"Dataset name must specify view (wrist or top): {cfg.dataset.dataset_name}")

        print(f"Converting UTD SO101 Clean Policy Ranking ({view} view) to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_utd_so101_clean_policy_ranking_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            view=view,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "usc_koch_human_robot_paired" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.usc_koch_human_robot_paired_loader import (
            convert_usc_koch_human_robot_paired_to_hf,
        )

        # Determine trajectory type from dataset name
        if "usc_koch_human_robot_paired_human" in cfg.dataset.dataset_name.lower():
            trajectory_type = "human"
        elif "usc_koch_human_robot_paired_robot" in cfg.dataset.dataset_name.lower():
            trajectory_type = "robot"
        else:
            raise ValueError(
                f"Dataset name must specify either 'usc_koch_human_robot_paired_human' or 'usc_koch_human_robot_paired_robot': {cfg.dataset.dataset_name}. "
            )

        print(f"Converting USC Koch Human-Robot Paired ({trajectory_type}) to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_usc_koch_human_robot_paired_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            trajectory_type=trajectory_type,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "usc_koch_p_ranking" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.usc_koch_p_ranking_loader import (  # type: ignore
            convert_usc_koch_p_ranking_to_hf,
        )

        print(f"Converting USC Koch P-Ranking to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_usc_koch_p_ranking_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "egocot" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.egocot_loader import load_egocot_dataset

        # Load the trajectories using the loader
        print(f"Loading EgoCoT dataset from: {cfg.dataset.dataset_path}")
        task_data = load_egocot_dataset(
            cfg.dataset.dataset_path,
        )
        trajectories = flatten_task_data(task_data)
    elif "humanoid_everyday" in cfg.dataset.dataset_name.lower():
        # Stream + convert directly (OXE-style)
        from dataset_unify.dataset_loaders.humanoid_everyday_loader import convert_humanoid_everyday_dataset_to_hf

        print(f"Converting Humanoid Everyday dataset directly to HF from: {cfg.dataset.dataset_path}")
        dataset = convert_humanoid_everyday_dataset_to_hf(
            dataset_path=cfg.dataset.dataset_path,
            dataset_name=cfg.dataset.dataset_name,
            output_dir=cfg.output.output_dir,
            max_trajectories=cfg.output.max_trajectories,
            max_frames=cfg.output.max_frames,
            fps=cfg.output.fps,
            num_workers=cfg.output.num_workers,
        )

        save_dataset_locally(dataset, cfg.output.output_dir, cfg.dataset.dataset_name)
        print("Dataset conversion complete!")
        return
    elif "motif" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.motif_loader import load_motif_dataset

        print(f"Loading MotIF dataset from: {cfg.dataset.dataset_path}")
        task_data = load_motif_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "failsafe" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.failsafe_loader import load_failsafe_dataset

        print(f"Loading FailSafe dataset from: {cfg.dataset.dataset_path}")
        task_data = load_failsafe_dataset(cfg.dataset.dataset_path)
        trajectories = flatten_task_data(task_data)
    elif "racer" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.racer_loader import load_racer_dataset

        print(f"Loading RACER dataset from: {cfg.dataset.dataset_path}")
        task_data = load_racer_dataset(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
        trajectories = flatten_task_data(task_data)
    elif "hand_paired" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.hand_paired_loader import load_hand_paired_dataset

        print(f"Loading HAND_paired dataset from: {cfg.dataset.dataset_path}")
        task_data = load_hand_paired_dataset(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
        trajectories = flatten_task_data(task_data)
    elif "roboreward" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.roboreward_loader import load_roboreward_dataset

        print(f"Loading RoboReward dataset from: {cfg.dataset.dataset_path}")
        task_data = load_roboreward_dataset(cfg.dataset.dataset_path, cfg.dataset.dataset_name)
        trajectories = flatten_task_data(task_data)
    elif "robofac" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.robofac_loader import load_robofac_dataset

        print(f"Loading RoboFAC dataset from: {cfg.dataset.dataset_path}")
        task_data = load_robofac_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    elif "rbm-1m-ood" in cfg.dataset.dataset_name.lower():
        from dataset_unify.dataset_loaders.rbm_1m_ood_loader import load_rbm_1m_ood_dataset

        print(f"Loading RBM-1M-ODD dataset from: {cfg.dataset.dataset_path}")
        task_data = load_rbm_1m_ood_dataset(
            cfg.dataset.dataset_path,
            max_trajectories=cfg.output.max_trajectories,
        )
        trajectories = flatten_task_data(task_data)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_name}")

    # Convert dataset (non-streaming datasets)
    convert_dataset_to_hf_format(
        trajectories=trajectories,
        hf_creator_fn=partial(
            create_hf_trajectory,
            dataset_name=cfg.dataset.dataset_name,
            use_video=cfg.output.use_video,
            fps=cfg.output.fps,
            shortest_edge_size=cfg.output.shortest_edge_size,
            center_crop=cfg.output.center_crop,
        ),
        output_dir=cfg.output.output_dir,
        dataset_name=cfg.dataset.dataset_name,
        max_trajectories=cfg.output.max_trajectories,
        max_frames=cfg.output.max_frames,
        use_video=cfg.output.use_video,
        fps=cfg.output.fps,
        num_workers=cfg.output.num_workers,
    )

    print("Dataset conversion complete!")


if __name__ == "__main__":
    main()
