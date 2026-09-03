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
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of the dataset (required)"})
    exclude_wrist_cam: bool = field(default=False, metadata={"help": "Exclude wrist camera views (MIT Franka only)"})
    view: str = field(default="external_main", metadata={"help": "Semantic camera view for simulator rollouts"})


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
    """Convert a registered source dataset to the standard Hugging Face format."""

    if not cfg.output.use_video:
        raise ValueError("output.use_video=false is not supported; dataset_unify currently writes MP4 videos only")
    if not cfg.dataset.dataset_name:
        raise ValueError("dataset.dataset_name must be provided")

    # Importing the catalog performs lightweight registrations; individual loaders remain lazy.
    from dataset_unify.converters import get_dataset_converter

    converter = get_dataset_converter(cfg.dataset.dataset_name)
    conversion_result = converter(cfg)

    if isinstance(conversion_result, Dataset):
        save_dataset_locally(conversion_result, cfg.output.output_dir, cfg.dataset.dataset_name)
    else:
        convert_dataset_to_hf_format(
            trajectories=conversion_result,
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
