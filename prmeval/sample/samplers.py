from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from typing import ClassVar

import numpy as np

from ..core.config import SamplingConfig
from ..core.registry import SAMPLERS, register_sampler
from ..core.schemas import PreferenceSample, ProgressSample, Trajectory
from .progress import compute_target_progress, linspace_indices, transform_indices
from .utils import load_frames, load_hf_trajectory_pool

logger = logging.getLogger(__name__)


class EvalSampler(ABC):
    eval_type: str

    def __init__(
        self,
        config: SamplingConfig,
        dataset_name: str,
        pool: list[Trajectory] | None = None,
    ):
        self.config = config
        self.dataset_name = dataset_name
        self.pool = pool if pool is not None else load_hf_trajectory_pool(config)
        self.pool_size = len(self.pool)

    @abstractmethod
    def sample(self) -> Iterable[ProgressSample | PreferenceSample]:
        raise NotImplementedError

    def _stable_sample_id(self, dataset: str, trajectory_ids: list[str], eval_type: str, indices: list[int]) -> str:
        raw = "|".join([dataset, *trajectory_ids, eval_type, ",".join(map(str, indices))])
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _subset_trajectory(
        self,
        traj: Trajectory,
        frames: np.ndarray,
        indices: list[int],
    ) -> Trajectory:
        total = len(frames)
        selected = frames[indices]
        source_target = self._has_target_progress(traj, total)
        if source_target is not None:
            target = [source_target[index] for index in indices]
        else:
            target = compute_target_progress(
                total,
                indices,
                self.config.progress_type,
                partial_success=traj.partial_success,
            )
        if len(selected) > self.config.base_frames:
            local = linspace_indices(len(selected), self.config.base_frames)
            selected = selected[local]
            indices = [indices[i] for i in local]
            target = [target[i] for i in local]
        return traj.model_copy(
            update={
                "frames": selected,
                "frame_indices": indices,
                "num_frames_total": total,
                "target_progress": target,
            }
        )

    @staticmethod
    def _has_target_progress(traj: Trajectory, total: int) -> list[float] | None:
        if traj.target_progress is None:
            return None
        if len(traj.target_progress) != total:
            raise ValueError(
                f"Simulation trajectory '{traj.id}' has {len(traj.target_progress)} target_progress values "
                f"for {total} frames"
            )
        return traj.target_progress


@register_sampler("progress")
class ProgressSampler(EvalSampler):
    eval_type = "progress"

    def sample(self):
        logger.info("总共采样到%d条成功轨迹", self.pool_size)
        for traj in self.pool:
            frames = load_frames(traj.frames)
            total = len(frames)
            indices = list(range(total))
            sampled = self._subset_trajectory(traj, frames, indices)
            yield ProgressSample(
                sample_id=self._stable_sample_id(
                    self.dataset_name,
                    [traj.data_source, traj.id],
                    self.eval_type,
                    indices,
                ),
                trajectory=sampled,
                eval_type=self.eval_type,
            )


@register_sampler("progress_temporal_variation")
class ProgressTemporalVariationSampler(EvalSampler):
    eval_type = "progress_temporal_variation"

    def _temporal_rng(self, seed: int, *parts: str) -> random.Random:
        digest = hashlib.sha256("|".join([str(seed), *parts]).encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def sample(self):
        temporal = self.config.temporal_robustness
        base_count = self.config.base_frames
        logger.info(
            "总共采样到%d条成功轨迹",
            len(self.pool) * temporal.variants_per_transform * len(temporal.transforms),
        )
        for trajectory in self.pool:
            frames = load_frames(trajectory.frames)
            total = len(frames)
            if total < base_count:
                continue
            base_indices = linspace_indices(total, base_count)
            source_progress = self._has_target_progress(trajectory, total)
            if source_progress is None:
                source_progress = compute_target_progress(total, list(range(total)), self.config.progress_type)
            variants = [("original", 0)]
            variants.extend(
                (transform, variant)
                for transform in temporal.transforms
                for variant in range(temporal.variants_per_transform)
            )
            for transform, variant in variants:
                rng = self._temporal_rng(
                    self.config.temporal_robustness.random_seed,
                    self.dataset_name,
                    trajectory.data_source,
                    trajectory.id,
                    transform,
                    str(variant),
                )
                indices, params = transform_indices(base_indices, transform, rng, temporal)
                target = [source_progress[index] for index in indices]
                metadata = {
                    **trajectory.metadata,
                    "synthetic_temporal": {
                        "transform": transform,
                        "variant": variant,
                        "parameters": params,
                        "base_frames": base_count,
                        "transformed_frames": len(indices),
                        "length_ratio": len(indices) / base_count,
                        "source_trajectory_id": trajectory.id,
                        "source_quality_label": trajectory.quality_label,
                        "source_partial_success": trajectory.partial_success,
                        "final_progress": target[-1],
                        "synthetic_success": transform not in {"rewind", "truncate"},
                    },
                }
                descriptor = json.dumps(
                    {"transform": transform, "variant": variant, "parameters": params},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                sampled = trajectory.model_copy(
                    update={
                        "frames": frames[indices],
                        "frame_indices": indices,
                        "num_frames_total": total,
                        "target_progress": target,
                        "metadata": metadata,
                        "quality_label": ("failure" if transform in {"rewind", "truncate"} else "successful"),
                        "partial_success": (target[-1] if transform in {"rewind", "truncate"} else 1.0),
                    }
                )
                yield ProgressSample(
                    sample_id=self._stable_sample_id(
                        self.dataset_name,
                        [trajectory.data_source, trajectory.id, descriptor],
                        self.eval_type,
                        indices,
                    ),
                    trajectory=sampled,
                    eval_type=self.eval_type,
                )


@register_sampler("policy_ranking")
class PolicyRankingSampler(EvalSampler):
    eval_type = "policy_ranking"

    def sample(self):
        groups: dict[str, dict[object, list[Trajectory]]] = defaultdict(lambda: defaultdict(list))
        uses_partial = any(t.partial_success is not None for t in self.pool)
        for traj in self.pool:
            if uses_partial:
                if traj.partial_success is None:
                    continue
                key = round(float(traj.partial_success), 2)
            else:
                key = traj.quality_label
            if key is not None:
                groups[traj.task][key].append(traj)
        eligible = [(task, values) for task, values in sorted(groups.items()) if len(values) > 1]
        rng = random.Random(self.config.temporal_robustness.random_seed)
        if self.config.max_tasks and len(eligible) > self.config.max_tasks:
            rng.shuffle(eligible)
            eligible = eligible[: self.config.max_tasks]
        selected: list[Trajectory] = []
        for _, values in eligible:
            if uses_partial and self.config.num_partial_successes:
                pools = [sorted(pool, key=lambda t: t.id) for _, pool in sorted(values.items())]
                task_selected: list[Trajectory] = []
                while pools and len(task_selected) < self.config.num_partial_successes:
                    for pool in pools:
                        if pool and len(task_selected) < self.config.num_partial_successes:
                            task_selected.append(pool.pop(rng.randrange(len(pool))))
                    pools = [pool for pool in pools if pool]
                selected.extend(task_selected)
            else:
                for pool in values.values():
                    limit = min(len(pool), self.config.num_examples_per_quality or len(pool))
                    selected.extend(rng.sample(pool, limit))
        for traj in selected:
            frames = load_frames(traj.frames)
            total = len(frames)
            indices = list(range(total))
            sampled = self._subset_trajectory(traj, frames, indices)
            yield ProgressSample(
                sample_id=self._stable_sample_id(
                    self.dataset_name,
                    [traj.data_source, traj.id],
                    self.eval_type,
                    indices,
                ),
                trajectory=sampled,
                eval_type=self.eval_type,
            )


@register_sampler("confusion_matrix")
class ConfusionMatrixSampler(EvalSampler):
    eval_type = "confusion_matrix"

    def sample(self):
        rng = random.Random(self.config.temporal_robustness.random_seed)
        tasks = sorted({trajectory.task for trajectory in self.pool})
        by_source: dict[str, dict[str, list[Trajectory]]] = defaultdict(lambda: defaultdict(list))
        for trajectory in self.pool:
            by_source[trajectory.data_source][trajectory.task].append(trajectory)
        selected: list[Trajectory] = []
        for task_groups in by_source.values():
            pools = [list(pool) for _, pool in sorted(task_groups.items())]
            for pool in pools:
                rng.shuffle(pool)
            limit = self.config.trajectories_per_source or sum(map(len, pools))
            source_selected: list[Trajectory] = []
            while pools and len(source_selected) < limit:
                for pool in pools:
                    if pool and len(source_selected) < limit:
                        source_selected.append(pool.pop())
                pools = [pool for pool in pools if pool]
            selected.extend(source_selected)
        for trajectory in selected:
            frames = load_frames(trajectory.frames)
            indices = list(range(len(frames)))
            for language_task in tasks:
                metadata = {
                    **trajectory.metadata,
                    "video_task": trajectory.task,
                    "lang_task": language_task,
                }
                overridden = trajectory.model_copy(update={"task": language_task, "metadata": metadata})
                sampled = self._subset_trajectory(overridden, frames, indices)
                yield ProgressSample(
                    sample_id=self._stable_sample_id(
                        self.dataset_name,
                        [trajectory.data_source, trajectory.id, language_task],
                        self.eval_type,
                        indices,
                    ),
                    trajectory=sampled,
                    eval_type=self.eval_type,
                )


@register_sampler("quality_preference")
class QualityPreferenceSampler(EvalSampler):
    eval_type = "quality_preference"
    quality_rank: ClassVar = {
        "successful": 2,
        "suboptimal": 1,
        "failure": 0,
        "failed": 0,
    }

    def sample(self):
        by_task: dict[str, dict[float, list[Trajectory]]] = defaultdict(lambda: defaultdict(list))
        for traj in self.pool:
            rank = traj.partial_success
            if rank is None:
                rank = traj.preference_rank
            if rank is None:
                rank = self.quality_rank.get(traj.quality_label or "")
            if rank is not None:
                by_task[traj.task][round(float(rank), 2)].append(traj)
        rng = random.Random(self.config.temporal_robustness.random_seed)
        pairs: list[tuple[Trajectory, Trajectory]] = []
        for rank_groups in by_task.values():
            task_pairs = []
            for first_rank, second_rank in itertools.combinations(sorted(rank_groups), 2):
                lower, higher = rank_groups[first_rank], rank_groups[second_rank]
                task_pairs.extend(itertools.product(higher, lower))
            if self.config.comparisons_per_task and len(task_pairs) > self.config.comparisons_per_task:
                task_pairs = rng.sample(task_pairs, self.config.comparisons_per_task)
            pairs.extend(task_pairs)
        if self.config.max_comparisons and len(pairs) > self.config.max_comparisons:
            pairs = rng.sample(pairs, self.config.max_comparisons)
        for chosen, rejected in pairs:
            chosen_frames = load_frames(chosen.frames)
            rejected_frames = load_frames(rejected.frames)
            chosen_indices = list(range(len(chosen_frames)))
            rejected_indices = list(range(len(rejected_frames)))
            yield PreferenceSample(
                sample_id=self._stable_sample_id(
                    self.dataset_name,
                    [chosen.data_source, chosen.id, rejected.data_source, rejected.id],
                    self.eval_type,
                    [*chosen_indices, -1, *rejected_indices],
                ),
                chosen_trajectory=self._subset_trajectory(chosen, chosen_frames, chosen_indices),
                rejected_trajectory=self._subset_trajectory(rejected, rejected_frames, rejected_indices),
                eval_type=self.eval_type,
            )


def create_samplers(
    config: SamplingConfig,
    pool: list[Trajectory] | None = None,
) -> list[EvalSampler]:
    return [SAMPLERS.get(eval_type)(config, config.dataset_name, pool=pool) for eval_type in config.eval_types]
