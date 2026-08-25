from __future__ import annotations

import math
import random
from itertools import pairwise
from typing import Any

import numpy as np

from ..core.config import TemporalRobustnessConfig


def length_bounds(base_frames: int, config: TemporalRobustnessConfig) -> tuple[int, int]:
    lower = math.ceil(config.min_length_ratio * base_frames)
    upper = min(math.floor(config.max_length_ratio * base_frames), config.max_frames)
    return lower, upper


def _sample_count(rng: random.Random, ratio_range: tuple[float, float], base: int, low: int, high: int) -> int:
    ratio = rng.uniform(*ratio_range)
    return min(high, max(low, round(ratio * base)))


def _piecewise_positions(control_points: list[int], length: int) -> list[int]:
    edge_count = length - 1
    segments = len(control_points) - 1
    counts = [1] * segments
    remaining = edge_count - segments
    distances = [max(1, abs(second - first)) for first, second in pairwise(control_points)]
    for _ in range(max(0, remaining)):
        scores = [distance / count for distance, count in zip(distances, counts, strict=True)]
        counts[int(np.argmax(scores))] += 1
    values = [control_points[0]]
    for (first, second), count in zip(pairwise(control_points), counts, strict=True):
        values.extend(np.rint(np.linspace(first, second, count + 1)[1:]).astype(int).tolist())
    return values


def transform_indices(
    base_indices: list[int],
    transform: str,
    rng: random.Random,
    config: TemporalRobustnessConfig,
) -> tuple[list[int], dict[str, Any]]:
    base = len(base_indices)
    lower, upper = length_bounds(base, config)
    params: dict[str, Any] = {}

    if transform == "original":
        return list(base_indices), params
    if transform in {"slow", "fast"}:
        gamma_range = config.slow_gamma_range if transform == "slow" else config.fast_gamma_range
        gamma = rng.uniform(*gamma_range)
        positions = np.floor((base - 1) * np.linspace(0, 1, base) ** gamma).astype(int).tolist()
        positions[-1] = base - 1
        return [base_indices[position] for position in positions], {"gamma": gamma}
    if transform == "pause":
        extra = _sample_count(rng, config.pause_extra_ratio_range, base, 1, upper - base)
        location = rng.randint(1, base - 2)
        inserted = [base_indices[location]] * extra
        result = [*base_indices[: location + 1], *inserted, *base_indices[location + 1 :]]
        params = {"extra_frames": extra, "location": location}
        return result, params
    if transform in {"rewind", "retry"}:
        extra_range = config.rewind_extra_ratio_range if transform == "rewind" else config.retry_extra_ratio_range
        length = base + _sample_count(rng, extra_range, base, 1, upper - base)
        peak_progress = rng.uniform(*config.peak_progress_range)
        retreat_ratio = rng.uniform(*config.retreat_ratio_range)
        peak = min(base - 2, max(2, round((base - 1) * peak_progress)))
        retreat = max(0, min(peak - 1, round(peak * (1 - retreat_ratio))))
        controls = [0, peak, retreat] if transform == "rewind" else [0, peak, retreat, base - 1]
        positions = _piecewise_positions(controls, length)
        params = {
            "extra_frames": length - base,
            "peak_progress": peak_progress,
            "retreat_ratio": retreat_ratio,
            "peak_position": peak,
            "retreat_position": retreat,
        }
        return [base_indices[position] for position in positions], params
    if transform == "truncate":
        retained_ratio = rng.uniform(*config.truncate_retained_ratio_range)
        length = min(base - 1, max(lower, round(base * retained_ratio)))
        return base_indices[:length], {"retained_ratio": retained_ratio}
    if transform == "skip":
        removable = base - lower
        removed = _sample_count(rng, config.skip_removed_ratio_range, base, 1, removable)
        start = rng.randint(1, base - removed - 1)
        result = [*base_indices[:start], *base_indices[start + removed :]]
        return result, {"removed_frames": removed, "start": start}
    raise ValueError(f"Unknown temporal transform: {transform}")
