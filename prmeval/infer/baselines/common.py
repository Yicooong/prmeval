"""Shared helpers for built-in inference baselines."""

from collections.abc import Sequence

import numpy as np


def trajectory_prefix_lengths(num_frames: int, num_prefix_samples: int) -> list[int]:
    """Return increasing, 1-based lengths for cumulative trajectory prefixes.

    The first and final frames are included whenever at least two samples are
    requested. A single requested sample evaluates the complete trajectory.
    """
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if not isinstance(num_prefix_samples, int) or isinstance(num_prefix_samples, bool) or num_prefix_samples < 1:
        raise ValueError("num_prefix_samples must be a positive integer")
    if num_frames == 0:
        return []
    count = min(num_prefix_samples, num_frames)
    if count == 1:
        return [num_frames]
    return sorted({int(value) for value in np.linspace(1, num_frames, count, dtype=int)})


def normalize_prefix_values(values: Sequence[float]) -> np.ndarray:
    """Min-max normalize scalar prefix values to [0, 1]."""
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return array
    if len(array) == 1:
        return np.ones_like(array)
    minimum = float(array.min())
    maximum = float(array.max())
    if maximum == minimum:
        return np.ones_like(array)
    return (array - minimum) / (maximum - minimum)


def interpolate_prefix_values(
    num_frames: int,
    prefix_lengths: Sequence[int],
    prefix_values: Sequence[float],
) -> np.ndarray:
    """Interpolate scalar values observed at prefix endpoints onto every frame."""
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if num_frames == 0:
        if len(prefix_lengths) or len(prefix_values):
            raise ValueError("empty trajectories cannot have prefix observations")
        return np.array([], dtype=np.float64)
    if len(prefix_lengths) != len(prefix_values):
        raise ValueError("prefix_lengths and prefix_values must have the same length")
    if len(prefix_lengths) == 0:
        raise ValueError("at least one prefix observation is required")

    lengths = np.asarray(prefix_lengths, dtype=np.int64)
    values = np.asarray(prefix_values, dtype=np.float64)
    if (lengths < 1).any() or (lengths > num_frames).any():
        raise ValueError("prefix lengths must be within the trajectory")
    if len(set(lengths.tolist())) != len(lengths) or (np.diff(lengths) <= 0).any():
        raise ValueError("prefix lengths must be unique and strictly increasing")
    if not np.isfinite(values).all():
        raise ValueError("prefix values must be finite")

    frame_numbers = np.arange(1, num_frames + 1, dtype=np.float64)
    return np.interp(frame_numbers, lengths.astype(np.float64), values).astype(np.float64)
