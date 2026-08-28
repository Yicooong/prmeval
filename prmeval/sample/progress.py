from __future__ import annotations

import numpy as np


def compute_target_progress(
    num_frames_total: int,
    frame_indices: list[int],
    progress_type: str = "absolute_first_frame",
    success_cutoff: float | None = None,
    partial_success: float | None = None,
) -> list[float]:
    if num_frames_total <= 0 or not frame_indices:
        return []
    cutoff_index = int(success_cutoff * num_frames_total) if success_cutoff and success_cutoff > 0 else None
    if progress_type == "absolute_wrt_total_frames":
        absolute = [1.0 if cutoff_index is not None and i >= cutoff_index else (i + 1) / num_frames_total for i in frame_indices]
    else:
        start = min(frame_indices)
        end = cutoff_index if cutoff_index is not None else num_frames_total
        denominator = max(1, end - start - 1)
        absolute = [min(1.0, (i - start) / denominator) for i in frame_indices]
    if partial_success is not None and not np.isclose(partial_success, 1.0):
        final_index = num_frames_total - 1
        if final_index in frame_indices:
            values = [0.0] * len(frame_indices)
            values[frame_indices.index(final_index)] = float(partial_success)
            return values
    if progress_type == "relative_first_frame":
        return np.diff(np.asarray([0.0, *absolute], dtype=float)).tolist()
    return [float(v) for v in absolute]


def linspace_indices(length: int, max_frames: int) -> list[int]:
    if length <= max_frames:
        return list(range(length))
    return np.linspace(0, length - 1, max_frames, dtype=int).tolist()
