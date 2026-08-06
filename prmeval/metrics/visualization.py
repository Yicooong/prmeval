"""Optional plots for completed evaluation runs."""

from __future__ import annotations

from pathlib import Path

from ..core.schemas import EvaluationRecord


def plot_progress(record: EvaluationRecord, output_path: str | Path) -> Path:
    """Plot target and predicted progress for one successful progress record."""
    if (
        record.prediction is None or record.prediction.kind != "progress"
        or record.target is None or record.target.kind != "progress"
    ):
        raise ValueError("plot_progress requires a successful progress record with targets")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires the 'viz' extra") from exc
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(record.target.values, label="target", marker="o")
    axis.plot(record.prediction.values, label="prediction", marker="o")
    axis.set(xlabel="sampled frame", ylabel="progress", ylim=(-0.05, 1.05), title=record.input.task)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
