"""Prepare locally downloaded Hugging Face datasets for evaluation.

This command intentionally does only evaluation-time normalization: decode a
trajectory video, sample a bounded number of frames, store those frames in an
``.npz`` file, and write a lightweight Hugging Face ``Dataset`` index.  Model
embeddings and training indices are not part of the evaluation contract.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REQUIRED_COLUMNS = ("id", "task")


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


def _uniform_indices(length: int, max_frames: int) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=int)
    if length <= max_frames:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, max_frames, dtype=int)


def _video_frames(value: Any, root: Path | None = None) -> np.ndarray:
    """Decode common Hugging Face Video values or accept existing frame arrays."""
    if isinstance(value, np.ndarray):
        frames = value
    elif isinstance(value, list):
        frames = np.asarray(value)
    elif isinstance(value, (str, Path)):
        path = Path(value)
        if root and not path.is_absolute():
            path = root / path
        if path.suffix.lower() == ".npz":
            with np.load(path) as archive:
                frames = np.asarray(archive["frames"])
        else:
            frames = _decode_video(path)
    elif isinstance(value, dict):
        if value.get("frames") is not None:
            frames = np.asarray(value["frames"])
        elif value.get("path"):
            path = Path(value["path"])
            if root and not path.is_absolute():
                path = root / path
            frames = _decode_video(path)
        elif value.get("bytes"):
            frames = _decode_video(io.BytesIO(value["bytes"]))
        else:
            raise ValueError(f"Unsupported video mapping fields: {sorted(value)}")
    else:
        try:
            frames = np.asarray(list(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Unsupported video value: {type(value)!r}") from exc
    if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected video frames shaped [T,H,W,C], got {frames.shape}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames


def _decode_video(source: Any) -> np.ndarray:
    try:
        import imageio.v3 as iio
        return np.asarray(list(iio.imiter(source, plugin="pyav")))
    except (ImportError, OSError, RuntimeError, ValueError):
        return _decode_video_with_ffmpeg(source)


def _decode_video_with_ffmpeg(source: Any) -> np.ndarray:
    """Decode a path or in-memory video with FFmpeg when PyAV is unavailable."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Video decoding requires PyAV or the ffmpeg and ffprobe executables")

    if isinstance(source, (str, Path)):
        input_bytes = None
        input_args = ["-i", str(source)]
    elif isinstance(source, io.BytesIO):
        input_bytes = source.getvalue()
        input_args = ["-i", "pipe:0"]
    else:
        raise TypeError(f"FFmpeg fallback does not support video source {type(source)!r}")

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            *input_args,
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
        ],
        input=input_bytes,
        capture_output=True,
        check=True,
    )
    width, height = (int(value) for value in probe.stdout.decode().strip().split("x"))
    decoded = subprocess.run(
        ["ffmpeg", "-loglevel", "error", *input_args, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        input=input_bytes,
        capture_output=True,
        check=True,
    )
    return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def _load_local_dataset(path: Path, subset: str | None, split: str):
    try:
        from datasets import Dataset, DatasetDict, Video, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("Dataset preparation requires the 'data' extra") from exc

    if (path / "state.json").exists() or (path / "dataset_dict.json").exists():
        dataset = load_from_disk(str(path))
    else:
        dataset = load_dataset(str(path), name=subset, split=split)
    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split '{split}' not found; available: {sorted(dataset)}")
        dataset = dataset[split]
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Hugging Face Dataset, got {type(dataset)!r}")

    # Decode videos ourselves so the normalized cache never depends on torchcodec.
    for column in ("frames_video", "video"):
        if column in dataset.column_names and column in dataset.features:
            try:
                dataset = dataset.cast_column(column, Video(decode=False))
            except (TypeError, ValueError):
                pass
    return dataset


def _source_frames(row: dict[str, Any], configured_column: str | None) -> Any:
    candidates = [configured_column] if configured_column else []
    candidates.extend(["frames", "frames_video", "video", "frames_path"])
    for column in candidates:
        if column and row.get(column) is not None:
            return row[column]
    raise ValueError("No frame/video column found (tried frames, frames_video, video, frames_path)")


def prepare_source(source: dict[str, Any], output_root: Path, max_frames: int, force: bool) -> dict[str, Any]:
    source_path = Path(source["path"]).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Local Hugging Face dataset not found: {source_path}")
    cache_name = source.get("cache_name") or _safe_name(
        "/".join(part for part in (source.get("dataset_id"), source.get("subset")) if part)
        or source_path.name
    )
    destination = output_root / cache_name
    processed_path = destination / "processed_dataset"
    if processed_path.exists() and not force:
        return {"cache_name": cache_name, "status": "skipped", "path": str(destination)}
    if destination.exists():
        shutil.rmtree(destination)
    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True)

    dataset = _load_local_dataset(source_path, source.get("subset"), source.get("split", "train"))
    missing = [column for column in REQUIRED_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset {source_path} is missing required columns: {', '.join(missing)}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    source_limit = source.get("max_trajectories")
    for index, item in enumerate(dataset):
        if source_limit is not None and len(rows) >= int(source_limit):
            break
        row = dict(item)
        try:
            frames = _video_frames(_source_frames(row, source.get("video_column")), source_path)
            selected = frames[_uniform_indices(len(frames), max_frames)]
            if not len(selected):
                raise ValueError("trajectory contains no frames")
            trajectory_id = str(row["id"])
            frame_path = frames_dir / f"{index:08d}_{_safe_name(trajectory_id)}.npz"
            np.savez_compressed(frame_path, frames=selected)
            for column in ("frames_video", "video", "frames_path"):
                row.pop(column, None)
            row["id"] = trajectory_id
            row["task"] = str(row["task"])
            row["frames"] = str(frame_path.resolve())
            row["num_frames"] = len(selected)
            row["data_source"] = str(row.get("data_source") or cache_name)
            rows.append(row)
        except Exception as exc:  # keep preparing other valid trajectories
            failures.append({"index": index, "id": str(row.get("id", "unknown")), "error": str(exc)})

    if not rows:
        shutil.rmtree(destination)
        raise RuntimeError(f"No valid trajectories were prepared from {source_path}")
    from datasets import Dataset

    Dataset.from_list(rows).save_to_disk(str(processed_path))
    summary = {
        "cache_name": cache_name,
        "source": str(source_path),
        "subset": source.get("subset"),
        "split": source.get("split", "train"),
        "trajectories": len(rows),
        "failed": len(failures),
        "max_frames": max_frames,
    }
    (destination / "prepare_manifest.json").write_text(
        json.dumps({**summary, "failures": failures}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {**summary, "status": "prepared", "path": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prmeval-data-preprocess",
        description="Normalize locally downloaded Hugging Face robot datasets for evaluation",
    )
    parser.add_argument("--config", required=True, help="YAML file containing output_dir and sources")
    parser.add_argument("--force", action="store_true", help="Replace existing prepared caches")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    output_root = Path(config["output_dir"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    max_frames = int(config.get("max_frames", 32))
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    sources = config.get("sources") or []
    if not sources:
        raise ValueError("Preparation config must contain at least one source")
    results = [prepare_source(source, output_root, max_frames, args.force) for source in sources]
    print(json.dumps({"output_dir": str(output_root), "sources": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
