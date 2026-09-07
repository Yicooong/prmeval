


from pathlib import Path
import re
import tempfile
from typing import Any

import cv2
import numpy as np


def load_frames(
    value: Any,
    stride: int = 1,
    start_frame: int = 0,
    end_frame: int = -1,
    npz_key: str = "frames",
) -> np.ndarray:
    """
    Unified frame loader. Returns RGB uint8 THWC numpy arrays.

    Supported sources:
        - Video files:     .mp4 .avi .mov .mkv .webm .m4v
        - Image directory: directory containing .jpg/.png/.bmp/.tiff/.webp files
        - NumPy archive:   .npy  (T,H,W,C)  or  .npz  (key=npz_key)
        - Hugging Face video mapping containing ``path`` or ``bytes``
    """

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    NUMPY_EXTENSIONS = {".npy", ".npz"}

    def load_frames_from_video(path: Path, stride: int, start: int, end: int) -> np.ndarray:
        """Load frames from a video file using OpenCV."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if end >= 0 and idx >= end:
                break
            if idx >= start and (idx - start) % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            idx += 1
        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from video: {path}")
        return np.asarray(frames)

    def load_frames_from_directory(path: Path, stride: int, start: int, end: int) -> np.ndarray:
        """Load images from a directory, sorted numerically by filename."""
        from PIL import Image

        def _numeric_sort_key(path: Path) -> tuple[int, tuple[int, ...], str]:
            """Sort key: extract all digit groups from the stem, then fall back to name."""
            numbers = tuple(int(number) for number in re.findall(r"\d+", path.stem))
            return (0 if numbers else 1, numbers, path.name)

        candidates = sorted(
            [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
            key=_numeric_sort_key,
        )
        if not candidates:
            raise ValueError(f"No images found in directory: {path}")

        frames = []
        for idx, img_path in enumerate(candidates):
            if end >= 0 and idx >= end:
                break
            if idx >= start and (idx - start) % stride == 0:
                img = Image.open(img_path).convert("RGB")
                frames.append(np.array(img))

        if not frames:
            raise ValueError(f"No frames selected from directory: {path}")
        return np.asarray(frames)

    def load_frames_from_numpy(path: Path, npz_key: str, stride: int, start: int, end: int) -> np.ndarray:
        """Load frames from a .npy or .npz file. Expected shape: (T, H, W, C)."""
        if path.suffix.lower() == ".npy":
            arr = np.load(str(path))
        else:
            with np.load(str(path)) as data:
                if npz_key not in data:
                    available = list(data.keys())
                    raise ValueError(f"Key '{npz_key}' not found in {path}. Available: {available}")
                arr = np.asarray(data[npz_key])

        if arr.ndim != 4:
            raise ValueError(f"Expected shape (T,H,W,C), got {arr.shape} from {path}")

        indices = range(start, (len(arr) if end < 0 else min(end, len(arr))), stride)
        frames = [arr[i] for i in indices]
        if not frames:
            raise ValueError(f"No frames selected from numpy file: {path}")
        return np.asarray(frames)

    if stride < 1:
        raise ValueError("stride must be at least 1")
    if start_frame < 0:
        raise ValueError("start_frame must not be negative")
    if end_frame >= 0 and end_frame < start_frame:
        raise ValueError("end_frame must not be less than start_frame")

    if isinstance(value, dict):
        if value.get("frames") is not None:
            value = value["frames"]
        elif value.get("path"):
            value = value["path"]
        elif value.get("bytes"):
            suffix = Path(value.get("path") or "video.mp4").suffix or ".mp4"
            with tempfile.NamedTemporaryFile(suffix=suffix) as video_file:
                video_file.write(value["bytes"])
                video_file.flush()
                return load_frames(
                    video_file.name,
                    stride=stride,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    npz_key=npz_key,
                )
        else:
            raise ValueError(f"Unsupported video mapping fields: {sorted(value)}")

    if isinstance(value, np.ndarray):
        if value.ndim not in (1, 4):
            raise ValueError(f"Expected frame references or shape (T,H,W,C), got {value.shape}")
        stop = len(value) if end_frame < 0 else min(end_frame, len(value))
        frames = value[start_frame:stop:stride]
    elif isinstance(value, list):
        frames_array = np.asarray(value)
        if frames_array.ndim not in (1, 4):
            raise ValueError("Frame lists must contain frame references or have shape (T,H,W,C)")
        stop = len(frames_array) if end_frame < 0 else min(end_frame, len(frames_array))
        frames = frames_array[start_frame:stop:stride]
    elif isinstance(value, (str, Path)):
        p = Path(value)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
        if p.is_dir():
            frames = load_frames_from_directory(p, stride, start_frame, end_frame)
        else:
            ext = p.suffix.lower()

            if ext in VIDEO_EXTENSIONS:
                frames = load_frames_from_video(p, stride, start_frame, end_frame)
            elif ext in NUMPY_EXTENSIONS:
                frames = load_frames_from_numpy(p, npz_key, stride, start_frame, end_frame)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
    else:
        raise TypeError(f"Unsupported frames value: {type(value)!r}")
    if not len(frames):
        raise ValueError("No frames selected")
    if frames.dtype.kind in {"O", "S", "U"}:
        return frames
    if frames.dtype != np.uint8:
        if np.issubdtype(frames.dtype, np.floating):
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
        else:
            frames = frames.astype(np.uint8)
    return frames
