from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np


def image_data_url(frame: Any, quality: int = 85) -> str:
    if isinstance(frame, str) and frame.startswith("data:image/"):
        return frame
    if isinstance(frame, (str, Path)):
        path = Path(frame)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Encoding numpy frames requires the 'data' extra (Pillow)") from exc
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"


def vision_content(frames: Any, prefix: str = "Frame") -> list[dict[str, Any]]:
    return [
        item
        for index, frame in enumerate(frames)
        for item in (
            {"type": "text", "text": f"{prefix} {index + 1}:"},
            {"type": "image_url", "image_url": {"url": image_data_url(frame), "detail": "low"}},
        )
    ]
