"""Shared helpers for cross-stage records and their on-disk artifacts."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel

T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Yield items in lists of at most ``batch_size`` elements."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch
        del batch


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSON Lines rows, returning an empty list for a missing file."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value
