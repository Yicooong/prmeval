from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class MatchMode(str, Enum):
    """How a registered pattern is matched against a dataset name."""

    CONTAINS = "contains"
    PREFIX = "prefix"
    EXACT = "exact"


@dataclass(frozen=True)
class _Registration(Generic[T]):
    name: str
    patterns: tuple[str, ...]
    match_mode: MatchMode
    item: T

    def matches(self, dataset_name: str) -> bool:
        if self.match_mode is MatchMode.EXACT:
            return dataset_name in self.patterns
        if self.match_mode is MatchMode.PREFIX:
            return dataset_name.startswith(self.patterns)
        return any(pattern in dataset_name for pattern in self.patterns)


class DatasetConverterRegistry(Generic[T]):
    """Ordered registry that resolves a converter from a user-facing dataset name."""

    def __init__(self) -> None:
        self._items: dict[str, _Registration[T]] = {}

    def register(
        self,
        name: str,
        *,
        patterns: tuple[str, ...] | None = None,
        match_mode: MatchMode = MatchMode.CONTAINS,
    ) -> Callable[[T], T]:
        key = name.strip().lower()
        normalized_patterns = tuple(pattern.strip().lower() for pattern in (patterns or (key,)))
        if not key or not normalized_patterns or any(not pattern for pattern in normalized_patterns):
            raise ValueError("Dataset converter names and patterns must not be empty")

        def decorator(item: T) -> T:
            if key in self._items:
                raise ValueError(f"Duplicate dataset converter registration: {key}")
            self._items[key] = _Registration(key, normalized_patterns, match_mode, item)
            return item

        return decorator

    def get(self, dataset_name: str) -> T:
        normalized_name = (dataset_name or "").strip().lower()
        for registration in self._items.values():
            if registration.matches(normalized_name):
                return registration.item

        available = ", ".join(self.names()) or "<none>"
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available converters: {available}")

    def names(self) -> list[str]:
        return sorted(self._items)


DATASET_CONVERTERS: DatasetConverterRegistry[Callable[[Any], Any]] = DatasetConverterRegistry()
register_dataset_converter = DATASET_CONVERTERS.register
