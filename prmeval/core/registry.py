from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        key = name.strip().lower()

        def decorator(item: T) -> T:
            if key in self._items:
                raise ValueError(f"Duplicate {self.kind} registration: {key}")
            self._items[key] = item
            return item

        return decorator

    def get(self, name: str) -> T:
        key = name.strip().lower()
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {available}")
        return self._items[key]

    def names(self) -> list[str]:
        return sorted(self._items)


SAMPLERS: Registry[Any] = Registry("sampler")
INFERS: Registry[Any] = Registry("infer")
METRICS: Registry[Any] = Registry("metric")

register_sampler = SAMPLERS.register
register_infer = INFERS.register
register_metric = METRICS.register
