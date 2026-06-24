"""Wrapper gate that freezes the decision once ready."""
from .base import BaseGate


class CacheOnceGate(BaseGate):
    """Wraps an inner gate and caches the decision once is_ready().

    After the inner gate reports is_ready(), all subsequent is_open()
    calls return the cached value without consulting the inner gate.
    This is useful for condition-then-steer patterns where the condition
    is evaluated only during the first forward pass (prompt encoding).

    Args:
        inner: The gate to wrap.
    """

    def __init__(self, inner: BaseGate):
        self._inner = inner
        self._cached: bool | None = None

    def reset(self) -> None:
        """Clear cached decision and reset inner gate."""
        self._inner.reset()
        self._cached = None

    def update(self, score: float, *, key: int | None = None) -> None:
        """Forward update to inner gate and cache when ready.

        Args:
            score: The computed score.
            key: Optional identifier for the source.
        """
        if self._cached is not None:
            return  # already frozen
        self._inner.update(score, key=key)
        if self._inner.is_ready():
            self._cached = self._inner.is_open()

    def is_open(self) -> bool:
        """Return cached decision or query inner gate."""
        if self._cached is not None:
            return self._cached
        return self._inner.is_open()

    def is_ready(self) -> bool:
        """Return True if decision is cached or inner gate is ready."""
        return self._cached is not None or self._inner.is_ready()
