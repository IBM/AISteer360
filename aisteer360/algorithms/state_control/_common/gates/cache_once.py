"""Wrapper gate that freezes the per-row decision once ready."""
import torch

from .base import BaseGate


class CacheOnceGate(BaseGate):
    """Wraps an inner gate and caches its `open_rows()` once `is_ready()`.

    After the inner gate reports ready, all subsequent `open_rows()` calls return the cached
    tensor and further updates are ignored. This is the condition-then-steer pattern: the
    condition is evaluated on the prompt (prefill) and the decision holds for the whole
    generation. Combined with the runtime's is_ready() early-out, the frozen gate also stops
    condition scoring entirely — "score the prompt once" is a consequence of the evidence
    contract, not a hook-side pass counter.

    The wrapped gate stays reachable via `inner` (for diagnostics such as threshold/evidence).

    Args:
        inner: The gate to wrap.
    """

    def __init__(self, inner: BaseGate):
        self.inner = inner
        self._cached: torch.BoolTensor | None = None

    def reset(self, num_rows: int = 1) -> None:
        """Clear the cached decision and reset the inner gate to `num_rows`."""
        super().reset(num_rows)
        self.inner.reset(num_rows)
        self._cached = None

    def update(self, scores: torch.Tensor | float, *, key: int | None = None) -> None:
        """Forward evidence to the inner gate; freeze the decision once it is ready."""
        if self._cached is not None:
            return  # already frozen
        self.inner.update(scores, key=key)
        if self.inner.is_ready():
            self._cached = self.inner.open_rows().clone()

    def open_rows(self) -> torch.BoolTensor:
        """Cached per-row decision, or the inner gate's live decision before freeze."""
        if self._cached is not None:
            return self._cached
        return self.inner.open_rows()

    def is_ready(self) -> bool:
        """True once the decision is frozen or the inner gate is ready."""
        return self._cached is not None or self.inner.is_ready()
