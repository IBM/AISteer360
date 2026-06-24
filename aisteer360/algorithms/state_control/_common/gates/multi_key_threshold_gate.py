"""Threshold gate that aggregates scores from multiple condition layers."""
from typing import Literal

from ..specs import Comparator
from .base import BaseGate


class MultiKeyThresholdGate(BaseGate):
    """Gate that opens based on threshold comparison of received scores.

    Supports multiple condition layers (keys). The gate aggregates
    per-key decisions using either "any" or "all" semantics.

    Args:
        threshold: Score threshold for comparison.
        comparator: "larger" means gate opens when score >= threshold.
            "smaller" means gate opens when score <= threshold.
        expected_keys: Set of keys (layer_ids) the gate expects to hear
            from. If None, the gate opens on first update.
        aggregate: "any" opens if any key passes. "all" requires all.
    """

    def __init__(
        self,
        threshold: float,
        comparator: Comparator,
        expected_keys: set[int] | None = None,
        aggregate: Literal["any", "all"] = "any",
    ):
        self.threshold = threshold
        self.comparator = comparator
        self.expected_keys = expected_keys
        self.aggregate = aggregate
        self._decisions: dict[int, bool] = {}

    def reset(self) -> None:
        """Clear all stored decisions."""
        self._decisions.clear()

    def update(self, score: float, *, key: int | None = None) -> None:
        """Record whether a score passes the threshold.

        Args:
            score: The computed score (e.g., cosine similarity).
            key: Layer id or other identifier for this signal.
        """
        if self.comparator == "larger":
            passed = score >= self.threshold
        else:
            passed = score <= self.threshold
        k = key if key is not None else 0
        self._decisions[k] = passed

    def is_open(self) -> bool:
        """Return True if the gate should allow the transform to fire."""
        if not self._decisions:
            return False
        values = self._decisions.values()
        if self.aggregate == "any":
            return any(values)
        return all(values)

    def is_ready(self) -> bool:
        """Return True if all expected keys have reported."""
        if self.expected_keys is None:
            return len(self._decisions) > 0
        return self.expected_keys <= self._decisions.keys()
