"""Threshold gate that aggregates per-row scores from multiple condition layers."""
from typing import Literal

import torch

from ..specs import ComparatorInput, normalize_comparator
from .base import BaseGate


class MultiKeyThresholdGate(BaseGate):
    """Row-vectorized gate that opens based on threshold comparison of received scores.

    Supports multiple condition layers (keys). Each `update()` records a per-row pass/fail
    decision for that key, and `open_rows()` aggregates across keys with "any"/"all" semantics,
    elementwise per row. Rows are gated independently, so one batched generation can steer some
    prompts and not others.

    The gate retains the raw per-key score tensors (`evidence()`) so callers can surface
    diagnostics without re-deriving them in hook code.

    Comparator semantics are inverted relative to the reference implementation at
    github.com/IBM/activation-steering. Here "larger" opens the gate when the score is
    greater than or equal to the threshold. Any (layer, threshold, comparator) tuple copied
    from the paper or the reference repository must flip the comparator. Prefer the aliases
    "score_above" and "score_below".

    Args:
        threshold: Score threshold for comparison.
        comparator: "larger"/"score_above" opens the gate when score >= threshold;
            "smaller"/"score_below" opens when score <= threshold.
        expected_keys: Set of keys (layer_ids) the gate expects to hear from. Drives
            `is_ready()`: the gate is ready once every expected key has reported. If None,
            the gate is ready after the first update.
        aggregate: "any" opens a row if any key passes for that row. "all" requires all keys.
    """

    def __init__(
        self,
        threshold: float,
        comparator: ComparatorInput,
        expected_keys: set[int] | None = None,
        aggregate: Literal["any", "all"] = "any",
    ):
        self.threshold = threshold
        self.comparator = normalize_comparator(comparator)
        self.expected_keys = expected_keys
        self.aggregate = aggregate
        self._decisions: dict[int, torch.BoolTensor] = {}
        self._scores: dict[int, torch.Tensor] = {}

    def reset(self, num_rows: int = 1) -> None:
        """Clear all stored decisions/scores and size the gate to the logical batch."""
        super().reset(num_rows)
        self._decisions.clear()
        self._scores.clear()

    def update(self, scores: torch.Tensor | float, *, key: int | None = None) -> None:
        """Record per-row pass/fail for one key.

        Args:
            scores: Per-row condition scores, shape `[num_rows]` (float allowed when
                `num_rows == 1`).
            key: Layer id or other identifier for this signal.
        """
        rows = self._coerce_scores(scores)
        if self.comparator == "larger":
            passed = rows >= self.threshold
        else:
            passed = rows <= self.threshold
        k = key if key is not None else 0
        self._decisions[k] = passed
        self._scores[k] = rows

    def open_rows(self) -> torch.BoolTensor:
        """Per-row decision aggregated across keys; all-closed before any evidence."""
        if not self._decisions:
            return torch.zeros(self.num_rows, dtype=torch.bool)
        stacked = torch.stack(list(self._decisions.values()), dim=0)  # [K, num_rows]
        if self.aggregate == "any":
            return stacked.any(dim=0)
        return stacked.all(dim=0)

    def is_ready(self) -> bool:
        """True once all expected keys have reported (or any key, when unspecified)."""
        if self.expected_keys is None:
            return len(self._decisions) > 0
        return self.expected_keys <= self._decisions.keys()

    def evidence(self) -> dict[int, torch.Tensor]:
        """Raw per-key score tensors (`[num_rows]` each) received this generation."""
        return dict(self._scores)
