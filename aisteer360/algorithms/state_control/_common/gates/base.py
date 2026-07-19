"""Base class for runtime gates that control transform application.

Gates are **row-vectorized**: a gate holds one open/closed decision per *logical batch row* (one
per prompt in the batch). Beam expansion is not the gate's concern — the runtime collapses
beam-expanded scores down to logical rows before `update()` and re-expands `open_rows()` when
masking hidden states. The scalar case is simply `num_rows == 1`.
"""
from abc import ABC, abstractmethod

import torch


class BaseGate(ABC):
    """Decides, per logical batch row, whether a transform should fire during a generation step.

    Lifecycle per generation call:
        1. reset(num_rows) - clear state from the previous generation and size the gate to the
           logical batch (number of prompts, not the beam-expanded batch).
        2. update(scores) - called from condition hooks as evidence arrives; `scores` is a
           `[num_rows]` tensor (a python float is accepted only when `num_rows == 1`).
        3. open_rows() - queried by behavior hooks; returns a `[num_rows]` bool tensor.

    `is_ready()` reports whether the gate has received all the evidence it expects. The runtime
    uses it to stop paying for condition scoring once the decision is complete (see
    `TransformHookRuntime.build_condition_hook`), so "score the prompt once, then freeze" is a
    property of the gate's evidence contract rather than a hook-side pass counter.
    """

    num_rows: int = 1

    def reset(self, num_rows: int = 1) -> None:
        """Clear all state and size the gate to `num_rows` logical batch rows."""
        if num_rows < 1:
            raise ValueError(f"num_rows must be >= 1; got {num_rows}.")
        self.num_rows = int(num_rows)

    @abstractmethod
    def update(self, scores: torch.Tensor | float, *, key: int | None = None) -> None:
        """Provide a new evidence signal to the gate.

        Args:
            scores: Per-row condition scores of shape `[num_rows]`. A bare float is accepted
                only when `num_rows == 1` (the scalar gate case); passing a float for a
                multi-row gate raises, because silently broadcasting one score across rows
                would make batched generation diverge from per-item generation.
            key: Optional identifier for the source (e.g., layer_id) when the gate
                aggregates signals from multiple sources.
        """
        ...

    @abstractmethod
    def open_rows(self) -> torch.BoolTensor:
        """Return a `[num_rows]` bool tensor; True where the transform should be applied."""
        ...

    def is_open(self) -> bool:
        """Scalar convenience: True if ANY row is open (exact for `num_rows == 1`)."""
        return bool(self.open_rows().any())

    def is_ready(self) -> bool:
        """Return True if the gate has received all expected evidence.

        Default returns True (gate is always ready to make a decision).
        Override for gates that wait for multiple signals before deciding.
        """
        return True

    def _coerce_scores(self, scores: torch.Tensor | float) -> torch.Tensor:
        """Normalize `scores` to a float32 `[num_rows]` CPU tensor, enforcing the row contract."""
        if isinstance(scores, (int, float)):
            if self.num_rows != 1:
                raise ValueError(
                    f"Gate has {self.num_rows} rows but received a scalar score; condition "
                    f"scorers must return per-row scores ([num_rows]) for batched generation."
                )
            return torch.tensor([float(scores)], dtype=torch.float32)
        t = torch.as_tensor(scores, dtype=torch.float32).reshape(-1).cpu()
        if t.numel() != self.num_rows:
            raise ValueError(
                f"Gate has {self.num_rows} rows but received {t.numel()} scores."
            )
        return t


class AlwaysOpenGate(BaseGate):
    """Gate that is always open for every row. Use when no conditional gating is needed.

    This avoids branching in hook logic: methods without conditions still
    go through the gate, but it is always open with zero overhead.
    """

    def update(self, scores: torch.Tensor | float, *, key: int | None = None) -> None:
        pass

    def open_rows(self) -> torch.BoolTensor:
        return torch.ones(self.num_rows, dtype=torch.bool)
