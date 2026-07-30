"""Candidate value from a linear-probe margin.

For each candidate token, forwards `prefix + candidate` through the pipeline's own model via
`CandidateForward` and reads the last hidden state `h`; the value is
`probe.direction . (h - probe.midpoint)`. This is the value underlying `SASA`.
"""
from __future__ import annotations

import torch

from aisteer360.algorithms.output_control._common.estimators.linear_probe import LinearProbe
from aisteer360.algorithms.output_control._common.values.base import (
    BaseCandidateValue,
    StepContext,
)
from aisteer360.algorithms.output_control._common.candidate_forward import CandidateForward


class SubspaceMarginValue(BaseCandidateValue):
    """Per-candidate margin against a fitted `LinearProbe`.

    This value forwards the pipeline's own model per candidate, so it declares
    `same_model_forwards=True`; the forwards run inside `auxiliary_pass(aligned=True)` (via
    `CandidateForward`), so state-control transforms apply to them at the candidates' true
    positions while condition scoring and gates ignore them.

    Args:
        probe: The fitted `LinearProbe` (`direction`, `midpoint`).

    Note:
        `scoring_cost="model_forward"` and `supports_batching` is False (the prefix cache tracks a
        single row).
    """

    supports_batching: bool = False
    scoring_cost = "model_forward"
    same_model_forwards: bool = True

    def __init__(self, probe: LinearProbe):
        self.probe = probe
        self._forward: CandidateForward | None = None
        self._aligned: tuple[torch.device, torch.dtype] | None = None
        self._direction: torch.Tensor | None = None
        self._midpoint: torch.Tensor | None = None

    def _align_probe(self, device: torch.device, dtype: torch.dtype) -> None:
        """Cache the probe tensors aligned to (device, dtype); the probe is fixed per generation."""
        if self._aligned != (device, dtype):
            self._direction = self.probe.direction.to(device, dtype)
            self._midpoint = self.probe.midpoint.to(device, dtype)
            self._aligned = (device, dtype)

    def score(self, ctx: StepContext) -> torch.Tensor:
        if ctx.model is None:
            raise RuntimeError("SubspaceMarginValue requires the pipeline model in StepContext.")
        if self._forward is None or self._forward.model is not ctx.model:
            self._forward = CandidateForward(ctx.model)
        hidden = self._forward.last_hidden_states(
            ctx.prefix_ids, ctx.candidate_ids, ctx.attention_mask
        )  # [K, H]
        self._align_probe(hidden.device, hidden.dtype)
        margins = (self._direction * (hidden - self._midpoint)).sum(dim=-1)  # [K]
        return margins.unsqueeze(0)  # [1, K]
