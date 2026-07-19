"""Scorers: packaged condition-scoring callables for gate condition paths.

A condition scorer maps a layer's hidden states to **per-row** condition scores:

    scorer(hidden [B, T, H], layer_id, *, prompt_mask [B, T] | None) -> Tensor [B]

`prompt_mask` is the pad-aware prompt attention mask (True at real tokens) and is supplied by
the runtime only on the prefill pass; on decode passes it is None and `hidden` is the newly
generated token(s). `B` here is the batch the hook observes (possibly beam-expanded); the
runtime collapses scores down to logical rows before feeding the gate. Returning a python
float is permitted only for single-prompt generation — for batches, scorers must be per-row so
batched generation matches per-item generation exactly.
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from ..specs import CompMode
from ..steering_vector import SteeringVector
from .utils.scores import (
    aggregate_condition_hidden,
    projected_cosine_similarity_tensor,
    rank_one_projector,
)


def _extract_directions(artifact: SteeringVector | Mapping[int, torch.Tensor], who: str) -> dict[int, torch.Tensor]:
    """Validate and extract a concrete per-layer directions mapping from an artifact."""
    if isinstance(artifact, SteeringVector):
        directions = dict(artifact.directions)
    elif isinstance(artifact, Mapping):
        directions = dict(artifact)
    else:
        raise TypeError(
            f"{who} expects a concrete SteeringVector or Mapping[int, Tensor]; "
            f"got {type(artifact).__name__}."
        )
    if not all(isinstance(v, torch.Tensor) for v in directions.values()):
        raise TypeError(f"{who} directions must be torch.Tensor values.")
    return directions


class CosineDirectionScorer:
    """Per-row cosine similarity between the last real token and a per-layer direction.

    Scores each row's **last real token** hidden state (per `prompt_mask` when given, else the
    final position) against the layer's steering direction (the first row when the artifact
    stores `[K, H]`). Returns zeros for layers absent from the artifact. Device/dtype casting is
    handled internally.

    Args:
        artifact: A `SteeringVector` or `Mapping[int, torch.Tensor]` of per-layer directions.

    Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
      Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
      [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
    """

    def __init__(self, artifact: SteeringVector | Mapping[int, torch.Tensor]):
        self.directions = _extract_directions(artifact, type(self).__name__)

    @torch.no_grad()
    def __call__(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return `[B]` last-real-token cosine similarities for `layer_id` (zeros if absent)."""
        direction = self.directions.get(layer_id)
        if direction is None:
            return torch.zeros(hidden.size(0), dtype=torch.float32)
        direction = direction.to(dtype=hidden.dtype, device=hidden.device)
        if direction.ndim == 2:
            direction = direction[0]  # first row when [K, H]
        last = aggregate_condition_hidden(hidden, "last", attention_mask=prompt_mask)  # [B, H]
        return F.cosine_similarity(last, direction.unsqueeze(0), dim=-1).float().cpu()


class ProjectedCosineScorer:
    """CAST's condition score: projected-cosine similarity of aggregated prompt states, per row.

    For each condition layer, hidden states are aggregated over real (non-pad) tokens —
    `"mean"` pools, `"last"` selects the last real token — and scored as the cosine similarity
    between the aggregate and its tanh'd rank-one projection onto the condition direction
    (`projected_cosine_similarity_tensor`). One score per batch row, so a `MultiKeyThresholdGate`
    downstream gates each prompt independently.

    Projectors are built lazily from the per-layer directions and cached per
    `(layer_id, device)`; directions with `[K, H]` storage use row 0.

    Args:
        artifact: Condition directions — a `SteeringVector` or `Mapping[int, torch.Tensor]`.
        comparison_mode: Aggregation over prompt tokens: `"mean"` or `"last"`.

    Reference:

    - "Programming Refusal with Conditional Activation Steering"
      Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin,
      Manish Nagireddy, Amit Dhurandhar
      [https://arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
    """

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor],
        comparison_mode: CompMode = "mean",
    ):
        self.directions = _extract_directions(artifact, type(self).__name__)
        self.comparison_mode: CompMode = comparison_mode
        self._projector_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _projector(self, layer_id: int, device: torch.device) -> torch.Tensor | None:
        """Cached `[H, H]` rank-one projector for a layer, or None when the layer is absent."""
        key = (layer_id, device)
        cached = self._projector_cache.get(key)
        if cached is None:
            direction = self.directions.get(layer_id)
            if direction is None:
                return None
            if direction.ndim == 2:
                direction = direction[0]  # [K, H] -> feature row
            cached = rank_one_projector(direction.to(device=device)).to(device)
            self._projector_cache[key] = cached
        return cached

    @torch.no_grad()
    def __call__(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return `[B]` projected-cosine condition scores for `layer_id` (zeros if absent)."""
        projector = self._projector(layer_id, hidden.device)
        if projector is None:
            return torch.zeros(hidden.size(0), dtype=torch.float32)
        aggregated = aggregate_condition_hidden(
            hidden, self.comparison_mode, attention_mask=prompt_mask
        )  # [B, H]
        scores = projected_cosine_similarity_tensor(aggregated, projector.to(aggregated.dtype))
        return scores.float().cpu()
