"""Condition scoring for the steering runtime's condition path.

A condition scorer maps a layer's hidden states to per-row condition scores:

    scorer(hidden [B, T, H], layer_id, *, prompt_mask [B, T] | None) -> Tensor [B]

`prompt_mask` is the pad-aware prompt attention mask (True at real tokens) and is supplied by
the runtime only on the prefill pass; on decode passes it is None and `hidden` is the newly
generated token(s). `B` here is the batch the hook observes (possibly beam-expanded); the
runtime collapses scores down to logical rows before feeding the gate. Returning a python
float is permitted only for single-prompt generation; for batches, scorers must return per-row
scores.

This module holds the `ConditionScorer` protocol, the packaged scorers
(`CosineDirectionScorer`, `ProjectedCosineScorer`, `ProbeContributionScorer`), the
`probe_condition()` factory that assembles a fitted probe into `ActivationAdapter`
condition-port kwargs, and the score math shared with `selectors/condition_point.py` so that
selector calibration and runtime scoring provably agree.
"""
from __future__ import annotations

from typing import Mapping, Protocol

import torch
import torch.nn.functional as F

from aisteer360.algorithms.core.internals.pooling import aggregate_condition_hidden
from aisteer360.algorithms.core.internals.probes.probe import Probe

from .gates.base import BaseGate
from .gates.cache_once import CacheOnceGate
from .gates.probe_sum import ProbeSumGate
from .specs import CompMode
from .steering_vector import SteeringVector


class ConditionScorer(Protocol):
    """Per-row condition scorer.

    Maps a layer's hidden states to one score per observed batch row. `prompt_mask` is the
    pad-aware prompt attention mask (True at real tokens), supplied only on the prefill pass and
    already aligned to the hidden batch; on decode passes it is None and `hidden` holds the newly
    generated token(s). A python float return is permitted only for single-prompt generation.
    Scorers may expose `location` and `model_fingerprint`; the adapter validates them when
    present.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | float: ...


def rank_one_projector(direction: torch.Tensor) -> torch.Tensor:
    """Build the rank-one projector `cc^T / (c^T c)` for a direction.

    Args:
        direction: Shape `[H]`.

    Returns:
        Projection matrix of shape `[H, H]`.
    """
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D [H]; got shape {tuple(direction.shape)}.")
    c = direction.float()
    return torch.outer(c, c) / (c @ c + 1e-8)


@torch.no_grad()
def projected_cosine_similarity_tensor(
    hidden: torch.Tensor,
    projector: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity between rows of `hidden` and their projections, one score per row.

    Args:
        hidden: Shape `[..., H]`.
        projector: Shape `[H, H]` outer-product projection matrix.

    Returns:
        Scores of shape `[...]` (float32).
    """
    hidden = hidden.float()
    projector = projector.float()
    projected = torch.tanh(hidden @ projector)  # projector is symmetric
    numerator = (hidden * projected).sum(dim=-1)
    denominator = hidden.norm(dim=-1) * projected.norm(dim=-1) + 1e-8
    return numerator / denominator


@torch.no_grad()
def projected_cosine_similarity(
    hidden_state: torch.Tensor,
    projector: torch.Tensor,
) -> float:
    """Compute cosine similarity between a vector and its projection.

    This function projects the hidden state through the condition subspace
    projector, applies tanh, then computes cosine similarity with the original.
    The CAST method uses this scoring function.

    Args:
        hidden_state: Shape [H] - aggregated hidden state.
        projector: Shape [H, H] - outer-product projection matrix.

    Returns:
        Cosine similarity as a float.
    """
    score = projected_cosine_similarity_tensor(hidden_state.unsqueeze(0), projector)[0]
    return float(score.item())


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
    """Signed cosine similarity between aggregated prompt states and a per-layer direction.

    For each condition layer, hidden states are aggregated over real (non-pad) tokens, where
    `"last"` selects the last real token per row and `"mean"` pools, then scored as the signed
    cosine similarity between the aggregate and the layer's steering direction (the first row
    when the artifact stores `[K, H]`). One score is produced per batch row. Returns zeros for
    layers absent from the artifact. Device/dtype casting is handled internally.

    The score is signed, unlike `ProjectedCosineScorer` whose score is approximately
    `|cos(h, d)|` (alignment with the line spanned by the direction, erasing which side of the
    direction a state lies on). For a mean-difference direction, positives score high, negatives
    score low, and unrelated content scores near zero, so a `"larger"` threshold fails closed on
    out-of-distribution inputs. Prefer this scorer for topic or domain gates whose calibration
    negatives cannot cover the deployment input space. Signed scores can be negative, so pair it
    with a `ConditionSearchSpec.threshold_range` that admits negative values, e.g. `(-1.0, 1.0)`.

    Args:
        artifact: A `SteeringVector` or `Mapping[int, torch.Tensor]` of per-layer directions.
        comparison_mode: Aggregation over prompt tokens: `"last"` or `"mean"`.

    Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
      Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
      [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
    """

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor],
        comparison_mode: CompMode = "last",
    ):
        self.directions = _extract_directions(artifact, type(self).__name__)
        self.comparison_mode: CompMode = comparison_mode

    @torch.no_grad()
    def __call__(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return `[B]` signed cosine condition scores for `layer_id` (zeros if absent)."""
        direction = self.directions.get(layer_id)
        if direction is None:
            return torch.zeros(hidden.size(0), dtype=torch.float32)
        direction = direction.to(dtype=hidden.dtype, device=hidden.device)
        if direction.ndim == 2:
            direction = direction[0]  # first row when [K, H]
        aggregated = aggregate_condition_hidden(
            hidden, self.comparison_mode, attention_mask=prompt_mask
        )  # [B, H]
        return F.cosine_similarity(aggregated, direction.unsqueeze(0), dim=-1).float().cpu()


class ProjectedCosineScorer:
    """Projected-cosine similarity of aggregated prompt states, per row.

    For each condition layer, hidden states are aggregated over real (non-pad) tokens, where
    `"mean"` pools and `"last"` selects the last real token, then scored as the cosine similarity
    between the aggregate and its tanh'd rank-one projection onto the condition direction
    (`projected_cosine_similarity_tensor`). One score is produced per batch row, so each prompt is
    gated independently downstream.

    Projectors are built lazily from the per-layer directions and cached per
    `(layer_id, device)`; directions with `[K, H]` storage use row 0.

    Args:
        artifact: Condition directions, a `SteeringVector` or `Mapping[int, torch.Tensor]`.
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


class ProbeContributionScorer:
    """Per-layer affine contribution of a probe, conforming to the `ConditionScorer` protocol.

    For each condition layer, hidden states are aggregated over real (non-pad) tokens per the
    probe's `pooling`, then scored as the dot product with that layer's weight vector, without
    the bias (the gate applies it once, at decision time). One score is produced per batch row.
    Returns zeros for layers absent from the probe.

    The scorer exposes `location` (the boundary the probe was fitted at) and
    `model_fingerprint` (the fitted model's identity, or None when the probe records none);
    `ActivationAdapter.steer()` validates both when present.

    Args:
        probe: The probe whose weights and pooling define the contribution.

    Attributes:
        location: The probe's capture boundary; validated against the adapter's `hook_point`.
        model_fingerprint: The probe's recorded model identity; None disarms the adapter's
            identity check.
    """

    def __init__(self, probe: Probe):
        self.probe = probe
        self.location: str = probe.location
        self.model_fingerprint: str | None = probe.meta.get("model_fingerprint")

    @torch.no_grad()
    def __call__(
        self,
        hidden: torch.Tensor,
        layer_id: int,
        *,
        prompt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return `[B]` per-layer contributions `w_l . x_l` for `layer_id` (zeros if absent)."""
        weights = self.probe.weights.get(layer_id)
        if weights is None:
            return torch.zeros(hidden.size(0), dtype=torch.float32)
        aggregated = aggregate_condition_hidden(
            hidden, self.probe.pooling, attention_mask=prompt_mask
        )  # [B, H]
        return (aggregated.to(torch.float32) @ weights.to(aggregated.device)).float().cpu()


def probe_condition(
    probe: Probe,
    *,
    cache_once: bool = True,
    allow_model_mismatch: bool = False,
) -> dict:
    """Condition-port kwargs for `ActivationAdapter`, driving steering with a probe.

    Args:
        probe: The fitted probe whose decision admits the intervention.
        cache_once: When True (default), the gate is wrapped in `CacheOnceGate`, so the
            decision is evaluated on the prompt during prefill and holds for the whole
            generation.
        allow_model_mismatch: When True, the scorer's `model_fingerprint` is set to None,
            which disarms the adapter's model-identity check.

    Returns:
        A dict with keys `"score_fn"`, `"gate"`, and `"condition_layer_ids"`, drop-in keyword
        arguments for `ActivationAdapter`.
    """
    scorer = ProbeContributionScorer(probe)
    if allow_model_mismatch:
        scorer.model_fingerprint = None
    gate: BaseGate = ProbeSumGate(probe)
    if cache_once:
        gate = CacheOnceGate(gate)
    return {
        "score_fn": scorer,
        "gate": gate,
        "condition_layer_ids": list(probe.layer_ids),
    }
