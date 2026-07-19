"""Alignment-adaptive filtering: a transform decorator that gates by feature alignment."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from .context import TransformContext


class AlignmentAdaptiveTransform(BaseTransform):
    """Applies an inner transform only at tokens aligned with a feature direction.

    Computes each token's projection onto a chosen direction and keeps only positions whose
    alignment exceeds `threshold`; those at or below are removed from the token mask before the
    inner transform runs. With `threshold=0.0` this reproduces the paper's Adaptive Angular
    Steering gate `mask = max(0, sign(h·d_feat))` (Eq. 3), which improves coherence on smaller
    models by rotating only tokens already positively aligned with the feature axis. Because it
    only narrows the mask, it composes with ANY `BaseTransform` (e.g., adaptive CAA/ITI for free).

    The transform carries two artifacts — the inner transform's, and its own alignment axis — and
    is bound iff both are bound. `bind` recurses into the inner and resolves its own source; the
    two may share the same source instance (memoization makes the shared fit run once).

    Args:
        inner: The transform to apply at the surviving positions.
        artifact: The steering artifact whose per-layer directions supply the alignment axis — a
            `SteeringVector`, a per-layer directions mapping, or an `ArtifactSource` (unbound until
            `bind(ctx)`). The axis used is `directions[layer_id][direction_index]`. Required.
        threshold: Alignment cutoff; positions with alignment `> threshold` survive.
        direction_index: Row of the per-layer direction tensor to use as the alignment axis
            (row 0 = feature axis for angular steering).
        use_cosine: If True, alignment is cosine similarity; otherwise it is the projection onto
            the unit-normalized direction.

    Reference:

    - "Angular Steering: Behavior Control via Rotation in Activation Space"
      Hieu M. Vu, Tan M. Nguyen
      [https://arxiv.org/abs/2510.26243](https://arxiv.org/abs/2510.26243)
    """

    def __init__(
        self,
        inner: BaseTransform,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        threshold: float = 0.0,
        direction_index: int = 0,
        use_cosine: bool = False,
    ):
        self.inner = inner
        self.threshold = float(threshold)
        self.direction_index = int(direction_index)
        self.use_cosine = bool(use_cosine)
        self._source: ArtifactSource | None = None
        self.steering_vector: SteeringVector | None = None

        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.steering_vector = artifact
        elif isinstance(artifact, Mapping):
            self.steering_vector = SteeringVector(model_type="unknown", directions=dict(artifact))
        else:
            raise TypeError(
                f"AlignmentAdaptiveTransform artifact must be a SteeringVector, a Mapping[int, Tensor], "
                f"or an ArtifactSource; got {type(artifact).__name__} (did you mean threshold=?)."
            )

    @property
    def is_bound(self) -> bool:
        return self.steering_vector is not None and self.inner.is_bound

    def bind(self, ctx: "TransformContext") -> "AlignmentAdaptiveTransform":
        if self.is_bound:
            return self
        own = self.steering_vector if self.steering_vector is not None else ctx.resolve(self._source)
        return AlignmentAdaptiveTransform(
            self.inner.bind(ctx),
            own,
            threshold=self.threshold,
            direction_index=self.direction_index,
            use_cosine=self.use_cosine,
        )

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return self.inner.covered_layer_ids

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Narrow the token mask by feature alignment, then delegate to the inner transform.

        Args:
            hidden_states: Shape `[B, T, H]`.
            layer_id: Which layer this is being applied at.
            token_mask: Shape `[B, T]`. True at positions the inner transform may modify.
            **kwargs: Passed through to the inner transform.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        dirs = self.steering_vector.directions.get(layer_id)
        if dirs is not None:
            direction = dirs[self.direction_index].to(hidden_states.device, hidden_states.dtype)
            if self.use_cosine:
                alignment = torch.nn.functional.cosine_similarity(
                    hidden_states, direction.view(1, 1, -1), dim=-1
                )
            else:
                unit = direction / (direction.norm() + 1e-8)
                alignment = hidden_states @ unit  # [B, T]
            token_mask = token_mask & (alignment > self.threshold)
        return self.inner.apply(hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs)
