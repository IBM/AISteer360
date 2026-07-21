"""Wrapper that rescales hidden states to preserve original norms."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .base import BaseTransform

if TYPE_CHECKING:
    from .context import TransformContext


class NormPreservingTransform(BaseTransform):
    """Wraps an inner transform and rescales to maintain original norms.

    After applying the inner transform, if the norm increased at any position,
    rescale those positions back to original norm. This prevents distribution
    shift from large steering vectors.

    Binding and coverage delegate to the inner transform: the wrapper is bound iff the inner is,
    `bind` returns a new wrapper around the bound inner, and coverage is the inner's coverage.

    Args:
        inner: The transform to wrap.
    """

    def __init__(self, inner: BaseTransform):
        self._inner = inner

    @property
    def is_bound(self) -> bool:
        return self._inner.is_bound

    def bind(self, ctx: "TransformContext") -> "NormPreservingTransform":
        if self.is_bound:
            return self
        return NormPreservingTransform(self._inner.bind(ctx))

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return self._inner.covered_layer_ids

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply inner transform then rescale to preserve norms.

        Args:
            hidden_states: Shape [B, T, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Passed to inner transform.

        Returns:
            Modified hidden states with preserved norms.

        Raises:
            ValueError: If NaN or Inf detected after transform.
        """
        self._require_bound()
        original_norm = hidden_states.norm(dim=-1, keepdim=True)
        modified = self._inner.apply(
            hidden_states, layer_id=layer_id, token_mask=token_mask, **kwargs
        )

        if torch.isnan(modified).any() or torch.isinf(modified).any():
            raise ValueError("NaN or Inf detected after transform application.")

        new_norm = modified.norm(dim=-1, keepdim=True)
        # only rescale where norm increased
        needs_rescale = new_norm > original_norm
        if needs_rescale.any():
            scale = torch.where(needs_rescale, original_norm / (new_norm + 1e-8), torch.ones_like(new_norm))
            modified = modified * scale

        return modified
