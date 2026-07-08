"""Wrapper that rescales hidden states to preserve original norms."""
import torch

from .base import BaseTransform


class NormPreservingTransform(BaseTransform):
    """Wraps an inner transform and rescales to maintain original norms.

    After applying the inner transform, if the norm increased at any position,
    rescale those positions back to original norm. This prevents distribution
    shift from large steering vectors.

    Args:
        inner: The transform to wrap.
    """

    def __init__(self, inner: BaseTransform):
        self._inner = inner

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
