"""Head-level additive transform for activation steering."""
import torch

from ..steering_vector import SteeringVector
from .base import BaseTransform


class HeadAdditiveTransform(BaseTransform):
    """Adds scaled direction vectors to specific head slices.

    For each selected (layer, head) pair, it adds a direction vector to the slice
    [h * head_dim : (h+1) * head_dim].

    For ITI, this operates in pre-o_proj space: the input to the output projection
    where each head_dim-sized slice corresponds to an individual attention head's
    output. The directions must be computed in the same space.

    Expects a SteeringVector whose directions are shaped [num_heads, head_dim]
    per layer, with num_heads and head_dim metadata set. Only head indices
    present in ``active_heads`` are applied; other heads are left untouched.

    Args:
        steering_vector: Unified SteeringVector with per-head directions.
        active_heads: Mapping from layer_id to set of head indices to intervene on.
        strength: Global scaling factor (alpha in ITI).
    """

    def __init__(
        self,
        steering_vector: SteeringVector,
        active_heads: dict[int, set[int]],
        strength: float = 1.0,
    ):
        if steering_vector.num_heads is None or steering_vector.head_dim is None:
            raise ValueError("HeadAdditiveTransform requires num_heads and head_dim metadata on the SteeringVector.")
        self.steering_vector = steering_vector
        self.active_heads = active_heads
        self.num_heads = steering_vector.num_heads
        self.head_dim = steering_vector.head_dim
        self.strength = strength

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply head-level additive steering.

        Args:
            hidden_states: Shape [B, T, H] where H = num_heads * head_dim.
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        heads = self.active_heads.get(layer_id)
        if not heads:
            return hidden_states

        dirs = self.steering_vector.directions.get(layer_id)
        if dirs is None:
            return hidden_states

        hidden_states = hidden_states.clone()

        for head_id in heads:
            start = head_id * self.head_dim
            end = start + self.head_dim
            direction = dirs[head_id]  # [head_dim]
            v = (self.strength * direction).to(dtype=hidden_states.dtype, device=hidden_states.device)
            # scale by token mask so unmasked positions are untouched
            delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
            hidden_states[:, :, start:end] = hidden_states[:, :, start:end] + delta

        return hidden_states
