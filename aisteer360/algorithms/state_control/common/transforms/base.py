"""Base class for hidden-state transforms."""
from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseTransform(ABC):
    """Applies a modification to hidden states at a given layer.

    All transforms receive:
        - hidden_states shaped [B, T, H]
        - the layer_id so the transform can index per-layer artifacts
        - a token_mask shaped [B, T] (True where the transform should apply)

    Transforms MUST NOT modify hidden_states in-place if the original tensor
    is needed later (e.g., for norm-preserving wrappers); return a new tensor.
    """

    @abstractmethod
    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the transform and return modified hidden states.

        Args:
            hidden_states: Shape [B, T, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T]. True at positions to modify.
            **kwargs: Transform-specific extra arguments.

        Returns:
            Modified hidden states, same shape as input.
        """
        ...
