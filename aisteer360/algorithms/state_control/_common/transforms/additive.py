"""Additive activation steering transform."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import torch

from ..sources import ArtifactSource
from ..steering_vector import SteeringVector
from .base import BaseTransform

if TYPE_CHECKING:
    from .context import TransformContext


class AdditiveTransform(BaseTransform):
    """Adds scaled direction vector(s) to hidden states.

    Supports two modes determined by the shape of the direction tensor:

    Non-positional (T=1, e.g., CAA):
        ``h'[pos] = h[pos] + mask[pos] * strength * direction[0]``

        The same vector is added at every masked position.
        The ``alignment`` parameter is ignored.

    Positional (T>1, e.g., ActAdd):
        ``h'[a+t] = h[a+t] + mask[a+t] * strength * direction[t]``

        Each vector is placed at its alignment-offset position.
        Positions outside [0, seq_len) are silently clipped.
        During KV-cached generation (seq_len=1), the alignment range
        [a, a+T) never intersects [0, 1), so no injection occurs —
        prefill-only semantics emerge from the geometry.

    Args:
        artifact: The steering artifact — a `SteeringVector`, a per-layer directions mapping
            (`Mapping[int, Tensor]`, each `[T, H]`), or an `ArtifactSource` (unbound until
            `bind(ctx)`). Required.
        strength: Global scaling factor.
        alignment: Starting position for positional injection (default: 0).
            Only used when T > 1.
    """

    def __init__(
        self,
        artifact: SteeringVector | Mapping[int, torch.Tensor] | ArtifactSource,
        strength: float = 1.0,
        alignment: int = 0,
    ):
        self.strength = strength
        self.alignment = alignment
        self._source: ArtifactSource | None = None
        self.directions: dict[int, torch.Tensor] | None = None

        if isinstance(artifact, ArtifactSource):
            self._source = artifact
        elif isinstance(artifact, SteeringVector):
            self.directions = artifact.directions
        elif isinstance(artifact, Mapping):
            self.directions = dict(artifact)
        else:
            raise TypeError(
                f"AdditiveTransform artifact must be a SteeringVector, a Mapping[int, Tensor], or an "
                f"ArtifactSource; got {type(artifact).__name__} (did you mean strength=?)."
            )

    @property
    def is_bound(self) -> bool:
        return self.directions is not None

    def bind(self, ctx: "TransformContext") -> "AdditiveTransform":
        if self.is_bound:
            return self
        return AdditiveTransform(ctx.resolve(self._source), strength=self.strength, alignment=self.alignment)

    @property
    def covered_layer_ids(self) -> set[int] | None:
        return set(self.directions.keys()) if self.directions is not None else None

    def apply(
        self,
        hidden_states: torch.Tensor,
        *,
        layer_id: int,
        token_mask: torch.BoolTensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply additive steering.

        Args:
            hidden_states: Shape [B, T_seq, H].
            layer_id: Which layer this is being applied at.
            token_mask: Shape [B, T_seq]. True at positions to modify.
            **kwargs: Ignored.

        Returns:
            Modified hidden states, same shape as input.
        """
        self._require_bound()
        direction = self.directions.get(layer_id)
        if direction is None:
            return hidden_states

        # handle both 1D [H] and 2D [T, H] directions for backward compatibility
        if direction.ndim == 1:
            direction = direction.unsqueeze(0)  # [H] -> [1, H]

        T_steer = direction.size(0)
        seq_len = hidden_states.size(1)

        if T_steer == 1:
            # broadcast mode (e.g., CAA); same vector at all masked positions
            v = (self.strength * direction.squeeze(0)).to(
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            delta = token_mask.unsqueeze(-1).to(hidden_states.dtype) * v.view(1, 1, -1)
            return hidden_states + delta

        # positional mode (e.g., ActAdd); aligned injection
        a = self.alignment
        inject_start = max(a, 0)
        inject_end = min(a + T_steer, seq_len)

        if inject_start >= inject_end:
            return hidden_states

        # slice the steering vector to match the clipped injection range
        vec_start = inject_start - a
        vec_end = vec_start + (inject_end - inject_start)

        v = (self.strength * direction[vec_start:vec_end]).to(
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )  # [inject_len, H]

        mask_slice = token_mask[:, inject_start:inject_end]  # [B, inject_len]
        gated_v = mask_slice.unsqueeze(-1).to(hidden_states.dtype) * v.unsqueeze(0)

        # add in-place at the injection slice
        out = hidden_states.clone()
        out[:, inject_start:inject_end] += gated_v
        return out
