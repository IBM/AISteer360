"""Base class for hidden-state transforms."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .context import TransformContext


class BaseTransform(ABC):
    """Applies a modification to hidden states at a given layer.

    All transforms receive:
        - hidden_states shaped [B, T, H]
        - the layer_id so the transform can index per-layer artifacts
        - a token_mask shaped [B, T] (True where the transform should apply)

    Transforms MUST NOT modify hidden_states in-place if the original tensor
    is needed later (e.g., for norm-preserving wrappers); return a new tensor.

    Artifact-carrying transforms take their artifact as the first positional argument, typed
    `SteeringVector | Mapping[int, Tensor] | ArtifactSource` and **required**. A concrete artifact
    binds the transform immediately (validated at `__init__`); an `ArtifactSource` leaves it
    *unbound* until `bind(ctx)` resolves the source (validated then). Subclass recipe:

        - store the concrete artifact and set `is_bound=True`, or store the source and set
          `is_bound=False`;
        - override `bind(ctx)` to return a **freshly constructed** bound instance (never mutate
          `self`, never `copy.copy` — derived caches would go stale);
        - override `covered_layer_ids` to report the layers the (bound) artifact covers;
        - call `self._require_bound()` as the first line of `apply`.

    Transforms with no steering artifact keep the defaults (always bound, no coverage).
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

    @property
    def is_bound(self) -> bool:
        """True when the transform can be applied as-is (artifact concrete).

        Default True: transforms that take no steering artifact are always bound.
        """
        return True

    def bind(self, ctx: "TransformContext") -> "BaseTransform":
        """Return a fully-bound transform for this context.

        Contract:

            - MUST NOT mutate `self` (instances/sources are shared across adapters and
              `Benchmark`/`ControlSpec` grid points, whose params objects are reused per point).
            - Returns `self` when already bound (idempotent).
            - When source-carrying, returns a NEW instance of the same class constructed with
              `ctx.resolve(self._source)` and all hyperparameters copied — a fresh construction,
              not `copy.copy`.

        Default: returns self.
        """
        return self

    def _require_bound(self) -> None:
        """Raise a clear `RuntimeError` if this transform is unbound.

        Called as the first line of every artifact-consuming `apply`.
        """
        if not self.is_bound:
            raise RuntimeError(
                f"{type(self).__name__} was constructed with an ArtifactSource and is unbound; pass "
                f"it through a control that resolves and binds it during steer() (ActivationAdapter, "
                f"or CAST via behavior_transform)."
            )

    @property
    def covered_layer_ids(self) -> set[int] | None:
        """Layers this transform can act on; None = unknown (opts out of adapter validation).

        Artifact transforms report `set(directions.keys())` when bound and None when unbound;
        wrappers delegate to their inner transform.
        """
        return None
