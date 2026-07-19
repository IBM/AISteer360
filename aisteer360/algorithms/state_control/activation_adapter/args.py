"""ActivationAdapter argument validation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.gates.base import BaseGate
from aisteer360.algorithms.state_control._common.selectors.base import BaseSelector
from aisteer360.algorithms.state_control._common.token_scope import TokenScope
from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform
from aisteer360.algorithms.state_control._common.transforms.context import TransformContext

_ARTIFACT_KWARG_HINTS = {
    "steering_vector": "pass it to the transform, e.g. AdditiveTransform(sv, strength=...) "
                       "or DirectionalAblationTransform(sv, alpha=...)",
    "data": "wrap it in a source on the transform: AdditiveTransform(ContrastiveFit(data=...), ...)",
    "train_spec": "its fields are ContrastiveFit kwargs "
                  "(method / accumulate / batch_size / prompt_format / location)",
    "estimator": "ContrastiveFit(estimator=...)",
    "estimator_kwargs": "ContrastiveFit(estimator_kwargs=...)",
    "strength": "AdditiveTransform(..., strength=...)",
    "normalize_vector": "ContrastiveFit(normalize=True) or SteeringVector.normalized()",
}


@dataclass
class ActivationAdapterArgs(BaseArgs):
    """Arguments for `ActivationAdapter`.

    The adapter is the single-behavior atom for activation steering: one transform (which carries
    its own artifact), one selector, one gate, one token scope. It exposes the `_common` component
    families as constructor slots so a recipe can be assembled without writing a new control class.

    The transform is the sole artifact carrier: it holds a concrete `SteeringVector`/directions
    mapping, or an `ArtifactSource` (e.g. `ContrastiveFit(data=...)`) resolved at `steer()` time.
    Artifact kwargs passed to the adapter (`steering_vector`, `data`, `train_spec`, `estimator`,
    `estimator_kwargs`, `strength`, `normalize_vector`) raise a `TypeError` naming the
    transform-based equivalent.

    Attributes:
        transform: A `BaseTransform` instance (bound, or source-carrying and bound at `steer()`), or
            a factory `Callable[[TransformContext], BaseTransform]`. Required.
        layer_ids: Explicit behavior layer(s) to steer (int or sequence). Exactly one of `layer_ids` /
            `layer_selector` must be supplied.
        layer_selector: A `BaseSelector` resolving the behavior layer(s) from `num_layers`.
        hook_point: `"layer_output"` (forward hooks) or `"layer_input"` (pre-hooks).
        gate: A `BaseGate`. A stateful gate requires the condition path (or `gate_driven_externally`);
            None defaults to always-open.
        gate_driven_externally: Follower mode. Set True when another control drives this (shared) gate
            instance; suppresses the stateful-gate condition-path requirement for this adapter.
        condition_layer_ids: Layers to score for the gate (requires `score_fn` and a stateful `gate`).
        score_fn: Per-row condition scorer with signature
            `(hidden [B, T, H], layer_id, *, prompt_mask [B, T] | None) -> Tensor[B] | float`
            (see `runtime.ConditionScorer`). Must return one score per row for batched
            generation; a float is accepted only for single-prompt calls. `prompt_mask` is the
            pad-aware prompt attention mask, supplied on the prefill pass only.
        token_scope: Which positions to steer (see `make_token_mask`).
        last_k: Required when `token_scope == "last_k"`.
        from_position: Required when `token_scope == "from_position"`.
    """

    # transform (sole artifact carrier); required
    transform: BaseTransform | Callable[[TransformContext], BaseTransform] = None

    # layer selection
    layer_ids: int | Sequence[int] | None = None
    layer_selector: BaseSelector | None = None

    # hook site
    hook_point: str = "layer_output"

    # gating (optional; all-or-nothing)
    gate: BaseGate | None = None
    gate_driven_externally: bool = False  # follower mode: another control drives this gate instance
    condition_layer_ids: Sequence[int] | None = None
    score_fn: Callable[..., "torch.Tensor | float"] | None = None  # ConditionScorer: (hidden, layer_id, *, prompt_mask) -> Tensor[B] | float

    # token scope
    token_scope: TokenScope = "after_prompt"
    last_k: int | None = None
    from_position: int | None = None

    @classmethod
    def validate(cls, _init_data: Any | None = None, **kwargs) -> "ActivationAdapterArgs":
        """Reject artifact kwargs with an error naming the transform-based equivalent."""
        merged = {**(_init_data if isinstance(_init_data, Mapping) else {}), **kwargs}
        rejected = [name for name in merged if name in _ARTIFACT_KWARG_HINTS]
        if rejected:
            lines = "; ".join(f"{name!r}: {_ARTIFACT_KWARG_HINTS[name]}" for name in rejected)
            raise TypeError(
                f"ActivationAdapter does not accept {rejected}; the transform is the sole artifact "
                f"carrier. {lines}."
            )
        return super().validate(_init_data, **kwargs)

    def __post_init__(self):
        if isinstance(self.layer_ids, int):
            self.layer_ids = [self.layer_ids]

        # transform required and of the right type
        if self.transform is None:
            raise ValueError(
                "transform is required; the transform carries the steering artifact "
                "(e.g. AdditiveTransform(vector_or_source, strength=...))."
            )
        if not isinstance(self.transform, BaseTransform) and not callable(self.transform):
            raise TypeError(
                f"transform must be a BaseTransform or a Callable[[TransformContext], BaseTransform]; "
                f"got {type(self.transform).__name__}."
            )

        has_condition = self.condition_layer_ids is not None or self.score_fn is not None
        stateful_gate = self.gate is not None and not isinstance(self.gate, AlwaysOpenGate)

        # layer selection (exactly one of layer_ids / layer_selector)
        if (self.layer_ids is None) == (self.layer_selector is None):
            raise ValueError("Provide exactly one of layer_ids or layer_selector.")

        # layer_ids negatives / duplicates
        if self.layer_ids is not None:
            ids = list(self.layer_ids)
            if any(lid < 0 for lid in ids):
                raise ValueError("layer_ids must all be >= 0.")
            if len(set(ids)) != len(ids):
                raise ValueError("layer_ids must not contain duplicates.")

        # condition-path completeness
        has_cond_layers = self.condition_layer_ids is not None
        has_score_fn = self.score_fn is not None

        # need both condition_layer_ids and score_fn
        if has_cond_layers != has_score_fn:
            raise ValueError(
                "A condition path requires both condition_layer_ids and score_fn; provide both or neither."
            )

        # a stateful gate with no condition path never receives evidence, unless another control
        # drives this (shared) gate instance
        if stateful_gate and not has_condition and not self.gate_driven_externally:
            raise ValueError(
                "a stateful gate requires condition_layer_ids and score_fn — or gate_driven_externally"
                "=True if another control drives this gate instance — otherwise it never receives "
                "evidence."
            )

        # condition path but no gate to consume the scores
        if has_condition and not stateful_gate:
            raise ValueError(
                "A condition path (condition_layer_ids + score_fn) requires a stateful gate; scores "
                "would otherwise be computed and ignored."
            )

        # a follower does not drive the gate, so it must not carry a condition path
        if self.gate_driven_externally and has_condition:
            raise ValueError(
                "gate_driven_externally=True marks a follower that does not drive the gate; drop the "
                "flag or the condition path."
            )

        # follower flag is inert without a stateful gate to follow
        if self.gate_driven_externally and not stateful_gate:
            warnings.warn(
                "gate_driven_externally is inert without a stateful gate to follow.", UserWarning
            )

        # token scope requirements
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")

        # last_k / from_position supplied under a scope that ignores them
        if self.last_k is not None and self.token_scope != "last_k":
            warnings.warn(f"last_k is inert under token_scope={self.token_scope!r}.", UserWarning)
        if self.from_position is not None and self.token_scope != "from_position":
            warnings.warn(f"from_position is inert under token_scope={self.token_scope!r}.", UserWarning)

        # hook_point validity
        if self.hook_point not in ("layer_output", "layer_input"):
            raise ValueError(
                f"hook_point must be 'layer_output' or 'layer_input'; got {self.hook_point!r}."
            )
