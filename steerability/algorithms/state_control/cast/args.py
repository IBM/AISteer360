"""CAST argument validation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.internals.data import ContrastivePairs, as_contrastive_pairs
from steerability.algorithms.state_control.common.fit_specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    VectorTrainSpec,
)
from steerability.algorithms.state_control.common.selectors.condition_point import ConditionPoint
from steerability.algorithms.state_control.common.steering_vector import SteeringVector
from steerability.algorithms.state_control.common.token_scope import ScopeKind
from steerability.algorithms.state_control.common.transforms.base import BaseTransform

if TYPE_CHECKING:
    from steerability.algorithms.state_control.common.transforms.context import TransformContext


@dataclass
class CASTArgs(BaseArgs):
    """Arguments for CAST (Conditional Activation Steering).

    The behavior artifact comes from `behavior_vector` (a precomputed vector), `behavior_data`
    (contrastive pairs fitted during `steer()`), or `behavior_transform` (a transform that provides
    its own artifact). With `behavior_vector` or `behavior_data`, the control builds an additive
    transform that scales each layer's direction by `behavior_vector_strength` and adds it. When
    both are given, `behavior_vector` is used. `behavior_transform` replaces the additive transform
    with any `BaseTransform` and cannot be combined with the other two.

    Layer indices are checked against the model in `steer()`.

    A configuration without a condition (no condition vector, data, layers, or threshold) steers
    every prompt. A partial condition point raises `ValueError`. The condition layers and the
    threshold must be given together. A `condition_vector` without condition layers requires
    `condition_data` and `search.auto_find=True`, since the condition point is then searched.

    Attributes:
        behavior_vector: Precomputed behavior steering vector.
        behavior_data: Contrastive pairs for fitting the behavior vector. A dict is converted to
            `ContrastivePairs`.
        behavior_fit: Fit configuration for the behavior vector. It is used only when the vector is
            fitted from `behavior_data`. A `behavior_transform` with a source configures its own fit
            (e.g., via `ContrastiveFit`).
        behavior_layer_ids: Layers to apply the behavior artifact to. If None, the late third of the
            model's layers is used.
        behavior_transform: A transform that replaces the default additive transform. Accepts a bound
            `BaseTransform` (e.g., `ProjectionTransform(vector, alpha=0.8)`), a `BaseTransform` with a
            source that `steer()` binds (e.g., `ProjectionTransform(ContrastiveFit(data=...))`), or a
            factory `Callable[[TransformContext], BaseTransform]`. It cannot be combined with
            `behavior_vector` or `behavior_data`. It also requires the defaults for
            `behavior_vector_strength`, `use_explained_variance`, and
            `use_ooi_preventive_normalization` (configure the transform instead). The transform is
            applied at the input of the behavior layers. An artifact that records its extraction
            location must record `"layer_input"`, since `steer()` raises `ValueError` for any other
            location. `ContrastiveFit` records its `location` (default `"layer_output"`) and therefore
            needs `location="layer_input"`.
        behavior_vector_strength: Scaling factor for the behavior vector. Positive values induce the
            target behavior and negative values subtract it (e.g., a negated refusal vector removes
            refusal), following the sign convention of the paper's behavior strength. Must be 1.0
            when `behavior_transform` is set.
        condition_vector: Precomputed condition steering vector.
        condition_data: Contrastive pairs for fitting the condition vector. A dict is converted to
            `ContrastivePairs`.
        condition_fit: Fit configuration for the condition vector. The default fits at the input of
            each layer (`location="layer_input"`), which is where `ConditionPointSelector` calibrates
            and where the runtime condition pre-hook scores.
        search: Configuration for the automatic condition point search.
        condition_point: A complete condition point given as one object. Accepts a `ConditionPoint`
            (e.g., from an earlier `ConditionPointSelector` search, or its `.flipped()` copy) or the
            dict returned by the `CAST.condition_point` property (keys `layer_ids`, `threshold`,
            `comparator`, and optionally `comparison_mode`). `__post_init__` expands it into
            `condition_layer_ids`, `condition_vector_threshold`, and
            `condition_comparator_threshold_is`, and into `condition_threshold_comparison_mode` when
            the point contains a comparison mode. A condition point disables the automatic search.
            `condition_layer_ids` and `condition_vector_threshold` must either both be None or both
            equal the point's layers and threshold (as in arguments that were already expanded).
            Any other combination raises `ValueError`.
        condition_layer_ids: Layers to check the condition on.
        condition_vector_threshold: Similarity threshold for condition detection.
        condition_comparator_threshold_is: When the gate opens. `"ge"` opens when
            score >= threshold and `"le"` opens when score <= threshold.
        condition_threshold_comparison_mode: How hidden states are aggregated for the comparison
            (`"mean"` or `"last"`).
        use_ooi_preventive_normalization: Apply out-of-distribution preventive normalization to
            maintain hidden-state magnitudes. Must be False when `behavior_transform` is set (wrap
            the transform in `NormPreservingTransform` instead).
        use_explained_variance: Scale each layer's behavior direction by its explained variance.
            Only the PCA methods (`"pca_center"`, `"pca_pairwise"`) produce explained variances.
            With `method="mean_diff"` the setting has no effect. Must be False when
            `behavior_transform` is set (scale the transform's artifact beforehand instead).
        token_scope: Which tokens to steer (`"all"`, `"after_prompt"`, `"last_k"`, or
            `"from_position"`). `"all"` (the default, matching the reference implementation) steers
            prompt tokens during prefill at the behavior layers above the condition layer, where the
            gate is already decided, and steers every decode token. `"after_prompt"` steers only
            generated tokens.
        last_k: Number of final positions of each forward pass to steer. Required (at least 1)
            when `token_scope` is `"last_k"`.
        from_position: The absolute position from which steering starts (e.g., for single-pass
            logit scoring). Required (at least 0) when `token_scope` is `"from_position"`.
    """

    # behavior
    behavior_vector: SteeringVector | None = None
    behavior_data: ContrastivePairs | dict | None = None
    behavior_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(method="pca_center", prompt_format="raw")
    )
    behavior_layer_ids: Sequence[int] | None = None
    behavior_transform: BaseTransform | Callable[["TransformContext"], BaseTransform] | None = None
    behavior_vector_strength: float = 1.0

    # condition
    condition_vector: SteeringVector | None = None
    condition_data: ContrastivePairs | dict | None = None
    condition_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(
            method="pca_center", accumulate="all", prompt_format="chat_prompt",
            location="layer_input",
        )
    )
    search: ConditionSearchSpec = field(default_factory=ConditionSearchSpec)
    condition_point: ConditionPoint | Mapping | None = None
    condition_layer_ids: Sequence[int] | None = None
    condition_vector_threshold: float | None = None
    condition_comparator_threshold_is: Comparator = "ge"
    condition_threshold_comparison_mode: CompMode = "mean"

    # hook behavior
    use_ooi_preventive_normalization: bool = False
    use_explained_variance: bool = False
    token_scope: ScopeKind = "all"
    last_k: int | None = None
    from_position: int | None = None

    def __post_init__(self):
        if self.behavior_vector is not None:
            self.behavior_vector.validate()
        if self.condition_vector is not None:
            self.condition_vector.validate()

        if self.condition_comparator_threshold_is not in ("ge", "le"):
            raise ValueError(
                f"condition_comparator_threshold_is must be 'ge' or 'le'; "
                f"got {self.condition_comparator_threshold_is!r}."
            )

        # expand a reusable condition point into the manual triple (supersedes search.auto_find)
        if self.condition_point is not None:
            if isinstance(self.condition_point, ConditionPoint):
                point_layer_ids: Sequence[int] = [self.condition_point.layer_id]
                point_threshold = self.condition_point.threshold
                point_comparator = self.condition_point.comparator
                point_comparison_mode = self.condition_point.comparison_mode
            elif isinstance(self.condition_point, Mapping):
                try:
                    point_layer_ids = list(self.condition_point["layer_ids"])
                    point_threshold = self.condition_point["threshold"]
                    point_comparator = self.condition_point["comparator"]
                except KeyError as exc:
                    raise ValueError(
                        "condition_point dict must have keys 'layer_ids', 'threshold', and "
                        f"'comparator' (missing {exc.args[0]!r})."
                    ) from None
                point_comparison_mode = self.condition_point.get("comparison_mode")
            else:
                raise TypeError(
                    "condition_point must be a ConditionPoint or a mapping with keys 'layer_ids', "
                    f"'threshold', 'comparator'; got {type(self.condition_point).__name__}."
                )

            if len(point_layer_ids) == 0:
                raise ValueError(
                    "condition_point carries no condition layers; a condition point must name at "
                    "least one layer (a complete manual configuration must not degrade to "
                    "unconditional steering)."
                )
            if point_comparison_mode is not None and point_comparison_mode not in ("mean", "last"):
                raise ValueError(
                    f"condition_point comparison_mode must be 'mean' or 'last', got "
                    f"{point_comparison_mode!r}."
                )

            if point_comparator not in ("ge", "le"):
                raise ValueError(
                    f"condition_point comparator must be 'ge' or 'le'; got {point_comparator!r}."
                )
            # args built from a condition point contain the expanded layers and threshold next to it
            expanded = (
                self.condition_layer_ids is not None
                and self.condition_vector_threshold is not None
                and list(self.condition_layer_ids) == list(point_layer_ids)
                and self.condition_vector_threshold == point_threshold
            )
            if not expanded and (self.condition_layer_ids is not None or self.condition_vector_threshold is not None):
                raise ValueError(
                    "condition_point already carries the layers and threshold; drop "
                    "condition_layer_ids and condition_vector_threshold, or drop condition_point."
                )
            self.condition_layer_ids = list(point_layer_ids)
            self.condition_vector_threshold = point_threshold
            self.condition_comparator_threshold_is = point_comparator
            if point_comparison_mode is not None:
                self.condition_threshold_comparison_mode = point_comparison_mode

        # normalize dict inputs to ContrastivePairs
        if self.behavior_data is not None and not isinstance(self.behavior_data, ContrastivePairs):
            object.__setattr__(self, "behavior_data", as_contrastive_pairs(self.behavior_data))
        if self.condition_data is not None and not isinstance(self.condition_data, ContrastivePairs):
            object.__setattr__(self, "condition_data", as_contrastive_pairs(self.condition_data))

        # behavior artifact source: exactly one of vector / data / transform
        if self.behavior_transform is not None:
            if not (isinstance(self.behavior_transform, BaseTransform) or callable(self.behavior_transform)):
                raise TypeError(
                    "behavior_transform must be a BaseTransform or a factory "
                    "Callable[[TransformContext], BaseTransform]; got "
                    f"{type(self.behavior_transform).__name__}."
                )
            if self.behavior_vector is not None or self.behavior_data is not None:
                raise ValueError(
                    "behavior_transform carries its own artifact; provide the vector, data, or source "
                    "on the transform (e.g. ProjectionTransform(ContrastiveFit(data=...))) "
                    "instead of behavior_vector/behavior_data."
                )
            if self.behavior_vector_strength != 1.0:
                raise ValueError(
                    "behavior_vector_strength scales the default additive path and has no referent "
                    "with behavior_transform; construct the transform with its own magnitude "
                    "(e.g. AdditiveTransform(..., strength=...) or ProjectionTransform(..., "
                    "alpha=...))."
                )
            if self.use_ooi_preventive_normalization:
                raise ValueError(
                    "use_ooi_preventive_normalization wraps the default additive path and does not "
                    "apply with behavior_transform; wrap the transform instead "
                    "(NormPreservingTransform(inner))."
                )
            if self.use_explained_variance:
                raise ValueError(
                    "use_explained_variance scales the default additive path and has no "
                    "behavior_transform equivalent; pre-scale the artifact the transform carries."
                )
        elif self.behavior_vector is None and self.behavior_data is None:
            raise ValueError("Provide behavior_vector, behavior_data, or behavior_transform.")

        # a partial condition point must fail rather than degrade to unconditional steering
        if self.condition_layer_ids is not None and self.condition_vector_threshold is None:
            raise ValueError(
                "condition_layer_ids given without condition_vector_threshold; provide a threshold, "
                "or drop condition_layer_ids and search the condition point from condition_data."
            )
        if self.condition_vector_threshold is not None and self.condition_layer_ids is None:
            raise ValueError(
                "condition_vector_threshold given without condition_layer_ids; provide condition "
                "layers, or drop the threshold and search the condition point from condition_data."
            )
        if self.condition_vector is not None and self.condition_layer_ids is None and (
            not self.search.auto_find or self.condition_data is None
        ):
            raise ValueError(
                "condition_vector given without condition_layer_ids; provide condition layers and a "
                "threshold (or a condition_point), or condition_data with search.auto_find=True."
            )
        if not self.search.auto_find and self.condition_data is not None and self.condition_layer_ids is None:
            raise ValueError(
                "condition_data given with search.auto_find=False and no manual condition point; "
                "enable search.auto_find or provide condition_layer_ids and a threshold."
            )

        # token scope cross-checks
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
        if self.token_scope == "from_position" and (self.from_position is None or self.from_position < 0):
            raise ValueError("from_position must be >= 0 when token_scope is 'from_position'.")
