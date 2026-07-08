"""CAST argument validation."""
from dataclasses import dataclass, field
from typing import Sequence

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control._common.specs import (
    Comparator,
    CompMode,
    ConditionSearchSpec,
    ContrastivePairs,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import TokenScope


@dataclass
class CASTArgs(BaseArgs):
    """Arguments for CAST (Conditional Activation Steering).

    Users provide EITHER pre-computed vectors OR training data. If data is
    provided, vectors are fitted during steer(). If vectors are provided,
    data is ignored.

    All layer validation happens in steer() once the model is known.

    Attributes:
        behavior_vector: Pre-computed behavior steering vector.
        behavior_data: Contrastive pairs for training the behavior vector.
        behavior_fit: Training configuration for behavior vector extraction.
        behavior_layer_ids: Layers to apply the behavior vector to. If None,
            defaults to the late third of the model's layers.
        behavior_vector_strength: Scaling factor for the behavior vector.
        condition_vector: Pre-computed condition steering vector.
        condition_data: Contrastive pairs for training the condition vector.
        condition_fit: Training configuration for condition vector extraction.
        search: Configuration for automatic condition point search.
        condition_layer_ids: Layers to check the condition on.
        condition_vector_threshold: Similarity threshold for condition detection.
        condition_comparator_threshold_is: Whether to activate when similarity
            is "larger" or "smaller" than threshold.
        condition_threshold_comparison_mode: How to aggregate hidden states
            for comparison ("mean" or "last").
        apply_behavior_on_first_call: Whether to apply behavior vector on the
            first forward call.
        use_ooi_preventive_normalization: Apply out-of-distribution preventive
            normalization to maintain hidden state magnitudes.
        use_explained_variance: Scale steering vectors by their explained
            variance for adaptive layer-wise control.
        token_scope: Which tokens to steer ("all", "after_prompt", "last_k").
        last_k: Required when token_scope == "last_k".
    """

    # behavior
    behavior_vector: SteeringVector | None = None
    behavior_data: ContrastivePairs | dict | None = None
    behavior_fit: VectorTrainSpec = field(default_factory=VectorTrainSpec)
    behavior_layer_ids: Sequence[int] | None = None
    behavior_vector_strength: float = 1.0

    # condition
    condition_vector: SteeringVector | None = None
    condition_data: ContrastivePairs | dict | None = None
    condition_fit: VectorTrainSpec = field(
        default_factory=lambda: VectorTrainSpec(accumulate="all")
    )
    search: ConditionSearchSpec = field(default_factory=ConditionSearchSpec)
    condition_layer_ids: Sequence[int] | None = None
    condition_vector_threshold: float | None = None
    condition_comparator_threshold_is: Comparator = "larger"
    condition_threshold_comparison_mode: CompMode = "mean"

    # hook behavior
    apply_behavior_on_first_call: bool = True
    use_ooi_preventive_normalization: bool = False
    use_explained_variance: bool = False
    token_scope: TokenScope = "all"
    last_k: int | None = None

    def __post_init__(self):
        if self.behavior_vector_strength < 0:
            raise ValueError("behavior_vector_strength must be >= 0.")

        if self.behavior_vector is not None:
            self.behavior_vector.validate()
        if self.condition_vector is not None:
            self.condition_vector.validate()

        # normalize dict inputs to ContrastivePairs
        if self.behavior_data is not None and not isinstance(self.behavior_data, ContrastivePairs):
            object.__setattr__(self, "behavior_data", as_contrastive_pairs(self.behavior_data))
        if self.condition_data is not None and not isinstance(self.condition_data, ContrastivePairs):
            object.__setattr__(self, "condition_data", as_contrastive_pairs(self.condition_data))

        # must have at least one source for behavior
        if self.behavior_vector is None and self.behavior_data is None:
            raise ValueError("Provide either behavior_vector or behavior_data.")

        # if condition vector is given, condition layers should also be given
        # (or search.auto_find should be True)
        if self.condition_vector is not None and self.condition_layer_ids is None and not self.search.auto_find:
            raise ValueError(
                "When condition_vector is provided without condition_layer_ids, "
                "search.auto_find must be True."
            )

        # token scope cross-check
        if self.token_scope == "last_k" and (self.last_k is None or self.last_k < 1):
            raise ValueError("last_k must be >= 1 when token_scope is 'last_k'.")
