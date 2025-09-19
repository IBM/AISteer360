from dataclasses import dataclass, field
from typing import List, Literal, Optional

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.state_control.cast.utils.steering_vector import (
    SteeringVector,
)


@dataclass
class CASTArgs(BaseArgs):

    behavior_vector: Optional[SteeringVector] = field(
        default=None,
        metadata={"help": "The vector representing the desired behavior change"}
    )
    behavior_layer_ids: List[int] = field(
        default_factory=[10, 11, 12, 13, 14, 15],
        metadata={"help": "Layers to apply the behavior vector to. Default is [10, 11, 12, 13, 14, 15]."}
    )
    behavior_vector_strength: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the behavior vector. Default is 1.0."}
    )
    condition_vector: SteeringVector = field(
        default=None,
        metadata={"help": "The vector representing the condition for applying the behavior."}
    )
    condition_layer_ids: List[int] = field(
        default=None,
        metadata={"help": "Layers to check the condition on."}
    )
    condition_vector_threshold: float = field(
        default=None,
        metadata={"help": "Layers to check the condition on."}
    )
    condition_comparator_threshold_is: Literal["larger", "smaller"] = field(
        default="larger",
        metadata={"help": 'Whether to activate when similarity is "larger" or "smaller" than threshold. Default is "larger".'}
    )
    condition_threshold_comparison_mode: Literal["mean", "last"] = field(
        default="mean",
        metadata={"help": 'How to compare thresholds, either "mean" or "last". Default is "mean".'}
    )
    use_explained_variance: bool = field(
        default=False,
        metadata={"help": "Whether to scale vectors by their explained variance. Default is False."}
    )
    use_ooi_preventive_normalization: bool = field(
        default=False,
        metadata={"help": "Whether to use out-of-input preventive normalization. Default is False."}
    )
    apply_behavior_on_first_call: bool = field(
        default=True,
        metadata={"help": "Whether to apply behavior vector on the first forward call. Default is True."}
    )

    # validate
    def __post_init__(self):

        if self.behavior_vector is not None:
            if not isinstance(self.behavior_vector, SteeringVector):
                raise ValueError("'behavior_vector' must be a SteeringVector.")

            self.behavior_vector.validate()

        if self.condition_vector is not None:
            if not isinstance(self.condition_vector, SteeringVector):
                raise ValueError("'condition_vector' must be a SteeringVector.")

            self.condition_vector.validate()

        if (self.condition_layer_ids is None) != (self.condition_vector is None):
            raise ValueError("condition_layer_ids and condition_vector must be both given or both not given")

        if self.condition_layer_ids is not None:
            _check_layer_ids(self.condition_layer_ids)

        if self.behavior_layer_ids:
            _check_layer_ids(self.behavior_layer_ids)

        if self.condition_comparator_threshold_is not in ["larger", "smaller"]:
            raise ValueError(f"{self.condition_comparator_threshold_is=} should be one of ['larger', 'smaller']")

        if self.condition_threshold_comparison_mode not in ["mean", "last"]:
            raise ValueError(f"{self.condition_threshold_comparison_mode=} should be one of ['mean', 'last']")


def _check_layer_ids(layer_ids):
    """
    Checks validity of layer_ids list

    Raises exception if elements are not int and <0, or elements are not unique.
    """
    for ii, vv in enumerate(layer_ids):
        if not isinstance(vv, int):
            raise ValueError(f"invalid layer_id[{ii}]={vv} is of type {type(vv)} instead of int.")
        if vv < 0:
            raise ValueError(f"invalid layer_id[{ii}]={vv} < 0, should be >=0.")

    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"{layer_ids=} has duplicate entries. layers ids should be unique")
