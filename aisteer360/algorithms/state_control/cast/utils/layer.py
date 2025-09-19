import typing
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class LayerControlParams:
    control: torch.Tensor | None = None
    operator: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = (
        lambda current, control: current + control
    )
    """
    A dataclass for layer control parameters.

    Attributes:
        control: Optional tensor for control.
        operator: Callable that defines how to apply the control to the current state.
    """

    @classmethod
    def default(cls):
        """
        Return a default instance of LayerControlParams.

        Returns:
            A LayerControlParams instance with default values.
        """
        return cls()


@dataclass
class LayerArgs():

    is_multi_steering: bool = field(
        default=False,
        metadata={"help": "Whether this layer is performing multi-condition steering."}
    )

    behavior_vector: torch.Tensor = field(
        default=None,
        metadata={"help": "The behavior vector to apply at this layer."}
    )

    condition_projector: torch.Tensor = field(
        default=None,
        metadata={"help": "The condition projector to use at this layer."}
    )

    threshold: float = field(
        default=0.0,
        metadata={"help": "The threshold for condition activation."}
    )

    use_ooi_preventive_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to use out-of-input preventive normalization. Default is False."}
    )
    apply_behavior_on_first_call: bool = field(
        default=True,
        metadata={"help": "Whether to apply behavior vector on the first forward call. Default is True."}
    )
    condition_comparator_threshold_is: Literal["larger", "smaller"] = field(
        default="larger",
        metadata={"help": 'Whether to activate when similarity is "larger" or "smaller" than threshold. Default is "larger".'}
    )
    condition_threshold_comparison_mode: Literal["mean", "last"] = field(
        default="mean",
        metadata={"help": 'How to compare thresholds, either "mean" or "last". Default is "mean".'}
    )

    params: LayerControlParams = field(
        default=None,
        metadata={"help": "Layer parameters controlling how to apply CAST offset. for information, see LayerControlParams dataclass"}
    )

    # validate
    def __post_init__(self):

        if self.is_multi_steering:
            raise ValueError("Only a simple condition steering is allowed for now.")
