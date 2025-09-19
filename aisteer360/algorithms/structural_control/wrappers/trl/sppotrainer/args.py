from dataclasses import dataclass, field

from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.args import (
    DPOArgs,
)


@dataclass
class SPPOArgs(DPOArgs):
    optim: str | None = field(default="rmsprop")
    logging_first_step: bool = field(default=True)
    beta: float = field(default=0.001)
    loss_type: str = field(default="sppo")

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
             super().__post_init__()

        self.training_args['optim'] = self.optim
        self.training_args['logging_first_step'] = self.logging_first_step
        self.training_args['beta'] = self.beta
        self.training_args['loss_type'] = self.loss_type
