from dataclasses import dataclass, field

from aisteer360.algorithms.structural_control.wrappers.trl.args import TRLArgs


@dataclass
class DPOArgs(TRLArgs):

    loss_type: str = field(default="sigmoid")
    beta: float = field(default=0.1)
    learning_rate: float = field(default=1e-6)
    max_prompt_length: int | None = field(default=512)
    max_length: int | None = field(default=1024)

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        self.training_args['beta'] = self.beta
        self.training_args['loss_type'] = self.loss_type
        self.training_args['max_prompt_length'] = self.max_prompt_length
        self.training_args['max_length'] = self.max_length
