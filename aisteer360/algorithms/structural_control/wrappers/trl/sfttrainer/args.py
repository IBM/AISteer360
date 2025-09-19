from dataclasses import dataclass, field

from aisteer360.algorithms.structural_control.wrappers.trl.args import TRLArgs


@dataclass
class SFTArgs(TRLArgs):

    max_seq_length: int = field(default=4096)

    def __post_init__(self) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        self.training_args['max_seq_length'] = self.max_seq_length
