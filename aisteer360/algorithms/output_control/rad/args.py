# aisteer360/algorithms/output_control/sasa/args.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class RADArgs(BaseArgs):
    """
    """

    beta: float = field(
        default=0.0,
        metadata={"help": "Steering intensity."},
    )
    reward_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the trained reward model. See https://github.com/r-three/RAD for details."},
    )

    # validation
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("'beta' must be non-negative.")
