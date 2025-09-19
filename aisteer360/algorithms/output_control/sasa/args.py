# aisteer360/algorithms/output_control/sasa/args.py
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class SASAArgs(BaseArgs):
    """
    """

    beta: float = field(
        default=0.0,
        metadata={"help": "Scaling coefficient for value redistribution."},
    )
    wv_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a saved steering-vector tensor."},
    )
    gen_wv_data_path: Optional[str] = field(
        default="Jigsaw_data/",
        metadata={"help": "Path to the value dataset, e.g. sentences with labeled toxicity."},
    )
    gen_wv_data: Optional[str] = field(
        default=None,
        metadata={"help": "List of data associated with a targed value dataset, e.g. sentences with labeled toxicity."},
    )
    gen_wv_length: Optional[int] = field(
        default=-1,
        metadata={"help": "The maximum number of samples used for preparing SASA steering if wv_path does not exist."}
    )
    gen_wv_batch_size: Optional[int] = field(
        default=4,
        metadata={"help": "The batch size used for preparing SASA steering if wv_path does not exist."}
    )

    # validation
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("'beta' must be non-negative.")
        if self.wv_path is not None and not self.wv_path.endswith(".pt"):
            raise ValueError("wv_path must point to a .pt file holding a tensor.")
        if self.wv_path is None and self.gen_wv_batch_size < 0:
            raise ValueError("'gen_wv_batch_size' must be non-negative.")
