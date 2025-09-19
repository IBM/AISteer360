from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class ThinkingInterventionArgs(BaseArgs):
    intervention: Callable[[str, dict]] = field(
        default=None,
    )
    # validation
    def __post_init__(self) -> None:
        if not callable(self.intervention):
            raise TypeError("`intervention` must be a callable.")
