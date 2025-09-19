from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer.args import (
    SFTArgs,
)
from aisteer360.algorithms.structural_control.wrappers.trl.sfttrainer.base_mixin import (
    SFTTrainerMixin,
)


class SFT(SFTTrainerMixin):
    """
    Structural control that applies a LoRA adapter.
    """
    Args = SFTArgs
