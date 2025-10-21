from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.args import (
    SPPOArgs,
)
from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.base_mixin import (
    SPPOTrainerMixin,
)


class SPPO(SPPOTrainerMixin):
    """
    SPPO controller.
    """
    Args = SPPOArgs
