from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.args import (
    SPPOArgs,
)
from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.control import (
    SPPO,
)

# __all__ = ["SPPO", "SPPOArgs"]

STEERING_METHOD = {
    "category": "structural_control",
    "name": "sppo",
    "control": SPPO,
    "args": SPPOArgs,
}
