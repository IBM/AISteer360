from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.args import (
    DPOArgs,
)
from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.control import DPO

# __all__ = ["DPO", "DPOArgs"]

STEERING_METHOD = {
    "category": "structural_control",
    "name": "dpo",
    "control": DPO,
    "args": DPOArgs,
}
