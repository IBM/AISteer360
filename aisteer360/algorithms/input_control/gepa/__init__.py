from aisteer360.algorithms.input_control.gepa.args import GEPAArgs
from aisteer360.algorithms.input_control.gepa.control import GEPA

STEERING_METHOD = {
    "category": "input_control",
    "name": "gepa",
    "control": GEPA,
    "args": GEPAArgs,
}
