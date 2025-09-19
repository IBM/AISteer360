from .args import PASTAArgs
from .control import PASTA

# __all__ = ["PASTA", "PASTAArgs"]

STEERING_METHOD = {
    "category": "state_control",
    "name": "pasta",
    "control": PASTA,
    "args": PASTAArgs,
}
