from .actions import Generate, Prefix, Respond, generate, prefix, respond
from .args import RoutedDecodingArgs
from .control import RoutedDecoding

STEERING_METHOD = {
    "category": "output_control",
    "name": "routed_decoding",
    "control": RoutedDecoding,
    "args": RoutedDecodingArgs,
}
