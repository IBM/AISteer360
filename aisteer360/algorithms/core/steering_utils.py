"""
Helper functions for steering.
"""
from collections import defaultdict
from typing import Iterable, Type

from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.base import InputControl, NoInputControl
from aisteer360.algorithms.output_control.base import NoOutputControl, OutputControl
from aisteer360.algorithms.state_control.base import NoStateControl, StateControl
from aisteer360.algorithms.structural_control.base import (
    NoStructuralControl,
    StructuralControl,
)

_DEFAULT_FACTORIES: dict[Type, callable] = {
    InputControl: NoInputControl,
    StructuralControl: NoStructuralControl,
    StateControl: NoStateControl,
    OutputControl: NoOutputControl,
}


def merge_controls(
        supplied: Iterable[StructuralControl | StateControl | InputControl | OutputControl]
) -> dict[str, object]:
    """Sort supplied controls by category and ensure at most one per category.

    Args:
       supplied: List of control instances to organize

    Returns:
       Dict mapping field names to control instances (with default no-ops for unspecified categories)

    Raises:
       ValueError: If multiple controls of the same category are supplied
       TypeError: If an unrecognized control type is supplied
    """
    bucket: dict[type, list] = defaultdict(list)
    for control in supplied:
        for category in _DEFAULT_FACTORIES:
            if isinstance(control, category):
                bucket[category].append(control)
                break
        else:
            raise TypeError(f"Unknown control type: {type(control)}")

    for category, controls in bucket.items():
        if len(controls) > 1:
            names = [type(control).__name__ for control in controls]
            raise ValueError(f"Multiple {category.__name__}s supplied: {names}")

    out: dict[str, object] = {}
    for category, factory in _DEFAULT_FACTORIES.items():
        instance = bucket.get(category, [factory()])[0]  # fresh instance every time
        out_key = (
            "input_control" if category is InputControl else
            "structural_control" if category is StructuralControl else
            "state_control" if category is StateControl else
            "output_control"
        )
        out[out_key] = instance
    return out


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Set pad token to eos token if not already defined.

    Args:
       tokenizer: HuggingFace tokenizer instance

    Returns:
       The same tokenizer with pad_token configured
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer
