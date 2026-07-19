"""Helpers for working with control objects: composition/validation and adapt-messages guards."""
import warnings
from collections import defaultdict
from typing import Iterable, Type

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
    """Sort supplied controls by category.

    The state category admits any number of controls (returned as an ordered list under
    `"state_controls"`, in encounter order); the input, structural, and output categories admit at
    most one each (returned as a single instance). Omitted categories fall back to a fresh no-op.

    Args:
       supplied: List of control instances to organize

    Returns:
       Dict with keys `"input_control"`, `"structural_control"`, `"output_control"` (single control
       instances) and `"state_controls"` (a list of state controls), with default no-ops for
       unspecified categories.

    Raises:
       ValueError: If the same control instance is supplied more than once, or if multiple input,
           structural, or output controls are supplied
       TypeError: If an unrecognized control type is supplied
    """
    supplied = list(supplied)

    # reject the same control instance supplied twice
    seen_ids: set[int] = set()
    for control in supplied:
        if id(control) in seen_ids:
            raise ValueError(
                f"The same {type(control).__name__} instance was supplied more than once. "
                "To apply a method twice, construct a second instance."
            )
        seen_ids.add(id(control))

    bucket: dict[type, list] = defaultdict(list)
    for control in supplied:
        for category in _DEFAULT_FACTORIES:
            if isinstance(control, category):
                bucket[category].append(control)
                break
        else:
            raise TypeError(f"Unknown control type: {type(control)}")

    # only the state category admits multiple controls; the others stay singular
    for category, controls in bucket.items():
        if category is not StateControl and len(controls) > 1:
            names = [type(control).__name__ for control in controls]
            raise ValueError(f"Multiple {category.__name__}s supplied: {names}")

    out: dict[str, object] = {}
    for category, factory in _DEFAULT_FACTORIES.items():
        if category is StateControl:
            # ordered list in encounter order; fresh no-op when none supplied
            out["state_controls"] = bucket.get(category) or [factory()]
            continue
        instance = bucket.get(category, [factory()])[0]  # fresh instance every time
        out_key = (
            "input_control" if category is InputControl else
            "structural_control" if category is StructuralControl else
            "output_control"
        )
        out[out_key] = instance
    return out


def warn_if_adapt_messages_bypassed(input_control: InputControl, already_warned: bool) -> bool:
    """Warn (UserWarning) when `input_control` overrides `adapt_messages` but the caller used
    tensor/text input, bypassing chat-template tokenization. Returns the updated warned-state.

    Args:
        input_control: The pipeline's input control.
        already_warned: Whether the bypass warning has already fired for this pipeline.

    Returns:
        The updated warned-state.
    """
    if already_warned:
        return already_warned
    cls = type(input_control)
    if cls.adapt_messages is not InputControl.adapt_messages:
        warnings.warn(
            f"{cls.__name__} overrides `adapt_messages` but received tensor/text input; "
            "the message-level adaptation will not run. Pass `list[dict]` or `list[list[dict]]` "
            "to engage `adapt_messages`.",
            UserWarning,
        )
        return True
    return already_warned
