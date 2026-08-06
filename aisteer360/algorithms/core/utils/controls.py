"""Helpers for working with control objects: composition/validation and adapt-messages guards."""
import warnings
from collections import defaultdict
from typing import Iterable, Type

from aisteer360.algorithms.input_control.base import InputControl, NoInputControl
from aisteer360.algorithms.output_control.base import DecodingDriver, OutputControl
from aisteer360.algorithms.state_control.base import NoStateControl, StateControl
from aisteer360.algorithms.structural_control.base import (
    NoStructuralControl,
    StructuralControl,
)

_DEFAULT_FACTORIES: dict[Type, callable] = {
    InputControl: NoInputControl,
    StructuralControl: NoStructuralControl,
    StateControl: NoStateControl,
    OutputControl: None,  # output has no phantom no-op; the pipeline owns a default driver
}


def merge_controls(
        supplied: Iterable[StructuralControl | StateControl | InputControl | OutputControl]
) -> dict[str, object]:
    """Sort supplied controls by category.

    Every category admits any number of controls, returned as ordered lists (in encounter order)
    under `"input_controls"`, `"structural_controls"`, `"state_controls"`, and `"output_controls"`.
    Omitted input/structural/state categories fall back to a single fresh no-op; an omitted output
    category stays an empty list (the pipeline supplies the default decoding driver as
    infrastructure, not a control entry).

    The output category additionally admits at most one enabled `DecodingDriver`; the decode loop
    does not compose. Input controls chain in two phases (message-level, then token-level); see
    `SteeringPipeline.generate` for the per-control application contract.

    Args:
       supplied: List of control instances to organize

    Returns:
       Dict with keys `"input_controls"`, `"structural_controls"`, `"state_controls"`, and
       `"output_controls"`, each an ordered list of controls, with a single default no-op for
       unspecified input/structural/state categories and an empty list for an unspecified output
       category.

    Raises:
       ValueError: If the same control instance is supplied more than once, or if more than one
           enabled `DecodingDriver` is supplied
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

    # at most one enabled DecodingDriver; the decode loop does not compose
    drivers = [
        control for control in bucket.get(OutputControl, [])
        if isinstance(control, DecodingDriver) and getattr(control, "enabled", True)
    ]
    if len(drivers) > 1:
        names = [type(control).__name__ for control in drivers]
        raise ValueError(
            f"Multiple decoding drivers supplied: {names}. The decode loop does not compose; "
            "keep one DecodingDriver and express the rest as logits processors or stopping criteria."
        )

    out: dict[str, object] = {}
    out["state_controls"] = bucket.get(StateControl) or [NoStateControl()]
    out["output_controls"] = list(bucket.get(OutputControl, []))  # empty stays empty
    out["input_controls"] = bucket.get(InputControl) or [NoInputControl()]
    out["structural_controls"] = bucket.get(StructuralControl) or [NoStructuralControl()]
    return out


def warn_if_adapt_messages_bypassed(input_controls: list[InputControl], already_warned: bool) -> bool:
    """Warn (UserWarning) when any control in `input_controls` overrides `adapt_messages` but the
    caller used tensor/text input, bypassing chat-template tokenization. The warning names each
    bypassed control class. Returns the updated warned-state.

    Args:
        input_controls: The pipeline's input controls, in list order.
        already_warned: Whether the bypass warning has already fired for this pipeline.

    Returns:
        The updated warned-state.
    """
    if already_warned:
        return already_warned
    bypassed = [
        type(control).__name__
        for control in input_controls
        if type(control).adapt_messages is not InputControl.adapt_messages
    ]
    if bypassed:
        warnings.warn(
            f"{', '.join(bypassed)} override(s) `adapt_messages` but received tensor/text input; "
            "the message-level adaptation will not run. Pass `list[dict]` or `list[list[dict]]` "
            "to engage `adapt_messages`.",
            UserWarning,
        )
        return True
    return already_warned
