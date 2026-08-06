"""Shared specification dataclasses for output control components.

`LabeledExamples` is a local copy of `state_control/_common/specs.LabeledExamples` so output-side
estimators (SASA's linear probe) take the same input surface as state-side estimators without a
cross-category import. Consolidating shared data specs into `core/` is a named non-goal; the lift
happens once, here, on purpose.
"""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LabeledExamples:
    """Independent positive/negative text data with binary labels.

    Does not require equal-length lists. Useful for methods where positive and negative examples
    are independent/unpaired (the estimator concatenates them).

    Attributes:
        positives: Texts exhibiting the target behavior (label=1).
        negatives: Texts not exhibiting the target behavior (label=0).
    """

    positives: Sequence[str]
    negatives: Sequence[str]

    def __post_init__(self):
        if len(self.positives) == 0 or len(self.negatives) == 0:
            raise ValueError("positives and negatives must each have at least one entry.")


def as_labeled_examples(x) -> LabeledExamples:
    """Normalize input to `LabeledExamples`.

    Accepts:

        - An existing `LabeledExamples` instance (returned as-is).
        - A dict with keys `"positives"` and `"negatives"`.

    Args:
        x: Input to normalize.

    Returns:
        A `LabeledExamples` instance.

    Raises:
        TypeError: If input is neither `LabeledExamples` nor a suitable dict.
    """
    if isinstance(x, LabeledExamples):
        return x
    if isinstance(x, dict):
        return LabeledExamples(**x)
    raise TypeError("Expected LabeledExamples or dict with positives/negatives.")
