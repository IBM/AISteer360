"""Contrastive data containers shared by detection and steering estimation."""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ContrastivePairs:
    """Paired positive/negative text data for contrastive estimation.

    The token sequence the model sees for each example is produced by
    `render_for_model` according to the consumer's prompt format, and the
    rendered string is tokenized with `add_special_tokens=False` for
    chat-templated text.

    Attributes:
        positives: Texts exhibiting the target behavior. Treated as completions
            under `prompt_format="chat_completion"`, and as standalone prompts
            under `prompt_format="chat_prompt"`.
        negatives: Texts not exhibiting the target behavior (see `positives`).
        prompts: Optional shared prompts. Used as the user turn under
            `prompt_format="chat_completion"` and as the prefix under
            `prompt_format="raw"`. Required when `accumulate == "suffix-only"`.
            Ignored under `prompt_format="chat_prompt"`.
    """

    positives: Sequence[str]
    negatives: Sequence[str]
    prompts: Sequence[str] | None = None

    def __post_init__(self):
        if len(self.positives) == 0 or len(self.negatives) == 0:
            raise ValueError("positives and negatives must each have at least one entry.")
        if len(self.positives) != len(self.negatives):
            raise ValueError(
                f"positives ({len(self.positives)}) and negatives ({len(self.negatives)}) "
                f"must have equal length."
            )
        if self.prompts is not None and len(self.prompts) != len(self.positives):
            raise ValueError("prompts must have the same length as positives/negatives.")


def as_contrastive_pairs(x) -> ContrastivePairs:
    """Normalize input to ContrastivePairs.

    Accepts:
        - An existing ContrastivePairs instance (returned as-is).
        - A dict with keys "positives", "negatives", and optionally "prompts".

    Args:
        x: Input to normalize.

    Returns:
        ContrastivePairs instance.

    Raises:
        TypeError: If input is neither ContrastivePairs nor a suitable dict.
    """
    if isinstance(x, ContrastivePairs):
        return x
    if isinstance(x, dict):
        return ContrastivePairs(**x)
    raise TypeError("Expected ContrastivePairs or dict with positives/negatives[/prompts].")


@dataclass(frozen=True)
class LabeledExamples:
    """Independent positive/negative text data with binary labels.

    The positive and negative lists need not be the same length. Useful for methods where
    positive and negative examples are independent and unpaired, and the estimator concatenates
    them.

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
    """Normalize input to LabeledExamples.

    Accepts:
        - An existing LabeledExamples instance (returned as-is).
        - A ContrastivePairs instance (converted; pairing is dropped).
        - A dict with keys "positives" and "negatives".

    Args:
        x: Input to normalize.

    Returns:
        LabeledExamples instance.

    Raises:
        TypeError: If input is not LabeledExamples, ContrastivePairs, or a suitable dict.
    """
    if isinstance(x, LabeledExamples):
        return x
    if isinstance(x, ContrastivePairs):
        return LabeledExamples(positives=x.positives, negatives=x.negatives)
    if isinstance(x, dict):
        return LabeledExamples(**x)
    raise TypeError("Expected LabeledExamples, ContrastivePairs, or dict with positives/negatives.")
