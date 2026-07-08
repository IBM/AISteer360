"""Shared specification dataclasses for state control components."""
from dataclasses import dataclass
from typing import Literal, Sequence

Comparator = Literal["larger", "smaller"]
CompMode = Literal["mean", "last"]


@dataclass(frozen=True)
class LabeledExamples:
    """Independent positive/negative text data with binary labels.

    Does not require equal-length lists (unlike ContrastivePairs).
    Useful for methods where positive and negative examples are independent/
    unpaired (and the estimator concatenates them, e.g., in ITI).

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


@dataclass(frozen=True)
class ContrastivePairs:
    """Paired positive/negative text data for contrastive estimation.

    Attributes:
        positives: Texts exhibiting the target behavior.
        negatives: Texts not exhibiting the target behavior.
        prompts: Optional shared prompts prepended to each text.
            Required when VectorTrainSpec.accumulate == "suffix-only".
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
class VectorTrainSpec:
    """Configuration for how to train/extract direction vectors.

    Attributes:
        method: Extraction algorithm.
            "pca_pairwise" uses PCA on paired differences of hidden states.
            "mean_diff" uses the mean difference of hidden states (CAA method).
        accumulate: How to select hidden state spans for aggregation.
            "all" uses the full sequence.
            "suffix-only" uses only the portion after the shared prompt.
            "last_token" uses only the final non-pad token position.
        batch_size: Batch size for hidden state extraction forward passes.
    """

    method: Literal["pca_pairwise", "mean_diff"] = "pca_pairwise"
    accumulate: Literal["all", "suffix-only", "last_token"] = "all"
    batch_size: int = 8

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")


@dataclass(frozen=True)
class ConditionSearchSpec:
    """Configuration for automatic condition point search.

    Attributes:
        auto_find: If True, run the search during steer(). If False, the
            user must provide condition_layer_ids and threshold manually.
        candidate_layers: Explicit layer ids to search over. If None, use
            layer_range.
        layer_range: (start, end) range of layers to consider. Ignored if
            candidate_layers is set.
        threshold_range: (min, max) for the threshold grid search.
        threshold_step: Step size for the threshold grid.
    """

    auto_find: bool = True
    candidate_layers: Sequence[int] | None = None
    layer_range: tuple[int, int] | None = None
    threshold_range: tuple[float, float] = (-1.0, 1.0)
    threshold_step: float = 0.05

    def __post_init__(self):
        lo, hi = self.threshold_range
        if lo >= hi:
            raise ValueError(f"threshold_range ({lo}, {hi}): min must be < max.")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be > 0.")
