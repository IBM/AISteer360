"""Shared specification dataclasses for state control components."""
from dataclasses import dataclass
from typing import Literal, Sequence

from aisteer360.utils.rendering import PromptFormat

Comparator = Literal["larger", "smaller"]
ComparatorInput = Literal["larger", "smaller", "score_above", "score_below"]
CompMode = Literal["mean", "last"]
HiddenStateLocation = Literal["layer_output", "layer_input"]

_COMPARATOR_ALIASES: dict[str, Comparator] = {
    "larger": "larger", "score_above": "larger",
    "smaller": "smaller", "score_below": "smaller",
}


def normalize_comparator(value: str) -> Comparator:
    """Map user-facing comparator names to the canonical internal values.

    Canonical semantics (THIS toolkit): "larger" opens the gate when score >= threshold; "smaller"
    when score <= threshold.

    WARNING — inverted vs the CAST reference implementation
    (github.com/IBM/activation-steering), where "larger" means "the THRESHOLD is larger" and fires
    when similarity < threshold. Settings copied from the paper or reference repo must flip the
    comparator. Prefer the unambiguous aliases "score_above" / "score_below".

    Args:
        value: One of "larger", "smaller", "score_above", "score_below".

    Returns:
        The canonical comparator ("larger" or "smaller").

    Raises:
        ValueError: If `value` is not a recognized comparator name.
    """
    try:
        return _COMPARATOR_ALIASES[value]
    except KeyError:
        raise ValueError(
            f"Unknown comparator {value!r}; expected one of {sorted(_COMPARATOR_ALIASES)}."
        ) from None


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

    The token sequence the model sees for each example is produced by
    `render_for_model` according to `VectorTrainSpec.prompt_format`, and the
    rendered string is tokenized with `add_special_tokens=False`.

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
class VectorTrainSpec:
    """Configuration for how to train/extract direction vectors.

    Attributes:
        method: Extraction algorithm.
            "pca_pairwise" uses PCA on paired differences of hidden states.
            "pca_center" uses PCA on all positive/negative hidden states centered
                by their grand mean (the CAST extraction from the paper).
            "mean_diff" uses the mean difference of hidden states (CAA method).
        accumulate: How to select hidden state spans for aggregation.
            "all" uses the full sequence.
            "suffix-only" uses only the portion after the shared prompt.
            "last_token" uses only the final non-pad token position.
        batch_size: Batch size for hidden state extraction forward passes.
        prompt_format: How to render contrastive examples into model-ready text
            (via `render_for_model`); the rendered string is tokenized with
            `add_special_tokens=False`.
            "chat_completion" renders `prompts` as user turns and appends
            positives/negatives as completions (prompt+answer pairs, e.g. CAA);
            falls back to "raw" when no `prompts` are provided.
            "chat_prompt" renders each positive/negative as a standalone user turn
            (standalone-prompt contrasts, e.g. the CAST condition); matches the
            inference rendering exactly.
            "raw" concatenates `prompts` + text verbatim with no chat template
            (base-model methods and standalone statements).
        location: Residual-stream boundary each layer key maps to. `outputs.hidden_states` is a
            tuple of `num_layers + 1` tensors: index 0 is the embedding output (the input to layer
            0) and index `i` is the output of layer `i - 1`.
            "layer_output" (default): key `l` maps to the output of layer `l`
            (`hidden_states[l + 1]`); matches controls that hook the layer output (e.g. CAA,
            DirectionalAblation, and `TransformHookRuntime(hook_point="layer_output")`).
            "layer_input": key `l` maps to the input of layer `l`, i.e. the output of layer `l - 1`
            (`hidden_states[l]`); matches pre-hook observers, in particular CAST's runtime condition
            scoring and the `ConditionPointSelector` calibration.
            A vector fit at one boundary is a distinct artifact from one fit at the other; fit it at
            the boundary the consuming control scores or applies it at.
    """

    method: Literal["pca_pairwise", "pca_center", "mean_diff"] = "pca_pairwise"
    accumulate: Literal["all", "suffix-only", "last_token"] = "all"
    batch_size: int = 8
    prompt_format: PromptFormat = "chat_completion"
    location: HiddenStateLocation = "layer_output"

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if self.prompt_format not in ("raw", "chat_completion", "chat_prompt"):
            raise ValueError(
                f"prompt_format must be one of raw/chat_completion/chat_prompt, got {self.prompt_format!r}."
            )
        if self.location not in ("layer_output", "layer_input"):
            raise ValueError(
                f"location must be 'layer_output' or 'layer_input', got {self.location!r}."
            )


@dataclass(frozen=True)
class ConditionSearchSpec:
    """Configuration for automatic condition point search.

    Attributes:
        auto_find: If True, run the search during steer(). If False, the
            user must provide condition_layer_ids and threshold manually.
        candidate_layers: Explicit layer ids to search over. If None, use
            layer_range.
        layer_range: 0-based (start, end) half-open range of layers to consider. Ignored if
            candidate_layers is set. Defaults to all layers.
        threshold_range: (min, max) for the threshold grid search (half-open, step-exact).
        threshold_step: Step size for the threshold grid.
    """

    auto_find: bool = True
    candidate_layers: Sequence[int] | None = None
    layer_range: tuple[int, int] | None = None
    threshold_range: tuple[float, float] = (0.0, 1.0)
    threshold_step: float = 0.01

    def __post_init__(self):
        lo, hi = self.threshold_range
        if lo >= hi:
            raise ValueError(f"threshold_range ({lo}, {hi}): min must be < max.")
        if self.threshold_step <= 0:
            raise ValueError("threshold_step must be > 0.")
