from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control._common.drivers.phased import Fixed, Generated, PhasedDriver
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.thinking_intervention.args import (
    ThinkingInterventionArgs,
)


class ThinkingIntervention(PhasedDriver):
    """
    Implementation of Thinking Intervention from Wu et al., 2025.

    `ThinkingIntervention` enables controlled text generation by injecting structured thinking processes into the model's
    reasoning chain. The method modifies the input prompt to include explicit thinking steps enclosed in special tags,
    allowing the model to engage in guided reasoning before producing the final output.

    The algorithm works in three phases:

    1. **Prompt Modification**: Transform the original prompt by applying an intervention function that injects thinking
    instructions, reasoning templates, or structured prompts to guide the model's internal reasoning process.

    2. **Guided Generation**: Generate text using the modified prompt, where the model first produces thinking content
    within special tags (e.g. <think>...</think>) before generating the actual response.

    3. **Output Extraction**: Parse the generated text to extract only the content after the thinking tags.

    ThinkingIntervention is a decoding driver: a thin preset of the generic `PhasedDriver`. Its plan is a single
    replacing `Fixed` phase (the intervention-rewritten prompt) followed by a `Generated` phase, with an
    `extract_after="</think>"` output rule that keeps the original prompt's token prefix and the re-tokenized remainder
    after the closing tag. Per-example `params` supplied as a dict-of-lists are sliced during plan construction.

    Batch-1 plans are constructed per example (the driver loops over rows), preserving the original batched behavior.

    Args:
        intervention (Callable[[str, dict], str]): Function that modifies the input prompt to include thinking
            instructions. Takes the original prompt string and parameter dict, returns the modified prompt string.

    Reference:
        "Effectively Controlling Reasoning Models through Thinking Intervention"
        Tong Wu, Chong Xiang, Jiachen T. Wang, G. Edward Suh, Prateek Mittal
        https://arxiv.org/abs/2503.24370
    """

    Args = ThinkingInterventionArgs

    supports_batching: bool = True

    tokenizer: PreTrainedTokenizer | None = None

    def __init__(self, *args, **kwargs):
        # route through OutputControl (validate ThinkingInterventionArgs, mirror fields, _configure)
        OutputControl.__init__(self, *args, **kwargs)

    def _configure(self) -> None:
        """Fix the phase-splice output rule (`extract_after` = the closing think tag)."""
        self.extract_after = "</think>"

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to re-tokenize the modified prompt."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def plan(self, prompt_text: str, params: dict) -> list:
        """Rewrite the prompt via `intervention`, then generate; keep the post-`</think>` remainder."""
        return [
            Fixed(self.intervention, replace=True, add_special_tokens=True),
            Generated(),
        ]
