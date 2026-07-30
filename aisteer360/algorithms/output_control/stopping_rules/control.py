from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control._common.criteria import (
    BudgetTokens,
    StopOnSubstring,
    StopOnTokens,
)
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.stopping_rules.args import StoppingRulesArgs


class StoppingRules(OutputControl):
    """Stopping criteria as configuration: substring, token, and budget stops.

    `StoppingRules` is the smallest member of the generic family and, without it, the only way to
    get a `StopOnSubstring` into a pipeline without writing a class. It participates through the
    stopping-criteria composition only and contributes no logits processors. Each configured rule
    becomes a fresh, prompt-anchored criterion per generation:

        - `stop_texts=["\\n\\nQ:"]` halts a row once its continuation contains the substring.
        - `stop_token_ids=[13]` halts a row once its last token is one of the ids.
        - `budget=64` halts a row once it has generated `budget` tokens past the prompt.

    `StoppingRules` is a step-level control: `get_stopping_criteria` returns fresh criteria anchored
    at the current prompt length, so two generations with different prompt lengths each stop relative
    to their own prompt. It contributes no logits processors.

    Semantics: criteria are not applied during `compute_logprobs` (there is no loop to stop), and
    under a segment or phase driver the composed criteria apply inside every rollout/phase with the
    prompt-anchored lengths fixed at composition time (a global stop, by design). `StopOnSubstring`
    decodes the continuation each step (the cost of a text-level stop).

    Args:
        stop_texts (list[str]): Substrings that halt a row.
        stop_token_ids (list[int]): Token ids that halt a row.
        budget (int | None): Max new tokens before a row halts. Defaults to None.
    """

    Args = StoppingRulesArgs

    supports_batching: bool = True

    tokenizer: PreTrainedTokenizer | None = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **_) -> PreTrainedModel:
        """Attach the tokenizer (required whenever `stop_texts` is configured)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.stop_texts and self.tokenizer is None:
            raise RuntimeError("StoppingRules requires a tokenizer when 'stop_texts' is configured.")
        return model

    def get_stopping_criteria(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """Return fresh criteria anchored at the current prompt length."""
        prompt_len = input_ids.size(1)
        criteria = []
        for text in self.stop_texts:
            criteria.append(StopOnSubstring(self.tokenizer, text, prompt_len))
        if self.stop_token_ids:
            criteria.append(StopOnTokens(self.stop_token_ids))
        if self.budget is not None:
            criteria.append(BudgetTokens(self.budget, prompt_len))
        return criteria
