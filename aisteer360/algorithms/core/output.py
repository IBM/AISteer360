"""The `Output` generation record and the per-row finish-reason inference that populates it."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class Output:
    """The result of one generation call.

    Attributes:
        output_ids: Generated token IDs as a `[batch, seq]` tensor, excluding the prompt (the same
            slice the pipeline returns to the caller by default).
        adapted_input_ids: The `input_ids` actually fed to the model after all input-control
            transformations. None if not provided by the producer.
        finish_reason: One of `"eos"`, `"length"`, or None when neither can be inferred (such as a
            custom stop-string termination).
    """
    output_ids: torch.Tensor
    adapted_input_ids: torch.Tensor | None = None
    finish_reason: str | None = None

    def decode(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode `output_ids` to text. Batch-aware."""
        return tokenizer.batch_decode(
            self.output_ids, skip_special_tokens=skip_special_tokens
        )


def infer_finish_reasons(
    new_tokens: torch.Tensor,
    gen_kwargs: dict,
    *,
    eos_token_id: int | list[int] | None,
    pad_token_id: int | None,
) -> list[str | None]:
    """Infer a per-row finish reason from generated token IDs.

    Args:
        new_tokens: Generated token IDs as a `[batch, gen_len]` tensor, right-padded by `generate`
            (the continuation only, with the prompt excluded).
        gen_kwargs: Generation parameters; only `max_new_tokens` is consulted.
        eos_token_id: End-of-sequence token ID(s); an int, a list of ints, or None. Normalized to a
            set of IDs internally.
        pad_token_id: Padding token ID used to right-pad short rows, or None.

    Returns:
        One reason per row, in order. Each is `"length"`, `"eos"`, or None. For row `i`, trailing
        `pad_token_id` positions are stripped to recover the true continuation length `n`, then:

            - `"length"` if `max_new_tokens` is set and `n >= max_new_tokens`;
            - `"eos"` if `n > 0` and the last unstripped token is in the eos set;
            - `"eos"` if `pad_token_id` is in the eos set and at least one trailing token was
              stripped (the pad-equals-eos configuration common to Llama-family tokenizers, where
              the first stripped token was the genuine EOS);
            - None otherwise (including zero-length rows).

    This is a heuristic. It does not observe Hugging Face stopping-criteria results, so custom
    stop-string terminations are reported as None rather than a distinct reason.
    """
    eos_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    elif eos_token_id is not None:
        eos_ids = {int(token_id) for token_id in eos_token_id}

    max_new = gen_kwargs.get("max_new_tokens")
    pad_equals_eos = pad_token_id is not None and pad_token_id in eos_ids

    reasons: list[str | None] = []
    for row in new_tokens:
        row_list = row.tolist()

        stripped_any = False
        if pad_token_id is not None:
            end = len(row_list)
            while end > 0 and row_list[end - 1] == pad_token_id:
                end -= 1
            stripped_any = end < len(row_list)
            row_list = row_list[:end]

        n = len(row_list)

        if max_new is not None and n >= max_new:
            reasons.append("length")
        elif n > 0 and row_list[-1] in eos_ids:
            reasons.append("eos")
        elif pad_equals_eos and stripped_any:
            reasons.append("eos")
        else:
            reasons.append(None)

    return reasons
