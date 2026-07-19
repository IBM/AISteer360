"""Shared value types for steering controls."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class Output:
    """The result of one generation call.

    Attributes:
        output_ids: Generated token IDs as a `[batch, seq]` tensor, excluding the prompt (the same slice the
            pipeline returns to the caller by default).
        adapted_input_ids: The `input_ids` actually fed to the model after all input-control transformations.
            Useful for inspection/debugging (e.g., to see the steered prompt). None if not provided by the pipeline.
        runtime_kwargs: The `runtime_kwargs` that produced this output. May be None if no runtime overrides were used.
        finish_reason: How generation ended; one of `"eos"`, `"length"`, `"stop_token"`, or None if not tracked.
        metadata: Open-ended dict for pipeline-attached extras (resolved gen_kwargs, timing, token counts).
            Keys are not part of the stable contract.
    """
    output_ids: torch.Tensor
    adapted_input_ids: torch.Tensor | None = None
    runtime_kwargs: dict | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None

    def decode(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        skip_special_tokens: bool = True,
    ) -> list[str]:
        """Decode `output_ids` to text. Batch-aware."""
        return tokenizer.batch_decode(
            self.output_ids, skip_special_tokens=skip_special_tokens
        )
