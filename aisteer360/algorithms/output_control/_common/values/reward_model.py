"""Candidate value from an auxiliary sequence classifier.

For each row-candidate, decodes `prefix + candidate` to text and scores it with an auxiliary reward
model. The value owns the scoring forward; loading and configuration of the reward model is done by
the owning control (RAD), which constructs this value with the loaded model, tokenizer, and score
function.
"""
from __future__ import annotations

from typing import Callable

import torch

from aisteer360.algorithms.output_control._common.values.base import (
    BaseCandidateValue,
    StepContext,
)


class RewardModelValue(BaseCandidateValue):
    """Score `prefix + candidate` text with an auxiliary reward model (RAD).

    Args:
        reward_model: The loaded auxiliary reward model (in eval mode).
        rm_tokenizer: Tokenizer for the reward model. Its `max_length` attribute (if set) bounds the
            reward-model input length.
        rm_score_fn: Extracts a scalar `[batch]` reward from the reward model's output. Defaults to
            the first output column.

    Note:
        `scoring_cost="aux_forward"` and `supports_batching=True`; candidate scoring already batches
        `B * K` rows through the auxiliary model.
    """

    supports_batching: bool = True
    scoring_cost = "aux_forward"

    def __init__(
        self,
        reward_model,
        rm_tokenizer,
        rm_score_fn: Callable | None = None,
    ):
        self.reward_model = reward_model
        self.rm_tokenizer = rm_tokenizer
        self.rm_score_fn = rm_score_fn if rm_score_fn is not None else (lambda output: output[:, 0])
        self._device = next(reward_model.parameters()).device

    @torch.inference_mode()
    def score(self, ctx: StepContext) -> torch.Tensor:
        batch_size = ctx.prefix_ids.size(0)
        num_candidates = ctx.candidate_ids.size(1)

        # build (prefix + candidate) ids for every row-candidate, then decode to text
        prefix = ctx.prefix_ids.unsqueeze(1).expand(-1, num_candidates, -1)  # [B, K, T]
        combined = torch.cat([prefix, ctx.candidate_ids.unsqueeze(-1)], dim=-1)  # [B, K, T+1]
        flat = combined.reshape(batch_size * num_candidates, -1)  # [B*K, T+1]
        texts = ctx.lm_tokenizer.batch_decode(flat, skip_special_tokens=True)

        max_length = getattr(self.rm_tokenizer, "max_length", None)
        inputs = self.rm_tokenizer.batch_encode_plus(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self._device)
        output = self.reward_model(**inputs)
        rewards = self.rm_score_fn(output)  # [B*K]
        return rewards.reshape(batch_size, num_candidates)
