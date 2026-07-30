"""Same-model forward of candidate continuations with prefix KV-cache reuse.

Some candidate values require forwarding the pipeline's own model mid-step to read candidate
hidden states; `CandidateForward` performs those forwards. These passes are marked as auxiliary
via `auxiliary_pass(aligned=True)`, so state-control accounting keeps them out of condition
scoring and gate updates while transforms still apply at the candidates' true positions (prefix
and candidate positions lie on the generation's own coordinate axis). At hook points where the
state runtime cannot read positions from `cache_position`, it skips transforming these passes and
warns once. Values scored by an auxiliary model are unaffected.
"""
from __future__ import annotations

import torch
from transformers import PreTrainedModel

from aisteer360.algorithms.core.utils.auxiliary_pass import auxiliary_pass
from aisteer360.algorithms.output_control._common.kv_cache import repeat_cache


class CandidateForward:
    """Batched forward of `(prefix + candidate_k)` for all K candidates on a given model.

    Maintains an internal prefix KV cache keyed on the prefix ids. When a call's prefix extends the
    cached one, only the delta tokens are forwarded to extend the cache (one 1-token forward per
    decode step in the common case); any non-extension (rewind, restart, new generation, scoring
    replay from an unrelated prefix) rebuilds from scratch. The candidate evaluation then repeats a
    fresh copy of the cache across the K candidates, so the incremental cache survives the call.
    Explicit `cache_position` is passed on every with-past forward (this is what `generate` does
    internally). Supports batch size 1 only.

    Args:
        model: The model to forward through (the pipeline's own model for same-model values).
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self._cached_ids: torch.Tensor | None = None  # [1, T_c]
        self._cached_mask: torch.Tensor | None = None  # [1, T_c]
        self._cache = None  # past_key_values covering _cached_ids

    def _extends(self, ids: torch.Tensor) -> bool:
        last = self._cached_ids
        if last is None or ids.size(1) < last.size(1):
            return False
        return bool(torch.equal(ids[:, : last.size(1)], last.to(ids.device)))

    @staticmethod
    def _full_mask(prefix_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """The provided mask right-extended with ones to the prefix length.

        The mask must span the full prefix; generated tokens are always real, so right-extension with
        ones is exact.
        """
        if attention_mask is None:
            return torch.ones_like(prefix_ids)
        pad = prefix_ids.size(1) - attention_mask.size(1)
        if pad < 0:
            raise ValueError("attention_mask is longer than prefix_ids.")
        if pad == 0:
            return attention_mask
        ones = torch.ones(
            attention_mask.size(0), pad, device=attention_mask.device, dtype=attention_mask.dtype
        )
        return torch.cat([attention_mask, ones], dim=1)

    def _sync_cache(self, prefix_ids: torch.Tensor, full_mask: torch.Tensor) -> None:
        """Bring the internal cache up to `prefix_ids` (extend by the delta, or rebuild)."""
        if not self._extends(prefix_ids):
            out = self.model(
                input_ids=prefix_ids, attention_mask=full_mask, use_cache=True, return_dict=True
            )
            self._cache = out.past_key_values
        else:
            cached_len = self._cached_ids.size(1)
            if prefix_ids.size(1) > cached_len:
                delta = prefix_ids[:, cached_len:]
                positions = torch.arange(cached_len, prefix_ids.size(1), device=prefix_ids.device)
                out = self.model(
                    input_ids=delta, attention_mask=full_mask, past_key_values=self._cache,
                    use_cache=True, cache_position=positions, return_dict=True,
                )
                self._cache = out.past_key_values
            # equal length + extends -> cache already current; nothing to do
        self._cached_ids = prefix_ids.detach()
        self._cached_mask = full_mask

    @torch.no_grad()
    def last_hidden_states(
        self,
        prefix_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return last-layer hidden states at the candidate position for each candidate.

        Args:
            prefix_ids: `[1, T]` prefix (batch size 1).
            candidate_ids: `[1, K]` candidate next tokens.
            attention_mask: Optional prefix mask; right-extended with ones to the prefix length.

        Returns:
            A tensor `[K, H]` of last-token hidden states, one per candidate.
        """
        if prefix_ids.size(0) != 1:
            raise ValueError("CandidateForward supports batch size 1 only.")

        num = candidate_ids.size(1)
        device = prefix_ids.device
        full_mask = self._full_mask(prefix_ids, attention_mask)
        with auxiliary_pass(aligned=True):
            self._sync_cache(prefix_ids, full_mask)

            prefix_len = prefix_ids.size(1)
            repeated = repeat_cache(self._cache, num, preserve_input=True)
            cand_tokens = candidate_ids.reshape(num, 1)
            cand_mask = torch.cat(
                [full_mask.repeat(num, 1), torch.ones(num, 1, device=device, dtype=full_mask.dtype)],
                dim=1,
            )
            positions = torch.arange(prefix_len, prefix_len + 1, device=device)
            outputs = self.model(
                input_ids=cand_tokens,
                attention_mask=cand_mask,
                past_key_values=repeated,
                use_cache=True,
                cache_position=positions,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1][:, -1, :]  # [K, H]
