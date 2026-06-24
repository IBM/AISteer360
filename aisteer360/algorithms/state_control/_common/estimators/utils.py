"""Shared utilities for steering vector estimators."""
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Tokenize a flat list of texts independently.

    Unlike tokenize_pairs(), this function tokenizes texts without interleaving.
    Use this for methods like ITI where positive and negative examples are
    independent and do not need co-padding for token alignment.

    Args:
        tokenizer: Tokenizer to use.
        texts: List of text strings.
        device: Target device.

    Returns:
        Dictionary with input_ids and attention_mask tensors.
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pos_texts: Sequence[str],
    neg_texts: Sequence[str],
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Tokenize positive/negative pairs together to ensure consistent padding.

    Interleaves pairs before tokenization so each (pos, neg) pair shares the same
    padding length. This ensures token alignment for shared prefixes, which is
    important because different padding can subtly change attention patterns.

    Args:
        tokenizer: Tokenizer to use.
        pos_texts: List of positive text strings.
        neg_texts: List of negative text strings (same length as pos_texts).
        device: Target device.

    Returns:
        Tuple of (enc_pos, enc_neg) dictionaries with input_ids and attention_mask.
    """
    # interleave: [pos0, neg0, pos1, neg1, ...]
    interleaved = []
    for pos, neg in zip(pos_texts, neg_texts):
        interleaved.append(pos)
        interleaved.append(neg)

    enc = tokenizer(
        interleaved,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # de-interleave: even indices are positive, odd indices are negative
    enc_pos = {k: v[0::2] for k, v in enc.items()}
    enc_neg = {k: v[1::2] for k, v in enc.items()}

    return enc_pos, enc_neg


def get_last_token_positions(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    num_samples: int,
) -> torch.LongTensor:
    """Find the last non-pad token position for each sample.

    Args:
        attention_mask: Shape [N, T] or None.
        seq_len: Sequence length T.
        num_samples: Number of samples N.

    Returns:
        Tensor of shape [N] with last token positions.
    """
    if attention_mask is None:
        # no padding, last token is at seq_len - 1
        return torch.full((num_samples,), seq_len - 1, dtype=torch.long)

    # for each sample, find the last position where attention_mask == 1
    # this handles both left-padded and right-padded sequences
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(num_samples, -1)
    # mask out padded positions with -1
    masked_positions = torch.where(attention_mask == 1, positions, torch.tensor(-1, device=attention_mask.device))
    return masked_positions.max(dim=1).values


def select_at_positions(
    hidden: torch.Tensor,
    positions: torch.LongTensor,
) -> torch.Tensor:
    """Select hidden states at specified positions for each sample.

    Args:
        hidden: Shape [N, T, H].
        positions: Shape [N] with position indices.

    Returns:
        Tensor of shape [N, H].
    """
    N, _, H = hidden.shape
    # gather at the specified positions
    idx = positions.view(N, 1, 1).expand(N, 1, H)
    return hidden.gather(dim=1, index=idx).squeeze(1)
