"""
Helper functions for steering.
"""
from collections import defaultdict
from typing import Iterable, Type

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.input_control.base import InputControl, NoInputControl
from aisteer360.algorithms.output_control.base import NoOutputControl, OutputControl
from aisteer360.algorithms.state_control.base import NoStateControl, StateControl
from aisteer360.algorithms.structural_control.base import (
    NoStructuralControl,
    StructuralControl,
)

_DEFAULT_FACTORIES: dict[Type, callable] = {
    InputControl: NoInputControl,
    StructuralControl: NoStructuralControl,
    StateControl: NoStateControl,
    OutputControl: NoOutputControl,
}


def merge_controls(
        supplied: Iterable[StructuralControl | StateControl | InputControl | OutputControl]
) -> dict[str, object]:
    """Sort supplied controls by category and ensure at most one per category.

    Args:
       supplied: List of control instances to organize

    Returns:
       Dict mapping field names to control instances (with default no-ops for unspecified categories)

    Raises:
       ValueError: If multiple controls of the same category are supplied
       TypeError: If an unrecognized control type is supplied
    """
    bucket: dict[type, list] = defaultdict(list)
    for control in supplied:
        for category in _DEFAULT_FACTORIES:
            if isinstance(control, category):
                bucket[category].append(control)
                break
        else:
            raise TypeError(f"Unknown control type: {type(control)}")

    for category, controls in bucket.items():
        if len(controls) > 1:
            names = [type(control).__name__ for control in controls]
            raise ValueError(f"Multiple {category.__name__}s supplied: {names}")

    out: dict[str, object] = {}
    for category, factory in _DEFAULT_FACTORIES.items():
        instance = bucket.get(category, [factory()])[0]  # fresh instance every time
        out_key = (
            "input_control" if category is InputControl else
            "structural_control" if category is StructuralControl else
            "state_control" if category is StateControl else
            "output_control"
        )
        out[out_key] = instance
    return out


def ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Set pad token to eos token if not already defined.

    Args:
       tokenizer: HuggingFace tokenizer instance

    Returns:
       The same tokenizer with pad_token configured
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def to_left_pad(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rearrange a batch to left-padded layout for correct batched causal scoring.

    Left-padding ensures that all sequences in the batch end at the same position, allowing a uniform logit slice
    after concatenation with reference tokens. Works correctly regardless of whether the input is right-padded,
    left-padded, or unpadded.

    Args:
        input_ids: Input token IDs [batch, seq_len]
        attention_mask: Corresponding attention mask [batch, seq_len]

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Left-padded (input_ids, attention_mask)
    """
    batch_size, max_len = input_ids.shape
    seq_lens = attention_mask.sum(dim=1)

    left_ids = input_ids.clone()
    left_mask = torch.zeros_like(attention_mask)

    for i in range(batch_size):
        length = seq_lens[i]
        pad = max_len - length
        if pad > 0:
            real_tokens = input_ids[i][attention_mask[i].bool()]
            pad_tokens = input_ids[i][~attention_mask[i].bool()]
            left_ids[i] = torch.cat([pad_tokens, real_tokens])
        left_mask[i, max_len - length:] = 1

    return left_ids, left_mask
