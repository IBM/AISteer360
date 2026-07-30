"""Helpers for `SteeringPipeline.generate()`: message-level adaptation and chat-template tokenization."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from aisteer360.algorithms.input_control.base import InputControl


def apply_adapt_messages_and_tokenize(
        input_controls: "list[InputControl]",
        tokenizer: "PreTrainedTokenizerBase",
        messages_batch: list[list[dict]],
        runtime_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor | None, set[int]]:
    """Fold every input control's `adapt_messages` over the message batch, then chat-template tokenize once.

    Controls run in list order. A non-None return becomes the input to the next control and marks
    that control as handled at message level; a None return passes the messages through unchanged
    and leaves the control unmarked, so the pipeline later runs its token-level `adapt` instead.
    Each control is therefore applied exactly once per call.

    Returns:
        tuple[input_ids, attention_mask, handled] where `handled` contains `id(control)` for each
        control whose `adapt_messages` returned a non-None result.
    """
    handled: set[int] = set()
    for control in input_controls:
        adapted = control.adapt_messages(
            messages_batch,
            runtime_kwargs=runtime_kwargs,
        )
        if adapted is not None:
            messages_batch = adapted
            handled.add(id(control))

    encoded = tokenizer.apply_chat_template(
        messages_batch,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
    return input_ids, attention_mask, handled
