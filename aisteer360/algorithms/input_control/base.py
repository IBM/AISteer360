"""Input control base classes.

This module provides the abstract base class for methods that modify prompts before they reach the model.

Two base classes are provided:

- `InputControl`: Base class for all input control methods.
- `NoInputControl`: Identity (null) control; used when no input control is defined in steering pipeline.

Input controls implement steering through prompt transformation σ(x), enabling behavior modification without altering
model parameters or architecture. These methods transform inputs before they reach the model, resulting in generations
following y ~ p_θ(σ(x)).

Examples of input controls:

- Few-shot learning (prepending examples)
- Prompt templates and formatting
- Soft prompts and prompt tuning
- Chain-of-thought prompting
- Iterative prompt refinement

See Also:

- `aisteer360.algorithms.input_control`: Implementations of input control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.core.base_args import BaseArgs

if TYPE_CHECKING:
    from aisteer360.algorithms.input_control._common.memory.base import Memory


class InputControl(ABC):
    """Abstract base class for input control steering methods.

    Transforms a prompt before it reaches the model, steering behavior via the input alone (no changes to weights,
    architecture, or runtime activations). Any preparation happens once in `steer()`; `adapt()` / `adapt_messages()`
    are a function of the input and the prepared state at inference time.

    Methods:
        adapt(input_ids, runtime_kwargs) -> input_ids: Tensor-level adaptation (required).
        adapt_messages(messages, runtime_kwargs) -> messages | None: Optional message-level adaptation,
            called BEFORE chat-template tokenization. Default returns None (no change).
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional).
        cleanup() -> None: Release resources allocated during steer (optional).

    Subclasses that produce an artifact in `steer()` (instructions, demonstrations, learned weights, ...) may expose it
    via the `memory` attribute, e.g., see `TextMemory`. 
    """

    Args: type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False

    memory: Memory | None = None  # subclasses populate in steer()

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is None:  # null control
            if args or kwargs:
                raise TypeError(f"{type(self).__name__} accepts no constructor arguments.")
            return

        self.args: BaseArgs = self.Args.validate(*args, **kwargs)

        # move fields to attributes
        for field in fields(self.args):
            setattr(self, field.name, getattr(self.args, field.name))

    @abstractmethod
    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Transform `input_ids` into a steered prompt.

        May read instance state (e.g. `self.memory`) that was populated by `steer()`.

        Args:
            input_ids: The user's prompt token IDs.
            runtime_kwargs: Per-call parameters.

        Returns:
            The transformed token IDs.
        """

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        """Optional message-level adaptation, called BEFORE chat-template tokenization.

        Default returns None (no message-level changes). Subclasses that modify chat structure (set/replace system
        prompt, insert example turns) override this. When this method returns a non-None result for a chat-input
        generation, the pipeline does NOT additionally call `adapt()` for that call; controls may therefore implement
        both entry points (message-level for chat input, token-level for raw text/tensor input) without being applied
        twice.

        Args:
            messages: A batch of chats; outer list is the batch, inner list is the message sequence for one chat.
            runtime_kwargs: Per-call parameters.

        Returns:
            The transformed messages, or None to indicate no change.
        """
        return None

    def steer(
        self,
        model=None,
        tokenizer=None,
        **kwargs,
    ) -> None:
        """Optional offline preparation. Default is no-op."""
        pass

    def cleanup(self) -> None:
        """Release resources allocated during `steer()`.

        Override this method in subclasses that allocate GPU memory or other resources during steering to ensure proper
        cleanup.
        """
        pass


class NoInputControl(InputControl):
    """Identity input control.

    Used as the default when no input control is needed. Returns input_ids unchanged.
    """
    enabled: bool = False
    supports_batching: bool = True
    tokenizer: PreTrainedTokenizerBase | None = None

    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        """Identity adapter; returns input_ids unchanged."""
        return input_ids

    def steer(
        self,
        model=None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ) -> None:
        """Null steer operation; attaches tokenizer."""
        self.tokenizer = tokenizer
