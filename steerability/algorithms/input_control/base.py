"""Input control base class.

This module provides `InputControl`, the abstract base class for methods that modify prompts before they reach the
model.

Input controls steer through a prompt transformation σ(x) and do not change model parameters or architecture.
Generations then follow y ~ p_θ(σ(x)).

Examples of input controls:

- Few-shot learning (prepending examples)
- Prompt templates and formatting
- Soft prompts and prompt tuning
- Chain-of-thought prompting
- Iterative prompt refinement

See Also:

- `steerability.algorithms.input_control`: Implementations of input control methods
- `steerability.algorithms.core.steering_pipeline`: Integration with steering pipeline
"""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedTokenizerBase

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.base_control import BaseControl
from steerability.algorithms.core.execution.contracts import Requirements

if TYPE_CHECKING:
    from steerability.algorithms.input_control.common.memory.base import Memory


class InputControl(BaseControl):
    """Abstract base class for input control steering methods.

    Transforms a prompt before it reaches the model, steering behavior via the input alone (no changes to weights,
    architecture, or runtime activations). Any preparation happens once in `steer()`; `adapt()` / `adapt_messages()`
    are a function of the input and the prepared state at inference time.

    A pipeline may hold several input controls; they chain in `controls`-list order, each control receiving the
    previous control's output within its phase (message-level for chat input, then token-level).

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
            The transformed token IDs, in the padding layout `input_ids` was received in. When the
            length changes, the pipeline infers the attention mask from the leading and trailing
            pad runs.
        """

    def adapt_messages(
        self,
        messages: list[list[dict]],
        runtime_kwargs: dict | None = None,
    ) -> list[list[dict]] | None:
        """Optional message-level adaptation, called BEFORE chat-template tokenization.

        Default returns None (no message-level changes). Subclasses that modify chat structure (set/replace system
        prompt, insert example turns) override this. When this method returns a non-None result for a chat-input
        generation, the pipeline does NOT additionally call this control's `adapt()` for that call; a control may
        therefore implement both entry points (message-level for chat input, token-level for raw text/tensor input)
        without being applied twice. Other input controls in the pipeline still run per their own phase.

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
        session=None,
        **kwargs,
    ) -> None:
        """Optional offline preparation. Default is no-op.

        `session` is a `SteeringSession` on the steering backend, provided by the pipeline.
        """
        pass

    def requirements(self) -> Requirements:
        """Return the backend requirements of this control for each phase.

        The base implementation returns empty requirements, since input controls transform the
        prompt on the client and need nothing beyond the session contract on any backend. A
        control that uses the in-process pipeline model at adapt time overrides this method with
        a generate-phase `Capability.IN_PROCESS_TORCH` requirement. The model access that
        `steer()` needs (e.g., for rollouts or scoring) is declared by `steer_access()`.

        Returns:
            The control's requirements, keyed by phase.
        """
        return Requirements()
