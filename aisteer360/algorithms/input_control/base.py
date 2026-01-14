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
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Any, Callable, Type

import torch
from transformers import PreTrainedTokenizerBase

from aisteer360.algorithms.core.base_args import BaseArgs


class InputControl(ABC):
    """Abstract base class for input control steering methods.

    Transforms prompts before model processing through a prompt adapter function that modifies input token sequences.

    Methods:
        get_prompt_adapter(runtime_kwargs) -> Callable: Return transformation function (required)
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional)
    """

    Args: Type[BaseArgs] | None = None

    enabled: bool = True
    supports_batching: bool = False

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
    def get_prompt_adapter(
        self,
        runtime_kwargs: dict | None = None
    ) -> Callable[[list[int] | torch.Tensor, dict[str, Any]], list[int] | torch.Tensor]:
        """Receives (input_ids, runtime_kwargs) and returns modified input_ids."""
        pass

    def steer(self,
              model=None,
              tokenizer=None,
              **kwargs) -> None:
        """Optional steering/preparation."""
        pass


class NoInputControl(InputControl):
    """Identity input control.

    Used as the default when no input control is needed. Returns input_ids.
    """
    enabled: bool = False
    supports_batching: bool = True
    tokenizer: PreTrainedTokenizerBase | None = None

    def get_prompt_adapter(
            self,
            runtime_kwargs: dict | None = None
    ):
        """Null adapter operation; returns identity map."""
        if self.tokenizer is None:
            return lambda ids, _: ids

        def adapter(input_ids: list[int] | torch.Tensor, runtime_kwargs) -> list[int] | torch.Tensor:
            return input_ids

        return adapter

    def steer(
            self,
            model=None,
            tokenizer: PreTrainedTokenizerBase | None = None,
            **kwargs
    ) -> None:
        """Null steer operation; attaches tokenizer."""
        self.tokenizer = tokenizer
