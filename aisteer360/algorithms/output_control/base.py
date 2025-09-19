"""Output control base classes.

This module provides the abstract base classes for methods that intervene during text generation (e.g., via modifying
logits, constraining the output space, or implementing alternative decoding strategies).

Two base classes are provided:

- `OutputControl`: Base class for all output control methods.
- `NoOutputControl`: Identity (null) control; used when no output control is defined in steering pipeline.

Output controls implement steering through decoding algorithms and constraints, modifying the sampling process to
produce generations y ~ᵈ p_θ(x), where ~ᵈ indicates the modified generation process.

Examples of output controls:

- Constrained beam search
- Reward-augmented decoding
- Grammar-constrained generation
- Token filtering and masking
- Classifier-guided generation
- Best-of-N sampling

See Also:

- `aisteer360.algorithms.output_control`: Implementations of output control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Type

import torch
from transformers import PreTrainedModel

from aisteer360.algorithms.core.base_args import BaseArgs


class OutputControl(ABC):
    """Abstract base class for output control steering methods.

    Overrides the generation process with custom logic.

    Methods:
        generate(input_ids, attention_mask, runtime_kwargs, model, **gen_kwargs) -> Tensor: Custom generation (required)
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional)
    """

    Args: Type[BaseArgs] | None = None

    enabled: bool = True

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
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        runtime_kwargs: dict | None,
        model: PreTrainedModel,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Custom generation logic."""
        pass

    def steer(self,
              model: PreTrainedModel,
              tokenizer=None,
              **kwargs) -> None:
        """Optional steering/preparation."""
        pass


class NoOutputControl(OutputControl):
    """Identity output control.

    Used as the default when no output control is needed. Calls (unsteered) model's generate.
    """
    enabled: bool = False

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        runtime_kwargs: dict | None,  # only for API compliance as runtime_kwargs are not used in HF models.
        model: PreTrainedModel,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Null generate operation; applies model's generate."""
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)
