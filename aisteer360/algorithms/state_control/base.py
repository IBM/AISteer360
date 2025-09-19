"""State control base classes.

This module provides the abstract base class for methods that register hooks into the model (e.g., to modify
intermediate representations during inference); does not change model weights.

Two base classes are provided:

- `StateControl`: Base class for all state control methods.
- `NoStateControl`: Identity (null) control; used when no state control is defined in steering pipeline.

State controls implement steering through runtime intervention in the model's forward pass, modifying internal states
(activations, attention patterns) to produce generations following y ~ p_θᵃ(x), where "p_θᵃ" is the model with state
controls.

Examples of state controls:

- Activation steering (e.g., adding direction vectors)
- Attention head manipulation and pruning
- Layer-wise activation editing
- Dynamic routing between components
- Representation engineering techniques

The base class provides automatic hook management through context managers (ensures cleanup and avoids memory leaks).

See Also:

- `aisteer360.algorithms.state_control`: Implementations of state control methods
- `aisteer360.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Callable, Type

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.base_args import BaseArgs

PreHook = Callable[[nn.Module, tuple], tuple | torch.Tensor]
ForwardHook = Callable[[nn.Module, tuple, torch.Tensor], torch.Tensor]
BackwardHook = Callable[[nn.Module, tuple, tuple], tuple]
HookSpec = dict[str, str | PreHook | ForwardHook | BackwardHook]


class StateControl(ABC):
    """Abstract base class for state control steering methods.

    Modifies internal model states during forward passes via hooks.

    Methods:
        get_hooks(input_ids, runtime_kwargs, **kwargs) -> dict: Create hook specs (required)
        steer(model, tokenizer, **kwargs) -> None: One-time preparation (optional)
        reset() -> None: Reset logic (optional)
        register_hooks(model) -> None: Attach hooks to model (provided)
        remove_hooks() -> None: Remove all registered hooks (provided)
    """

    Args: Type[BaseArgs] | None = None

    enabled: bool = True
    _model_ref: PreTrainedModel | None = None

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is None:  # null control
            if args or kwargs:
                raise TypeError(f"{type(self).__name__} accepts no constructor arguments.")
            return

        self.args: BaseArgs = self.Args.validate(*args, **kwargs)

        # move fields to attributes
        for field in fields(self.args):
            setattr(self, field.name, getattr(self.args, field.name))

        self.hooks: dict[str, list[HookSpec]] = {"pre": [], "forward": [], "backward": []}
        self.registered: list[torch.utils.hooks.RemovableHandle] = []

    @abstractmethod
    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **kwargs,
    ) -> dict[str, list[HookSpec]]:
        """Create hook specifications for the current generation."""
        pass

    def steer(self,
              model: PreTrainedModel,
              tokenizer: PreTrainedTokenizerBase = None,
              **kwargs) -> None:
        """Optional steering/preparation."""
        pass

    def register_hooks(self, model: PreTrainedModel) -> None:
        """Attach hooks to model."""
        for phase in ("pre", "forward", "backward"):
            for spec in self.hooks[phase]:
                module = model.get_submodule(spec["module"])
                if phase == "pre":
                    handle = module.register_forward_pre_hook(spec["hook_func"], with_kwargs=True)
                elif phase == "forward":
                    handle = module.register_forward_hook(spec["hook_func"], with_kwargs=True)
                else:
                    handle = module.register_full_backward_hook(spec["hook_func"])
                self.registered.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self.registered:
            handle.remove()
        self.registered.clear()

    def set_hooks(self, hooks: dict[str, list[HookSpec]]):
        """Update the hook specifications to be registered."""
        self.hooks = hooks

    def __enter__(self):
        """Context manager entry: register hooks to model.

        Raises:
            RuntimeError: If model reference not set by pipeline
        """
        if self._model_ref is None:
            raise RuntimeError("Model reference not set before entering context.")
        self.register_hooks(self._model_ref)

        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit: clean up all hooks."""
        self.remove_hooks()

    def reset(self):
        """Optional reset call for state control."""
        pass


    def reset(self):
        """Optional reset call for state control"""
        pass


class NoStateControl(StateControl):
    """Identity state control.

    Used as the default when no state control is needed. Returns empty hook dictionaries and skips registration.
    """
    enabled: bool = False

    def get_hooks(self, *_, **__) -> dict[str, list[HookSpec]]:
        """Return empty hooks."""
        return {"pre": [], "forward": [], "backward": []}

    def steer(self,
              model: PreTrainedModel,
              tokenizer=None,
              **kwargs) -> None:
        """Null steering operation."""
        pass

    def register_hooks(self, *_):
        """Null registration operation."""
        pass

    def remove_hooks(self, *_):
        """Null removal operation."""
        pass

    def set_hooks(self, hooks: dict[str, list[HookSpec]]):
        """Null set operation."""
        pass

    def reset(self):
        """Null reset operation."""
        pass
