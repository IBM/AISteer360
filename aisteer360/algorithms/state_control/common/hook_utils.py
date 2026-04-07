"""Utilities for hook registration and model inspection."""
import torch
import torch.nn as nn
from transformers import PreTrainedModel


def get_model_layer_list(model: PreTrainedModel) -> tuple[list, list[str]]:
    """Return (layer_modules, layer_name_strings) for a HuggingFace model.

    Supports llama/mistral/gemma-style (model.model.layers) and
    GPT2-style (model.transformer.h) architectures.

    Args:
        model: A HuggingFace causal LM.

    Returns:
        Tuple of (list of nn.Module layers, list of dotted name strings).

    Raises:
        ValueError: If model architecture is not recognized.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        modules = list(model.model.layers)
        prefix = "model.layers"
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        modules = list(model.transformer.h)
        prefix = "transformer.h"
    else:
        raise ValueError(
            f"Cannot determine layer list for {type(model).__name__}. "
            f"Expected model.model.layers or model.transformer.h."
        )
    names = [f"{prefix}.{i}" for i in range(len(modules))]
    return modules, names


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Return the dtype of a model's parameters.

    Works with both HuggingFace ``PreTrainedModel`` (which exposes
    ``model.dtype``) and vLLM ``nn.Module`` by falling back to the 
    first parameter's dtype.
    """
    if hasattr(model, "dtype"):
        return model.dtype
    return next(model.parameters()).dtype


def _hidden_states_index(input_args: tuple) -> int | None:
    """Return the index of the hidden_states tensor in positional args.

    HF layers: ``forward(hidden_states, ...)`` → index 0.
    vLLM layers: ``forward(positions, hidden_states, residual)`` → index 1.

    Uses ``ndim >= 2`` to skip 1-D tensors like positions.
    Returns ``None`` if no suitable tensor is found.
    """
    for i, arg in enumerate(input_args):
        if isinstance(arg, torch.Tensor) and arg.ndim >= 2:
            return i
    return None


def extract_hidden_states(input_args: tuple, input_kwargs: dict) -> torch.Tensor | None:
    """Extract hidden_states tensor from a pre-hook's arguments.

    Works with both HuggingFace layers (hidden_states as first arg)
    and vLLM layers (hidden_states as second arg after positions).

    Args:
        input_args: Positional args from the pre-hook.
        input_kwargs: Keyword args from the pre-hook.

    Returns:
        The hidden_states tensor, or None if not found.
    """
    if input_args:
        idx = _hidden_states_index(input_args)
        if idx is not None:
            return input_args[idx]
    return input_kwargs.get("hidden_states")


def replace_hidden_states(
    input_args: tuple,
    input_kwargs: dict,
    new_hidden: torch.Tensor,
) -> tuple[tuple, dict]:
    """Return modified (input_args, input_kwargs) with hidden_states replaced.

    Works with both HuggingFace and vLLM layer argument patterns.

    Args:
        input_args: Original positional args.
        input_kwargs: Original keyword args.
        new_hidden: Replacement hidden states tensor.

    Returns:
        Tuple of (new_input_args, new_input_kwargs).
    """
    if input_args:
        idx = _hidden_states_index(input_args)
        if idx is not None:
            args_list = list(input_args)
            args_list[idx] = new_hidden
            return tuple(args_list), input_kwargs
    input_kwargs = dict(input_kwargs)
    input_kwargs["hidden_states"] = new_hidden
    return input_args, input_kwargs
