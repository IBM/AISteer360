"""Utilities for hook registration and model inspection."""
import torch
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


def extract_hidden_states(input_args: tuple, input_kwargs: dict) -> torch.Tensor | None:
    """Extract hidden_states tensor from a pre-hook's arguments.

    HuggingFace transformer layers receive hidden_states either as the
    first positional argument or as a keyword argument.

    Args:
        input_args: Positional args from the pre-hook.
        input_kwargs: Keyword args from the pre-hook.

    Returns:
        The hidden_states tensor, or None if not found.
    """
    if input_args:
        return input_args[0]
    return input_kwargs.get("hidden_states")


def replace_hidden_states(
    input_args: tuple,
    input_kwargs: dict,
    new_hidden: torch.Tensor,
) -> tuple[tuple, dict]:
    """Return modified (input_args, input_kwargs) with hidden_states replaced.

    Args:
        input_args: Original positional args.
        input_kwargs: Original keyword args.
        new_hidden: Replacement hidden states tensor.

    Returns:
        Tuple of (new_input_args, new_input_kwargs).
    """
    if input_args:
        return (new_hidden, *input_args[1:]), input_kwargs
    input_kwargs = dict(input_kwargs)
    input_kwargs["hidden_states"] = new_hidden
    return input_args, input_kwargs
