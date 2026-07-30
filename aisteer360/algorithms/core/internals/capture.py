"""Hooked/forward capture of hidden states at module boundaries."""
from typing import Callable, Literal

import torch
from transformers import PreTrainedModel

HiddenStateLocation = Literal["layer_output", "layer_input"]


@torch.no_grad()
def layerwise_tokenwise_hidden(
    model: PreTrainedModel,
    enc: dict[str, torch.Tensor],
    batch_size: int = 8,
    on_batch: Callable[[], None] | None = None,
    *,
    location: HiddenStateLocation = "layer_output",
) -> dict[int, torch.Tensor]:
    """Extract per-layer hidden states for all tokens.

    `outputs.hidden_states` is a tuple of `num_layers + 1` tensors: index 0 is the embedding output
    (the input to layer 0) and index `i` is the output of layer `i - 1`.

    - `location="layer_output"`: key `l` maps to the output of layer `l` (`hidden_states[l + 1]`).
    - `location="layer_input"`: key `l` maps to the input of layer `l`, i.e. the output of layer
        `l - 1` (`hidden_states[l]`), the boundary a layer pre-hook observes.

    Args:
        model: The model to extract from.
        enc: Tokenized input with input_ids and attention_mask.
        batch_size: Batch size for forward passes.
        on_batch: Optional callable invoked after each batch finishes. Used by callers to surface
            progress to the UI.
        location: Which residual-stream boundary each layer key maps to.

    Returns:
        Dict mapping layer_id (`0 .. num_layers - 1`) to tensor of shape [N, T, H].

    Raises:
        ValueError: If `location` is unsupported or the number of mapped states does not equal the
            model's layer count.
    """
    if location not in ("layer_output", "layer_input"):
        raise ValueError(f"Unsupported hidden-state location: {location!r}.")

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N = input_ids.size(0)

    # collect states per layer
    all_hidden: dict[int, list[torch.Tensor]] = {}
    num_layers: int | None = None

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_ids = input_ids[start:end]
        batch_mask = attention_mask[start:end] if attention_mask is not None else None

        outputs = model(
            input_ids=batch_ids,
            attention_mask=batch_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

        num_layers = len(outputs.hidden_states) - 1
        layer_states = outputs.hidden_states[1:] if location == "layer_output" else outputs.hidden_states[:-1]
        for layer_idx, hs in enumerate(layer_states):
            all_hidden.setdefault(layer_idx, []).append(hs.cpu())

        if on_batch is not None:
            on_batch()

    result = {layer_idx: torch.cat(tensors, dim=0) for layer_idx, tensors in all_hidden.items()}

    if num_layers is not None and len(result) != num_layers:
        raise ValueError(
            f"Expected {num_layers} mapped hidden states for location={location!r}, got {len(result)}."
        )

    return result
