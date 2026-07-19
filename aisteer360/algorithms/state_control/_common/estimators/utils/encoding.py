"""Shared utilities for steering vector estimators."""
from typing import Callable, Literal, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control._common.specs import HiddenStateLocation
from aisteer360.utils.rendering import PromptFormat, has_chat_template, render_for_model


def tokenize_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device | str,
    *,
    add_special_tokens: bool = True,
) -> dict[str, torch.Tensor]:
    """Tokenize a flat list of texts independently.

    Unlike tokenize_pairs(), this function tokenizes texts without interleaving.
    Use this for methods like ITI where positive and negative examples are
    independent and do not need co-padding for token alignment.

    Args:
        tokenizer: Tokenizer to use.
        texts: List of text strings.
        device: Target device.
        add_special_tokens: Whether to add special tokens (e.g. BOS). Pass False
            for chat-templated text that already contains them.

    Returns:
        Dictionary with input_ids and attention_mask tensors.
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    return {k: v.to(device) for k, v in enc.items()}


def tokenize_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pos_texts: Sequence[str],
    neg_texts: Sequence[str],
    device: torch.device | str,
    *,
    add_special_tokens: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Tokenize positive/negative pairs together to ensure consistent padding.

    Interleaves pairs before tokenization so each (pos, neg) pair shares the same
    padding length. This ensures token alignment for shared prefixes, which is
    important because different padding can subtly change attention patterns.

    Args:
        tokenizer: Tokenizer to use.
        pos_texts: List of positive text strings.
        neg_texts: List of negative text strings (same length as pos_texts).
        device: Target device.
        add_special_tokens: Whether to add special tokens (e.g. BOS). Pass False
            for chat-templated text that already contains them.

    Returns:
        Tuple of (enc_pos, enc_neg) dictionaries with input_ids and attention_mask.
    """
    # interleave: [pos0, neg0, pos1, neg1, ...]
    interleaved = []
    for pos, neg in zip(pos_texts, neg_texts):
        interleaved.append(pos)
        interleaved.append(neg)

    enc = tokenizer(
        interleaved,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=add_special_tokens,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # de-interleave: even indices are positive, odd indices are negative
    enc_pos = {k: v[0::2] for k, v in enc.items()}
    enc_neg = {k: v[1::2] for k, v in enc.items()}

    return enc_pos, enc_neg


def get_last_token_positions(
    attention_mask: torch.Tensor | None,
    seq_len: int,
    num_samples: int,
) -> torch.LongTensor:
    """Find the last non-pad token position for each sample.

    Args:
        attention_mask: Shape [N, T] or None.
        seq_len: Sequence length T.
        num_samples: Number of samples N.

    Returns:
        Tensor of shape [N] with last token positions.
    """
    if attention_mask is None:
        # no padding, last token is at seq_len - 1
        return torch.full((num_samples,), seq_len - 1, dtype=torch.long)

    # for each sample, find the last position where attention_mask == 1
    # this handles both left-padded and right-padded sequences
    positions = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0).expand(num_samples, -1)
    # mask out padded positions with -1
    masked_positions = torch.where(attention_mask == 1, positions, torch.tensor(-1, device=attention_mask.device))
    return masked_positions.max(dim=1).values


def select_at_positions(
    hidden: torch.Tensor,
    positions: torch.LongTensor,
) -> torch.Tensor:
    """Select hidden states at specified positions for each sample.

    Args:
        hidden: Shape [N, T, H].
        positions: Shape [N] with position indices.

    Returns:
        Tensor of shape [N, H].
    """
    N, _, H = hidden.shape
    # gather at the specified positions
    idx = positions.view(N, 1, 1).expand(N, 1, H)
    return hidden.gather(dim=1, index=idx).squeeze(1)


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
        `l - 1` (`hidden_states[l]`). CAST's runtime condition pre-hook observes this boundary.

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


@torch.no_grad()
def measure_residual_norms(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_ids: Sequence[int] | None,
    prompts: Sequence[str],
    *,
    location: HiddenStateLocation = "layer_output",
    stat: Literal["median", "mean"] = "median",
    prompt_format: PromptFormat = "chat_prompt",
    batch_size: int = 8,
) -> dict[int, float]:
    """Typical per-token residual-stream norm at each requested layer boundary.

    The statistic is aggregated over the real (non-pad) tokens of `prompts` at the residual-stream
    boundary named by `location`. Residual norms grow with depth, so a fixed additive strength is a
    very different intervention at different layers; measuring the typical norm at each layer lets a
    caller rescale a direction to a fixed *fraction* of it (see `SteeringVector.scaled_to_norms`).

    Prompts are rendered via `render_for_model(tokenizer, prompt=p, mode=prompt_format)`. The default
    `prompt_format="chat_prompt"` renders with `add_generation_prompt=True`, byte-identical to what
    inference produces for a prompt; the rendered string is tokenized with `add_special_tokens=False`
    whenever a chat template was applied (the rendering contract), matching how the estimators consume
    contrastive text.

    Args:
        model: Model to extract hidden states from.
        tokenizer: Tokenizer for rendering and encoding the prompts.
        layer_ids: Layers to measure (0-based). `None` measures every layer (no extra cost — all
            layers are extracted regardless).
        prompts: Calibration prompts. Use fit-distribution, inference-style prompts, not a held-out
            evaluation set, so measurement never leaks eval into fitting.
        location: Residual-stream boundary each layer key maps to. `"layer_output"` (default) matches
            the toolkit-wide `VectorTrainSpec` default and controls that hook the layer output (CAA);
            `"layer_input"` matches pre-hook observers, in particular CAST's behavior application.
        stat: Aggregation over the pooled real-token norms, `"median"` (default, robust) or `"mean"`.
        prompt_format: How each prompt is rendered into model-ready text (via `render_for_model`).
        batch_size: Batch size for the extraction forward passes.

    Returns:
        Mapping from layer id to the aggregated per-token residual norm (a plain float).

    Raises:
        ValueError: If `prompts` is empty, `stat` is unsupported, or any requested layer id is out of
            range.
    """
    if len(prompts) == 0:
        raise ValueError("prompts must contain at least one prompt.")
    if stat not in ("median", "mean"):
        raise ValueError(f"stat must be 'median' or 'mean', got {stat!r}.")

    device = next(model.parameters()).device

    # render then tokenize with add_special_tokens=False when a template was applied
    rendered = [render_for_model(tokenizer, prompt=p, mode=prompt_format) for p in prompts]
    template_applied = has_chat_template(tokenizer) and prompt_format != "raw"
    enc = tokenize_texts(tokenizer, rendered, device, add_special_tokens=not template_applied)

    hidden = layerwise_tokenwise_hidden(model, enc, batch_size=batch_size, location=location)
    num_layers = len(hidden)

    if layer_ids is None:
        target_layers = sorted(hidden.keys())
    else:
        target_layers = [int(lid) for lid in layer_ids]
        for lid in target_layers:
            if not 0 <= lid < num_layers:
                raise ValueError(f"layer id {lid} out of range [0, {num_layers}).")

    # real-token mask
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        keep = attention_mask.to("cpu", torch.bool)  # [N, T]
    else:
        keep = None

    norms: dict[int, float] = {}
    for lid in target_layers:
        per_token = hidden[lid].to(torch.float32).norm(dim=-1)  # [N, T]
        values = per_token[keep] if keep is not None else per_token.reshape(-1)
        agg = values.median() if stat == "median" else values.mean()
        norms[lid] = float(agg)

    return norms
