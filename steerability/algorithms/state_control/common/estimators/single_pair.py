"""Single-pair estimator for ActAdd steering vectors."""
import logging
from typing import TYPE_CHECKING

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.internals.capture import capture_hidden
from steerability.algorithms.core.internals.fingerprint import artifact_provenance_meta, session_artifact_identity

from ..steering_vector import SteeringVector
from .base import BaseEstimator

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession

logger = logging.getLogger(__name__)


class SinglePairEstimator(BaseEstimator[SteeringVector]):
    """Extracts per-token positional steering vectors from a single prompt pair.

    Given one positive prompt and one negative prompt, it computes the per-token
    activation difference at every layer (or a specified subset of layers) at the
    layer-input boundary, i.e. the residual stream a layer pre-hook observes,
    preserving the full positional structure of the contrast.

    The prompts are tokenized without special tokens, and the shorter prompt is right-padded
    with attended space tokens (the tokenizer's pad token when `" "` encodes to no token). Row
    `t` of the result is the difference at content token `t` of the pair. The tokenizer is not
    modified. The result is a `[T, H]` direction matrix per layer, where `T` is the token
    length of the longer prompt, and the artifact records `meta["location"] = "layer_input"`.
    """

    def fit(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase,
        *,
        positive_prompt: str,
        negative_prompt: str,
        layer_ids: list[int] | None = None,
        session: "SteeringSession | None" = None,
    ) -> SteeringVector:
        """Extract positional steering vector from a single prompt pair.

        Args:
            model: Model to extract hidden states from, or None to extract through `session`.
            tokenizer: Tokenizer for encoding the prompts.
            positive_prompt: Prompt representing the desired direction
                (e.g., "Love", "I talk about weddings constantly").
            negative_prompt: Prompt representing the opposite direction
                (e.g., "Hate", "I do not talk about weddings constantly").
            layer_ids: If provided, only compute directions for these layers.
                If None, compute for all layers.
            session: A `SteeringSession` serving hidden-state capture when no live model is
                available.

        Returns:
            SteeringVector with [T, H] directions per layer.
        """
        device = next(model.parameters()).device if model is not None else torch.device("cpu")
        if model is not None:
            model_type = getattr(model.config, "model_type", "unknown")
            session_meta: dict = {}
        else:
            model_type, session_meta = session_artifact_identity(session)

        logger.debug("Tokenizing prompt pair: positive=%r, negative=%r", positive_prompt, negative_prompt)

        # right-pad the shorter prompt with attended space tokens so the two prompts align
        # position by position; pad tokens such as eos produce different activations
        positive_ids = tokenizer.encode(positive_prompt, add_special_tokens=False, truncation=True)
        negative_ids = tokenizer.encode(negative_prompt, add_special_tokens=False, truncation=True)
        space = (tokenizer.encode(" ", add_special_tokens=False) or [tokenizer.pad_token_id])[:1]
        width = max(len(positive_ids), len(negative_ids))
        rows = [ids + space * (width - len(ids)) for ids in (positive_ids, negative_ids)]
        enc = {
            "input_ids": torch.tensor(rows, dtype=torch.long, device=device),
            "attention_mask": torch.ones(2, width, dtype=torch.long, device=device),
        }

        logger.debug("Running forward pass to extract hidden states")

        hidden, _ = capture_hidden(enc, model=model, session=session, location="layer_input")

        directions: dict[int, torch.Tensor] = {}

        num_layers = len(hidden)
        logger.debug("Computing per-token difference for %d layers", num_layers)

        for layer_idx in range(num_layers):
            if layer_ids is not None and layer_idx not in layer_ids:
                continue

            hs = hidden[layer_idx]  # [2, T, H]
            h_pos = hs[0]  # [T, H]
            h_neg = hs[1]  # [T, H]

            direction = (h_pos - h_neg).cpu().to(dtype=torch.float32)  # [T, H]

            directions[layer_idx] = direction

        logger.debug("Finished fitting single-pair directions with T=%d tokens", direction.size(0))
        meta = artifact_provenance_meta(model, tokenizer) if model is not None else session_meta
        meta["location"] = "layer_input"
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            meta=meta,
        )
