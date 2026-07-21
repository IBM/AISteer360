"""Mean difference estimator for CAA steering vectors."""
import logging
import math
from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control._common.estimators.base import BaseEstimator
from aisteer360.algorithms.state_control._common.estimators.utils import (
    get_last_token_positions,
    layerwise_tokenwise_hidden,
    select_at_positions,
    tokenize_pairs,
)
from aisteer360.algorithms.state_control._common.gates.utils.scores import masked_mean as _masked_mean
from aisteer360.algorithms.state_control._common.render import render_contrastive
from aisteer360.algorithms.state_control._common.specs import ContrastivePairs, VectorTrainSpec
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector

logger = logging.getLogger(__name__)


class MeanDifferenceEstimator(BaseEstimator[SteeringVector]):
    """Learns per-layer steering vectors using the mean difference method.

    For each layer, computes:
        v_L = mean(a_L(positive) - a_L(negative))

    where activations are extracted at the last non-pad token position
    of each example (the answer letter in the CAA prompt format).

    This differs from ContrastiveDirectionEstimator which uses PCA on the
    pairwise differences. Mean difference takes the centroid directly, while
    PCA finds the direction of maximum variance. They converge when difference
    vectors are nearly collinear.

    Examples are rendered via `render_for_model` according to `spec.prompt_format`
    and tokenized with `add_special_tokens=False` for chat-templated text.
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> SteeringVector:
        """Extract steering vectors using mean difference.

        Args:
            model: Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data: The positive/negative text pairs.
            spec: Training configuration (method, accumulate, batch_size).
            on_progress: Optional `(completed, total)` callback fired as each forward-pass batch
                finishes. `total` covers both positive and negative passes.

        Returns:
            SteeringVector with one direction per layer.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")

        # render full texts according to prompt_format (shared with inference)
        rendered = render_contrastive(tokenizer, data, spec.prompt_format)

        logger.debug(
            "Tokenizing %d positive and %d negative examples", len(rendered.pos_texts), len(rendered.neg_texts)
        )

        # tokenize pairs together to ensure consistent padding and token alignment
        enc_pos, enc_neg = tokenize_pairs(
            tokenizer, rendered.pos_texts, rendered.neg_texts, device, add_special_tokens=rendered.add_special_tokens
        )

        # extract hidden states
        logger.debug("Extracting hidden states with batch_size=%d", spec.batch_size)
        n_pos = enc_pos["input_ids"].size(0)
        n_neg = enc_neg["input_ids"].size(0)
        total_batches = math.ceil(n_pos / spec.batch_size) + math.ceil(n_neg / spec.batch_size)
        completed = {"n": 0}

        def _tick() -> None:
            completed["n"] += 1
            if on_progress is not None:
                on_progress(completed["n"], total_batches)

        if on_progress is not None:
            on_progress(0, total_batches)
        hs_pos = layerwise_tokenwise_hidden(
            model, enc_pos, batch_size=spec.batch_size, on_batch=_tick, location=spec.location
        )
        hs_neg = layerwise_tokenwise_hidden(
            model, enc_neg, batch_size=spec.batch_size, on_batch=_tick, location=spec.location
        )

        num_samples = len(rendered.pos_texts)
        num_layers = len(hs_pos)
        logger.debug("Computing mean difference directions for %d layers", num_layers)

        # determine how to aggregate hidden states based on accumulate mode
        directions: dict[int, torch.Tensor] = {}

        # get attention masks for position selection
        attn_pos = enc_pos.get("attention_mask")
        attn_neg = enc_neg.get("attention_mask")
        if attn_pos is not None:
            attn_pos = attn_pos.cpu()
        if attn_neg is not None:
            attn_neg = attn_neg.cpu()

        for layer_id in range(num_layers):
            hp = hs_pos[layer_id]  # [N, T, H]
            hn = hs_neg[layer_id]  # [N, T, H]

            if spec.accumulate == "last_token":
                # select activation at last non-pad token
                pos_positions = get_last_token_positions(attn_pos, hp.size(1), num_samples)
                neg_positions = get_last_token_positions(attn_neg, hn.size(1), num_samples)
                hp_agg = select_at_positions(hp, pos_positions)  # [N, H]
                hn_agg = select_at_positions(hn, neg_positions)  # [N, H]
            elif spec.accumulate == "all":
                # mean pool over real tokens only; pooling pad positions would bias the direction
                hp_agg = _masked_mean(hp, attn_pos)  # [N, H]
                hn_agg = _masked_mean(hn, attn_neg)  # [N, H]
            else:
                raise ValueError(f"MeanDifferenceEstimator does not support accumulate='{spec.accumulate}'")

            # compute mean difference: v = mean(h_pos - h_neg)
            diffs = hp_agg - hn_agg  # [N, H]
            direction = diffs.mean(dim=0)  # [H]

            directions[layer_id] = direction.unsqueeze(0).to(dtype=torch.float32)  # [1, H]

        logger.debug("Finished fitting mean difference directions")
        return SteeringVector(
            model_type=model_type,
            directions=directions,
        )
