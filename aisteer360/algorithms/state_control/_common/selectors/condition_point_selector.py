"""Condition point search: find optimal (layer, threshold, comparator)."""
import logging
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..specs import Comparator, ConditionSearchSpec, ContrastivePairs, VectorTrainSpec
from .base import BaseSelector
from ..estimators.contrastive_direction_estimator import (
    _layerwise_tokenwise_hidden,
    _pool_over_spans,
    _select_spans,
    _tokenize,
)

logger = logging.getLogger(__name__)


@dataclass
class ConditionPoint:
    """Result of a condition point search."""

    layer_id: int
    threshold: float
    comparator: Comparator
    f1: float


@torch.no_grad()
def _proj_sim(h: torch.Tensor, c: torch.Tensor) -> float:
    """Compute projected cosine similarity.

    Args:
        h: Hidden state vector of shape [H].
        c: Condition direction vector of shape [H].

    Returns:
        Cosine similarity as a float.
    """
    P = torch.outer(c, c) / (c @ c + 1e-8)
    proj = torch.tanh(P @ h)
    return float((h @ proj) / (h.norm() * proj.norm() + 1e-8))


class ConditionPointSelector(BaseSelector[ConditionPoint]):
    """Grid-searches for the (layer, threshold, comparator) that best
    separates positive from negative examples using projected cosine
    similarity.
    """

    def select(
        self,
        *,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        condition_directions: dict[int, torch.Tensor],
        data: ContrastivePairs,
        fit_spec: VectorTrainSpec,
        search_spec: ConditionSearchSpec,
    ) -> ConditionPoint:
        """Run the grid search.

        Args:
            model: Model for hidden state extraction.
            tokenizer: Tokenizer for encoding.
            condition_directions: Per-layer direction tensors (from estimator).
            data: Contrastive pairs to evaluate separation on.
            fit_spec: How hidden states were accumulated (for span selection).
            search_spec: Search grid configuration.

        Returns:
            ConditionPoint with the best (layer, threshold, comparator, f1).
        """
        device = next(model.parameters()).device

        # build full texts
        if data.prompts is not None:
            pos_texts = [p + c for p, c in zip(data.prompts, data.positives)]
            neg_texts = [p + c for p, c in zip(data.prompts, data.negatives)]
        else:
            pos_texts = list(data.positives)
            neg_texts = list(data.negatives)

        # tokenize
        enc_pos = _tokenize(tokenizer, pos_texts, device)
        enc_neg = _tokenize(tokenizer, neg_texts, device)

        # extract hidden states
        hs_pos = _layerwise_tokenwise_hidden(model, enc_pos, batch_size=fit_spec.batch_size)
        hs_neg = _layerwise_tokenwise_hidden(model, enc_neg, batch_size=fit_spec.batch_size)

        # move encodings to CPU for span selection
        enc_pos_cpu = {k: v.cpu() for k, v in enc_pos.items()}
        enc_neg_cpu = {k: v.cpu() for k, v in enc_neg.items()}

        # tokenize prompts separately if needed
        prompt_enc = None
        if fit_spec.accumulate == "suffix-only" and data.prompts is not None:
            prompt_enc = _tokenize(tokenizer, list(data.prompts), device)
            prompt_enc = {k: v.cpu() for k, v in prompt_enc.items()}

        spans_pos = _select_spans(enc_pos_cpu, prompt_enc, fit_spec.accumulate)
        spans_neg = _select_spans(enc_neg_cpu, prompt_enc, fit_spec.accumulate)

        # determine layers to search
        if search_spec.candidate_layers is not None:
            layers = list(search_spec.candidate_layers)
        else:
            start, end = search_spec.layer_range or (1, len(hs_pos))
            layers = list(range(start, min(end, len(hs_pos))))

        # build threshold grid
        thr_min, thr_max = search_spec.threshold_range
        steps = max(1, int((thr_max - thr_min) / search_spec.threshold_step))
        if steps > 1:
            grid = torch.linspace(thr_min, thr_max, steps=steps)
        else:
            grid = torch.tensor([thr_min])

        best = {"f1": -1.0, "layer": 0, "thr": 0.0, "direction": "larger"}

        logger.debug("Searching %d layers with %d threshold values", len(layers), len(grid))

        for lid in layers:
            if lid not in condition_directions:
                continue

            Hp = _pool_over_spans(hs_pos[lid], spans_pos)
            Hn = _pool_over_spans(hs_neg[lid], spans_neg)
            c = condition_directions[lid].to(Hp.dtype)
            # squeeze [K, D] → [D] for K=1 (unified SteeringVector format)
            if c.ndim == 2 and c.shape[0] == 1:
                c = c.squeeze(0)

            # compute similarities
            sims_p = torch.tensor([_proj_sim(h, c) for h in Hp], dtype=torch.float32)
            sims_n = torch.tensor([_proj_sim(h, c) for h in Hn], dtype=torch.float32)

            for cmp in ("larger", "smaller"):
                for thr in grid:
                    if cmp == "larger":
                        yhat_p = sims_p >= thr
                        yhat_n = sims_n >= thr
                    else:
                        yhat_p = sims_p <= thr
                        yhat_n = sims_n <= thr

                    tp = int(yhat_p.sum().item())
                    fp = int(yhat_n.sum().item())
                    fn = len(sims_p) - tp

                    prec = tp / (tp + fp + 1e-8)
                    rec = tp / (tp + fn + 1e-8)
                    f1 = 0.0 if prec + rec < 1e-8 else 2 * prec * rec / (prec + rec)

                    if f1 > best["f1"]:
                        best.update(f1=f1, layer=lid, thr=float(thr), direction=cmp)

        logger.debug(
            "Best condition point: layer=%d, threshold=%.3f, comparator=%s, f1=%.3f",
            best["layer"],
            best["thr"],
            best["direction"],
            best["f1"],
        )

        return ConditionPoint(
            layer_id=best["layer"],
            threshold=best["thr"],
            comparator=best["direction"],
            f1=best["f1"],
        )
