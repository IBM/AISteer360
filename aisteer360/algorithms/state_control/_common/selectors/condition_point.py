"""Condition point search: find optimal (layer, threshold, comparator)."""
import logging
import warnings
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..gates.utils.scores import projected_cosine_similarity_tensor, rank_one_projector
from ..render import render_contrastive
from ..specs import Comparator, CompMode, ConditionSearchSpec, ContrastivePairs, VectorTrainSpec
from .base import BaseSelector
from ..estimators.contrastive_direction import (
    _pool_over_spans,
    _select_spans,
    _tokenize,
)
from ..estimators.utils import layerwise_tokenwise_hidden

logger = logging.getLogger(__name__)


@dataclass
class ConditionPoint:
    """Result of a condition point search, reusable as a self-contained CAST condition argument.

    A `ConditionPoint` carries everything CAST needs to gate without re-searching: the layer,
    threshold, comparator, and (when populated by the selector) the runtime aggregation mode. Pass
    one directly to `CASTArgs.condition_point` to reuse a searched point, or `flipped()` to condition
    on the complement.

    Attributes:
        layer_id: The condition layer (0-based).
        threshold: The gate threshold.
        comparator: The canonical gate comparator ("larger" opens when score >= threshold; "smaller"
            when score <= threshold).
        f1: F1 of the class separation achieved by the search at this point.
        margin: Geometric margin: distance from the decision boundary to the nearest calibration
            point. Positive means the calibration classes are cleanly separated at this threshold.
        comparison_mode: The runtime token-aggregation mode ("mean" or "last") the search was run
            for, or None when unset. Populated by `ConditionPointSelector.select`.
    """

    layer_id: int
    threshold: float
    comparator: Comparator
    f1: float
    margin: float = 0.0
    comparison_mode: CompMode | None = None

    def flipped(self) -> "ConditionPoint":
        """Return a copy with the comparator inverted ("larger" <-> "smaller").

        Conditions on the complement of this point (e.g. gate on scores at or below the threshold
        instead of at or above it). All other fields are preserved: `f1` and `margin` describe the
        original search and are carried over unchanged.
        """
        flipped_comparator: Comparator = "smaller" if self.comparator == "larger" else "larger"
        return ConditionPoint(
            layer_id=self.layer_id,
            threshold=self.threshold,
            comparator=flipped_comparator,
            f1=self.f1,
            margin=self.margin,
            comparison_mode=self.comparison_mode,
        )


def _threshold_grid(threshold_range: tuple[float, float], step: float) -> torch.Tensor:
    """Build a half-open, step-exact threshold grid `[low, high)`.

    Args:
        threshold_range: `(low, high)` bounds.
        step: Grid increment (must be positive).

    Returns:
        1-D tensor of candidate thresholds; `[low]` when the range admits no full step.
    """
    low, high = threshold_range
    grid = torch.arange(low, high, step, dtype=torch.float64)
    if grid.numel() == 0:
        grid = torch.tensor([low], dtype=torch.float64)
    return grid


def _best_point_for_layer(
    sims_p: torch.Tensor,
    sims_n: torch.Tensor,
    grid: torch.Tensor,
) -> dict:
    """Best (f1, margin, threshold, comparator) for one layer's calibration scores.

    F1 saturates at 1.0 across many (threshold, comparator) points when the calibration set is
    small -- 10 pairs against a 100-point grid and 2 comparators leaves thousands of candidates,
    so a perfect fit carries little evidence. Ties are therefore broken by the geometric margin:
    the distance from the decision boundary to the nearest calibration point. Maximising it
    centres the threshold in the widest gap and prefers layers whose classes are genuinely far
    apart over layers that separate by luck.

    Args:
        sims_p: [N_pos] condition scores for positives.
        sims_n: [N_neg] condition scores for negatives.
        grid: Candidate thresholds.

    Returns:
        dict with keys "f1", "margin", "thr", "comparator".
    """
    best = {"f1": -1.0, "margin": float("-inf"), "thr": 0.0, "comparator": "larger"}

    for cmp in ("larger", "smaller"):
        for thr in grid:
            thr_f = float(thr)

            if cmp == "larger":  # gate opens when score >= threshold
                tp = int((sims_p >= thr_f).sum().item())
                fp = int((sims_n >= thr_f).sum().item())
                margin = min(
                    float((sims_p - thr_f).min().item()),
                    float((thr_f - sims_n).min().item()),
                )
            else:  # gate opens when score <= threshold
                tp = int((sims_p <= thr_f).sum().item())
                fp = int((sims_n <= thr_f).sum().item())
                margin = min(
                    float((thr_f - sims_p).min().item()),
                    float((sims_n - thr_f).min().item()),
                )

            fn = int(sims_p.numel()) - tp
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 0.0 if prec + rec < 1e-8 else 2 * prec * rec / (prec + rec)

            if (f1, margin) > (best["f1"], best["margin"]):
                best = {"f1": f1, "margin": margin, "thr": thr_f, "comparator": cmp}

    return best


class ConditionPointSelector(BaseSelector[ConditionPoint]):
    """Grid-searches for the (layer, threshold, comparator) that best
    separates positive from negative examples using projected cosine
    similarity.

    Examples are rendered via `render_for_model` according to
    `fit_spec.prompt_format` and tokenized with `add_special_tokens=False` for
    chat-templated text (matching the inference rendering of the condition gate).

    The returned `comparator` is always one of the canonical values "larger"/"smaller" (this
    toolkit's semantics: "larger" opens the gate when score >= threshold), consumed directly by
    `MultiKeyThresholdGate` with no normalization needed. These are NOT the reference
    implementation's semantics; see `normalize_comparator` and `MultiKeyThresholdGate`.
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
        comparison_mode: CompMode | None = None,
    ) -> ConditionPoint:
        """Run the grid search.

        Calibration extracts hidden states at `location="layer_input"`, matching the residual-stream
        boundary CAST's runtime condition pre-hook observes, and scores with the shared
        `projected_cosine_similarity_tensor` so selector and runtime scores agree for the same input.
        A `fit_spec` whose `location` is not `"layer_input"` means the condition direction was fit
        at a different boundary than it is calibrated and scored against, and triggers a
        `UserWarning`.

        Args:
            model: Model for hidden state extraction.
            tokenizer: Tokenizer for encoding.
            condition_directions: Per-layer direction tensors (from estimator).
            data: Contrastive pairs to evaluate separation on.
            fit_spec: How hidden states were accumulated (for span selection).
            search_spec: Search grid configuration.
            comparison_mode: The runtime condition aggregation mode CAST will use. Accepted for
                caller symmetry with CAST; calibration pools over `fit_spec.accumulate` spans.

        Returns:
            ConditionPoint with the best (layer, threshold, comparator, f1).
        """
        device = next(model.parameters()).device

        if fit_spec.location != "layer_input":
            warnings.warn(
                "ConditionPointSelector calibrates on layer-input hidden states (the boundary CAST's "
                f"runtime condition pre-hook observes), but fit_spec.location={fit_spec.location!r}. "
                "The condition direction was fit at a different residual-stream boundary than it is "
                "calibrated and scored against. Fit the condition vector with location='layer_input' "
                "to align fit, calibration, and runtime scoring.",
                UserWarning,
            )

        # render full texts according to prompt_format (shared with inference)
        rendered = render_contrastive(tokenizer, data, fit_spec.prompt_format)

        # tokenize
        enc_pos = _tokenize(tokenizer, rendered.pos_texts, device, add_special_tokens=rendered.add_special_tokens)
        enc_neg = _tokenize(tokenizer, rendered.neg_texts, device, add_special_tokens=rendered.add_special_tokens)

        # extract hidden states at the layer-input boundary the runtime pre-hook observes
        hs_pos = layerwise_tokenwise_hidden(model, enc_pos, batch_size=fit_spec.batch_size, location="layer_input")
        hs_neg = layerwise_tokenwise_hidden(model, enc_neg, batch_size=fit_spec.batch_size, location="layer_input")

        # move encodings to CPU for span selection
        enc_pos_cpu = {k: v.cpu() for k, v in enc_pos.items()}
        enc_neg_cpu = {k: v.cpu() for k, v in enc_neg.items()}

        # tokenize prompts separately if needed
        prompt_enc = None
        if fit_spec.accumulate == "suffix-only" and rendered.prompt_texts is not None:
            prompt_enc = _tokenize(
                tokenizer, rendered.prompt_texts, device, add_special_tokens=rendered.add_special_tokens
            )
            prompt_enc = {k: v.cpu() for k, v in prompt_enc.items()}

        spans_pos = _select_spans(enc_pos_cpu, prompt_enc, fit_spec.accumulate)
        spans_neg = _select_spans(enc_neg_cpu, prompt_enc, fit_spec.accumulate)

        # determine layers to search (0-based, matching runtime condition layer ids)
        if search_spec.candidate_layers is not None:
            layers = list(search_spec.candidate_layers)
        else:
            start, end = search_spec.layer_range or (0, len(hs_pos))
            layers = list(range(start, min(end, len(hs_pos))))

        grid = _threshold_grid(search_spec.threshold_range, search_spec.threshold_step)

        best = {"f1": -1.0, "margin": float("-inf"), "layer": 0, "thr": 0.0, "direction": "larger"}

        logger.debug("Searching %d layers with %d threshold values", len(layers), len(grid))

        for lid in layers:
            if lid not in condition_directions:
                continue

            Hp = _pool_over_spans(hs_pos[lid], spans_pos)
            Hn = _pool_over_spans(hs_neg[lid], spans_neg)
            c = condition_directions[lid].to(device=Hp.device, dtype=Hp.dtype)
            # squeeze [K, D] → [D] for K=1 (unified SteeringVector format)
            if c.ndim == 2 and c.shape[0] == 1:
                c = c.squeeze(0)

            projector = rank_one_projector(c)
            sims_p = projected_cosine_similarity_tensor(Hp, projector)
            sims_n = projected_cosine_similarity_tensor(Hn, projector)

            cand = _best_point_for_layer(sims_p, sims_n, grid)
            if (cand["f1"], cand["margin"]) > (best["f1"], best["margin"]):
                best.update(
                    f1=cand["f1"],
                    margin=cand["margin"],
                    layer=lid,
                    thr=cand["thr"],
                    direction=cand["comparator"],
                )

        logger.debug(
            "Best condition point: layer=%d, threshold=%.3f, comparator=%s, f1=%.3f, margin=%.4f",
            best["layer"], best["thr"], best["direction"], best["f1"], best["margin"],
        )

        if fit_spec.method == "mean_diff" and best["direction"] == "smaller":
            warnings.warn(
                f"Condition search selected comparator 'smaller' at layer {best['layer']} "
                f"(margin {best['margin']:.4f}). The direction was fit as mean(positives) - "
                "mean(negatives), so positives are expected to score HIGHER. An inverted "
                "comparator usually means the calibration set is too small or this layer carries "
                "little signal for the condition. Consider a larger calibration set, a narrower "
                "layer_range, or pinning condition_layer_ids/condition_vector_threshold.",
                UserWarning,
                stacklevel=2,
            )

        return ConditionPoint(
            layer_id=best["layer"],
            threshold=best["thr"],
            comparator=best["direction"],
            f1=best["f1"],
            margin=best["margin"],
            comparison_mode=comparison_mode,
        )
