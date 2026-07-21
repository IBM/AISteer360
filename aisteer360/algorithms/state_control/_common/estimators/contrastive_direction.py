"""Contrastive direction estimator using paired PCA."""
import logging
import math
from typing import Callable, Literal, Sequence

import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..render import render_contrastive
from ..specs import ContrastivePairs, VectorTrainSpec
from ..steering_vector import SteeringVector
from .base import BaseEstimator
from .utils import layerwise_tokenwise_hidden

logger = logging.getLogger(__name__)

PcaMethod = Literal["pca_pairwise", "pca_center"]


def _prepare_pca_samples(
    positive: torch.Tensor,
    negative: torch.Tensor,
    method: PcaMethod,
) -> torch.Tensor:
    """Build the PCA sample matrix from pooled positive/negative activations.

    - `pca_pairwise`: centers each pair `(H^+_i, H^-_i)` at its midpoint, yielding the two samples
        `±(H^+_i - H^-_i)/2`.
    - `pca_center`: stacks positive and negative activations and centers by their grand mean.

    Args:
        positive: Pooled positive activations, shape `[N, H]`.
        negative: Pooled negative activations, shape `[N, H]`.
        method: Which sample construction to use.

    Returns:
        A float32 sample matrix of shape `[2N, H]`.

    Raises:
        ValueError: If the shapes disagree, the method is unsupported, or the samples are non-finite.
    """
    if positive.shape != negative.shape:
        raise ValueError(
            "positive and negative pooled activations must have equal shape; "
            f"got {tuple(positive.shape)} and {tuple(negative.shape)}."
        )

    positive = positive.float()
    negative = negative.float()

    if method == "pca_pairwise":
        delta = positive - negative
        samples = torch.cat((0.5 * delta, -0.5 * delta), dim=0)
    elif method == "pca_center":
        stacked = torch.cat((positive, negative), dim=0)
        samples = stacked - stacked.mean(dim=0, keepdim=True)
    else:
        raise ValueError(f"Unknown PCA method: {method!r}.")

    if not torch.isfinite(samples).all():
        raise ValueError("PCA samples contain non-finite values.")
    return samples


def _orient_direction(
    direction: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    """Orient `direction` so the positive class projects above the negative class.

    Uses a majority vote over pairs, breaking ties by the sign of the mean projection margin.

    Args:
        direction: Direction to orient, shape `[H]`.
        positive: Pooled positive activations, shape `[N, H]`.
        negative: Pooled negative activations, shape `[N, H]`.

    Returns:
        The direction, flipped if positives projected below negatives.
    """
    direction = direction.float()
    positive_projection = positive.float() @ direction
    negative_projection = negative.float() @ direction

    positive_wins = (positive_projection > negative_projection).float().mean()
    if positive_wins < 0.5:
        return -direction
    if positive_wins == 0.5:
        mean_margin = (positive_projection - negative_projection).mean()
        if mean_margin < 0:
            return -direction
    return direction


def _tokenize(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device | str,
    *,
    add_special_tokens: bool = True,
) -> dict[str, torch.Tensor]:
    """Tokenize a list of texts and move to device.

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


def _select_spans(
    enc: dict[str, torch.Tensor],
    prompt_enc: dict[str, torch.Tensor] | None,
    accumulate: str,
) -> list[tuple[int, int]]:
    """Determine token spans to pool over for each sample.

    Args:
        enc: Tokenized full sequences (prompts + completions).
        prompt_enc: Tokenized prompts only (if accumulate == "suffix-only").
        accumulate: "all", "suffix-only", or "last_token".

    Returns:
        List of (start, end) tuples, one per sample.
    """
    if accumulate not in ("all", "suffix-only", "last_token"):
        raise ValueError(
            f"_select_spans does not support accumulate='{accumulate}'. "
            "Expected one of: 'all', 'suffix-only', 'last_token'."
        )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N, T = input_ids.shape

    spans = []
    for i in range(N):
        # number of prompt tokens to skip for suffix-only pooling (a count, not an absolute index)
        if accumulate == "suffix-only" and prompt_enc is not None:
            prompt_len = (
                int(prompt_enc["attention_mask"][i].sum().item())
                if "attention_mask" in prompt_enc
                else prompt_enc["input_ids"].size(1)
            )
        else:
            prompt_len = 0

        # derive both bounds from the mask so the span excludes pads on either padding side
        if attention_mask is not None:
            non_pad = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            if len(non_pad) > 0:
                first = int(non_pad[0].item())
                last = int(non_pad[-1].item())
            else:
                first, last = 0, T - 1
            if accumulate == "last_token":
                # one-token span at the final non-pad position (mask-derived, so pad-side agnostic)
                start, end = last, last + 1
            else:
                start = first + prompt_len
                end = last + 1
        else:
            if accumulate == "last_token":
                start, end = T - 1, T
            else:
                start = prompt_len
                end = T

        spans.append((start, end))

    return spans


def _pool_over_spans(
    hidden: torch.Tensor,
    spans: list[tuple[int, int]],
) -> torch.Tensor:
    """Mean-pool hidden states over specified spans.

    Args:
        hidden: Shape [N, T, H].
        spans: List of (start, end) tuples.

    Returns:
        Pooled tensor of shape [N, H].
    """
    N, T, H = hidden.shape
    pooled = []
    for i, (start, end) in enumerate(spans):
        if start >= end:
            # fallback: use last token
            pooled.append(hidden[i, -1, :])
        else:
            pooled.append(hidden[i, start:end, :].mean(dim=0))
    return torch.stack(pooled, dim=0)


class ContrastiveDirectionEstimator(BaseEstimator[SteeringVector]):
    """Learns per-layer direction vectors from contrastive text pairs via PCA.

    Two PCA variants are supported, selected by `spec.method`:

    - `pca_pairwise`: centers each pair `(H_l^+, H_l^-)` at its midpoint, giving the samples
        `±(H_l^+ - H_l^-)/2`, and takes the first principal component of that symmetric set.
    - `pca_center`: fits PCA on the union of positive and negative pooled activations centered by
        their grand mean: `vector_l = PCA(H_l^+ - mu_l, H_l^- - mu_l)` with `mu_l` the mean over all
        examples of both classes.

    For both methods the first principal component is oriented so positive examples project above
    negative examples (see `_orient_direction`).

    Examples are rendered via `render_for_model` according to `spec.prompt_format` and tokenized with
    `add_special_tokens=False` for chat-templated text.
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
        """Extract contrastive direction vectors.

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

        # tokenize
        enc_pos = _tokenize(tokenizer, rendered.pos_texts, device, add_special_tokens=rendered.add_special_tokens)
        enc_neg = _tokenize(tokenizer, rendered.neg_texts, device, add_special_tokens=rendered.add_special_tokens)

        # tokenize prompts separately if needed for suffix-only
        prompt_enc = None
        if spec.accumulate == "suffix-only" and rendered.prompt_texts is not None:
            prompt_enc = _tokenize(
                tokenizer, rendered.prompt_texts, device, add_special_tokens=rendered.add_special_tokens
            )
            prompt_enc = {k: v.cpu() for k, v in prompt_enc.items()}

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

        # move encodings to CPU for span selection
        enc_pos_cpu = {k: v.cpu() for k, v in enc_pos.items()}
        enc_neg_cpu = {k: v.cpu() for k, v in enc_neg.items()}

        # select spans
        spans_pos = _select_spans(enc_pos_cpu, prompt_enc, spec.accumulate)
        spans_neg = _select_spans(enc_neg_cpu, prompt_enc, spec.accumulate)

        # compute directions via PCA
        directions: dict[int, torch.Tensor] = {}
        explained_variances: dict[int, float] = {}

        num_layers = len(hs_pos)
        logger.debug("Computing directions for %d layers", num_layers)

        for layer_id in range(num_layers):
            # pool over spans
            Hp = _pool_over_spans(hs_pos[layer_id], spans_pos)  # [N, H]
            Hn = _pool_over_spans(hs_neg[layer_id], spans_neg)  # [N, H]

            samples = _prepare_pca_samples(Hp, Hn, spec.method)  # [2N, H]

            pca = PCA(n_components=1)
            pca.fit(samples.numpy())
            direction = torch.from_numpy(pca.components_[0]).float()  # [H]
            variance = float(pca.explained_variance_ratio_[0])

            direction = _orient_direction(direction, Hp, Hn)
            if not torch.isfinite(direction).all():
                raise ValueError(f"Non-finite direction produced for layer {layer_id}.")

            directions[layer_id] = direction.unsqueeze(0)  # [1, H]
            explained_variances[layer_id] = variance

        logger.debug("Finished fitting contrastive directions")
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            explained_variances=explained_variances,
        )
