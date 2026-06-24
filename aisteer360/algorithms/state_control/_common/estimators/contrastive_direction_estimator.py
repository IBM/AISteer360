"""Contrastive direction estimator using paired PCA."""
import logging
from typing import Sequence

import torch
from sklearn.decomposition import PCA
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..specs import ContrastivePairs, VectorTrainSpec
from ..steering_vector import SteeringVector
from .base import BaseEstimator

logger = logging.getLogger(__name__)


def _tokenize(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Tokenize a list of texts and move to device.

    Args:
        tokenizer: Tokenizer to use.
        texts: List of text strings.
        device: Target device.

    Returns:
        Dictionary with input_ids and attention_mask tensors.
    """
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def _layerwise_tokenwise_hidden(
    model: PreTrainedModel,
    enc: dict[str, torch.Tensor],
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """Extract hidden states from all layers for all tokens.

    Args:
        model: The model to extract from.
        enc: Tokenized input with input_ids and attention_mask.
        batch_size: Batch size for forward passes.

    Returns:
        Dict mapping layer_id to tensor of shape [N, T, H] where N is total samples.
    """
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N = input_ids.size(0)

    # collect outputs per layer
    all_hidden: dict[int, list[torch.Tensor]] = {}

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

        # outputs.hidden_states is a tuple of (num_layers+1) tensors
        # index 0 is embedding output, indices 1..num_layers are layer outputs
        for layer_idx, hs in enumerate(outputs.hidden_states[1:]):
            if layer_idx not in all_hidden:
                all_hidden[layer_idx] = []
            all_hidden[layer_idx].append(hs.cpu())

    # concatenate all batches
    result = {}
    for layer_idx, tensors in all_hidden.items():
        result[layer_idx] = torch.cat(tensors, dim=0)

    return result


def _select_spans(
    enc: dict[str, torch.Tensor],
    prompt_enc: dict[str, torch.Tensor] | None,
    accumulate: str,
) -> list[tuple[int, int]]:
    """Determine token spans to pool over for each sample.

    Args:
        enc: Tokenized full sequences (prompts + completions).
        prompt_enc: Tokenized prompts only (if accumulate == "suffix-only").
        accumulate: "all" or "suffix-only".

    Returns:
        List of (start, end) tuples, one per sample.
    """
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    N, T = input_ids.shape

    spans = []
    for i in range(N):
        # find actual sequence length (non-padded)
        if attention_mask is not None:
            seq_len = int(attention_mask[i].sum().item())
        else:
            seq_len = T

        if accumulate == "suffix-only" and prompt_enc is not None:
            # start after the prompt
            prompt_len = int(prompt_enc["attention_mask"][i].sum().item()) if "attention_mask" in prompt_enc else prompt_enc["input_ids"].size(1)
            start = prompt_len
        else:
            start = 0

        # end is the actual sequence length (left-padded sequences: end at seq_len)
        # for left-padded, positions 0..T-seq_len are padding, so we want T-seq_len..T
        if attention_mask is not None:
            # find first non-pad position
            first_non_pad = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            if len(first_non_pad) > 0:
                first_pos = int(first_non_pad[0].item())
            else:
                first_pos = 0
            end = T
            # adjust start for left-padding
            start = max(start + first_pos, first_pos)
        else:
            end = seq_len

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
    """Learns per-layer direction vectors from contrastive text pairs.

    Uses PCA on the difference of mean hidden states between positive and
    negative examples to extract the principal direction of variation at each
    layer.
    """

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: ContrastivePairs,
        spec: VectorTrainSpec,
    ) -> SteeringVector:
        """Extract contrastive direction vectors.

        Args:
            model: Model to extract hidden states from.
            tokenizer: Tokenizer for encoding the contrastive pairs.
            data: The positive/negative text pairs.
            spec: Training configuration (method, accumulate, batch_size).

        Returns:
            SteeringVector with one direction per layer.
        """
        device = next(model.parameters()).device
        model_type = getattr(model.config, "model_type", "unknown")

        # build full texts
        if data.prompts is not None:
            pos_texts = [p + c for p, c in zip(data.prompts, data.positives)]
            neg_texts = [p + c for p, c in zip(data.prompts, data.negatives)]
        else:
            pos_texts = list(data.positives)
            neg_texts = list(data.negatives)

        logger.debug("Tokenizing %d positive and %d negative examples", len(pos_texts), len(neg_texts))

        # tokenize
        enc_pos = _tokenize(tokenizer, pos_texts, device)
        enc_neg = _tokenize(tokenizer, neg_texts, device)

        # tokenize prompts separately if needed for suffix-only
        prompt_enc = None
        if spec.accumulate == "suffix-only" and data.prompts is not None:
            prompt_enc = _tokenize(tokenizer, list(data.prompts), device)
            prompt_enc = {k: v.cpu() for k, v in prompt_enc.items()}

        # extract hidden states
        logger.debug("Extracting hidden states with batch_size=%d", spec.batch_size)
        hs_pos = _layerwise_tokenwise_hidden(model, enc_pos, batch_size=spec.batch_size)
        hs_neg = _layerwise_tokenwise_hidden(model, enc_neg, batch_size=spec.batch_size)

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

            # compute pairwise differences
            diffs = (Hp - Hn).numpy()  # [N, H]

            if spec.method == "pca_pairwise":
                # fit PCA to get principal direction
                pca = PCA(n_components=1)
                pca.fit(diffs)
                direction = pca.components_[0]  # shape [H]
                variance = float(pca.explained_variance_ratio_[0])
            else:
                raise ValueError(f"Unknown method: {spec.method}")

            directions[layer_id] = torch.tensor(direction, dtype=torch.float32).unsqueeze(0)  # [1, H]
            explained_variances[layer_id] = variance

        logger.debug("Finished fitting contrastive directions")
        return SteeringVector(
            model_type=model_type,
            directions=directions,
            explained_variances=explained_variances,
        )
