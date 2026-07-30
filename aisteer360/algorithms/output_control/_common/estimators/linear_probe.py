"""`LinearProbe` artifact + `LinearProbeEstimator` (SASA's subspace fit).

The estimator fits a Bayes-optimal linear discriminant over pooled last-token embeddings of labeled
texts — SASA's `_setup_wv` math verbatim (class means, pooled within-class covariance, SVD-reduced
direction, normalized). The artifact mirrors `state_control/_common/steering_vector.SteeringVector`
(dataclass with `validate` / `save` / `load` / `to`), specialized to the `(direction, midpoint)`
pair a subspace-margin value consumes.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.output_control._common.specs import (
    LabeledExamples,
    as_labeled_examples,
)

logger = logging.getLogger(__name__)


@dataclass
class LinearProbe:
    """A linear discriminant in a model's last-hidden-state space.

    Attributes:
        direction: The (normalized) discriminant direction `[H]`. The margin of a hidden state `h`
            is `direction . (h - midpoint)`.
        midpoint: The midpoint between the two class means `[H]`.
    """

    direction: torch.Tensor
    midpoint: torch.Tensor

    def validate(self) -> None:
        """Validate that the probe tensors are populated and shape-compatible.

        Raises:
            ValueError: If either tensor is missing or their shapes differ.
        """
        if self.direction is None or self.midpoint is None:
            raise ValueError("direction and midpoint must be provided.")
        if self.direction.shape != self.midpoint.shape:
            raise ValueError(
                f"direction {tuple(self.direction.shape)} and midpoint "
                f"{tuple(self.midpoint.shape)} must have the same shape."
            )

    def to(self, device=None, dtype=None) -> "LinearProbe":
        """Move/cast the probe tensors in place and return self (mutating, like `Tensor.to`)."""
        self.direction = self.direction.to(device=device, dtype=dtype)
        self.midpoint = self.midpoint.to(device=device, dtype=dtype)
        return self

    def save(self, file_path: str) -> None:
        """Save the probe to a JSON file (`.probe` extension added if absent)."""
        if not file_path.endswith(".probe"):
            file_path += ".probe"
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "direction": self.direction.detach().cpu().tolist(),
            "midpoint": self.midpoint.detach().cpu().tolist(),
        }
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.debug("Saved LinearProbe to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "LinearProbe":
        """Load a probe from a JSON file (`.probe` extension added if absent)."""
        if not file_path.endswith(".probe"):
            file_path += ".probe"
        with open(file_path) as f:
            data = json.load(f)
        return cls(
            direction=torch.tensor(data["direction"], dtype=torch.float32),
            midpoint=torch.tensor(data["midpoint"], dtype=torch.float32),
        )

    @classmethod
    def from_legacy_wv(cls, wv: dict) -> "LinearProbe":
        """Adapt a legacy SASA `{'wv', 'mu_mu'}` checkpoint into a `LinearProbe`.

        Args:
            wv: A dict with keys `"wv"` (the direction) and `"mu_mu"` (the midpoint).

        Returns:
            The equivalent `LinearProbe`.
        """
        return cls(direction=wv["wv"].float(), midpoint=wv["mu_mu"].float())

    @classmethod
    def load_any(cls, file_path: str) -> "LinearProbe":
        """Load a probe from any supported checkpoint form.

        Dispatches across the three forms in the tree:

            - A `.probe` JSON file (via `load`).
            - A legacy `{'wv', 'mu_mu'}` torch checkpoint (via `from_legacy_wv`).
            - A pickled `LinearProbe` object (returned as-is).

        Args:
            file_path: Path to the checkpoint.

        Returns:
            The loaded `LinearProbe`.

        Raises:
            ValueError: If the checkpoint is not one of the supported forms.
        """
        if file_path.endswith(".probe"):
            return cls.load(file_path)
        loaded = torch.load(file_path, map_location="cpu")
        if isinstance(loaded, LinearProbe):
            return loaded
        if isinstance(loaded, dict) and "wv" in loaded and "mu_mu" in loaded:
            return cls.from_legacy_wv(loaded)
        raise ValueError(
            f"Unrecognized probe checkpoint at {file_path!r}; expected a .probe JSON file, a legacy "
            "{'wv', 'mu_mu'} torch checkpoint, or a pickled LinearProbe."
        )


class LinearProbeEstimator:
    """Fit a `LinearProbe` from labeled texts using a closed-form Bayes-optimal discriminant.

    Pooling is over the last non-pad token of each example (the only pooling SASA used).
    """

    def __init__(self, pooling: str = "last_token"):
        if pooling != "last_token":
            raise ValueError("LinearProbeEstimator supports pooling='last_token' only.")
        self.pooling = pooling

    def _pool(self, model, tokenizer, sentences, batch_size, max_length, device) -> torch.Tensor:
        """Last-non-pad-token hidden states for `sentences`, batched. Returns `[N, H]` on CPU."""
        embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch_texts = sentences[start:start + batch_size]
            batch = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            batch.pop("token_type_ids", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1]
            lengths = batch["attention_mask"].sum(-1) - 1
            pooled = last_hidden[range(len(last_hidden)), lengths]
            embeddings.append(pooled.detach().cpu())
        return torch.vstack(embeddings)

    def fit(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        data: LabeledExamples | dict,
        batch_size: int = 4,
        max_length: int = 1024,
        save_path: str | None = None,
    ) -> LinearProbe:
        """Fit the probe and return it.

        Args:
            model: The model whose hidden-state space the probe lives in.
            tokenizer: Tokenizer for encoding the labeled texts.
            data: Labeled positive/negative texts (`LabeledExamples` or a dict).
            batch_size: Forward-pass batch size for embedding extraction.
            max_length: Truncation length for embedding extraction.
            save_path: When provided, save the fitted probe to this path (no write otherwise).

        Returns:
            The fitted `LinearProbe`.
        """
        data = as_labeled_examples(data)
        device = next(model.parameters()).device

        # sort by descending length to minimize padding waste (SASA behavior)
        pos = sorted(data.positives, key=lambda z: -len(z))
        neg = sorted(data.negatives, key=lambda z: -len(z))

        x1 = self._pool(model, tokenizer, pos, batch_size, max_length, device)
        x2 = self._pool(model, tokenizer, neg, batch_size, max_length, device)
        x1 = x1[~torch.isnan(x1).any(dim=1)]
        x2 = x2[~torch.isnan(x2).any(dim=1)]

        # closed-form Bayes-optimal linear classifier
        mu_1 = torch.mean(x1, dim=0)
        cov = torch.cov(x1.T) * (x1.shape[0] - 1)
        mu_2 = torch.mean(x2, dim=0)
        cov += torch.cov(x2.T) * (x2.shape[0] - 1)
        cov = cov / (x1.shape[0] + x2.shape[0] - 2)

        F, D, _ = torch.svd(cov, some=True)
        F = F[:, D > 1e-6].float()
        D = D[D > 1e-6].float()
        D_inv = torch.diag(D ** (-1))

        mu = torch.matmul(F.t(), (mu_1 - mu_2) / 2)
        midpoint = (mu_1 + mu_2) / 2
        w_0 = torch.matmul(D_inv, mu)
        direction = torch.matmul(F, w_0)
        direction = direction / torch.norm(direction)

        probe = LinearProbe(direction=direction.float(), midpoint=midpoint.float())
        probe.validate()
        if save_path is not None:
            probe.save(save_path)
        return probe
