"""`RewardModelScorer` — score `prompt + continuation` with an HF sequence classifier."""
from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class RewardModelScorer:
    """Score each continuation with an HF sequence classifier over `prompt + continuation`.

    The scored texts are truncated to the tokenizer's `max_length` attribute when it has one (the
    sequence-classifier loader sets it from the model's limit), and to the tokenizer's
    `model_max_length` otherwise.

    Args:
        model: The reward model (HF sequence classifier, in eval mode).
        tokenizer: Tokenizer for the reward model.
        score_index: Logit index to read as the scalar reward.
        batch_size: Batch size for scoring.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        score_index: int = 0,
        batch_size: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.score_index = score_index
        self.batch_size = batch_size
        self._device = next(model.parameters()).device

    @torch.inference_mode()
    def __call__(self, prompt: str, continuations: list[str], params: dict) -> list[float]:
        texts = [f"{prompt}{continuation}" for continuation in continuations]
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=getattr(self.tokenizer, "max_length", None),
            ).to(self._device)
            logits = self.model(**inputs).logits
            scores.extend(logits[:, self.score_index].detach().cpu().tolist())
        return scores
