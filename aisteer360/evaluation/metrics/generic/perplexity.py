from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.evaluation.metrics.base import Metric


class Perplexity(Metric):
    """Compute token-level perplexity for a batch of sentences.

    Perplexity is the exponentiated mean cross-entropy between the language model’s predicted distribution and the true
    next token. Lower is better.

    Args:
        model_or_id (str | torch.nn.Module): Hugging Face model ID or an already-instantiated causal language model.
        tokenizer (transformers.PreTrainedTokenizer | None, optional):
            Tokenizer to use.  Leave ``None`` when passing a model ID to automatically load the matching tokenizer.
            Defaults to ``None``.
        batch_size (int, optional): Number of sentences per forward pass. Higher is faster until GPU memory becomes the
            bottleneck. Defaults to ``16``.
        add_bos (bool, optional): Whether to prepend the tokenizer’s BOS token so the first word in each sentence is
            also scored. Ignored if the tokenizer has no BOS token. Defaults to ``True``.
        max_length (int | None, optional): If set, truncate inputs to this length so they fit the model’s context
            window. ``None`` disables truncation. Defaults to ``None``.
        device (str | None, optional): ``"cuda"`` or ``"cpu"``. When ``None``, automatically selects GPU if available.
            Defaults to ``None``.

    Attributes:
        add_bos (bool): Whether a BOS token is prepended before scoring.
        batch_size (int): Number of sentences processed per forward pass.
        device (str): The device actually selected for computation (``"cuda"`` or ``"cpu"``).
        max_length (int | None): Truncation length for inputs, or ``None`` for no truncation.
        model (transformers.PreTrainedModel): The loaded causal language model used to score tokens.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for encoding, padding, and BOS handling.
    """

    def __init__(
        self,
        model_or_id: str | torch.nn.Module,
        tokenizer: Any | None = None,
        batch_size: int = 16,
        add_bos: bool = True,
        max_length: int | None = None,
        device: str | None = None,
    ):
        super().__init__()

        if isinstance(model_or_id, str):
            self.model = AutoModelForCausalLM.from_pretrained(model_or_id)
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id)
        else:  # model object
            self.model = model_or_id
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_or_id.config._name_or_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.batch_size = batch_size
        self.add_bos = add_bos and (self.tokenizer.bos_token_id is not None)
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
                or self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            )

    @torch.no_grad()
    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute perplexity for each response (and the mean across the batch).

        Args:
            responses (list[str]): Text sequences to score.
            prompts (list[str] | None, optional): Unused here; present for a uniform metric API.

        Returns:
            dict[str, float]: A dict with keys:

                - ``"mean_perplexity"``: mean perplexity over all inputs.
                - ``"perplexities"``: list of per-sample perplexities in input order.
        """
        perplexities: list[float] = []
        local_batch_size = self.batch_size

        for i in range(0, len(responses), local_batch_size):
            batch = responses[i : i + local_batch_size]

            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=self.max_length is not None,
                max_length=self.max_length,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            input_ids = encoding["input_ids"]

            if self.add_bos:
                bos_tokens = torch.full(
                    (input_ids.size(0), 1),
                    self.tokenizer.bos_token_id,
                    device=self.device,
                )
                input_ids = torch.cat([bos_tokens, input_ids], dim=1)

            logits = self.model(input_ids).logits[:, :-1]
            labels = input_ids[:, 1:]

            loss_per_token = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).view(labels.size())

            mask = labels.ne(self.tokenizer.pad_token_id)
            seq_loss = (loss_per_token * mask).sum(1) / mask.sum(1)

            perplexities.extend(torch.exp(seq_loss).cpu().tolist())

        return {
            "mean_perplexity": sum(perplexities) / len(perplexities),
            "perplexities": perplexities,
        }
