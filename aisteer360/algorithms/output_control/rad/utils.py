"""RAD helpers: the legacy GPT-2 reward model.

`GPT2RewardModel` is the original RAD toxicity reward head (a GPT-2 backbone with the LM head
replaced by a linear classification head). `RAD.steer()` loads it when `reward_path` is used.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2LMHeadModel


class GPT2RewardModel(nn.Module):
    """GPT-2 based reward model for scoring text toxicity or other attributes.

    Modified GPT-2 architecture where the language modeling head is replaced with a classification
    head. Used to score text sequences for desired attributes during RAD-guided generation.

    Args:
        reward_model_name (str): Base GPT-2 model variant to use. Defaults to "gpt2".
        out_features (int): Number of output classes/attributes. Defaults to 1.
        cache_dir (str): Cache directory for the base GPT-2 weights.
    """

    def __init__(self, reward_model_name="gpt2", out_features=1, cache_dir="./"):
        super().__init__()
        model = GPT2LMHeadModel.from_pretrained(reward_model_name, cache_dir=cache_dir)
        model.lm_head = nn.Linear(in_features=model.lm_head.in_features, out_features=out_features, bias=True)
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: tuple[torch.FloatTensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
    ):
        """Forward pass; returns classification scores at each sequence's last valid token.

        Returns:
            torch.Tensor: Classification scores of shape `[batch_size, out_features]`, extracted from
                the last non-padding position of each sequence.
        """
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        logits = outputs["logits"]
        sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(logits.device)
        scores = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]
        return scores
