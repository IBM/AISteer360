from __future__ import annotations

from typing import Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.thinking_intervention.args import (
    ThinkingInterventionArgs,
)


class ThinkingIntervention(OutputControl):
    """
    Implementation of Thinking Intervention from Wu et al., 2025.

    `ThinkingIntervention` enables controlled text generation by injecting structured thinking processes into the model's
    reasoning chain. The method modifies the input prompt to include explicit thinking steps enclosed in special tags,
    allowing the model to engage in guided reasoning before producing the final output.

    The algorithm works in three phases:

    1. **Prompt Modification**: Transform the original prompt by applying an intervention function that injects thinking
    instructions, reasoning templates, or structured prompts to guide the model's internal reasoning process.

    2. **Guided Generation**: Generate text using the modified prompt, where the model first produces thinking content
    within special tags (e.g., <think>...</think>) before generating the actual response.

    3. **Output Extraction**: Parse the generated text to extract only the content after the thinking tags.

    Args:
        intervention (Callable[[str, dict], str]): Function that modifies the input prompt to include thinking
            instructions. Takes the original prompt string and parameter dict, returns the modified prompt string.

    Reference:
        "Effectively Controlling Reasoning Models through Thinking Intervention"
        Tong Wu, Chong Xiang, Jiachen T. Wang, G. Edward Suh, Prateek Mittal
        https://arxiv.org/abs/2503.24370
    """

    Args = ThinkingInterventionArgs

    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    base_generate: Callable | None = None

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **_
    ) -> PreTrainedModel:
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.base_generate = model.generate
        return model

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        runtime_kwargs: dict | None,
        model: PreTrainedModel,
        **gen_kwargs,
    ) -> torch.Tensor:
        runtime_kwargs = runtime_kwargs or {}
        self.tag_ids = self.tokenizer("</think>", add_special_tokens=False).input_ids
        # Paper says interventions are best at the beginning
        intervention = self.intervention
        input_params = {**runtime_kwargs.get('params', {})}

        base_generate = runtime_kwargs.get("base_generate", self.base_generate)

        original_prompt_ids = input_ids[0]
        original_input_len = original_prompt_ids.size(0)

        prompt_str = self.tokenizer.decode(
            original_prompt_ids, skip_special_tokens=True
        )
        modified_prompt_str = intervention(prompt_str, input_params)

        new_input = self.tokenizer(modified_prompt_str, return_tensors="pt").to(self.model.device)

        gen_kwargs["return_dict_in_generate"] = False
        output_ids = base_generate(**new_input, **gen_kwargs)[0]
        keep_prefix = output_ids[: original_input_len]

        decoded   = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        remainder_txt = decoded.rsplit("</think>", 1)[-1].lstrip()

        remainder = (
            self.tokenizer(
                remainder_txt,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"]
            .to(output_ids.device)
            .squeeze(0)
        )

        final_ids = torch.cat([keep_prefix, remainder], dim=0)
        return final_ids.unsqueeze(0) if final_ids.dim() == 1 else final_ids
