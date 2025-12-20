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

    supports_batching: bool = True

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
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("ThinkingIntervention requires .steer() first.")

        runtime_kwargs = runtime_kwargs or {}
        base_generate = runtime_kwargs.get("base_generate", self.base_generate)
        if base_generate is None:
            raise RuntimeError("ThinkingIntervention: base_generate is not set.")

        intervention = self.intervention

        # self.tag_ids = self.tokenizer("</think>", add_special_tokens=False).input_ids

        # normalize to [batch, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        batch_size = input_ids.size(0)

        # params handling
        params_agg = runtime_kwargs.get("params", None)
        if params_agg is None:
            params_per_example = [{} for _ in range(batch_size)]
        elif isinstance(params_agg, dict) and any(
                isinstance(v, (list, tuple)) for v in params_agg.values()
        ):
            # aggregated dict-of-lists; slice each list per example
            params_per_example: list[dict] = []
            for i in range(batch_size):
                p_i = {}
                for k, v in params_agg.items():
                    if isinstance(v, (list, tuple)):
                        if len(v) != batch_size:
                            raise ValueError(
                                f"ThinkingIntervention: params['{k}'] has length {len(v)}, but batch size is {batch_size}."
                            )
                        p_i[k] = v[i]
                    else:
                        p_i[k] = v
                params_per_example.append(p_i)
        else:
            # Simple case: same params dict for every example
            params_per_example = [params_agg] * batch_size

        # build modified prompts
        original_lengths = [ids.size(0) for ids in input_ids]
        original_prompts = self.tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        modified_prompts = [
            intervention(prompt, params_per_example[i])
            for i, prompt in enumerate(original_prompts)
        ]

        new_input = self.tokenizer(
            modified_prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        gen_kwargs = dict(gen_kwargs)
        gen_kwargs["return_dict_in_generate"] = False

        outputs = base_generate(
            input_ids=new_input["input_ids"],
            attention_mask=new_input.get("attention_mask", None),
            **gen_kwargs,
        )

        if isinstance(outputs, torch.Tensor):
            output_ids = outputs
        else:
            output_ids = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        final_sequences: list[torch.Tensor] = []

        for i in range(batch_size):
            out_ids = output_ids[i]
            keep_prefix = out_ids[: original_lengths[i]]

            decoded = self.tokenizer.decode(out_ids, skip_special_tokens=False)
            remainder_txt = decoded.rsplit("</think>", 1)[-1].lstrip()

            remainder_ids = (
                self.tokenizer(
                    remainder_txt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"]
                .to(out_ids.device)
                .squeeze(0)
            )

            final_ids = torch.cat([keep_prefix, remainder_ids], dim=0)
            final_sequences.append(final_ids)

        padded = self.tokenizer.pad(
            {"input_ids": [seq.tolist() for seq in final_sequences]},
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # return [batch, max_len]
        return padded["input_ids"]
