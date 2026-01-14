"""
Few-shot learning control for prompt adaptation.
"""
import warnings
from typing import Any, Callable, Sequence

import torch
from transformers import PreTrainedTokenizer

from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.few_shot.args import FewShotArgs
from aisteer360.algorithms.input_control.few_shot.selectors import SELECTOR_REGISTRY
from aisteer360.algorithms.input_control.few_shot.selectors.base import Selector
from aisteer360.algorithms.input_control.few_shot.selectors.random_selector import (
    RandomSelector,
)


class FewShot(InputControl):
    """
    Implementation of few-shot learning control for prompt adaptation.

    FewShot enables selective behavioral steering by prepending specific examples to user prompts, guiding model
    responses through demonstration.

    The method operates in two modes:

    1. **Pool-based sampling**: Maintains pools of positive and negative examples from which k examples are dynamically
        selected using configurable sampling strategies (random, semantic similarity, etc.).

    2. **Runtime injection**: Accepts examples directly at inference time through runtime_kwargs, enabling
        context-specific demonstrations without predefined pools. Useful for dynamic or user-provided examples.

    The selected examples are formatted into a system prompt with clear positive/negative labels and prepended to the
    user query using the model's chat template, allowing the model to learn the desired behavior pattern from the
    demonstrations.

    Args:
        directive (str, optional): Instruction text that precedes the examples, explaining the task or desired behavior.
            Defaults to None.
        positive_example_pool (Sequence[dict], optional): Pool of positive examples demonstrating desired behavior.
            Each dict can contain multiple key-value pairs. Defaults to None.
        negative_example_pool (Sequence[dict], optional): Pool of negative examples showing undesired behavior to avoid.
            Each dict can contain multiple key-value pairs. Defaults to None.
        k_positive (int, optional): Number of positive examples to sample from the pool per query.
            Defaults to None.
        k_negative (int, optional): Number of negative examples to sample from the pool per query.
            Defaults to None.
        selector_name (str, optional): Name of the selection strategy ('random', 'semantic', etc.).
            Determines how examples are chosen from pools. Defaults to 'random'.
        template (str, optional): Custom template for formatting the system prompt. Should contain
            {directive} and {example_blocks} placeholders. Defaults to built-in template.

    Runtime keyword arguments:

    - `positive_examples` (`list[dict]`, `optional`): Positive examples to use for this specific query (overrides pool-based
    selection).
    - `negative_examples` (`list[dict]`, `optional`): Negative examples to use for this specific query (overrides pool-based
    selection).

    Notes:

    - Requires a tokenizer with chat_template support for optimal formatting
    - Examples are automatically labeled as "### Positive example" or "### Negative example"
    - When both pools and runtime examples are available, runtime examples take precedence
    - If no examples are provided, the original input is returned unchanged
    """

    Args = FewShotArgs

    # default templates
    _SYSTEM_PROMPT_TEMPLATE = "{directive}: \n{example_blocks}\n\n"
    _POSITIVE_EXAMPLE_TEMPLATE = "### Positive example (behavior to follow)\n{content}\n"
    _NEGATIVE_EXAMPLE_TEMPLATE = "### Negative example (behavior to avoid)\n{content}\n"

    # placeholders
    tokenizer: PreTrainedTokenizer | None = None
    selector_name: str | None = None
    directive: str | None = None
    positive_example_pool: Sequence[dict] | None = None
    negative_example_pool: Sequence[dict] | None = None
    k_positive: int | None = None
    k_negative: int | None = None
    selector: Selector | None = None

    def steer(
            self,
            model=None,
            tokenizer: PreTrainedTokenizer | None = None,
            **kwargs
    ) -> None:
        self.tokenizer = tokenizer

        # initialize selector if using pool mode
        if self.positive_example_pool is not None or self.negative_example_pool is not None:
            if self.selector_name:
                selector_cls = SELECTOR_REGISTRY.get(self.selector_name, RandomSelector)
                self.selector = selector_cls()
            else:
                self.selector = RandomSelector()

    def get_prompt_adapter(
        self,
        runtime_kwargs: dict | None = None
    ) -> Callable[[list[int] | torch.Tensor, dict[str, Any]], list[int] | torch.Tensor]:
        """Return a prompt adapter function that adds few-shot examples to the model's system prompt. Creates and
        returns a closure that modifies input token sequences by prepending few-shot examples.

        The returned adapter function performs the following steps:

        1. Determines operational mode (runtime examples take precedence over pools)
        2. Decodes input tokens to retrieve the original user message
        3. Selects or retrieves appropriate examples based on mode
        4. Formats examples with positive/negative labels
        5. Constructs a system prompt containing the examples
        6. Applies the model's chat template (if available) to combine system prompt and user message
        7. Re-encodes the adapted text to tokens

        Returns:
            A prompt adapter function.

        Raises:
            RuntimeError: If tokenizer is not set (requires calling `steer()` first)

        Warnings:
            UserWarning: Issued when:

                - No examples available from either pools or runtime_kwargs
                - No examples remain after selection/sampling
                - Tokenizer lacks chat_template support (falls back to direct prepending)
        """

        if self.tokenizer is None:
            raise RuntimeError("FewShot needs a tokenizer; call .steer() first.")

        def adapter(input_ids: list[int] | torch.Tensor, runtime_kwargs: dict[str, Any]) -> list[int] | torch.Tensor:

            # infer mode from arguments
            using_runtime_examples = (runtime_kwargs and ("positive_examples" in runtime_kwargs or
                                                          "negative_examples" in runtime_kwargs))
            using_pool_mode = self.positive_example_pool is not None or self.negative_example_pool is not None

            if not using_runtime_examples and not using_pool_mode:
                warnings.warn(
                    "FewShot: No examples provided via runtime_kwargs or example pools. "
                    "Returning original input unchanged.",
                    UserWarning
                )
                return input_ids

            # decode to retrieve user message
            if isinstance(input_ids, torch.Tensor):
                input_ids_list = input_ids.tolist()[0]
            else:
                input_ids_list = input_ids

            original_text = self.tokenizer.decode(input_ids_list, skip_special_tokens=True)

            # get examples based on mode
            if using_runtime_examples:
                examples = self._gather_runtime_examples(runtime_kwargs)
            else:
                examples = self._sample_from_pools()

            if not examples:
                warnings.warn(
                    "FewShot: No examples available after selection. Returning original input unchanged.",
                    UserWarning
                )
                return input_ids

            examples_text = self._format_examples(examples)

            # apply chat template
            if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                messages = [
                    {"role": "system", "content": examples_text},
                    {"role": "user", "content": original_text}
                ]
                adapted_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                warnings.warn(
                    "No chat template found for tokenizer. Prepending few-shot examples directly to user query.",
                    UserWarning
                )
                adapted_text = examples_text + original_text

            # encode the adapted text
            adapted_tokens = self.tokenizer.encode(
                adapted_text,
                add_special_tokens=False,
                return_tensors="pt" if isinstance(input_ids, torch.Tensor) else None
            )

            if isinstance(input_ids, torch.Tensor):
                return adapted_tokens.squeeze(0) if adapted_tokens.dim() > 1 else adapted_tokens
            else:
                return adapted_tokens

        return adapter

    def _sample_from_pools(self) -> list[dict[str, Any]]:
        """Sample examples from the pools."""
        all_examples = []

        if self.positive_example_pool and self.k_positive and self.k_positive > 0:
            positive_samples = self.selector.sample(
                self.positive_example_pool,
                self.k_positive
            )
            for example in positive_samples:
                all_examples.append({**example, "_label": "positive"})

        if self.negative_example_pool and self.k_negative and self.k_negative > 0:
            negative_samples = self.selector.sample(
                self.negative_example_pool,
                self.k_negative
            )
            for example in negative_samples:
                all_examples.append({**example, "_label": "negative"})

        return all_examples

    def _format_examples(self, examples: list[dict[str, Any]]) -> str:
        """Format examples for system prompt."""
        if not examples:
            return ""

        example_blocks = []
        for example in examples:
            is_positive = example.get("_label", "positive") == "positive"
            content = self._format_example_content(example)

            if is_positive:
                example_blocks.append(self._POSITIVE_EXAMPLE_TEMPLATE.format(content=content))
            else:
                example_blocks.append(self._NEGATIVE_EXAMPLE_TEMPLATE.format(content=content))

        template = getattr(self, 'template', None) or self._SYSTEM_PROMPT_TEMPLATE
        formatted_blocks = "\n".join(example_blocks)

        return template.format(directive=self.directive or "", example_blocks=formatted_blocks)

    @staticmethod
    def _gather_runtime_examples(runtime_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        """Gather examples from runtime_kwargs."""
        examples = []
        if "positive_examples" in runtime_kwargs:
            for example in runtime_kwargs["positive_examples"]:
                examples.append({**example, "_label": "positive"})
        if "negative_examples" in runtime_kwargs:
            for example in runtime_kwargs["negative_examples"]:
                examples.append({**example, "_label": "negative"})
        return examples

    @staticmethod
    def _format_example_content(example: dict[str, Any]) -> str:
        segments = []
        for key, value in example.items():
            if key == "_label":
                continue
            formatted_key = key.replace("_", " ").title()
            segments.append(f"{formatted_key}: {value}")

        return "\n".join(segments)
