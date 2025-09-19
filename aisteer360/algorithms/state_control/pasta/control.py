from __future__ import annotations

from functools import partial
from typing import Sequence

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.pasta.args import PASTAArgs


class PASTA(StateControl):
    """
    Implementation of PASTA (Post-hoc Attention STeering Approach) from Zhang et al., 2023.

    PASTA performs controlled text generation by dynamically modifying attention patterns during inference to amplify or
    suppress the influence of specific text spans. This allows for fine-grained steering of model behavior without
    requiring model retraining or parameter updates.

    The algorithm works by:

    1. **Substring Identification**: Locate target substrings within the input prompt using tokenizer offset mapping to
    determine precise token ranges.

    2. **Attention Modification**: Inject scaling factors into the attention mask of specified layers and heads to
    increase or decrease attention weights for the identified token ranges.

    3. **Dynamic Steering**: Apply different scaling strategies (include, exclude, or generation-focused) to control how
    the model attends to relevant spans during text generation.

    This approach enables real-time control over model focus and can be used for tasks like concept amplification, bias
    mitigation, or content filtering without architectural changes.

    Args:
        alpha (float): Scaling factor for attention modification. Positive values increase attention, negative values
            decrease attention. Defaults to 1.0.
        head_config (dict | list): Configuration specifying which layers/heads to modify. If dict, maps layer indices
            to lists of head indices. If list, applies to all heads in specified layers.
        scale_position (str): Strategy for applying attention scaling. Options:

            - "include": Scale attention TO the target substrings
            - "exclude": Scale attention AWAY FROM the target substrings
            - "generation": Scale attention during generation phase

            Defaults to "include".

    Reference:
    - "PASTA: Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs"
    Qingru Zhang, Chandan Singh, Liyuan Liu, Xiaodong Liu, Bin Yu, Jianfeng Gao, Tuo Zhao
    [https://arxiv.org/abs/2311.02262](https://arxiv.org/abs/2311.02262)
    """

    Args = PASTAArgs

    # placeholders
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    device: torch.device | str | None = None

    _head_map: dict[int, list[int]] | None = None
    _layers: list[int] | None = None
    _scale_constant: torch.Tensor | None = None

    def steer(
        self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **__
    ) -> PreTrainedModel:
        """Initialize PASTA by configuring attention head mappings and model references.

        Sets up the layer and head configurations that will be modified during generation.
        Validates head configurations against model architecture.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for substring identification.
                If None, attempts to retrieve from model attributes.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model (unchanged).
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device
        self._setup_head_config(self.head_config)
        return model

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **__,
    ) -> dict[str, list]:
        """Create attention modification hooks for specified substrings.

        Identifies token ranges corresponding to target substrings and prepares hooks that will modify attention weights
        during the forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            runtime_kwargs (dict | None): Must contain "substrings" key with target text spans:

                - str: Single substring applied to all batch items
                - list[str]: List of substrings applied to all batch items
                - list[list[str]]: Per-batch substring groups
            **__: Additional arguments (unused).

        Returns:
            dict[str, list]: Hook specifications with "pre", "forward", "backward" keys. Only "pre" hooks are populated for attention modification.

        Raises:
            ValueError: If "substrings" not in runtime_kwargs or batch size mismatch.
        """
        if not runtime_kwargs or "substrings" not in runtime_kwargs:
            raise ValueError("PASTA requires 'substrings' inside runtime_kwargs")

        substrings = runtime_kwargs["substrings"]
        batch_size = input_ids.size(0)

        # normalize substrings to shape (batch, group, str)
        if isinstance(substrings, str):
            substrings = [[substrings]] * batch_size
        elif substrings and isinstance(substrings[0], str):
            substrings = [substrings] * batch_size
        elif len(substrings) != batch_size:
            raise ValueError(
                f"Need {batch_size} substring groups (one per prompt); got {len(substrings)}"
            )

        # decode and get offsets
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # Have to encode & decode substrings along with prompts, since we observed prompts getting changed due to
        # tokenization (e.g. spaces removed); and we need to replicate the same effect in the substrings to ensure they
        # actually match
        for idx, substring in enumerate(substrings):
            try:
                substrings[idx] = self.tokenizer.batch_decode(
                    self.tokenizer(substring, return_tensors="pt", padding=True)['input_ids'],
                    skip_special_tokens=True
                )
            except:
                breakpoint()

        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        tokenized: BatchEncoding = self.tokenizer(
            prompts,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=True,
        ).to(self.device)

        offset_mapping = tokenized.pop("offset_mapping")
        input_len = tokenized["input_ids"].size(-1)

        token_ranges = self._token_ranges_from_batch(
            prompts, substrings, offset_mapping
        )

        if self._scale_constant is None:
            self._scale_constant = torch.tensor(
                [self.alpha],
                device=self.device,
                dtype=tokenized.input_ids.dtype,
            ).log()

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for layer in self._layers:
            hooks["pre"].append(
                {
                    "module": f"model.layers.{layer}.self_attn",
                    "hook_func": partial(
                        self._attention_pre_hook,
                        head_idx=self._head_map[layer],
                        token_ranges=token_ranges,
                        input_len=input_len,
                    ),
                }
            )

        return hooks

    def _setup_head_config(self, head_config):
        """Parse and validate attention head configuration.

        Converts various configuration formats into internal layer-head mappings and validates against model architecture.

        Args:
            head_config: Configuration specifying which layers/heads to modify:

                - dict: Maps layer indices to lists of head indices
                - list: Layer indices (applies to all heads in those layers)

        Raises:
            ValueError: If configuration format invalid or heads out of range.
        """
        if isinstance(head_config, dict):
            self._head_map = {int(l): list(h) for l, h in head_config.items()}
            self._layers = sorted(self._head_map.keys())
        elif isinstance(head_config, list):
            self._layers = [int(l) for l in head_config]
            self._head_map = {
                l: list(range(self.model.config.num_attention_heads))
                for l in self._layers
            }
        else:
            raise ValueError(f"Invalid head configuration: {head_config!r}")

        num_heads = self.model.config.num_attention_heads
        for layer, heads in self._head_map.items():
            for head in heads:
                if not 0 <= head < num_heads:
                    raise ValueError(
                        f"Head {head} out of range for layer {layer} (0–{num_heads-1})"
                    )

    @staticmethod
    def _find_token_range(
        string: str,
        substring: str,
        offset_mapping: Sequence[tuple[int, int]],
        occurrence: int = 0,
    ) -> tuple[int, int]:
        """Map a substring to its token index range using offset mapping.

        Locates the character positions of a substring and converts them to token indices using the tokenizer's offset mapping.

        Args:
            string: Full text to search within.
            substring: Target substring to locate.
            offset_mapping: List of (start_char, end_char) tuples for each token.
            occurrence: Which occurrence to find if substring appears multiple times.
                Defaults to 0 (first occurrence).

        Returns:
            tuple[int, int]: Start (inclusive) and end (exclusive) token indices.

        Raises:
            ValueError: If substring cannot be mapped to token range.
        """
        if substring not in string:
            print(f"'{substring}' not found in input {string}")
            return 0, 0

        char_index = -1
        for _ in range(occurrence + 1):
            char_index = string.index(substring, char_index + 1)
        char_start = char_index
        char_end = char_start + len(substring)

        token_start = token_end = None
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if token_start is None and start_char <= char_start < end_char:
                token_start = token_idx
            if token_end is None and start_char < char_end <= end_char:
                token_end = token_idx

        if token_start is None or token_end is None:
            raise ValueError("Could not map substring to token range")

        return token_start, token_end + 1

    def _token_ranges_from_batch(
        self,
        texts: Sequence[str],
        groups: Sequence[Sequence[str]],
        offsets_mapping: Sequence[Sequence[tuple[int, int]]],
        occurrence: int = 0,
    ) -> list[torch.Tensor]:
        """Convert batch of substring groups to token ranges.

        Maps multiple substrings across batch items to their corresponding token index ranges for attention modification.

        Args:
            texts: Decoded text for each batch item.
            groups: Groups of substrings for each batch item.
            offsets_mapping: Token offset mappings for each batch item.
            occurrence: Which occurrence to find for repeated substrings.

        Returns:
            list[torch.Tensor]: Token range tensors for each batch item.
                Each tensor has shape [num_substrings, 2] with [start, end] pairs.
        """
        token_ranges: list[torch.Tensor] = []

        for text, substrings, offsets in zip(texts, groups, offsets_mapping):
            substring_ranges = [
                torch.tensor(
                    self._find_token_range(text, substring, offsets, occurrence)
                )
                for substring in substrings
            ]
            token_ranges.append(torch.stack(substring_ranges))

        return token_ranges

    def _attention_pre_hook(
        self,
        module,
        input_args: tuple,
        input_kwargs: dict,
        head_idx: list[int],
        token_ranges: list[torch.Tensor],
        input_len: int,
    ):
        """Modify attention mask to steer focus toward/away from target tokens.

        Pre-forward hook that adjusts attention weights by adding scaling factors to the attention mask for specified token ranges and attention heads.

        Args:
            module: The attention module being hooked.
            input_args: Positional arguments to the forward pass.
            input_kwargs: Keyword arguments to the forward pass.
            head_idx: List of attention head indices to modify.
            token_ranges: Token index ranges to apply scaling to.
            input_len: Length of input sequence (for generation positioning).

        Returns:
            Tuple of potentially modified (input_args, input_kwargs).

        Raises:
            RuntimeError: If hidden states cannot be located.
            ValueError: If scale_position is invalid.
        """
        hidden_states = (
            input_args[0] if input_args else input_kwargs.get("hidden_states")
        )
        if hidden_states is None:
            raise RuntimeError("PASTA: could not locate hidden states")

        attention_mask = input_kwargs.get("attention_mask")
        if attention_mask is None:  # build it
            batch_size, sequence_len, _ = hidden_states.size()
            num_heads = self.model.config.num_attention_heads
            causal = torch.triu(
                hidden_states.new_full((sequence_len, sequence_len), float("-inf")),
                diagonal=1,
            )
            attention_mask = causal[None, None]  # (1,1,q,k)
            attention_mask = attention_mask.expand(
                batch_size, num_heads, -1, -1
            ).contiguous()
            input_kwargs["attention_mask"] = attention_mask

        attention_mask = attention_mask.to(hidden_states.dtype).contiguous()
        if attention_mask.size(1) == 1:
            attention_mask = attention_mask.expand(
                -1,
                self.model.config.num_attention_heads,
                -1,
                -1,
            ).contiguous()

        batch_size = attention_mask.size(0)
        for batch_index in range(batch_size):
            for start_idx, end_idx in token_ranges[batch_index].tolist():
                if start_idx == end_idx:
                    continue
                if self.scale_position == "include":
                    attention_mask[
                        batch_index, head_idx, :, start_idx:end_idx
                    ] += self._scale_constant
                elif self.scale_position == "exclude":
                    attention_mask[
                        batch_index, head_idx, :, :start_idx
                    ] += self._scale_constant
                    attention_mask[
                        batch_index, head_idx, :, end_idx:input_len
                    ] += self._scale_constant
                elif self.scale_position == "generation":
                    attention_mask[
                        batch_index, head_idx, :, :input_len
                    ] += self._scale_constant

                else:
                    raise ValueError(f"Unknown scale_position '{self.scale_position}'")

        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :input_len] -= self._scale_constant

        input_kwargs["attention_mask"] = attention_mask
        return input_args, input_kwargs
