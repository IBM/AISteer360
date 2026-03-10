"""Inference-Time Intervention (ITI) state control."""
from __future__ import annotations

from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control.common.head_steering_vector import HeadSteeringVector
from aisteer360.algorithms.state_control.common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control.common.selectors import TopKHeadSelector
from aisteer360.algorithms.state_control.common.token_scope import compute_prompt_lens, make_token_mask
from aisteer360.algorithms.state_control.common.transforms import HeadAdditiveTransform, NormPreservingTransform

from .args import ITIArgs
from .utils import ProbeMassShiftEstimator


class ITI(StateControl):
    """Inference-Time Intervention (ITI).

    Steers model behavior by shifting activations at a sparse set of attention heads 
    during inference. The intervention operates at the residual stream level by adding 
    direction vectors to head-associated slices of the hidden dimension.

    ITI operates in two phases:

    1. **Offline (during steer())**: For every attention head across all layers,
       extract the head's output activations on labeled true/false statements.
       Train a per-head linear probe; rank heads by probe accuracy. For the
       top-K heads, compute the mass mean shift: direction = mean(activations_true)
       - mean(activations_false).

    2. **Online (during generation)**: At each generated token, for each selected
       (layer, head) pair, add alpha * direction to that head's slice of the
       residual stream. The intervention fires unconditionally on every token
       in the specified token_scope.

    Reference:

    - "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model"
    Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
    [https://arxiv.org/abs/2306.03341](https://arxiv.org/abs/2306.03341)
    """

    Args = ITIArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._head_steering_vector: HeadSteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] = []
        self._oproj_names: list[str] = []
        self._active_layer_ids: set[int] = set()
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None

        # tracks cumulative position for KV-cached generation
        self._position_offset: int = 0
        self._initial_seq_len: int = 0
        self._current_mask: torch.BoolTensor | None = None

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize ITI by training or loading the head steering vector.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data.

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names

        # determine o_proj suffix based on architecture
        if layer_names[0].startswith("model.layers"):
            oproj_suffix = ".self_attn.o_proj"
        elif layer_names[0].startswith("transformer.h"):
            oproj_suffix = ".attn.c_proj"
        else:
            raise ValueError(f"Unrecognized model architecture: {layer_names[0]}")

        self._oproj_names = [name + oproj_suffix for name in layer_names]

        num_heads = model.config.num_attention_heads
        hidden_size = model.config.hidden_size
        head_dim = hidden_size // num_heads

        # resolve head steering vector
        if self.head_steering_vector is not None:
            hsv = self.head_steering_vector
        else:
            estimator = ProbeMassShiftEstimator()
            hsv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # move to device
        hsv = hsv.to(device, dtype=model.dtype)

        self._head_steering_vector = hsv

        # resolve head selection
        if self.selected_heads is not None:
            selected = self.selected_heads
        else:
            if hsv.probe_accuracies is None:
                raise ValueError(
                    "head_steering_vector has no probe_accuracies. "
                    "Either provide selected_heads explicitly or use data to train a new vector."
                )
            selector = TopKHeadSelector(self.num_heads)
            selected = selector.select(probe_accuracies=hsv.probe_accuracies)

        # group selected directions by layer: {layer_id: {head_id: direction}}
        grouped: dict[int, dict[int, torch.Tensor]] = {}
        for layer_id, head_id in selected:
            if layer_id not in grouped:
                grouped[layer_id] = {}
            grouped[layer_id][head_id] = hsv.directions[(layer_id, head_id)]

        self._active_layer_ids = set(grouped.keys())

        # build transform
        transform = HeadAdditiveTransform(
            head_directions=grouped,
            num_heads=num_heads,
            head_dim=head_dim,
            strength=self.alpha,
        )
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)
        self._transform = transform

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        return model

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,  # noqa: ARG002
        **__,
    ) -> dict[str, list]:
        """Create pre-hooks on o_proj modules for pre-projection intervention.

        Registers pre-hooks on each active layer's o_proj. The pre-hook modifies
        the input to o_proj (the concatenated per-head attention outputs) by adding
        direction vectors to the appropriate head slices. This matches the paper's
        intervention point: after Att, before Q^h_l (the output projection).

        Args:
            input_ids: Input token IDs.
            runtime_kwargs: Runtime parameters (currently unused).

        Returns:
            Hook specifications with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        prompt_lens = compute_prompt_lens(ids, self._pad_token_id)

        self._initial_seq_len = ids.size(1)
        self._position_offset = 0
        self._current_mask = None

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        if not self._active_layer_ids:
            return hooks

        # pre-hook on the earliest active o_proj for mask computation
        earliest_layer = min(self._active_layer_ids)
        hooks["pre"].append({
            "module": self._oproj_names[earliest_layer],
            "hook_func": partial(
                self._mask_pre_hook,
                prompt_lens=prompt_lens,
                control_ref=self,
            ),
        })

        # pre-hooks on each active o_proj to apply the head-level transform
        for layer_id in sorted(self._active_layer_ids):
            hooks["pre"].append({
                "module": self._oproj_names[layer_id],
                "hook_func": partial(
                    self._transform_pre_hook,
                    layer_id=layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    control_ref=self,
                ),
            })

        return hooks

    @staticmethod
    def _mask_pre_hook(
        _module,
        args,
        kwargs,
        *,
        prompt_lens: torch.LongTensor,
        control_ref: "ITI",
    ):
        """Pre-hook on o_proj to compute token mask once per forward pass.

        This hook fires first (on the earliest active o_proj) and computes the
        token mask that all transform pre-hooks will use. It also handles position
        offset tracking for KV-cached generation.

        Args:
            prompt_lens: Per-batch prompt lengths.
            control_ref: Reference to the ITI control for state management.

        Returns:
            None (does not modify inputs).
        """
        # o_proj input is the first positional arg: [B, T, num_heads * head_dim]
        hidden = args[0] if args else None
        if hidden is None:
            return None

        seq_len = hidden.size(1)

        if seq_len < control_ref._initial_seq_len:
            position_offset = control_ref._position_offset
            control_ref._position_offset += seq_len
        else:
            position_offset = 0
            control_ref._position_offset = seq_len

        control_ref._current_mask = make_token_mask(
            control_ref.token_scope,
            seq_len=seq_len,
            prompt_lens=prompt_lens.to(hidden.device),
            last_k=control_ref.last_k,
            from_position=control_ref.from_position,
            position_offset=position_offset,
        )

        return None

    @staticmethod
    def _transform_pre_hook(
        _module,
        args,
        kwargs,
        *,
        layer_id: int,
        transform,
        gate,
        control_ref: "ITI",
    ):
        """Pre-hook on o_proj to apply head-level intervention to the o_proj input.

        Modifies the concatenated per-head attention outputs before the output
        projection, matching the paper's intervention point.

        Args:
            layer_id: Index of the target layer.
            transform: The head additive transform to apply.
            gate: The gate (always open for ITI).
            control_ref: Reference to the ITI control for cached mask access.

        Returns:
            Modified (args, kwargs) tuple or None if no modification needed.
        """
        hidden = args[0] if args else None
        if hidden is None:
            return None

        mask = control_ref._current_mask
        if mask is None:
            return None

        if gate.is_open():
            modified = transform.apply(
                hidden,
                layer_id=layer_id,
                token_mask=mask,
            )
            # return modified args tuple so o_proj sees the steered input
            return (modified,) + args[1:], kwargs

        return None

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        self._position_offset = 0
        self._current_mask = None
