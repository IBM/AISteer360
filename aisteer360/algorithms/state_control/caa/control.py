from __future__ import annotations

from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control.common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control.common.selectors import FixedLayerSelector, FractionalDepthSelector
from aisteer360.algorithms.state_control.common.token_scope import compute_prompt_lens, make_token_mask
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform

from aisteer360.algorithms.state_control.common.estimators import MeanDifferenceEstimator
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector

from .args import CAAArgs


class CAA(StateControl):
    """Contrastive Activation Addition (CAA).

    Steers model behavior by adding a learned mean-difference direction
    vector to the residual stream at a single layer during generation.

    CAA operates in two phases:

    1. **Training (offline)**: Given contrastive prompt pairs where each pair
       shares the same question but ends with opposite answer tokens, extract
       residual stream activations at the answer-token position. The steering
       vector is the mean difference between positive and negative activations.

    2. **Inference (online)**: Add `multiplier * v_L` to the residual stream
       at a chosen layer L, at all token positions after the user's prompt.
       A positive multiplier increases the target behavior; negative decreases it.

    Reference:

    - "Steering Llama 2 via Contrastive Activation Addition"
    Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
    [https://arxiv.org/abs/2312.06681](https://arxiv.org/abs/2312.06681)
    """

    Args = CAAArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] = []
        self._layer_id: int = 0
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None

        # tracks cumulative position for KV-cached generation
        self._position_offset: int = 0
        self._initial_seq_len: int = 0

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize CAA by training or loading the steering vector.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data.

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # resolve steering vector
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            estimator = MeanDifferenceEstimator()
            sv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # move to device
        sv = sv.to(device, dtype=model.dtype)

        # optionally normalize the vector
        if self.normalize_vector:
            for layer_id, direction in sv.directions.items():
                norm = direction.norm()
                if norm > 0:
                    sv.directions[layer_id] = direction / norm

        self._steering_vector = sv

        # resolve layer_id via selector
        if self.layer_id is not None:
            selector = FixedLayerSelector(self.layer_id)
        else:
            # heuristic: ~40% depth (paper finds layer 13/32 optimal)
            selector = FractionalDepthSelector(fraction=0.4)
        self._layer_id = selector.select(num_layers=num_layers)

        # validate layer is present in steering vector
        if self._layer_id not in sv.directions:
            raise ValueError(f"Steering vector has no direction for layer {self._layer_id}.")

        # build transform
        transform = AdditiveTransform(
            directions=sv.directions,
            strength=self.multiplier,
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
        runtime_kwargs: dict | None,
        **__,
    ) -> dict[str, list]:
        """Create forward hook for activation addition at the target layer.

        Registers a forward hook that adds the steering vector to the output of
        the target layer, modifying the residual stream at that point.

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

        # store initial sequence length for position tracking
        self._initial_seq_len = ids.size(1)
        self._position_offset = 0

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        # register forward hook on the target layer to modify its output
        hooks["forward"].append({
            "module": self._layer_names[self._layer_id],
            "hook_func": partial(
                self._forward_hook,
                layer_id=self._layer_id,
                transform=self._transform,
                gate=self._gate,
                token_scope=self.token_scope,
                prompt_lens=prompt_lens,
                last_k=self.last_k,
                from_position=self.from_position,
                control_ref=self,
            ),
        })

        return hooks

    def _forward_hook(
        self,
        module,
        args,
        kwargs,
        output,
        *,
        layer_id: int,
        transform,
        gate,
        token_scope: str,
        prompt_lens: torch.LongTensor,
        last_k: int | None,
        from_position: int | None,
        control_ref: "CAA",
    ):
        """Apply activation addition to the layer output.

        Args:
            module: The layer module being hooked.
            args: Positional arguments passed to the forward call.
            kwargs: Keyword arguments passed to the forward call.
            output: The layer's output (hidden_states, ...) or just hidden_states.
            layer_id: Index of the target layer.
            transform: The additive transform to apply.
            gate: The gate (always open for CAA).
            token_scope: Which tokens to steer.
            prompt_lens: Per-batch prompt lengths.
            last_k: Number of last tokens when token_scope is "last_k".
            from_position: Starting position when token_scope is "from_position".
            control_ref: Reference to the CAA control for position tracking.

        Returns:
            Modified output with steering vector added to hidden states.
        """
        # extract hidden states from layer output
        # transformer layers return (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if hidden is None:
            return output

        seq_len = hidden.size(1)

        # determine position offset for KV-cached generation
        # during the first forward pass, seq_len == initial_seq_len and offset is 0
        # during subsequent passes with KV cache, seq_len is typically 1 (or small)
        # and we need to compute the absolute position
        if seq_len < control_ref._initial_seq_len:
            # KV-cached generation mode
            position_offset = control_ref._position_offset
            control_ref._position_offset += seq_len
        else:
            # first forward pass or non-cached generation
            position_offset = 0
            control_ref._position_offset = seq_len

        mask = make_token_mask(
            token_scope,
            seq_len=seq_len,
            prompt_lens=prompt_lens.to(hidden.device),
            last_k=last_k,
            from_position=from_position,
            position_offset=position_offset,
        )

        if gate.is_open():
            hidden = transform.apply(
                hidden,
                layer_id=layer_id,
                token_mask=mask,
            )

        # return modified output in the same format
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        self._position_offset = 0
