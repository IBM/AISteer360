from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import FixedLayerSelector, FractionalDepthSelector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform, NormPreservingTransform

from aisteer360.algorithms.state_control._common.estimators import ContrastiveDirectionEstimator, MeanDifferenceEstimator
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector

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
        self._runtime = TransformHookRuntime(hook_point="layer_output")

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
            if self.train_spec.method == "pca_pairwise":
                estimator = ContrastiveDirectionEstimator()
            else:
                estimator = MeanDifferenceEstimator()
            sv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # clone before the in-place move/normalize so a caller-supplied vector is never mutated
        sv = sv.clone().to(device, dtype=model.dtype)

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
            sv.directions,
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
        self._runtime.reset(prompt_lens)

        return {
            "pre": [],
            "forward": [{
                "module": self._layer_names[self._layer_id],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=self._layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=True,  # single-layer control: its only hook opens the pass
                ),
            }],
            "backward": [],
        }

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        if self._runtime._prompt_lens is not None:
            self._runtime.reset(self._runtime._prompt_lens)
