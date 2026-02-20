"""ActAdd (Activation Addition) control implementation."""
from __future__ import annotations

from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.common.estimators import SinglePairEstimator
from aisteer360.algorithms.state_control.common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control.common.hook_utils import (
    get_model_layer_list,
    extract_hidden_states,
    replace_hidden_states,
)
from aisteer360.algorithms.state_control.common.selectors import FixedLayerSelector, FractionalDepthSelector
from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.common.transforms import AdditiveTransform, NormPreservingTransform

from .args import ActAddArgs


class ActAdd(StateControl):
    """Activation Addition (ActAdd).

    Steers model behavior by adding a positional steering vector — computed
    from a single contrast pair of short prompts — to the residual stream
    at a single layer during the initial forward pass.

    Reference:

        - "Steering Language Models With Activation Engineering"
          Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez,
          Ulisse Mini, Monte MacDiarmid
          [https://arxiv.org/abs/2308.10248](https://arxiv.org/abs/2308.10248)
    """

    Args = ActAddArgs
    supports_batching = False  # ActAdd uses positional alignment which breaks with left-padding

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] = []
        self._layer_id: int = 0
        self._gate = AlwaysOpenGate()

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Extract or load the steering vector and build the transform.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding the prompt pair.

        Returns:
            The input model, unchanged.
        """
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # resolve steering vector
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            estimator = SinglePairEstimator()
            sv = estimator.fit(
                model,
                tokenizer,
                positive_prompt=self.positive_prompt,
                negative_prompt=self.negative_prompt,
            )

        device = next(model.parameters()).device
        sv = sv.to(device, dtype=model.dtype)

        # resolve layer_id via selector
        if self.layer_id is not None:
            selector = FixedLayerSelector(self.layer_id)
        else:
            # heuristic: ~20% depth (paper uses layer 6/48 for GPT-2-XL)
            selector = FractionalDepthSelector(fraction=0.2, minimum=1)
        self._layer_id = selector.select(num_layers=num_layers)

        if self._layer_id not in sv.directions:
            raise ValueError(f"Steering vector has no direction for layer {self._layer_id}.")

        # optionally normalize per-position vectors
        if self.normalize_vector:
            d = sv.directions[self._layer_id]  # [T, H]
            norms = d.norm(dim=-1, keepdim=True)  # [T, 1]
            sv.directions[self._layer_id] = d / (norms + 1e-8)

        self._steering_vector = sv

        # build transform
        transform = AdditiveTransform(
            directions=sv.directions,
            strength=self.multiplier,
            alignment=self.alignment,
        )
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)
        self._transform = transform

        return model

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        **__,
    ) -> dict[str, list]:
        """Register a pre-hook on the target layer.

        The paper's Algorithm 1 specifies adding the steering vector to the
        residual stream *before* the target layer processes it (h_l input),
        not after (h_l output). Using a pre-hook ensures correct layer alignment.

        Args:
            input_ids: Input token IDs (used only for mask computation).
            runtime_kwargs: Unused.

        Returns:
            Hook specifications.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)

        # for ActAdd, token_scope is effectively "all" — the alignment-based
        # positioning in the transform handles spatial control, and the
        # mask just provides a uniform gate.
        mask = torch.ones(ids.size(0), ids.size(1), dtype=torch.bool, device=ids.device)

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        hooks["pre"].append({
            "module": self._layer_names[self._layer_id],
            "hook_func": partial(
                self._pre_hook,
                layer_id=self._layer_id,
                transform=self._transform,
                gate=self._gate,
                token_mask=mask,
            ),
        })

        return hooks

    def _pre_hook(
        self,
        module,
        args,
        kwargs,
        *,
        layer_id: int,
        transform,
        gate,
        token_mask: torch.BoolTensor,
    ):
        """Apply positional activation addition to the layer input (residual stream)."""
        hidden = extract_hidden_states(args, kwargs)
        if hidden is None:
            return args, kwargs

        # during KV-cached generation, the mask from get_hooks() was sized
        # for the full prompt. Resize to current seq_len.
        seq_len = hidden.size(1)
        batch_size = hidden.size(0)
        if token_mask.size(1) != seq_len:
            token_mask = torch.ones(
                batch_size, seq_len, dtype=torch.bool, device=hidden.device,
            )

        if gate.is_open():
            hidden = transform.apply(
                hidden,
                layer_id=layer_id,
                token_mask=token_mask,
            )

        return replace_hidden_states(args, kwargs, hidden)

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
