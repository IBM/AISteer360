"""Inference-Time Intervention (ITI) state control."""
from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.model_layout import resolve_model_layout
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import TopKHeadSelector
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import HeadAdditiveTransform, NormPreservingTransform

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
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] = []
        self._oproj_names: list[str] = []
        self._active_layer_ids: set[int] = set()
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_input")

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize ITI by training or loading the steering vector.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data.

        Returns:
            The input model, unchanged.
        """
        device = next(model.parameters()).device
        layout = resolve_model_layout(model)
        self._layer_names = layout.layer_names
        self._oproj_names = layout.oproj_names

        # resolve steering vector
        if self.steering_vector is not None:
            sv = self.steering_vector
        else:
            estimator = ProbeMassShiftEstimator()
            sv = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # move to device
        sv = sv.to(device, dtype=model.dtype)
        self._steering_vector = sv

        # resolve head selection
        if self.selected_heads is not None:
            selected = self.selected_heads
        else:
            if sv.probe_accuracies is None:
                raise ValueError(
                    "steering_vector has no probe_accuracies. "
                    "Either provide selected_heads explicitly or use data to train a new vector."
                )
            selector = TopKHeadSelector(self.num_heads)
            selected = selector.select(steering_vector=sv)

        # group selected heads by layer
        active_heads: dict[int, set[int]] = {}
        for layer_id, head_id in selected:
            active_heads.setdefault(layer_id, set()).add(head_id)

        self._active_layer_ids = set(active_heads.keys())

        # build transform
        transform = HeadAdditiveTransform(
            sv,
            active_heads=active_heads,
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
        """Create pre-hooks on active o_proj modules for pre-projection intervention.

        Registers a pre-hook on each active layer's o_proj. Each pre-hook modifies the input to
        o_proj (the concatenated per-head attention outputs) by adding direction vectors to the
        appropriate head slices, at the positions selected by `token_scope`. This matches the
        paper's intervention point: after Att, before Q^h_l (the output projection). Position
        bookkeeping is delegated to the shared runtime (the lowest active layer opens the pass).

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

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        active = sorted(self._active_layer_ids)
        if not active:
            return hooks

        opener = active[0]
        for layer_id in active:
            hooks["pre"].append({
                "module": self._oproj_names[layer_id],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=layer_id,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=(layer_id == opener),
                ),
            })

        return hooks

    def reset(self):
        """Reset internal state between generation calls."""
        self._gate.reset()
        if self._runtime._prompt_lens is not None:
            self._runtime.reset(self._runtime._prompt_lens)
