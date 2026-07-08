"""CAST control: conditional activation steering via composable components."""
from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.estimators import ContrastiveDirectionEstimator
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate, CacheOnceGate, MultiKeyThresholdGate
from aisteer360.algorithms.state_control._common.gates.scores import projected_cosine_similarity
from aisteer360.algorithms.state_control._common.hook_utils import (
    extract_hidden_states,
    get_model_layer_list,
    replace_hidden_states,
)
from aisteer360.algorithms.state_control._common.selectors import ConditionPointSelector
from aisteer360.algorithms.state_control._common.selectors.layer_heuristics import late_third
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens, make_token_mask
from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform, NormPreservingTransform

from .args import CASTArgs

logger = logging.getLogger(__name__)


def _squeeze_direction(d: torch.Tensor) -> torch.Tensor:
    """Squeeze a [1, H] direction to [H] for scalar operations.

    Handles both 1D [H] and 2D [K, D] tensors. For K=1, squeezes to [D].
    For K>1, returns as-is (caller must handle).
    """
    if d.ndim == 2 and d.shape[0] == 1:
        return d.squeeze(0)
    return d


class CAST(StateControl):
    """Conditional Activation Steering (CAST).

    CAST enables selective control of LLM behavior by conditionally applying
    activation steering based on input context. It operates in two phases:

    1. **Condition Detection**: Analyzes hidden state activation patterns at
       specified layers during inference to detect if the input matches target
       conditions.

    2. **Conditional Behavior Modification**: When conditions are met, applies
       steering vectors to hidden states at designated behavior layers.

    It composes:

    - ContrastiveDirectionEstimator: learns per-layer direction vectors from contrastive text pairs via PCA.
    - ConditionPointSelector: grid-searches for the layer, threshold, and comparator that best separate positive from negative examples using projected cosine similarity.
    - AdditiveTransform: adds scaled direction vectors to hidden states at designated layers.
    - MultiKeyThresholdGate (wrapped by CacheOnceGate): opens when projected scores cross a threshold,
          then caches the decision for subsequent tokens.

    Reference:

    - "Programming Refusal with Conditional Activation Steering"
    Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, Amit Dhurandhar
    [https://arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
    """

    Args = CASTArgs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._layer_names: list[str] = []
        self._behavior_layer_set: set[int] = set()
        self._condition_layer_set: set[int] = set()
        self._gate: AlwaysOpenGate | CacheOnceGate = AlwaysOpenGate()
        self._transform: AdditiveTransform | NormPreservingTransform | None = None
        self._cond_projectors: dict[int, torch.Tensor] = {}
        self._cond_mode: str = "mean"

        # per-generation state
        self._forward_calls: dict[int, int] = defaultdict(int)
        self._prompt_lens: torch.LongTensor | None = None

    def reset(self):
        """Reset internal state tracking between generation calls."""
        self._forward_calls = defaultdict(int)
        self._prompt_lens = None
        self._gate.reset()

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize CAST by configuring condition detection and behavior modification.

        Sets up steering vectors (from pre-computed artifacts or by training on
        data), condition projectors, and layer-specific parameters for
        conditional activation steering.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data. If None, attempts
                to retrieve from model attributes.

        Returns:
            The input model, unchanged.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # fit behavior vector if needed
        behavior_vec = self.behavior_vector
        if behavior_vec is None and self.behavior_data is not None:
            estimator = ContrastiveDirectionEstimator()
            behavior_vec = estimator.fit(
                model, tokenizer, data=self.behavior_data, spec=self.behavior_fit
            )
        if behavior_vec is not None:
            behavior_vec.to(device, dtype=model.dtype)

        # fit condition vector if needed
        condition_vec = self.condition_vector
        has_condition = condition_vec is not None or self.condition_data is not None
        if has_condition and condition_vec is None and self.condition_data is not None:
            estimator = ContrastiveDirectionEstimator()
            condition_vec = estimator.fit(
                model, tokenizer, data=self.condition_data, spec=self.condition_fit
            )
            condition_vec.to(device, dtype=model.dtype)

        # choose behavior layers
        behavior_layer_ids = self.behavior_layer_ids
        if behavior_layer_ids is None:
            behavior_layer_ids = late_third(num_layers)
        self._behavior_layer_set = set(behavior_layer_ids)

        for lid in self._behavior_layer_set:
            if not 0 <= lid < num_layers:
                raise ValueError(f"behavior_layer_id {lid} out of range [0, {num_layers}).")

        # choose condition point
        condition_layer_ids = self.condition_layer_ids
        condition_threshold = self.condition_vector_threshold
        condition_comparator = self.condition_comparator_threshold_is

        if has_condition and condition_vec is not None:
            if self.search.auto_find and condition_layer_ids is None and self.condition_data is not None:
                searcher = ConditionPointSelector()
                result = searcher.select(
                    model=model,
                    tokenizer=tokenizer,
                    condition_directions=condition_vec.directions,
                    data=self.condition_data,
                    fit_spec=self.condition_fit,
                    search_spec=self.search,
                )
                condition_layer_ids = [result.layer_id]
                condition_threshold = result.threshold
                condition_comparator = result.comparator

        self._condition_layer_set = set(condition_layer_ids or [])

        for lid in self._condition_layer_set:
            if not 0 <= lid < num_layers:
                raise ValueError(f"condition_layer_id {lid} out of range [0, {num_layers}).")

        # build gate
        if self._condition_layer_set and condition_threshold is not None:
            inner_gate = MultiKeyThresholdGate(
                threshold=condition_threshold,
                comparator=condition_comparator,
                expected_keys=self._condition_layer_set,
                aggregate="any",
            )
            self._gate = CacheOnceGate(inner_gate)
        else:
            self._gate = AlwaysOpenGate()

        # pre-compute condition projectors
        self._cond_projectors = {}
        if condition_vec is not None:
            for lid in self._condition_layer_set:
                if lid in condition_vec.directions:
                    c = _squeeze_direction(
                        condition_vec.directions[lid].to(device=device, dtype=model.dtype)
                    )
                    self._cond_projectors[lid] = torch.outer(c, c) / (c @ c + 1e-8)
        self._cond_mode = self.condition_threshold_comparison_mode

        # build transform
        directions: dict[int, torch.Tensor] = {}
        if behavior_vec is not None:
            for lid in self._behavior_layer_set:
                d = behavior_vec.directions.get(lid)
                if d is None:
                    continue
                d = _squeeze_direction(d)
                if self.use_explained_variance and behavior_vec.explained_variances:
                    scale = float(behavior_vec.explained_variances.get(lid, 1.0))
                    d = d * scale
                directions[lid] = d

        base_transform = AdditiveTransform(directions, strength=self.behavior_vector_strength)
        if self.use_ooi_preventive_normalization:
            self._transform = NormPreservingTransform(base_transform)
        else:
            self._transform = base_transform

        return model

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None,
        **__,
    ) -> dict[str, list]:
        """Create pre-forward hooks for conditional activation steering.

        Args:
            input_ids: Input token IDs.
            runtime_kwargs: Runtime parameters (currently unused).

        Returns:
            Hook specifications with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) if self.tokenizer else None
        self._prompt_lens = compute_prompt_lens(ids, pad_id)

        # hook every layer that is either a condition or behavior layer
        active_layers = self._condition_layer_set | self._behavior_layer_set
        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for lid, path in enumerate(self._layer_names):
            if lid in active_layers:
                hooks["pre"].append({
                    "module": path,
                    "hook_func": partial(self._pre_hook, layer_id=lid),
                })
        return hooks

    def _pre_hook(self, module, input_args, input_kwargs, layer_id: int):
        """Apply conditional activation steering as a pre-forward hook.

        Args:
            module: The layer module being hooked.
            input_args: Positional arguments to the forward pass.
            input_kwargs: Keyword arguments to the forward pass.
            layer_id: Index of the current layer.

        Returns:
            Tuple of potentially modified (input_args, input_kwargs).
        """
        hidden = extract_hidden_states(input_args, input_kwargs)
        if hidden is None:
            return input_args, input_kwargs

        self._forward_calls[layer_id] += 1
        is_first_call = self._forward_calls[layer_id] == 1
        _, T, _ = hidden.shape

        # condition evaluation (first pass only)
        if layer_id in self._condition_layer_set and is_first_call:
            projector = self._cond_projectors.get(layer_id)
            if projector is not None:
                if self._cond_mode == "mean":
                    agg = hidden.mean(dim=1)  # [B, H]
                else:
                    agg = hidden[:, -1, :]  # [B, H]
                # score per batch item (take first item for single-batch case)
                score = projected_cosine_similarity(agg[0], projector)
                self._gate.update(score, key=layer_id)

        # behavior application
        if layer_id in self._behavior_layer_set and self._gate.is_open():
            if is_first_call and not self.apply_behavior_on_first_call:
                pass  # skip first call
            else:
                mask = make_token_mask(
                    self.token_scope,
                    seq_len=T,
                    prompt_lens=self._prompt_lens.to(hidden.device),
                    last_k=self.last_k,
                )
                hidden = self._transform.apply(
                    hidden, layer_id=layer_id, token_mask=mask
                )

        return replace_hidden_states(input_args, input_kwargs, hidden)
