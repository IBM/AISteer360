from __future__ import annotations

from collections import defaultdict
from functools import partial
from typing import Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control.cast.args import CASTArgs
from aisteer360.algorithms.state_control.cast.utils.layer import (
    LayerArgs,
    LayerControlParams,
)
from aisteer360.algorithms.state_control.cast.utils.steering_vector import (
    SteeringVector,
)


class CAST(StateControl):
    """
    Implementation of CAST (Conditional Activation Steering) from Lee et al., 2024.

    CAST enables selective control of LLM behavior by conditionally applying activation steering based on input context,
    allowing fine-grained control without affecting responses to non-targeted content.

    The method operates in two phases:

    1. **Condition Detection**: Analyzes hidden state activation patterns at specified layers during inference to detect
        if the input matches target conditions. This is done by projecting hidden states onto a condition subspace and
        computing similarity scores against a threshold.

    2. **Conditional Behavior Modification**: When conditions are met, applies steering vectors to hidden states at
        designated behavior layers. This selectively modifies the model's internal representations to produce desired
        behavioral changes while preserving normal functionality for non-matching inputs.

    Args:
        condition_vector (SteeringVector, optional): Steering vector defining the condition subspace for detecting
            target input patterns. Defaults to None.
        behavior_vector (SteeringVector, optional): Steering vector applied to modify behavior when conditions are met.
            Defaults to None.
        condition_layer_ids (list[int], optional): Layer indices where condition detection occurs. Defaults to None.
        behavior_layer_ids (list[int], optional): Layer indices where behavior modification is applied. Defaults to None.
        condition_vector_threshold (float, optional): Similarity threshold for condition detection. Higher values
            require stronger pattern matches. Defaults to 0.5.
        behavior_vector_strength (float, optional): Scaling factor for the behavior steering vector. Controls the
            intensity of behavioral modification. Defaults to 1.0.
        condition_comparator_threshold_is (str, optional): Comparison mode for threshold ('larger' or 'smaller').
            Determines if condition is met when similarity is above or below threshold. Defaults to 'larger'.
        condition_threshold_comparison_mode (str, optional): How to aggregate hidden states for comparison ('mean'
            or 'last'). Defaults to 'mean'.
        apply_behavior_on_first_call (bool, optional): Whether to apply behavior steering on the first forward pass.
            Defaults to True.
        use_ooi_preventive_normalization (bool, optional): Apply out-of-distribution preventive normalization to
            maintain hidden state magnitudes. Defaults to False.
        use_explained_variance (bool, optional): Scale steering vectors by their explained variance for adaptive
            layer-wise control. Defaults to False.

    Reference:

    - "Programming Refusal with Conditional Activation Steering"
    Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, Amit Dhurandhar
    [https://arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
    """

    Args = CASTArgs

    # placeholders
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    device: torch.device | str | None = None

    # layers list reference
    _layers: list | None = None
    _layers_names: list | None = None
    _layers_states: dict[int, LayerArgs] | None = None

    # Boolean lists for condition and behavior layers
    _condition_layers: dict[int, bool] | None = None
    _behavior_layers: dict[int, bool] | None = None

    # Logic flags
    _condition_met: dict[int, bool] = defaultdict(bool)
    _forward_calls: dict[int, int] = defaultdict(int)

    # condition similarity record
    _condition_similarities: dict = defaultdict(lambda: defaultdict(float))

    def reset(self):
        """Reset internal state tracking between generation calls.

        Clears condition detection flags, forward call counters, and similarity scores.
        """
        self._condition_met = defaultdict(bool)
        self._forward_calls = defaultdict(int)
        self._condition_similarities = defaultdict(lambda: defaultdict(float))

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | None = None,
        **__
    ) -> PreTrainedModel:
        """Initialization by configuring condition detection and behavior modification layers.

        Sets up steering vectors, condition projectors, and layer-specific parameters for conditional activation
        steering. Pre-computes projection matrices and behavior vectors.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer (currently unused but maintained
                for API consistency). If None, attempts to retrieve from model attributes.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model, unchanged.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device

        self._setup(self.model)

        return model

    def get_hooks(
            self,
            input_ids: torch.Tensor,
            runtime_kwargs: dict | None,
            **__,
    ) -> dict[str, list]:
        """Create pre-forward hooks for conditional activation steering.

        Generates hook specifications for all model layers that will conditionally detect patterns and apply behavior
        modifications during the forward pass.

        Args:
            input_ids (torch.Tensor): Input token IDs (unused but required by interface).
            runtime_kwargs (dict | None): Runtime parameters (currently unused).
            **__: Additional arguments (unused).

        Returns:
            dict[str, list]: Hook specifications with "pre", "forward", "backward" keys.
                Only "pre" hooks are populated with CAST steering logic.
        """

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for layer_id, layer_name in enumerate(self._layers_names):
            hooks["pre"].append(
                {
                    "module": layer_name,  # "model.layers.0"
                    "hook_func": partial(
                        self._cast_pre_hook,
                        layer_id=layer_id,
                    ),
                }
            )

        return hooks

    def get_model_layer_list(self, model: PreTrainedModel) -> list:
        """Extract the list of transformer layers from the model.

        Args:
            model (PreTrainedModel): Model to extract layers from.

        Returns:

            List of layers for given model
            List of layers module name prefix for given model
        """
        layers = []
        layers_names = []

        model_layers = None
        model_layers_prefix = ''

        if hasattr(model, "model"):  # mistral-, llama-, gemma-like models
            model_layers = model.model.layers
            model_layers_prefix = "model.layers"
        elif hasattr(model, "transformer"):  # gpt2-like models
            model_layers = model.transformer.h
            model_layers_prefix = "transformer.h"
        else:
            raise ValueError(f"Don't know how to get layer list from model for {type(model)=}")

        for idx, layer in enumerate(model_layers):
            layers.append(layer)
            layers_names.append(f"{model_layers_prefix}.{idx}")

        return layers, layers_names

    def _setup(self, model: PreTrainedModel):
        """Configure all CAST internals for the given model.

        Pre-computes steering vectors, condition projectors, and layer configurations to minimize runtime overhead during generation.

        Process:

        1. Identifies condition and behavior layers from configuration
        2. Computes condition projection matrices for detection layers
        3. Prepares scaled behavior vectors for modification layers
        4. Stores layer-specific parameters in _layer_states

        Args:
            model (PreTrainedModel): Model to configure CAST for.
        """
        self._layers, self._layers_names = self.get_model_layer_list(model)
        num_layers = len(self._layers)

        # Creating dicts for condition and behavior layers
        condition_layers = [False] * num_layers
        behavior_layers = [False] * num_layers

        if self.condition_vector is not None and self.condition_layer_ids is not None:
            for layer_id in self.condition_layer_ids:
                condition_layers[layer_id] = True

        if self.behavior_vector is not None:
            for layer_id in self.behavior_layer_ids:
                behavior_layers[layer_id] = True

        self._condition_layers = {i: v for i, v in enumerate(condition_layers)}
        self._behavior_layers = {i: v for i, v in enumerate(behavior_layers)}

        # Precompute behavior vectors and condition projectors
        condition_layer_ids_set = set(self.condition_layer_ids) if self.condition_layer_ids is not None else set()
        behavior_layer_ids_set = set(self.behavior_layer_ids)

        self._layer_states = {}

        for layer_id in range(num_layers):
            # layer = self._layers[layer_id]
            behavior_tensor = None
            if self.behavior_vector is not None:
                if layer_id in behavior_layer_ids_set:
                    if self.use_explained_variance:
                        behavior_direction = self._use_explained_variance_func(self.behavior_vector)
                    else:
                        behavior_direction = self.behavior_vector.directions[layer_id]

                    behavior_tensor = torch.tensor(self.behavior_vector_strength * behavior_direction, dtype=self.model.dtype).to(self.model.device)

            condition_projector = None
            if self.condition_vector is not None and layer_id in condition_layer_ids_set:
                condition_direction = self.condition_vector.directions[layer_id]
                if self.use_explained_variance:
                    condition_direction = self._use_explained_variance_func(self.condition_vector)
                else:
                    condition_direction = self.condition_vector.directions[layer_id]

                condition_tensor = torch.tensor(condition_direction, dtype=self.model.dtype).to(self.model.device)
                condition_projector = torch.ger(condition_tensor, condition_tensor) / torch.dot(condition_tensor, condition_tensor)

            layer_control_params = LayerControlParams()

            layer_args = LayerArgs(
                behavior_vector=behavior_tensor,
                condition_projector=condition_projector,
                threshold=self.condition_vector_threshold,
                use_ooi_preventive_normalization=self.use_ooi_preventive_normalization,
                apply_behavior_on_first_call=self.apply_behavior_on_first_call,
                condition_comparator_threshold_is=self.condition_comparator_threshold_is,
                condition_threshold_comparison_mode=self.condition_threshold_comparison_mode,
                params=layer_control_params,
            )

            self._layer_states[layer_id] = layer_args

    def _use_explained_variance_func(self, vector: SteeringVector, layer_id: int) -> np.ndarray:
        """Scale steering vector by its explained variance for adaptive control.

        This method scales the steering vector based on its explained variance,
        potentially adjusting its impact on different layers of the model.

        Args:
            vector (SteeringVector): Steering vector containing directions and variances.
            layer_id (int): Layer index to retrieve variance scaling for.

        Returns:
            np.ndarray: Direction vector scaled by explained variance.
        """

        if hasattr(vector, 'explained_variances'):
            variance_scale = vector.explained_variances.get(layer_id, 1)
            direction = vector.directions.get(layer_id, 1)
            direction = direction * variance_scale

        return direction

    def _cast_pre_hook(
        self,
        module,
        input_args: Tuple,
        input_kwargs: dict,
        layer_id: int,
    ):
        """Apply conditional activation steering as a pre-forward hook.

        Detect conditions and applies behavior modifications during the model's forward pass. Processes each layer
        independently based on its configuration.

        Process:

        1. Extract hidden states from arguments
        2. If condition layer: detect if input matches target pattern
        3. If behavior layer and conditions met: apply steering vector
        4. Optionally apply OOI normalization to prevent distribution shift

        Args:
            module: The layer module being hooked.
            input_args: Positional arguments to the forward pass.
            input_kwargs: Keyword arguments to the forward pass.
            layer_id (int): Index of the current layer.

        Returns:
            Tuple of potentially modified (input_args, input_kwargs).

        Raises:
            RuntimeError: If hidden states cannot be located.
        """
        hidden_states = input_args[0] if input_args else input_kwargs.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError("CAST: could not locate hidden states")

        self._forward_calls[layer_id] += 1
        batch_size, seq_length, hidden_dim = hidden_states.shape

        if self._condition_layers is None:
            # CASE 1 -> no steering
            is_condition_layer = False
            is_behavior_layer = False
        else:
            # CASE 2 -> steering
            is_condition_layer = self._condition_layers[layer_id]
            is_behavior_layer = self._behavior_layers[layer_id]

        original_norm = hidden_states.norm(dim=-1, keepdim=True)

        if is_condition_layer:
            self._process_single_condition(hidden_states[0], layer_id)

        if is_behavior_layer:
            self._apply_single_behavior(hidden_states, layer_id)

        if self.use_ooi_preventive_normalization and is_behavior_layer:
            hidden_states = self._apply_ooi_normalization(hidden_states, original_norm)

        if input_args:
            input_list = list(input_args)
            input_list[0] = hidden_states
            my_input_args = tuple(input_list)
        else:
            my_input_args = input_args
            input_kwargs["hidden_states"] = hidden_states

        return my_input_args, input_kwargs

    def _process_single_condition(self, hidden_state, layer_id: int):
        """Detect if input matches target condition pattern.

        Projects hidden states onto condition subspace and compares similarity against threshold to determine if
        steering should be activated.

        Process:

        1. Aggregate hidden states (mean or last token based on config)
        2. Project onto condition subspace using precomputed projector
        3. Compute cosine similarity between original and projected
        4. Compare against threshold with specified comparator

        Args:
            hidden_state: Hidden state tensor to analyze [seq_len, hidden_dim].
            layer_id (int): Current layer index.
        """
        layer_args = self._layer_states[layer_id]

        if not self._condition_met[0] and self._forward_calls[layer_id] == 1:
            if layer_args.condition_threshold_comparison_mode == "mean":
                hidden_state = hidden_state.mean(dim=0)
            elif layer_args.condition_threshold_comparison_mode == "last":
                hidden_state = hidden_state[-1, :]

            projected_hidden_state = torch.tanh(torch.matmul(layer_args.condition_projector, hidden_state))
            condition_similarity = self._compute_similarity(hidden_state, projected_hidden_state)
            self._condition_similarities[0][layer_id] = condition_similarity

            if layer_args.condition_comparator_threshold_is == "smaller":
                condition_met = (condition_similarity >= layer_args.threshold)
            elif layer_args.condition_comparator_threshold_is == "larger":
                condition_met = (condition_similarity < layer_args.threshold)
            else:
                raise ValueError(f"invalid {layer_args.condition_comparator_threshold_is}")

            self._condition_met[0] = condition_met

            print(f"layer {layer_id}:  similarity: {condition_similarity} "
                  f"threshold: {layer_args.threshold} "
                  f"condition comparator threshold '{layer_args.condition_comparator_threshold_is}' -- "
                  f"Condition Met: {condition_met}")

    def _apply_single_behavior(self, hidden_states, layer_id: int):
        """Apply behavior steering vector when conditions are met.

        Modifies hidden states by adding scaled steering vectors to shift model behavior toward desired outputs.

        Args:
            hidden_states: Hidden states to modify [batch, seq_len, hidden_dim].
            layer_id (int): Current layer index.
        """
        layer_args = self._layer_states[layer_id]

        should_apply = not any(self._condition_layers.values()) or self._condition_met[0]

        # print(f"Should Apply Behavior: {should_apply}")

        if should_apply:
            control = layer_args.behavior_vector.to(dtype=hidden_states.dtype)
            if self._forward_calls[layer_id] == 1:
                if layer_args.apply_behavior_on_first_call:
                    hidden_states[0] = layer_args.params.operator(hidden_states[0], control)
                else:
                    print("apply_behavior_on_first_call is False, skipping behavior vector application")
            else:
                hidden_states[0] = layer_args.params.operator(hidden_states[0], control)
                # print(f"{layer_id=}: Applying behavior vector to all tokens")

    def _compute_similarity(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Compute the cosine similarity between two tensors.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            The cosine similarity as a float.
        """
        cossim = torch.dot(x.flatten(), y.flatten()) / (torch.norm(x) * torch.norm(y))
        return float(cossim.item())

    def _apply_ooi_normalization(self, hidden_states, original_norm):
        """Apply out-of-distribution preventive normalization.

        Prevents hidden states from drifting too far from original distribution by rescaling to maintain norm magnitudes after steering.

        Args:
            hidden_states: Modified hidden states to normalize.
            original_norm: Original norm before modifications.

        Returns:
            torch.Tensor: Normalized hidden states.

        Raises:
            ValueError: If NaN or Inf detected in hidden states.
        """
        new_norm = hidden_states.norm(dim=-1, keepdim=True)
        max_ratio = (new_norm / original_norm).max().item()
        has_nan_inf = torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any()

        if has_nan_inf:
            # NaN propagates, decided to raise instead of just applying norm as in original code.
            raise ValueError(f"NaN: {torch.isnan(hidden_states).any()} or Inf: {torch.isinf(hidden_states).any()} dectected in hidden_states")

        if max_ratio > 1 or has_nan_inf:
            print(f"Applying OOI preventive normalization. Max_ratio was {max_ratio}")
            hidden_states = hidden_states * (original_norm / new_norm)
        else:
            print(f"No OOI preventive normalization. Max_ratio was {max_ratio}")

        return hidden_states
