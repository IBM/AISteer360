"""Directional Ablation control: projects a learned direction out of the residual stream."""
from __future__ import annotations

import logging

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control._common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import FractionalDepthSelector
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import (
    DirectionalAblationTransform,
    NormPreservingTransform,
)
from aisteer360.algorithms.state_control.base import StateControl

from .args import DirectionalAblationArgs

logger = logging.getLogger(__name__)


class DirectionalAblation(StateControl):
    """Directional Ablation (feature removal via projection).

    Removes a learned feature direction from the residual stream at one or more layers during
    generation, `h' = h - alpha * (d_hat^T h) d_hat` at masked positions. This is the
    "abliteration" technique of Arditi et al.: learn a direction (difference-in-means over
    contrastive data, exactly as CAA) and project it out.

    The method operates in two phases:

    1. **Training (offline)**: identical to CAA. Extract residual activations for contrastive
       pairs and take the mean difference (or PCA of paired differences) as the feature direction.
       A precomputed direction (or an orthonormal subspace, `K > 1`) may be supplied directly.

    2. **Inference (online)**: at each target layer's output, project the direction out of the
       residual stream at masked positions. `alpha = 1.0` fully removes the component
       (`h'.d_hat == 0`); `alpha < 1.0` gives graded partial suppression.

    Ablation is a projection (idempotent at `alpha=1`, norm-reducing). 
    
    It can compose with the alignment-adaptive gate (`AlignmentAdaptiveTransform`) to ablate only 
    where the feature is present.

    Reference:

    - "Refusal in Language Models Is Mediated by a Single Direction"
    Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda
    [https://arxiv.org/abs/2406.11717](https://arxiv.org/abs/2406.11717)
    """

    Args = DirectionalAblationArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self._steering_vector: SteeringVector | None = None
        self._transform = None
        self._layer_names: list[str] = []
        self._layer_ids: list[int] = []
        self._gate = AlwaysOpenGate()
        self._pad_token_id: int | None = None
        self._runtime = TransformHookRuntime(hook_point="layer_output")

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Fit or load the feature direction and resolve the layers to ablate.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data (when fitting the direction).

        Returns:
            The input model, unchanged.

        Raises:
            ValueError: If no target layer has a direction in the steering vector.
        """
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # resolve the direction (identical to CAA)
        if self.steering_vector is not None:
            source = self.steering_vector
        else:
            if self.train_spec.method == "pca_pairwise":
                estimator = ContrastiveDirectionEstimator()
            else:
                estimator = MeanDifferenceEstimator()
            source = estimator.fit(model, tokenizer, data=self.data, spec=self.train_spec)

        # copy directions into a fresh vector (never mutate a caller-supplied steering_vector in
        # place; a precomputed direction may be reused across controls with different filters)
        start, end = self.layer_range if self.layer_range is not None else (None, None)
        directions = {
            lid: d.clone().to(device=device, dtype=model.dtype)
            for lid, d in source.directions.items()
            if self.layer_range is None or start <= lid < end
        }
        sv = SteeringVector(
            model_type=source.model_type,
            directions=directions,
            explained_variances=source.explained_variances,
        )
        self._steering_vector = sv

        # resolve target layers
        if self.layer_ids is not None:
            target_ids = sorted(set(self.layer_ids))
        else:
            # heuristic: single layer at ~40% depth (matches CAA)
            target_ids = [FractionalDepthSelector(fraction=0.4).select(num_layers=num_layers)]

        self._layer_ids = [lid for lid in target_ids if lid in sv.directions]
        if not self._layer_ids:
            raise ValueError(
                f"No target layer has a direction in the steering vector "
                f"(requested {target_ids}, available {sorted(sv.directions.keys())})."
            )

        # build the transform
        transform = DirectionalAblationTransform(sv.directions, alpha=self.alpha)
        if self.use_norm_preservation:
            transform = NormPreservingTransform(transform)
        self._transform = transform

        # store tokenizer info for hook generation
        self._pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None

        return model

    def get_hooks(
        self,
        input_ids: torch.Tensor,
        runtime_kwargs: dict | None = None,
        **__,
    ) -> dict[str, list]:
        """Create a forward hook on each target layer's output to ablate the residual stream.

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

        # the lowest hooked layer opens the pass and advances the shared KV offset once per forward pass
        opener = min(self._layer_ids) if self._layer_ids else None

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        for layer_id in self._layer_ids:
            hooks["forward"].append({
                "module": self._layer_names[layer_id],
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
