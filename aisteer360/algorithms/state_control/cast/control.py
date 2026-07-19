"""CAST control: conditional activation steering, composed from `_common` components."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.utils.tokenization import infer_attention_mask_from_ids
from aisteer360.algorithms.state_control.base import StateControl
from aisteer360.algorithms.state_control._common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from aisteer360.algorithms.state_control._common.gates import (
    AlwaysOpenGate,
    CacheOnceGate,
    MultiKeyThresholdGate,
    ProjectedCosineScorer,
)
from aisteer360.algorithms.state_control._common.hook_utils import get_model_layer_list
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.selectors import ConditionPointSelector
from aisteer360.algorithms.state_control._common.selectors.utils.layer_heuristics import late_third
from aisteer360.algorithms.state_control._common.specs import Comparator, CompMode, VectorTrainSpec
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms import (
    AdditiveTransform,
    NormPreservingTransform,
    resolve_transform_slot,
)
from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform

from .args import CASTArgs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionPointConfig:
    """Fully-resolved condition point produced once in `steer()`.

    Attributes:
        layer_ids: Condition layer ids (0-based), empty when unconditional.
        threshold: Gate threshold, or None when unconditional.
        comparator: Gate comparator.
        comparison_mode: Runtime token aggregation for condition scoring ("mean" or "last").
        enabled: Whether conditional gating is active. When False, behavior steering is always on.
    """

    layer_ids: frozenset[int]
    threshold: float | None
    comparator: Comparator
    comparison_mode: CompMode
    enabled: bool


@dataclass(frozen=True)
class CASTDecision:
    """Diagnostics snapshot of the most recent condition decision.

    Attributes:
        scores: Per-condition-layer score for the first logical row (single-prompt convenience).
        scores_per_row: Per-condition-layer scores for every logical row.
        threshold: The gate threshold in effect.
        comparator: The gate comparator in effect.
        open_per_row: Whether the gate opened, per logical row.
    """

    scores: dict[int, float]
    scores_per_row: dict[int, tuple[float, ...]]
    threshold: float | None
    comparator: Comparator | None
    open_per_row: tuple[bool, ...]

    @property
    def is_open(self) -> bool:
        """True if the gate opened for the first logical row (single-prompt convenience)."""
        return bool(self.open_per_row and self.open_per_row[0])


def _make_estimator(spec: VectorTrainSpec):
    """Dispatch a fit spec to its estimator.

    Raises:
        ValueError: If `method == "mean_diff"` is combined with `accumulate == "suffix-only"`,
            which the mean-difference estimator does not support.
    """
    if spec.method == "mean_diff":
        if spec.accumulate == "suffix-only":
            raise ValueError(
                "method='mean_diff' does not support accumulate='suffix-only'; "
                "use accumulate='all' or 'last_token', or method='pca_pairwise'/'pca_center'."
            )
        return MeanDifferenceEstimator()
    return ContrastiveDirectionEstimator()


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

    CAST enables selective control of LLM behavior by conditionally applying activation steering
    based on input context. It operates in two phases:

    1. **Condition Detection**: Scores hidden-state activation patterns at the condition layer(s)
       against a learned condition direction to detect whether the prompt matches the target
       condition.

    2. **Conditional Behavior Modification**: When the condition is met, applies a behavior
       transform to hidden states at the behavior layers.

    The control is a thin recipe over the `_common` component families — everything at hook time
    runs through the shared `TransformHookRuntime`:

    - `ContrastiveDirectionEstimator` / `MeanDifferenceEstimator`: learn per-layer direction
      vectors from contrastive text pairs.
    - `ConditionPointSelector`: grid-searches the (layer, threshold, comparator) that best
      separates positive from negative calibration examples.
    - `ProjectedCosineScorer`: the runtime condition scorer — pad-aware aggregation of prompt
      hidden states ("mean" or "last"), scored per row via projected cosine similarity.
    - `CacheOnceGate(MultiKeyThresholdGate)`: row-vectorized gating. Each prompt in a batch is
      gated independently; beam-expanded rows of one prompt share that prompt's decision; the
      decision freezes after the prefill pass (the runtime stops condition scoring once the gate
      reports ready).
    - The behavior transform: `AdditiveTransform` (scaled direction addition, optionally wrapped
      in `NormPreservingTransform`) by default, or any `BaseTransform` supplied via
      `behavior_transform` (e.g. `DirectionalAblationTransform` for conditional ablation).

    Layer convention. Behavior directions are estimated at the *output* of layer l
    (`hidden_states[l+1]`) and applied at the *input* of layer l (the output of layer l-1) — the
    runtime is constructed with `hook_point="layer_input"`, a deliberate one-layer skew matching
    the CAST reference implementation. Condition directions are estimated by default at the
    *input* of layer l (`VectorTrainSpec(location="layer_input")` in `CASTArgs.condition_fit`),
    the same boundary the `ConditionPointSelector` calibrates on and the runtime condition
    pre-hook scores, so condition fit, calibration, and runtime are aligned.

    Timing. Within the prefill pass, hooks fire in layer order: a behavior layer *below* the
    condition layer sees a still-closed gate (no evidence yet), while a behavior layer *above* it
    sees the decided gate. When the calibrated condition layer sits above the behavior layers,
    prompt tokens therefore pass the behavior layers unsteered and only decode steps are steered —
    faithful to the reference, and why conditional steering is gentler than unconditional. Token
    scope composes with this: `"all"` permits prompt-token steering wherever the gate is already
    decided during prefill (the reference behavior); `"after_prompt"` restricts steering to
    generated tokens regardless of layer order.

    Batching is native: `supports_batching = True`. Row-vectorized gates mean one batched
    `generate` call gates and steers each prompt exactly as separate calls would. One in-flight
    generation is supported per control instance (gate and runtime state are per-instance,
    cleared by `reset()`).

    Reference:

    - "Programming Refusal with Conditional Activation Steering"
    Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin,
    Manish Nagireddy, Amit Dhurandhar
    [https://arxiv.org/abs/2409.05907](https://arxiv.org/abs/2409.05907)
    """

    Args = CASTArgs
    supports_batching = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # populated in steer()
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self._layer_names: list[str] = []
        self._behavior_layer_ids: list[int] = []
        self._cond_config: ConditionPointConfig | None = None
        self._transform: BaseTransform | None = None
        self._scorer: ProjectedCosineScorer | None = None
        self._gate: CacheOnceGate | AlwaysOpenGate = AlwaysOpenGate()
        self._threshold_gate: MultiKeyThresholdGate | None = None  # inner gate, for diagnostics
        self._runtime = TransformHookRuntime(hook_point="layer_input")

    @property
    def latest_decision(self) -> CASTDecision | None:
        """The most recent condition decision, or None before the condition has been evaluated.

        Assembled on demand from the gate's retained evidence; cleared by `reset()` at the start
        of the next generation.
        """
        inner = self._threshold_gate
        if inner is None or not self._gate.is_ready():
            return None
        evidence = inner.evidence()
        if not evidence:
            return None
        open_rows = self._gate.open_rows()
        return CASTDecision(
            scores={lid: float(rows[0]) for lid, rows in evidence.items()},
            scores_per_row={lid: tuple(float(x) for x in rows) for lid, rows in evidence.items()},
            threshold=inner.threshold,
            comparator=inner.comparator,
            open_per_row=tuple(bool(x) for x in open_rows.tolist()),
        )

    @property
    def condition_point(self) -> dict | None:
        """The resolved condition configuration, or None when the control is unconditional.

        Populated by `steer()` from either the auto-search or the caller-supplied
        `condition_layer_ids` / `condition_vector_threshold` / `condition_comparator_threshold_is`.
        """
        cfg = self._cond_config
        if cfg is None or not cfg.enabled:
            return None
        return {
            "layer_ids": sorted(cfg.layer_ids),
            "threshold": cfg.threshold,
            "comparator": cfg.comparator,
            "comparison_mode": cfg.comparison_mode,
        }

    def reset(self):
        """Reset gate and runtime position/prefill state between generation calls."""
        self._gate.reset(max(self._runtime.num_logical_rows, 1))
        if self._runtime._prompt_lens is not None:
            self._runtime.reset(self._runtime._prompt_lens, self._runtime._prompt_mask)

    def steer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **__,
    ) -> PreTrainedModel:
        """Initialize CAST by fitting artifacts and assembling the runtime components.

        Fits (or clones) the behavior and condition vectors, resolves the condition point
        (auto-search or manual), and builds the scorer, gate, and behavior transform that the
        shared runtime will drive at generation time.

        Args:
            model: The base language model to be steered.
            tokenizer: Tokenizer for encoding training data. If None, attempts to retrieve from
                model attributes.

        Returns:
            The input model, unchanged.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        device = next(model.parameters()).device
        _, layer_names = get_model_layer_list(model)
        self._layer_names = layer_names
        num_layers = len(layer_names)

        # clone a caller-supplied vector so the in-place .to() below never mutates it
        behavior_vec = self.behavior_vector.clone() if self.behavior_vector is not None else None
        if behavior_vec is None and self.behavior_data is not None:
            estimator = _make_estimator(self.behavior_fit)
            behavior_vec = estimator.fit(
                model, tokenizer, data=self.behavior_data, spec=self.behavior_fit
            )
        if behavior_vec is not None:
            behavior_vec = behavior_vec.to(device, dtype=model.dtype)

        # fit condition vector if needed (same clone-if-caller-supplied rule as behavior)
        condition_vec = self.condition_vector.clone() if self.condition_vector is not None else None
        has_condition = condition_vec is not None or self.condition_data is not None
        if has_condition and condition_vec is None and self.condition_data is not None:
            estimator = _make_estimator(self.condition_fit)
            condition_vec = estimator.fit(
                model, tokenizer, data=self.condition_data, spec=self.condition_fit
            )
            condition_vec = condition_vec.to(device, dtype=model.dtype)

        # choose behavior layers
        behavior_layer_ids = self.behavior_layer_ids
        if behavior_layer_ids is None:
            behavior_layer_ids = late_third(num_layers)
        self._behavior_layer_ids = sorted(set(int(lid) for lid in behavior_layer_ids))

        for lid in self._behavior_layer_ids:
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
                    comparison_mode=self.condition_threshold_comparison_mode,
                )
                condition_layer_ids = [result.layer_id]
                condition_threshold = result.threshold
                condition_comparator = result.comparator

        condition_layer_set = set(int(lid) for lid in (condition_layer_ids or []))
        for lid in condition_layer_set:
            if not 0 <= lid < num_layers:
                raise ValueError(f"condition_layer_id {lid} out of range [0, {num_layers}).")

        # resolve conditional vs unconditional mode; a partial config must not silently open the gate
        conditional = bool(condition_layer_set) and condition_threshold is not None
        if conditional and condition_vec is None:
            raise ValueError("Conditional CAST requires a condition vector.")

        self._cond_config = ConditionPointConfig(
            layer_ids=frozenset(condition_layer_set) if conditional else frozenset(),
            threshold=condition_threshold if conditional else None,
            comparator=condition_comparator,
            comparison_mode=self.condition_threshold_comparison_mode,
            enabled=conditional,
        )

        # assemble scorer + gate for the condition path
        if conditional:
            missing = [lid for lid in condition_layer_set if lid not in condition_vec.directions]
            if missing:
                raise ValueError(f"Condition vector has no direction for condition layer(s) {missing}.")
            self._scorer = ProjectedCosineScorer(
                {lid: condition_vec.directions[lid] for lid in condition_layer_set},
                comparison_mode=self.condition_threshold_comparison_mode,
            )
            self._threshold_gate = MultiKeyThresholdGate(
                threshold=condition_threshold,
                comparator=condition_comparator,
                expected_keys=set(condition_layer_set),
                aggregate="any",
            )
            self._gate = CacheOnceGate(self._threshold_gate)
        else:
            self._scorer = None
            self._threshold_gate = None
            self._gate = AlwaysOpenGate()

        # build behavior transform: pluggable slot (artifact-carrier) or default additive path
        if self.behavior_transform is not None:
            self._transform = resolve_transform_slot(
                self.behavior_transform, model, tokenizer, self._behavior_layer_ids
            )
        else:
            directions: dict[int, torch.Tensor] = {}
            if behavior_vec is not None:
                for lid in self._behavior_layer_ids:
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
        attention_mask: torch.Tensor | None = None,
        **__,
    ) -> dict[str, list]:
        """Emit condition (read-only) and behavior pre-hooks for the current generation.

        Condition hooks are appended before behavior hooks so that, at a shared layer, the gate
        update precedes the transform application (registration order = execution order for
        same-module hooks of the same phase).

        Args:
            input_ids: Input token IDs.
            runtime_kwargs: Runtime parameters (currently unused).
            attention_mask: The prompt attention mask matching `input_ids` (forwarded by the
                pipeline). Used as the pad-aware condition-scoring mask so runtime condition
                scores align with the selector's calibration mask. When None, the mask is inferred
                from the prompt ids (leading and trailing pad runs only) rather than by token
                identity, so an interior pad==eos token (e.g. ChatML `<|im_end|>`) is not wrongly
                masked.

        Returns:
            Hook specifications with "pre", "forward", "backward" keys.
        """
        ids = input_ids if isinstance(input_ids, torch.Tensor) else input_ids["input_ids"]
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) if self.tokenizer else None
        prompt_lens = compute_prompt_lens(ids, pad_id)
        batch_size = ids.size(0)

        # pad-aware condition-scoring mask: prefer the pipeline-supplied attention mask (identical
        # to the selector's calibration mask); otherwise infer one from the prompt ids without
        # masking interior pad==eos tokens
        if attention_mask is not None:
            am = attention_mask if isinstance(attention_mask, torch.Tensor) else torch.as_tensor(attention_mask)
            if am.ndim == 1:
                am = am.unsqueeze(0)
            prompt_mask = am.to(torch.bool)
        elif pad_id is not None:
            prompt_mask = infer_attention_mask_from_ids(ids, pad_id).to(torch.bool)
        else:
            prompt_mask = None

        self._gate.reset(batch_size)
        self._runtime.reset(prompt_lens, prompt_mask)

        cfg = self._cond_config
        condition_layers = sorted(cfg.layer_ids) if (cfg is not None and cfg.enabled) else []
        all_layers = condition_layers + self._behavior_layer_ids
        opener = min(all_layers) if all_layers else None
        # exactly one hook may open each pass; at a shared opener layer the condition hook
        # registers (and therefore fires) first, so it takes the opener role
        behavior_opener = opener if opener not in condition_layers else None

        hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}

        # condition hooks first (so gate.update precedes transform at a shared layer)
        for lid in condition_layers:
            hooks["pre"].append({
                "module": self._layer_names[lid],
                "hook_func": self._runtime.build_condition_hook(
                    layer_id=lid,
                    scorer=self._scorer,
                    gate=self._gate,
                    is_pass_opener=(lid == opener),
                ),
            })

        for lid in self._behavior_layer_ids:
            hooks["pre"].append({
                "module": self._layer_names[lid],
                "hook_func": self._runtime.build_behavior_hook(
                    layer_id=lid,
                    transform=self._transform,
                    gate=self._gate,
                    token_scope=self.token_scope,
                    last_k=self.last_k,
                    from_position=self.from_position,
                    is_pass_opener=(lid == behavior_opener),
                ),
            })

        return hooks

    def cleanup(self) -> None:
        """Drop references to fitted artifacts and runtime state."""
        self._transform = None
        self._scorer = None
        self.model = None
        self._runtime = TransformHookRuntime(hook_point="layer_input")  # drop stored prompt state
