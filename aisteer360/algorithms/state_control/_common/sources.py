"""Artifact sources: recipes that resolve to a `SteeringVector` for a given model.

This module provides the `ArtifactSource` protocol and `ContrastiveFit`. A transform holds either 
a concrete artifact (a `SteeringVector` or a per-layer directions mapping) or a source; the adapter 
resolves the source at `steer()` time and binds the transform to the resulting vector. 
`resolve` returns a defensive clone, and the underlying fit is memoized per model.
"""
from __future__ import annotations

import warnings
import weakref
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.state_control._common.estimators import (
    ContrastiveDirectionEstimator,
    MeanDifferenceEstimator,
)
from aisteer360.algorithms.state_control._common.estimators.base import BaseEstimator
from aisteer360.algorithms.state_control._common.specs import (
    ContrastivePairs,
    HiddenStateLocation,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.utils.rendering import PromptFormat


@runtime_checkable
class ArtifactSource(Protocol):
    """A recipe for obtaining a steering artifact for a specific model.

    Implementations MUST return a defensive clone from `resolve` (callers may move/mutate their
    copy) and SHOULD memoize the underlying fit per model so repeated resolves against one model
    (e.g., a parameter sweep) fit only once.
    """

    def resolve(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> SteeringVector:
        """Return the steering artifact for this model (a fresh clone each call)."""
        ...


@dataclass
class ContrastiveFit:
    """A fit recipe: contrastive pairs plus how to extract a per-layer direction.

    The five spec fields (`method`, `accumulate`, `batch_size`, `prompt_format`, `location`) mirror
    `VectorTrainSpec` and drive the built-in estimators: `"mean_diff"` dispatches to
    `MeanDifferenceEstimator`, everything else to `ContrastiveDirectionEstimator`. When a custom
    `estimator` is supplied, the spec fields are ignored (a warning is emitted) and fitting delegates
    to `estimator.fit(model, tokenizer, data=<coerced pairs>, **(estimator_kwargs or {}))`.

    The fitted master vector is memoized in a single weakref slot keyed by model identity: the same
    model resolved repeatedly fits once; a different model refits; alternating models (A→B→A) refit
    on each switch. Every `resolve` returns an independent clone, so a consumer may freely
    `.to(...)`/mutate its copy without touching the master.

    Attributes:
        data: Contrastive pairs (or a dict coerced via `as_contrastive_pairs`).
        method: Direction-extraction method (mirrors `VectorTrainSpec.method`).
        accumulate: Hidden-state span selection (mirrors `VectorTrainSpec.accumulate`).
        batch_size: Forward-pass batch size (mirrors `VectorTrainSpec.batch_size`).
        prompt_format: How to render pairs into model-ready text (mirrors
            `VectorTrainSpec.prompt_format`).
        location: Residual-stream boundary to fit at (mirrors `VectorTrainSpec.location`).
        normalize: L2-normalize the fitted master per layer once, before caching.
        estimator: Optional custom `BaseEstimator`; when set, the spec fields are ignored.
        estimator_kwargs: Extra kwargs forwarded to a custom `estimator.fit(...)`.
    """

    data: ContrastivePairs | dict
    method: str = "pca_pairwise"
    accumulate: str = "all"
    batch_size: int = 8
    prompt_format: PromptFormat = "chat_completion"
    location: HiddenStateLocation = "layer_output"
    normalize: bool = False
    estimator: BaseEstimator | None = None
    estimator_kwargs: dict | None = None

    _model_ref: "weakref.ref | None" = field(default=None, init=False, repr=False, compare=False)
    _master: SteeringVector | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.data, ContrastivePairs):
            self.data = as_contrastive_pairs(self.data)

        spec_customized = (
            self.method != "pca_pairwise"
            or self.accumulate != "all"
            or self.batch_size != 8
            or self.prompt_format != "chat_completion"
            or self.location != "layer_output"
        )
        if self.estimator is not None and spec_customized:
            warnings.warn(
                "method/accumulate/batch_size/prompt_format/location are ignored when a custom "
                "estimator is supplied; the estimator owns its config via estimator_kwargs.",
                UserWarning,
            )
        if self.estimator is None and self.estimator_kwargs is not None:
            warnings.warn("estimator_kwargs is inert without a custom estimator.", UserWarning)

    def _fit(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> SteeringVector:
        """Fit the master steering vector (no caching, no cloning)."""
        if self.estimator is not None:
            master = self.estimator.fit(model, tokenizer, data=self.data, **(self.estimator_kwargs or {}))
        else:
            spec = VectorTrainSpec(
                method=self.method,
                accumulate=self.accumulate,
                batch_size=self.batch_size,
                prompt_format=self.prompt_format,
                location=self.location,
            )
            estimator = MeanDifferenceEstimator() if self.method == "mean_diff" else ContrastiveDirectionEstimator()
            master = estimator.fit(model, tokenizer, data=self.data, spec=spec)

        if self.normalize:
            master = master.normalized()
        return master

    def resolve(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> SteeringVector:
        """Return a fresh clone of the fitted artifact for `model`, fitting once and memoizing.

        Args:
            model: The model to fit against (or a memo hit for the same model).
            tokenizer: Tokenizer used to encode the contrastive pairs when fitting.

        Returns:
            An independent `SteeringVector` clone the caller owns.
        """
        if self._model_ref is not None and self._model_ref() is model and self._master is not None:
            return self._master.clone()
        master = self._fit(model, tokenizer)
        self._model_ref = weakref.ref(model)
        self._master = master
        return master.clone()


class _Precomputed:
    """A trivially-resolved source wrapping a concrete `SteeringVector` (internal).

    Lets the adapter's resolver treat concrete artifacts and sources uniformly. Not part of the
    public API — users pass vectors, mappings, or sources directly.
    """

    def __init__(self, steering_vector: SteeringVector):
        self._steering_vector = steering_vector

    def resolve(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> SteeringVector:
        return self._steering_vector.clone()


def _as_artifact_source(x) -> ArtifactSource:
    """Coerce a concrete artifact or source into an `ArtifactSource` (internal).

    Accepts an `ArtifactSource` (returned as-is), a `SteeringVector` (wrapped in `_Precomputed`), or
    a `Mapping[int, torch.Tensor]` of per-layer directions (wrapped in a `SteeringVector` with
    `model_type="unknown"` then `_Precomputed`). Anything else raises `TypeError`.
    """
    if isinstance(x, ArtifactSource):
        return x
    if isinstance(x, SteeringVector):
        return _Precomputed(x)
    if isinstance(x, Mapping):
        directions = {int(k): v for k, v in x.items()}
        if not all(isinstance(v, torch.Tensor) for v in directions.values()):
            raise TypeError("Mapping artifact must map layer ids to torch.Tensor directions.")
        return _Precomputed(SteeringVector(model_type="unknown", directions=directions))
    raise TypeError(
        f"Expected a SteeringVector, a Mapping[int, Tensor], or an ArtifactSource; got "
        f"{type(x).__name__}."
    )
