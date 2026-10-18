"""Base class for estimators that learn artifacts during steer()."""
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.access import ModelAccess

T = TypeVar("T")


class BaseEstimator(ABC, Generic[T]):
    """Learns an artifact from a model and data during the steer() phase.

    The type parameter T is the artifact type (e.g., SteeringVector). The class attribute `access`
    declares the model access a fit needs. The default `ModelAccess.CAPTURE` covers estimators
    that extract hidden states through `capture_hidden`, which runs on the loaded model or through a
    capture-capable session. An estimator that reads the module tree declares
    `ModelAccess.MODULE`.

    Attributes:
        access: The model access a fit needs, read by the fit sources that wrap the estimator.
    """

    access: ClassVar[ModelAccess] = ModelAccess.CAPTURE

    @abstractmethod
    def fit(
        self,
        model: PreTrainedModel | None,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs: Any,
    ) -> T:
        """Fit the estimator and return the learned artifact.

        Fit sources call `fit(model, tokenizer, data=..., session=session, **estimator_kwargs)`.
        An estimator accepts the `session` keyword (or `**kwargs`), and with `model=None` it
        extracts hidden states through the session. A `SteeringVector` artifact records the
        residual-stream boundary it was extracted at as `meta["location"]`.

        Args:
            model: The model being steered, or None when the fit runs through a session.
            tokenizer: Tokenizer for encoding training data.
            **kwargs: Estimator inputs, including `data` and `session`.

        Returns:
            The learned artifact of type T.
        """
        ...
