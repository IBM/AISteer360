from abc import ABC, abstractmethod
from typing import Any


class Metric(ABC):
    """
    Base-class for evaluation metrics.

    Provides a standardized interface for computing evaluation scores on model-generated responses. Subclasses should
    define their specific scoring logic in `compute()` and can accept additional configuration through constructor
    arguments stored in `extras`.

    Args:
        **extras
            Required extras for the metric (e.g., LLM, tokenizer, etc.)
    """
    def __init__(self, **extras: Any) -> None:
        self.name: str = self.__class__.__name__
        self.extras: dict[str, Any] = extras

    @abstractmethod
    def compute(
        self,
        responses: list[Any],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Base compute method."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
