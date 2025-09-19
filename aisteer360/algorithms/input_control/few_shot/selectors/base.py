"""
Base interface for few-shot example selection strategies.
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence


class Selector(ABC):
    """
    Base class for example selector.
    """

    @abstractmethod
    def sample(
        self,
        pool: Sequence[dict],
        k: int,
        **kwargs: Any
    ) -> list[dict]:
        """Return k items chosen from pool."""
        raise NotImplementedError
