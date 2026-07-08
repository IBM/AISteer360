"""Base class for runtime gates that control transform application."""
from abc import ABC, abstractmethod


class BaseGate(ABC):
    """Decides whether a transform should fire during a generation step.

    Lifecycle per generation call:
        1. reset() - clear state from previous generation
        2. update() - called from hooks as evidence (scores) arrive
        3. is_open() - queried by hooks to decide whether to apply transform
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all state. Called once at the start of each generation."""
        ...

    @abstractmethod
    def update(self, score: float, *, key: int | None = None) -> None:
        """Provide a new evidence signal to the gate.

        Args:
            score: The computed score (e.g., cosine similarity).
            key: Optional identifier for the source (e.g., layer_id) when
                the gate aggregates signals from multiple sources.
        """
        ...

    @abstractmethod
    def is_open(self) -> bool:
        """Return True if the transform should be applied."""
        ...

    def is_ready(self) -> bool:
        """Return True if the gate has received all expected evidence.

        Default returns True (gate is always ready to make a decision).
        Override for gates that wait for multiple signals before deciding.
        """
        return True


class AlwaysOpenGate(BaseGate):
    """Gate that is always open. Use when no conditional gating is needed.

    This avoids branching in hook logic: methods without conditions still
    go through the gate, but it is always open with zero overhead.
    """

    def reset(self) -> None:
        pass

    def update(self, score: float, *, key: int | None = None) -> None:
        pass

    def is_open(self) -> bool:
        return True
