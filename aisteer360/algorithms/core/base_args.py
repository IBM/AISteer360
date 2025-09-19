"""
Base argument validation for steering method configuration.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Type, TypeVar

T = TypeVar("T", bound="BaseArgs")


@dataclass
class BaseArgs:
    """Base class for all method's args classes."""

    @classmethod
    def validate(cls: Type[T], data: Any | None = None, **kwargs) -> T:
        """Create and validate an Args instance from dict, kwargs, or existing instance.

        Args:
            data: Existing instance, dict of args, or None
            **kwargs: Additional args (override values in data if both provided)

        Returns:
            Validated instance of the Args class
        """

        if isinstance(data, cls):
            return data

        if isinstance(data, Mapping):
            kwargs = {**data, **kwargs}

        return cls(**kwargs)
