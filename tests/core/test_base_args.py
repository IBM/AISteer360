"""
Tests for BaseArgs validation functionality.

Tests cover:

- Basic instantiation
- Validation from dict
- Validation from kwargs
- Validation from existing instance
- Post-init validation (in subclasses)
"""
from dataclasses import dataclass, field

import pytest

from tests.conftest import BaseArgs


# Test Args Subclasses
@dataclass
class SimpleArgs(BaseArgs):
    """Simple args for basic testing."""
    name: str = "default"
    value: int = 0


@dataclass
class ArgsWithValidation(BaseArgs):
    """Args with post-init validation."""
    alpha: float = 1.0
    mode: str = "default"

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.mode not in ("default", "strict", "relaxed"):
            raise ValueError(f"Invalid mode: {self.mode}")


@dataclass
class ArgsWithComplexDefaults(BaseArgs):
    """Args with mutable default values."""
    items: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    layers: list = field(default_factory=lambda: [0, 1])


# Test Cases
class TestBaseArgsBasic:
    """Tests for basic BaseArgs functionality."""

    def test_default_instantiation(self):
        """Test creating args with default values."""
        args = SimpleArgs()
        assert args.name == "default"
        assert args.value == 0

    def test_instantiation_with_values(self):
        """Test creating args with custom values."""
        args = SimpleArgs(name="custom", value=42)
        assert args.name == "custom"
        assert args.value == 42

    def test_partial_values(self):
        """Test creating args with some custom values."""
        args = SimpleArgs(value=100)
        assert args.name == "default"
        assert args.value == 100


class TestBaseArgsValidate:
    """Tests for BaseArgs.validate() class method."""

    def test_validate_from_none(self):
        """Test validate with no data creates defaults."""
        args = SimpleArgs.validate()
        assert args.name == "default"
        assert args.value == 0

    def test_validate_from_kwargs(self):
        """Test validate from keyword arguments."""
        args = SimpleArgs.validate(name="from_kwargs", value=10)
        assert args.name == "from_kwargs"
        assert args.value == 10

    def test_validate_from_dict(self):
        """Test validate from dictionary."""
        data = {"name": "from_dict", "value": 20}
        args = SimpleArgs.validate(data)
        assert args.name == "from_dict"
        assert args.value == 20

    def test_validate_from_existing_instance(self):
        """Test validate passes through existing instance."""
        original = SimpleArgs(name="original", value=30)
        validated = SimpleArgs.validate(original)
        assert validated is original

    def test_validate_dict_with_kwargs_override(self):
        """Test that kwargs override dict values."""
        data = {"name": "from_dict", "value": 20}
        args = SimpleArgs.validate(data, value=99)
        assert args.name == "from_dict"
        assert args.value == 99

    def test_validate_empty_dict(self):
        """Test validate with empty dict uses defaults."""
        args = SimpleArgs.validate({})
        assert args.name == "default"
        assert args.value == 0

    def test_validate_mapping_type(self):
        """Test validate works with Mapping types."""
        from collections import OrderedDict
        data = OrderedDict([("name", "ordered"), ("value", 50)])
        args = SimpleArgs.validate(data)
        assert args.name == "ordered"
        assert args.value == 50


class TestArgsWithPostInitValidation:
    """Tests for args that have __post_init__ validation."""

    def test_valid_args_pass_validation(self):
        """Test that valid args pass post-init validation."""
        args = ArgsWithValidation(alpha=0.5, mode="strict")
        assert args.alpha == 0.5
        assert args.mode == "strict"

    def test_invalid_alpha_raises(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ArgsWithValidation(alpha=-1.0)

    def test_zero_alpha_raises(self):
        """Test that zero alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ArgsWithValidation(alpha=0)

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ArgsWithValidation(mode="invalid")

    def test_validate_with_invalid_values_raises(self):
        """Test validate with invalid values still raises."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ArgsWithValidation.validate({"alpha": -1.0})


class TestArgsWithMutableDefaults:
    """Tests for args with mutable default values."""

    def test_mutable_defaults_are_independent(self):
        """Test that mutable defaults don't share state."""
        args1 = ArgsWithComplexDefaults()
        args2 = ArgsWithComplexDefaults()

        args1.items.append("item1")

        assert "item1" in args1.items
        assert "item1" not in args2.items

    def test_custom_mutable_values(self):
        """Test passing custom mutable values."""
        custom_items = ["a", "b", "c"]
        args = ArgsWithComplexDefaults(items=custom_items)
        assert args.items == ["a", "b", "c"]

    def test_factory_defaults(self):
        """Test that factory defaults work correctly."""
        args = ArgsWithComplexDefaults()
        assert args.layers == [0, 1]

    def test_validate_with_mutable_values(self):
        """Test validate preserves mutable values correctly."""
        data = {"items": [1, 2, 3], "config": {"key": "value"}}
        args = ArgsWithComplexDefaults.validate(data)
        assert args.items == [1, 2, 3]
        assert args.config == {"key": "value"}


class TestArgsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_kwargs_raises(self):
        """Test that unknown kwargs raise TypeError."""
        with pytest.raises(TypeError):
            SimpleArgs(unknown_param="value")

    def test_validate_with_unknown_kwargs_raises(self):
        """Test validate with unknown kwargs raises TypeError."""
        with pytest.raises(TypeError):
            SimpleArgs.validate(unknown_param="value")

    def test_validate_with_unknown_dict_keys_raises(self):
        """Test validate with unknown dict keys raises TypeError."""
        with pytest.raises(TypeError):
            SimpleArgs.validate({"unknown_key": "value"})

    def test_none_values_are_preserved(self):
        """Test that None values are preserved when explicitly passed."""
        @dataclass
        class ArgsWithOptional(BaseArgs):
            optional_value: str | None = None

        args = ArgsWithOptional.validate({"optional_value": None})
        assert args.optional_value is None

    def test_type_coercion_not_automatic(self):
        """Test that types are not automatically coerced."""
        # Dataclasses don't coerce types by default
        args = SimpleArgs(name=123, value="not_an_int")
        assert args.name == 123  # stored as-is
        assert args.value == "not_an_int"  # stored as-is


class TestArgsInheritance:
    """Tests for args class inheritance."""

    def test_subclass_inherits_validate(self):
        """Test that subclasses inherit validate method."""
        @dataclass
        class ChildArgs(SimpleArgs):
            extra: str = "extra_default"

        args = ChildArgs.validate(name="child", value=5, extra="custom")
        assert args.name == "child"
        assert args.value == 5
        assert args.extra == "custom"

    def test_subclass_can_override_defaults(self):
        """Test that subclasses can override parent defaults."""
        @dataclass
        class ChildArgs(SimpleArgs):
            name: str = "child_default"

        args = ChildArgs()
        assert args.name == "child_default"
        assert args.value == 0  # inherited default
