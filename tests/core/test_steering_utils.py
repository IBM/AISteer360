"""
Tests for steering utility functions.

Tests cover:

- merge_controls function
- ensure_pad_token function
- Control categorization and validation
"""
from unittest.mock import MagicMock

import pytest

from tests.conftest import (  # Base classes; Mock controls; Utilities
    HFGenerateDriver,
    InputControl,
    MockInputControl,
    MockOutputControl,
    MockStateControl,
    MockStructuralControl,
    NoInputControl,
    NoStateControl,
    NoStructuralControl,
    OutputControl,
    StateControl,
    StructuralControl,
    ensure_pad_token,
    merge_controls,
)


# merge_controls Tests
class TestMergeControlsEmpty:
    """Tests for merge_controls with empty or minimal input."""

    def test_empty_list_returns_all_defaults(self):
        """Test that empty list returns all null controls."""
        result = merge_controls([])

        assert "input_controls" in result
        assert "structural_controls" in result
        assert "state_controls" in result
        assert "output_controls" in result

        assert len(result["input_controls"]) == 1
        assert isinstance(result["input_controls"][0], NoInputControl)
        assert len(result["structural_controls"]) == 1
        assert isinstance(result["structural_controls"][0], NoStructuralControl)
        assert len(result["state_controls"]) == 1
        assert isinstance(result["state_controls"][0], NoStateControl)
        assert result["output_controls"] == []

    def test_empty_iterable_returns_defaults(self):
        """Test that any empty iterable returns defaults."""
        result = merge_controls(iter([]))
        assert isinstance(result["input_controls"][0], NoInputControl)


class TestMergeControlsSingleCategory:
    """Tests for merge_controls with single category controls."""

    def test_single_input_control(self):
        """Test merging single input control."""
        control = MockInputControl()
        result = merge_controls([control])

        assert result["input_controls"] == [control]
        # Other categories should be defaults
        assert isinstance(result["structural_controls"][0], NoStructuralControl)
        assert isinstance(result["state_controls"][0], NoStateControl)
        assert result["output_controls"] == []

    def test_single_structural_control(self):
        """Test merging single structural control."""
        control = MockStructuralControl()
        result = merge_controls([control])

        assert result["structural_controls"] == [control]
        assert isinstance(result["input_controls"][0], NoInputControl)

    def test_single_state_control(self):
        """Test merging single state control."""
        control = MockStateControl()
        result = merge_controls([control])

        assert isinstance(result["state_controls"][0], MockStateControl)
        assert result["state_controls"] == [control]

    def test_single_output_control(self):
        """Test merging single output control."""
        control = MockOutputControl()
        result = merge_controls([control])

        assert result["output_controls"] == [control]


class TestMergeControlsMultipleCategories:
    """Tests for merge_controls with multiple category controls."""

    def test_two_different_categories(self):
        """Test merging controls from two categories."""
        input_ctrl = MockInputControl()
        state_ctrl = MockStateControl()

        result = merge_controls([input_ctrl, state_ctrl])

        assert result["input_controls"] == [input_ctrl]
        assert result["state_controls"] == [state_ctrl]
        assert isinstance(result["structural_controls"][0], NoStructuralControl)
        assert result["output_controls"] == []

    def test_all_four_categories(self):
        """Test merging controls from all four categories."""
        input_ctrl = MockInputControl()
        structural_ctrl = MockStructuralControl()
        state_ctrl = MockStateControl()
        output_ctrl = MockOutputControl()

        result = merge_controls([input_ctrl, structural_ctrl, state_ctrl, output_ctrl])

        assert result["input_controls"] == [input_ctrl]
        assert result["structural_controls"] == [structural_ctrl]
        assert result["state_controls"] == [state_ctrl]
        assert result["output_controls"] == [output_ctrl]

    def test_order_independent_across_categories(self):
        """Test that cross-category control order doesn't matter."""
        input_ctrl = MockInputControl()
        state_ctrl = MockStateControl()

        result1 = merge_controls([input_ctrl, state_ctrl])
        result2 = merge_controls([state_ctrl, input_ctrl])

        assert result1["input_controls"] == result2["input_controls"]
        assert result1["state_controls"] == result2["state_controls"]


class TestMergeControlsErrors:
    """Tests for merge_controls error handling."""

    def test_multiple_input_controls_returned_in_order(self):
        """Multiple input controls are allowed and returned in encounter order."""
        ctrl1 = MockInputControl()
        ctrl2 = MockInputControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["input_controls"] == [ctrl1, ctrl2]

    def test_multiple_structural_controls_returned_in_order(self):
        """Multiple structural controls are allowed and returned in encounter order."""
        ctrl1 = MockStructuralControl()
        ctrl2 = MockStructuralControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["structural_controls"] == [ctrl1, ctrl2]

    def test_multiple_state_controls_returned_in_order(self):
        """Multiple state controls are allowed and returned in encounter order."""
        ctrl1 = MockStateControl()
        ctrl2 = MockStateControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["state_controls"] == [ctrl1, ctrl2]

    def test_multiple_output_controls_returned_in_order(self):
        """Multiple output controls are allowed and returned in encounter order."""
        ctrl1 = MockOutputControl()
        ctrl2 = MockOutputControl()

        result = merge_controls([ctrl1, ctrl2])
        assert result["output_controls"] == [ctrl1, ctrl2]

    def test_multiple_decoding_drivers_raises(self):
        """Two enabled decoding drivers raise (the decode loop does not compose)."""

        class DriverA(HFGenerateDriver):
            pass

        class DriverB(HFGenerateDriver):
            pass

        with pytest.raises(ValueError, match="decoding drivers"):
            merge_controls([DriverA(), DriverB()])

    def test_step_level_control_plus_driver_allowed(self):
        """A step-level control alongside a single driver is fine (returned in order)."""
        step_level_control = MockOutputControl()
        driver = HFGenerateDriver()

        result = merge_controls([step_level_control, driver])
        assert result["output_controls"] == [step_level_control, driver]

    def test_unknown_control_type_raises(self):
        """Test that unknown control type raises TypeError."""
        class UnknownControl:
            pass

        with pytest.raises(TypeError, match="Unknown control type"):
            merge_controls([UnknownControl()])

    def test_duplicate_instance_error_includes_control_name(self):
        """The same-instance-twice error message includes the control class name."""
        ctrl = MockInputControl()

        with pytest.raises(ValueError) as exc_info:
            merge_controls([ctrl, ctrl])

        assert "MockInputControl" in str(exc_info.value)


class TestMergeControlsWithSubclasses:
    """Tests for merge_controls with control subclasses."""

    def test_subclass_recognized_as_parent_category(self):
        """Test that subclasses are recognized by parent category."""
        # Create a custom subclass
        class CustomInputControl(InputControl):
            def get_prompt_adapter(self, runtime_kwargs=None):
                return lambda ids, _: ids

        control = CustomInputControl()
        result = merge_controls([control])

        assert isinstance(result["input_controls"][0], CustomInputControl)

    def test_different_subclasses_of_same_category_compose_in_order(self):
        """Different subclasses of the same category are returned in encounter order."""
        class CustomInput1(InputControl):
            pass

        class CustomInput2(InputControl):
            pass

        first, second = CustomInput1(), CustomInput2()
        result = merge_controls([first, second])
        assert result["input_controls"] == [first, second]


# ensure_pad_token Tests
class TestEnsurePadToken:
    """Tests for ensure_pad_token utility function."""

    def test_sets_pad_token_when_none(self):
        """Test that pad_token is set when None."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 1
        tokenizer.eos_token = "</s>"

        result = ensure_pad_token(tokenizer)

        assert result.pad_token_id == 1
        assert result.pad_token == "</s>"

    def test_preserves_existing_pad_token(self):
        """Test that existing pad_token is preserved."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = "<pad>"
        tokenizer.eos_token_id = 1

        result = ensure_pad_token(tokenizer)

        assert result.pad_token_id == 0
        assert result.pad_token == "<pad>"

    def test_returns_same_tokenizer(self):
        """Test that the same tokenizer instance is returned."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        result = ensure_pad_token(tokenizer)

        assert result is tokenizer


# Control Type Detection Tests
class TestControlTypeDetection:
    """Tests for control type detection in merge_controls."""

    def test_detects_input_control(self):
        """Test detection of InputControl type."""
        control = MockInputControl()
        result = merge_controls([control])
        assert "input_controls" in result
        assert result["input_controls"] == [control]

    def test_detects_by_inheritance(self):
        """Test that detection works via inheritance chain."""
        # MockInputControl inherits from InputControl
        control = MockInputControl()
        assert isinstance(control, InputControl)

        result = merge_controls([control])
        assert result["input_controls"] == [control]

    def test_null_controls_have_correct_types(self):
        """Test that null controls are correctly typed."""
        result = merge_controls([])

        assert isinstance(result["input_controls"][0], InputControl)
        assert isinstance(result["structural_controls"][0], StructuralControl)
        assert isinstance(result["state_controls"][0], StateControl)
        assert result["output_controls"] == []  # no phantom output no-op


# Edge Cases
class TestMergeControlsEdgeCases:
    """Edge case tests for merge_controls."""

    def test_generator_input(self):
        """Test that generator input works."""
        def control_generator():
            yield MockInputControl()
            yield MockStateControl()

        result = merge_controls(control_generator())
        assert isinstance(result["input_controls"][0], MockInputControl)
        assert isinstance(result["state_controls"][0], MockStateControl)

    def test_single_element_list(self):
        """Test single element list."""
        control = MockInputControl()
        result = merge_controls([control])
        assert result["input_controls"] == [control]

    def test_preserves_control_state(self):
        """Test that control state is preserved after merging."""
        control = MockInputControl(prefix="test_", num_examples=5)
        result = merge_controls([control])

        assert result["input_controls"][0].prefix == "test_"
        assert result["input_controls"][0].num_examples == 5

    def test_controls_not_modified(self):
        """Test that original controls are not modified."""
        control = MockInputControl()
        original_enabled = control.enabled

        merge_controls([control])

        assert control.enabled == original_enabled
