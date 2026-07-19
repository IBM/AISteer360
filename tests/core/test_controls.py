"""
Tests for base control classes (Input, Structural, State, Output).

Tests cover:

- Control initialization with and without Args
- Control method signatures and behavior
- Null/identity control behavior
- Control properties (enabled, supports_batching)
- Hook management for StateControl
"""
from unittest.mock import MagicMock

import pytest
import torch

from tests.conftest import (  # Base classes; Mock controls; Utilities
    InputControl,
    MockInputArgs,
    MockInputControl,
    MockOutputArgs,
    MockOutputControl,
    MockStateArgs,
    MockStateControl,
    MockStructuralArgs,
    MockStructuralControl,
    NoInputControl,
    NoOutputControl,
    NoStateControl,
    NoStructuralControl,
    OutputControl,
    StateControl,
    StructuralControl,
    create_mock_model,
    create_mock_tokenizer,
)


# Input Control Tests
class TestInputControlBase:
    """Tests for InputControl base class."""

    def test_base_class_defaults(self):
        """Test InputControl default properties."""
        assert InputControl.enabled is True
        assert InputControl.supports_batching is False
        assert InputControl.Args is None

    def test_get_prompt_adapter_returns_identity(self):
        """Test that base get_prompt_adapter returns identity function."""
        control = InputControl()
        adapter = control.get_prompt_adapter()

        input_ids = [1, 2, 3, 4]
        result = adapter(input_ids, {})
        assert result == input_ids

    def test_steer_is_optional(self):
        """Test that steer can be called without error."""
        control = InputControl()
        control.steer(model=None, tokenizer=None)  # Should not raise


class TestMockInputControl:
    """Tests for MockInputControl implementation."""

    def test_initialization_with_args(self):
        """Test MockInputControl initializes with args."""
        control = MockInputControl(prefix="test_", num_examples=5)
        assert control.prefix == "test_"
        assert control.num_examples == 5

    def test_initialization_from_dict(self):
        """Test MockInputControl initializes from dict."""
        control = MockInputControl({"prefix": "dict_", "num_examples": 3})
        assert control.prefix == "dict_"
        assert control.num_examples == 3

    def test_get_prompt_adapter_tracks_calls(self):
        """Test that adapter tracks call count."""
        control = MockInputControl()
        adapter = control.get_prompt_adapter()

        assert control._adapter_call_count == 0
        adapter([1, 2, 3], {})
        assert control._adapter_call_count == 1
        adapter([4, 5, 6], {})
        assert control._adapter_call_count == 2

    def test_steer_stores_references(self, mock_model, mock_tokenizer):
        """Test that steer stores model and tokenizer."""
        control = MockInputControl()
        control.steer(model=mock_model, tokenizer=mock_tokenizer)

        assert control.model is mock_model
        assert control.tokenizer is mock_tokenizer


class TestNoInputControl:
    """Tests for NoInputControl (identity control)."""

    def test_properties(self):
        """Test NoInputControl properties."""
        assert NoInputControl.enabled is False
        assert NoInputControl.supports_batching is True

    def test_returns_input_unchanged(self):
        """Test that adapter returns input unchanged."""
        control = NoInputControl()
        adapter = control.get_prompt_adapter()

        input_ids = torch.tensor([1, 2, 3, 4])
        result = adapter(input_ids, {})
        assert torch.equal(result, input_ids)


# Structural Control Tests
class TestStructuralControlBase:
    """Tests for StructuralControl base class."""

    def test_base_class_defaults(self):
        """Test StructuralControl default properties."""
        assert StructuralControl.enabled is True
        assert StructuralControl.supports_batching is True
        assert StructuralControl.Args is None

    def test_steer_returns_model(self, mock_model):
        """Test that base steer returns the model."""
        control = StructuralControl()
        result = control.steer(mock_model)
        assert result is mock_model


class TestMockStructuralControl:
    """Tests for MockStructuralControl implementation."""

    def test_initialization_with_args(self):
        """Test MockStructuralControl initializes with args."""
        control = MockStructuralControl(learning_rate=1e-3, num_epochs=5)
        assert control.learning_rate == 1e-3
        assert control.num_epochs == 5

    def test_steer_tracks_call(self, mock_model, mock_tokenizer):
        """Test that steer tracks whether it was called."""
        control = MockStructuralControl()
        assert control._steer_called is False

        control.steer(mock_model, tokenizer=mock_tokenizer)
        assert control._steer_called is True

    def test_steer_returns_model(self, mock_model):
        """Test that steer returns the model."""
        control = MockStructuralControl()
        result = control.steer(mock_model)
        assert result is mock_model


class TestNoStructuralControl:
    """Tests for NoStructuralControl (identity control)."""

    def test_properties(self):
        """Test NoStructuralControl properties."""
        assert NoStructuralControl.enabled is False
        assert NoStructuralControl.supports_batching is True

    def test_steer_returns_model_unchanged(self, mock_model):
        """Test that steer returns model unchanged."""
        control = NoStructuralControl()
        result = control.steer(mock_model)
        assert result is mock_model


# State Control Tests
class TestStateControlBase:
    """Tests for StateControl base class."""

    def test_base_class_defaults(self):
        """Test StateControl default properties."""
        assert StateControl.enabled is True
        assert StateControl.supports_batching is False
        assert StateControl.Args is None

    def test_hooks_initialized_empty(self):
        """Test that hooks are initialized as empty dicts."""
        control = StateControl()
        assert control.hooks == {"pre": [], "forward": [], "backward": []}
        assert control.registered == []

    def test_get_hooks_returns_empty(self):
        """Test that base get_hooks returns empty hook dict."""
        control = StateControl()
        hooks = control.get_hooks(torch.tensor([[1, 2, 3]]), {})
        assert hooks == {"pre": [], "forward": [], "backward": []}

    def test_set_hooks(self):
        """Test that set_hooks updates hooks."""
        control = StateControl()
        new_hooks = {"pre": [{"module": "test"}], "forward": [], "backward": []}
        control.set_hooks(new_hooks)
        assert control.hooks == new_hooks

    def test_context_manager_protocol(self):
        """Test that StateControl implements context manager."""
        control = StateControl()
        control._model_ref = MagicMock()

        with control as c:
            assert c is control

    def test_reset_callable(self):
        """Test that reset can be called."""
        control = StateControl()
        control.reset()  # Should not raise


class TestMockStateControl:
    """Tests for MockStateControl implementation."""

    def test_initialization_with_args(self):
        """Test MockStateControl initializes with args."""
        control = MockStateControl(target_layers=[0, 2, 4], scale_factor=2.0)
        assert control.target_layers == [0, 2, 4]
        assert control.scale_factor == 2.0

    def test_get_hooks_creates_hooks_for_layers(self):
        """Test that get_hooks creates hooks for target layers."""
        control = MockStateControl(target_layers=[0, 1])
        hooks = control.get_hooks(torch.tensor([[1, 2, 3]]), {})

        assert control._hooks_created is True
        assert len(hooks["pre"]) == 2  # One for each layer

    def test_get_hooks_stores_runtime_kwargs(self):
        """Test that get_hooks stores runtime_kwargs."""
        control = MockStateControl()
        runtime_kwargs = {"param": "value"}
        control.get_hooks(torch.tensor([[1, 2, 3]]), runtime_kwargs)

        assert control._runtime_kwargs_received == runtime_kwargs

    def test_steer_stores_device(self, mock_model, mock_tokenizer):
        """Test that steer stores device reference."""
        control = MockStateControl()
        control.steer(mock_model, mock_tokenizer)

        assert control.model is mock_model
        assert control.device == mock_model.device


class TestNoStateControl:
    """Tests for NoStateControl (identity control)."""

    def test_properties(self):
        """Test NoStateControl properties."""
        assert NoStateControl.enabled is False
        assert NoStateControl.supports_batching is True

    def test_get_hooks_returns_empty(self):
        """Test that get_hooks returns empty."""
        control = NoStateControl()
        hooks = control.get_hooks(torch.tensor([[1]]), {})
        assert hooks == {"pre": [], "forward": [], "backward": []}

    def test_operations_are_noop(self):
        """Test that all operations are no-ops."""
        control = NoStateControl()
        control.register_hooks(None)  # No error
        control.remove_hooks()  # No error
        control.set_hooks({"pre": [1, 2, 3], "forward": [], "backward": []})  # No error
        control.reset()  # No error


# Output Control Tests
class TestOutputControlBase:
    """Tests for OutputControl base class."""

    def test_base_class_defaults(self):
        """Test OutputControl default properties."""
        assert OutputControl.enabled is True
        assert OutputControl.supports_batching is False
        assert OutputControl.Args is None

    def test_generate_delegates_to_model(self, mock_model):
        """Test that base generate delegates to model."""
        control = OutputControl()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        control.generate(input_ids, attention_mask, {}, mock_model, max_new_tokens=10)
        mock_model.generate.assert_called_once()


class TestMockOutputControl:
    """Tests for MockOutputControl implementation."""

    def test_initialization_with_args(self):
        """Test MockOutputControl initializes with args."""
        control = MockOutputControl(temperature=0.5, top_k=30)
        assert control.temperature == 0.5
        assert control.top_k == 30

    def test_generate_tracks_call(self, mock_model):
        """Test that generate tracks whether it was called."""
        control = MockOutputControl()
        assert control._generate_called is False

        input_ids = torch.tensor([[1, 2, 3]])
        control.generate(input_ids, torch.ones_like(input_ids), {"key": "val"}, mock_model)

        assert control._generate_called is True

    def test_generate_stores_runtime_kwargs(self, mock_model):
        """Test that generate stores runtime_kwargs."""
        control = MockOutputControl()
        runtime_kwargs = {"constraint": "test"}
        input_ids = torch.tensor([[1, 2, 3]])

        control.generate(input_ids, torch.ones_like(input_ids), runtime_kwargs, mock_model)

        assert control._runtime_kwargs_received == runtime_kwargs


class TestNoOutputControl:
    """Tests for NoOutputControl (identity control)."""

    def test_properties(self):
        """Test NoOutputControl properties."""
        assert NoOutputControl.enabled is False
        assert NoOutputControl.supports_batching is True

    def test_generate_uses_model_generate(self, mock_model):
        """Test that generate uses model.generate."""
        control = NoOutputControl()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        control.generate(input_ids, attention_mask, {}, mock_model)
        mock_model.generate.assert_called_once()


# Control Args Integration Tests
class TestControlArgsIntegration:
    """Tests for how controls integrate with their Args classes."""

    def test_input_control_args_fields_become_attributes(self):
        """Test that Args fields become control attributes."""
        control = MockInputControl(prefix="test", suffix="_end", num_examples=10)

        assert hasattr(control, "prefix")
        assert hasattr(control, "suffix")
        assert hasattr(control, "num_examples")
        assert control.prefix == "test"
        assert control.suffix == "_end"
        assert control.num_examples == 10

    def test_state_control_args_fields_become_attributes(self):
        """Test that StateControl Args fields become attributes."""
        control = MockStateControl(target_layers=[5, 6], scale_factor=0.1, mode="multiply")

        assert control.target_layers == [5, 6]
        assert control.scale_factor == 0.1
        assert control.mode == "multiply"

    def test_control_preserves_args_reference(self):
        """Test that control preserves reference to args."""
        control = MockInputControl(prefix="test")

        assert hasattr(control, "args")
        assert isinstance(control.args, MockInputArgs)
        assert control.args.prefix == "test"


# Control Lifecycle Tests
class TestControlLifecycle:
    """Tests for control lifecycle patterns."""

    def test_input_control_full_lifecycle(self, mock_model, mock_tokenizer):
        """Test full lifecycle of input control."""
        control = MockInputControl(prefix=">>", num_examples=2)

        # Steer phase
        control.steer(mock_model, mock_tokenizer)

        # Generate phase - get adapter
        adapter = control.get_prompt_adapter({"key": "value"})

        # Use adapter
        result = adapter([1, 2, 3], {})

        assert control.model is mock_model
        assert control.tokenizer is mock_tokenizer
        assert result == [1, 2, 3]

    def test_state_control_full_lifecycle(self, mock_model, mock_tokenizer):
        """Test full lifecycle of state control."""
        control = MockStateControl(target_layers=[0])

        # Steer phase
        control.steer(mock_model, mock_tokenizer)

        # Generate phase - get hooks
        input_ids = torch.tensor([[1, 2, 3]])
        hooks = control.get_hooks(input_ids, {"runtime": "kwargs"})

        # Set and use hooks
        control.set_hooks(hooks)
        control._model_ref = mock_model

        with control:
            # Would normally call model.generate here
            pass

        # Reset for next generation
        control.reset()

    def test_structural_control_full_lifecycle(self, mock_model, mock_tokenizer):
        """Test full lifecycle of structural control."""
        control = MockStructuralControl(learning_rate=1e-4, num_epochs=1)

        # Steer phase (modifies model)
        returned_model = control.steer(mock_model, mock_tokenizer)

        assert control._steer_called
        assert returned_model is mock_model

    def test_output_control_full_lifecycle(self, mock_model, mock_tokenizer):
        """Test full lifecycle of output control."""
        control = MockOutputControl(temperature=0.8)

        # Steer phase
        control.steer(mock_model, mock_tokenizer)

        # Generate phase
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        output = control.generate(
            input_ids, attention_mask, {"key": "val"}, mock_model, max_new_tokens=5
        )

        assert control._generate_called
        assert output is not None


# Real StateControl.register_hooks unwind (Issue 8) + beam-expansion mask (Issue 4)
import torch.nn as nn  # noqa: E402

from aisteer360.algorithms.state_control.base import StateControl as RealStateControl  # noqa: E402


class _ProbeStateControl(RealStateControl):
    """Minimal concrete state control that registers a caller-supplied hook spec list."""

    Args = None

    def __init__(self, hook_specs):
        super().__init__()
        # the null-control (Args=None) branch of the base __init__ skips these; set them so the
        # provided register_hooks/remove_hooks machinery works
        self.hooks = {"pre": [], "forward": [], "backward": []}
        self.registered = []
        self._specs = hook_specs

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        return self._specs


class _TinyModel(nn.Module):
    """Two named submodules to hook, with a no-op forward."""

    def __init__(self):
        super().__init__()
        self.good = nn.Identity()
        self.also_good = nn.Identity()

    def forward(self, x):
        return x


class TestRegisterHooksUnwind:
    """register_hooks must not leak handles when registration fails partway (Issue 8)."""

    def _noop_pre_hook(self, module, args, kwargs):
        return None

    def test_partial_failure_removes_valid_handles(self):
        model = _TinyModel()
        control = _ProbeStateControl({
            "pre": [
                {"module": "good", "hook_func": self._noop_pre_hook},
                {"module": "does_not_exist", "hook_func": self._noop_pre_hook},
            ],
            "forward": [],
            "backward": [],
        })
        control.set_hooks(control.get_hooks(None, None))

        with pytest.raises(AttributeError):
            control.register_hooks(model)

        # the valid module has no lingering hooks and the registry is empty
        assert len(model.good._forward_pre_hooks) == 0
        assert len(model.good._forward_hooks) == 0
        assert control.registered == []

    def test_successful_registration_then_removal(self):
        model = _TinyModel()
        control = _ProbeStateControl({
            "pre": [
                {"module": "good", "hook_func": self._noop_pre_hook},
                {"module": "also_good", "hook_func": self._noop_pre_hook},
            ],
            "forward": [],
            "backward": [],
        })
        control.set_hooks(control.get_hooks(None, None))
        control.register_hooks(model)
        assert len(control.registered) == 2
        assert len(model.good._forward_pre_hooks) == 1

        control.remove_hooks()
        assert control.registered == []
        assert len(model.good._forward_pre_hooks) == 0


class TestBeamExpansionMask:
    """CAA under batched beam search: masks align to the repeat_interleave-expanded batch (Issue 4)."""

    def test_caa_batch2_beams2_completes_and_steers(self):
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
        from aisteer360.algorithms.state_control.caa.control import CAA
        from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

        torch.manual_seed(0)
        hidden, layers = 32, 4
        model = tiny_llama(num_layers=layers, hidden=hidden)
        tokenizer = wordlevel_tokenizer()

        g = torch.Generator().manual_seed(1)
        sv = SteeringVector(
            model_type="llama",
            directions={lid: torch.randn(1, hidden, generator=g) for lid in range(layers)},
        )
        applied = {"count": 0, "batches": []}
        control = CAA(steering_vector=sv, layer_id=1, multiplier=1.0, token_scope="after_prompt")

        pipeline = SteeringPipeline(controls=[control], lazy_init=True)
        pipeline.model = model
        pipeline.tokenizer = tokenizer
        pipeline.steer()

        inner = control._transform

        class _Spy:
            def apply(self, hidden_states, *, layer_id, token_mask, **kw):
                # the mask must have been aligned to the hidden batch (no broadcast mismatch)
                assert token_mask.size(0) == hidden_states.size(0)
                applied["count"] += 1
                applied["batches"].append(hidden_states.size(0))
                return inner.apply(hidden_states, layer_id=layer_id, token_mask=token_mask, **kw)

        control._transform = _Spy()

        input_ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 3]], dtype=torch.long)
        out = pipeline.generate(
            input_ids=input_ids,
            max_new_tokens=4,
            num_beams=2,
            do_sample=False,
            eos_token_id=None,
        )

        assert out.size(0) == 2  # two prompts in, two sequences out
        assert applied["count"] > 0
        # during decode the hidden batch is expanded to batch * beams = 4
        assert 4 in applied["batches"]

    def test_plain_batch2_no_beams_unchanged(self):
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
        from aisteer360.algorithms.state_control.caa.control import CAA
        from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

        torch.manual_seed(0)
        hidden, layers = 32, 4
        model = tiny_llama(num_layers=layers, hidden=hidden)
        tokenizer = wordlevel_tokenizer()
        g = torch.Generator().manual_seed(2)
        sv = SteeringVector(
            model_type="llama",
            directions={lid: torch.randn(1, hidden, generator=g) for lid in range(layers)},
        )
        control = CAA(steering_vector=sv, layer_id=1, token_scope="after_prompt")
        pipeline = SteeringPipeline(controls=[control], lazy_init=True)
        pipeline.model = model
        pipeline.tokenizer = tokenizer
        pipeline.steer()

        input_ids = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 3]], dtype=torch.long)
        out = pipeline.generate(input_ids=input_ids, max_new_tokens=4, do_sample=False, eos_token_id=None)
        assert out.size(0) == 2
