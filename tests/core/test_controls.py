"""
Tests for the control base classes (input, structural, state, output).

Tests cover:

- Base-class defaults, abstractness, and optional lifecycle hooks
- Args validation and field mirroring on construction
- Null/identity control behavior
- Hook management for `StateControl`, including partial-registration unwind
- The default decoding driver
- Batched beam search under a state control
"""
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from transformers import LogitsProcessorList, StoppingCriteriaList

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl, NoInputControl
from aisteer360.algorithms.output_control.base import (
    DecodingDriver,
    HFGenerateDriver,
    OutputControl,
)
from aisteer360.algorithms.state_control._common.steering_vector import SteeringVector
from aisteer360.algorithms.state_control.base import NoStateControl, StateControl
from aisteer360.algorithms.state_control.caa.control import CAA
from aisteer360.algorithms.structural_control.base import (
    NoStructuralControl,
    StructuralControl,
)
from tests.conftest import (
    MockInputArgs,
    MockInputControl,
    MockOutputControl,
    MockStateControl,
    MockStructuralControl,
)
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer


class _MinimalInputControl(InputControl):
    """Concrete input control with an identity `adapt`."""

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


class _MinimalStateControl(StateControl):
    """Concrete state control returning empty hook specs."""

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        return {"pre": [], "forward": [], "backward": []}


# Input Control Tests
class TestInputControlBase:
    """Tests for the `InputControl` base class."""

    def test_base_class_defaults(self):
        assert InputControl.enabled is True
        assert InputControl.supports_batching is False
        assert InputControl.Args is None

    def test_adapt_is_abstract(self):
        with pytest.raises(TypeError):
            InputControl()

    def test_adapt_messages_default_returns_none(self):
        control = _MinimalInputControl()
        assert control.adapt_messages([[{"role": "user", "content": "hi"}]]) is None

    def test_steer_is_optional(self):
        control = _MinimalInputControl()
        control.steer(model=None, tokenizer=None)  # no-op by default


class TestMockInputControl:
    """Tests for the recording `MockInputControl`."""

    def test_initialization_with_args(self):
        control = MockInputControl(prefix="test_", num_examples=5)
        assert control.prefix == "test_"
        assert control.num_examples == 5

    def test_initialization_from_dict(self):
        control = MockInputControl({"prefix": "dict_", "num_examples": 3})
        assert control.prefix == "dict_"
        assert control.num_examples == 3

    def test_adapt_tracks_calls(self):
        control = MockInputControl()

        assert control._adapt_call_count == 0
        control.adapt([1, 2, 3], {})
        assert control._adapt_call_count == 1
        control.adapt([4, 5, 6], {})
        assert control._adapt_call_count == 2

    def test_steer_stores_references(self, mock_model, mock_tokenizer):
        control = MockInputControl()
        control.steer(model=mock_model, tokenizer=mock_tokenizer)

        assert control.model is mock_model
        assert control.tokenizer is mock_tokenizer


class TestNoInputControl:
    """Tests for `NoInputControl` (identity control)."""

    def test_properties(self):
        assert NoInputControl.enabled is False
        assert NoInputControl.supports_batching is True

    def test_returns_input_unchanged(self):
        control = NoInputControl()
        input_ids = torch.tensor([1, 2, 3, 4])
        result = control.adapt(input_ids, {})
        assert torch.equal(result, input_ids)

    def test_steer_attaches_tokenizer(self, mock_tokenizer):
        control = NoInputControl()
        control.steer(tokenizer=mock_tokenizer)
        assert control.tokenizer is mock_tokenizer


# Structural Control Tests
class TestStructuralControlBase:
    """Tests for the `StructuralControl` base class."""

    def test_base_class_defaults(self):
        assert StructuralControl.enabled is True
        assert StructuralControl.supports_batching is True
        assert StructuralControl.Args is None

    def test_steer_is_abstract(self):
        with pytest.raises(TypeError):
            StructuralControl()


class TestMockStructuralControl:
    """Tests for the recording `MockStructuralControl`."""

    def test_initialization_with_args(self):
        control = MockStructuralControl(learning_rate=1e-3, num_epochs=5)
        assert control.learning_rate == 1e-3
        assert control.num_epochs == 5

    def test_steer_tracks_call(self, mock_model, mock_tokenizer):
        control = MockStructuralControl()
        assert control._steer_called is False

        control.steer(mock_model, tokenizer=mock_tokenizer)
        assert control._steer_called is True

    def test_steer_returns_model(self, mock_model):
        control = MockStructuralControl()
        result = control.steer(mock_model)
        assert result is mock_model


class TestNoStructuralControl:
    """Tests for `NoStructuralControl` (identity control)."""

    def test_properties(self):
        assert NoStructuralControl.enabled is False
        assert NoStructuralControl.supports_batching is True

    def test_steer_returns_model_unchanged(self, mock_model):
        control = NoStructuralControl()
        result = control.steer(mock_model)
        assert result is mock_model


# State Control Tests
class TestStateControlBase:
    """Tests for the `StateControl` base class."""

    def test_base_class_defaults(self):
        assert StateControl.enabled is True
        assert StateControl.supports_batching is False
        assert StateControl.Args is None

    def test_get_hooks_is_abstract(self):
        with pytest.raises(TypeError):
            StateControl()

    def test_hooks_initialized_empty(self):
        control = _MinimalStateControl()
        assert control.hooks == {"pre": [], "forward": [], "backward": []}
        assert control.registered == []

    def test_set_hooks(self):
        control = _MinimalStateControl()
        new_hooks = {"pre": [{"module": "test"}], "forward": [], "backward": []}
        control.set_hooks(new_hooks)
        assert control.hooks == new_hooks

    def test_context_manager_requires_model_ref(self):
        control = _MinimalStateControl()
        with pytest.raises(RuntimeError, match="Model reference not set"):
            with control:
                pass

    def test_context_manager_protocol(self):
        control = _MinimalStateControl()
        control._model_ref = MagicMock()

        with control as c:
            assert c is control

    def test_reset_default_is_noop(self):
        control = _MinimalStateControl()
        control.reset()  # no gate or runtime present

    def test_reset_clears_gate_and_runtime(self):
        """`reset` clears a `_gate` and re-clears a `_runtime`'s per-generation counters when
        the control exposes them."""
        control = _MinimalStateControl()
        control._gate = MagicMock()
        control._runtime = MagicMock()

        control.reset()

        control._gate.reset.assert_called_once_with()
        control._runtime.reset_between_generations.assert_called_once_with()


class TestMockStateControl:
    """Tests for the recording `MockStateControl`."""

    def test_initialization_with_args(self):
        control = MockStateControl(target_layers=[0, 2, 4], scale_factor=2.0)
        assert control.target_layers == [0, 2, 4]
        assert control.scale_factor == 2.0

    def test_get_hooks_creates_hooks_for_layers(self):
        control = MockStateControl(target_layers=[0, 1])
        hooks = control.get_hooks(torch.tensor([[1, 2, 3]]), {})

        assert control._hooks_created is True
        assert len(hooks["pre"]) == 2  # one per layer

    def test_get_hooks_stores_runtime_kwargs(self):
        control = MockStateControl()
        runtime_kwargs = {"param": "value"}
        control.get_hooks(torch.tensor([[1, 2, 3]]), runtime_kwargs)

        assert control._runtime_kwargs_received == runtime_kwargs

    def test_steer_stores_device(self, mock_model, mock_tokenizer):
        control = MockStateControl()
        control.steer(mock_model, mock_tokenizer)

        assert control.model is mock_model
        assert control.device == mock_model.device


class TestNoStateControl:
    """Tests for `NoStateControl` (identity control)."""

    def test_properties(self):
        assert NoStateControl.enabled is False
        assert NoStateControl.supports_batching is True

    def test_get_hooks_returns_empty(self):
        control = NoStateControl()
        hooks = control.get_hooks(torch.tensor([[1]]), {})
        assert hooks == {"pre": [], "forward": [], "backward": []}

    def test_operations_are_noop(self):
        control = NoStateControl()
        control.register_hooks(None)
        control.remove_hooks()
        control.set_hooks({"pre": [1, 2, 3], "forward": [], "backward": []})
        control.reset()


# Output Control Tests
class TestOutputControlBase:
    """Tests for the `OutputControl` base class."""

    def test_base_class_defaults(self):
        assert OutputControl.enabled is True
        assert OutputControl.supports_batching is False
        assert OutputControl.Args is None
        assert OutputControl.include_in_scoring is True
        assert OutputControl.same_model_forwards is False

    def test_base_output_hooks_default_empty(self):
        control = OutputControl()
        assert control.get_logits_processors(torch.tensor([[1, 2, 3]]), {}) == []
        assert control.get_stopping_criteria(torch.tensor([[1, 2, 3]]), {}) == []

    def test_decoding_driver_is_abstract(self):
        with pytest.raises(TypeError):
            DecodingDriver()


class TestMockOutputControl:
    """Tests for the recording `MockOutputControl` (step-level)."""

    def test_initialization_with_args(self):
        control = MockOutputControl(temperature=0.5, top_k=30)
        assert control.temperature == 0.5
        assert control.top_k == 30

    def test_get_logits_processors_tracks_call(self):
        control = MockOutputControl()
        assert control._processors_requested is False

        processors = control.get_logits_processors(torch.tensor([[1, 2, 3]]), {"key": "val"})

        assert control._processors_requested is True
        assert len(processors) == 1  # contributes one identity processor

    def test_get_logits_processors_stores_runtime_kwargs(self):
        control = MockOutputControl()
        runtime_kwargs = {"constraint": "test"}

        control.get_logits_processors(torch.tensor([[1, 2, 3]]), runtime_kwargs)

        assert control._runtime_kwargs_received == runtime_kwargs


class TestHFGenerateDriver:
    """Tests for `HFGenerateDriver` (default decoding driver)."""

    def test_properties(self):
        assert HFGenerateDriver.enabled is True
        assert HFGenerateDriver.supports_batching is True
        assert issubclass(HFGenerateDriver, DecodingDriver)

    def test_decode_uses_model_generate(self, mock_model):
        driver = HFGenerateDriver()
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones_like(input_ids)

        driver.decode(
            input_ids, attention_mask, mock_model,
            LogitsProcessorList(), StoppingCriteriaList(), None,
        )
        mock_model.generate.assert_called_once()

    def test_empty_stacks_are_not_forwarded(self, mock_model):
        """Empty processor and criteria stacks are omitted from the `model.generate` kwargs."""
        driver = HFGenerateDriver()
        input_ids = torch.tensor([[1, 2, 3]])

        driver.decode(
            input_ids, torch.ones_like(input_ids), mock_model,
            LogitsProcessorList(), StoppingCriteriaList(), None,
        )
        kwargs = mock_model.generate.call_args.kwargs
        assert "logits_processor" not in kwargs
        assert "stopping_criteria" not in kwargs

    def test_nonempty_stacks_are_forwarded(self, mock_model):
        driver = HFGenerateDriver()
        input_ids = torch.tensor([[1, 2, 3]])
        processors = LogitsProcessorList([lambda prefix_ids, scores: scores])

        driver.decode(
            input_ids, torch.ones_like(input_ids), mock_model,
            processors, StoppingCriteriaList(), None,
        )
        kwargs = mock_model.generate.call_args.kwargs
        assert kwargs["logits_processor"] is processors


# Control Args Integration Tests
class TestControlArgsIntegration:
    """Tests for how controls integrate with their `Args` classes."""

    def test_input_control_args_fields_become_attributes(self):
        control = MockInputControl(prefix="test", suffix="_end", num_examples=10)

        assert hasattr(control, "prefix")
        assert hasattr(control, "suffix")
        assert hasattr(control, "num_examples")
        assert control.prefix == "test"
        assert control.suffix == "_end"
        assert control.num_examples == 10

    def test_state_control_args_fields_become_attributes(self):
        control = MockStateControl(target_layers=[5, 6], scale_factor=0.1, mode="multiply")

        assert control.target_layers == [5, 6]
        assert control.scale_factor == 0.1
        assert control.mode == "multiply"

    def test_control_preserves_args_reference(self):
        control = MockInputControl(prefix="test")

        assert hasattr(control, "args")
        assert isinstance(control.args, MockInputArgs)
        assert control.args.prefix == "test"


# Control Lifecycle Tests
class TestControlLifecycle:
    """Tests for control lifecycle patterns."""

    def test_input_control_full_lifecycle(self, mock_model, mock_tokenizer):
        control = MockInputControl(prefix=">>", num_examples=2)

        control.steer(mock_model, mock_tokenizer)
        result = control.adapt([1, 2, 3], {"key": "value"})

        assert control.model is mock_model
        assert control.tokenizer is mock_tokenizer
        assert result == [1, 2, 3]

    def test_state_control_full_lifecycle(self, mock_model, mock_tokenizer):
        control = MockStateControl(target_layers=[0])

        control.steer(mock_model, mock_tokenizer)

        input_ids = torch.tensor([[1, 2, 3]])
        hooks = control.get_hooks(input_ids, {"runtime": "kwargs"})

        control.set_hooks(hooks)
        control._model_ref = mock_model

        with control:
            pass

        control.reset()

    def test_structural_control_full_lifecycle(self, mock_model, mock_tokenizer):
        control = MockStructuralControl(learning_rate=1e-4, num_epochs=1)

        returned_model = control.steer(mock_model, mock_tokenizer)

        assert control._steer_called
        assert returned_model is mock_model

    def test_output_control_full_lifecycle(self, mock_model, mock_tokenizer):
        control = MockOutputControl(temperature=0.8)

        control.steer(mock_model, mock_tokenizer)

        input_ids = torch.tensor([[1, 2, 3]])
        processors = control.get_logits_processors(input_ids, {"key": "val"})

        assert control._processors_requested
        assert len(processors) == 1


# StateControl.register_hooks unwind
class _ProbeStateControl(StateControl):
    """Concrete state control that registers a caller-supplied hook spec dict."""

    def __init__(self, hook_specs):
        super().__init__()
        self._specs = hook_specs

    def get_hooks(self, input_ids, runtime_kwargs, **kwargs):
        return self._specs


class _TinyModel(nn.Module):
    """Two named submodules to hook, with a pass-through forward."""

    def __init__(self):
        super().__init__()
        self.good = nn.Identity()
        self.also_good = nn.Identity()

    def forward(self, x):
        return x


class TestRegisterHooksUnwind:
    """`register_hooks` must not leak handles when registration fails partway."""

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
    """CAA under batched beam search: masks align to the `repeat_interleave`-expanded batch."""

    def test_caa_batch2_beams2_completes_and_steers(self):
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
