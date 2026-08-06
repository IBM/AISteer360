"""
Tests for SteeringPipeline functionality.

Tests cover:

- Pipeline initialization
- Control merging and assignment
- Steer method behavior
- Generate method behavior
- Compute logprobs method behavior
- Runtime kwargs handling
- Supports batching property
- Error handling
"""
import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline as RealSteeringPipeline
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

from tests.conftest import (  # Base classes; Mock controls; Utilities
    DecodingDriver,
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
    create_mock_model,
    create_mock_tokenizer,
    ensure_pad_token,
    merge_controls,
)


# Mock SteeringPipeline
@dataclass
class MockSteeringPipeline:
    """
    Mock SteeringPipeline for testing.
    """
    model_name_or_path: str | None = None
    controls: list = field(default_factory=list)
    tokenizer_name_or_path: str | None = None
    device_map: str = "auto"
    device: torch.device | str | None = None
    hf_model_kwargs: dict = field(default_factory=dict)
    lazy_init: bool = False

    def __post_init__(self):
        self._is_steered = False

        # Merge controls
        controls_merged = merge_controls(self.controls)
        self.structural_controls = controls_merged["structural_controls"]
        self.input_controls = controls_merged["input_controls"]
        self.state_controls = controls_merged["state_controls"]
        self.output_controls = controls_merged["output_controls"]
        self._default_driver = HFGenerateDriver()

        # Mock model and tokenizer
        if not self.lazy_init:
            self.model = create_mock_model()
            self.tokenizer = create_mock_tokenizer()
            self.device = self.model.device
        else:
            self.model = None
            self.tokenizer = None

    @property
    def supports_batching(self) -> bool:
        """Return True if all enabled controls support batching."""
        controls = (
            *self.structural_controls,
            *self.input_controls,
            *self.state_controls,
            *self.output_controls,
        )
        return all(
            getattr(c, "supports_batching", False)
            for c in controls
            if getattr(c, "enabled", True)
        )

    def steer(self, **kwargs) -> None:
        """Apply all steering controls to the model."""
        if self._is_steered:
            return

        for control in (
            *self.structural_controls,
            *self.input_controls,
            *self.state_controls,
            *self.output_controls,
        ):
            steer_fn = getattr(control, "steer", None)
            if callable(steer_fn):
                maybe_new_model = steer_fn(self.model, tokenizer=self.tokenizer, **kwargs)
                if maybe_new_model is not None and hasattr(maybe_new_model, 'generate'):
                    self.model = maybe_new_model

        if self.model is None:
            raise RuntimeError("No model available after steering.")

        self._is_steered = True

    def _prepare_inputs(
            self,
            input_ids: list | torch.Tensor,
            attention_mask: torch.Tensor | None,
            runtime_kwargs: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply input control and normalize input tensors."""
        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        # Apply input control adapters (chained in list order)
        steered_input_ids = input_ids
        for control in self.input_controls:
            adapter = control.get_prompt_adapter(runtime_kwargs)
            steered_input_ids = adapter(steered_input_ids, runtime_kwargs)

        # Normalize input_ids to 2D tensor
        if isinstance(steered_input_ids, list):
            steered_input_ids = torch.tensor(steered_input_ids, dtype=torch.long)
        if steered_input_ids.ndim == 1:
            steered_input_ids = steered_input_ids.unsqueeze(0)
        steered_input_ids = steered_input_ids.to(device)

        # Normalize attention_mask
        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.shape[-1] != steered_input_ids.shape[-1]:
                attention_mask = None

        if attention_mask is None:
            if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
                attention_mask = (steered_input_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(steered_input_ids, dtype=torch.long)

        attention_mask = attention_mask.to(dtype=steered_input_ids.dtype, device=device)

        return steered_input_ids, attention_mask

    def _setup_state_control(
            self,
            steered_input_ids: torch.Tensor,
            runtime_kwargs: dict | None,
            **kwargs,
    ) -> None:
        """Configure state control hooks for the current forward/generate call."""
        for state_control in self.state_controls:
            hooks = state_control.get_hooks(steered_input_ids, runtime_kwargs, **kwargs)
            state_control.set_hooks(hooks)
            state_control._model_ref = self.model
            state_control.reset()

    def generate(
        self,
        input_ids: list | torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        runtime_kwargs: dict | None = None,
        **gen_kwargs
    ) -> torch.Tensor:
        """Generate text with all steering controls applied."""
        if not self._is_steered:
            raise RuntimeError("Must call .steer() before .generate()")

        runtime_kwargs = runtime_kwargs or {}

        # Input control
        steered_input_ids, attention_mask = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            runtime_kwargs=runtime_kwargs,
        )

        # State control
        self._setup_state_control(steered_input_ids, runtime_kwargs, **gen_kwargs)

        # output controls: compose the processor and criteria stacks (list order), then drive the loop
        logits_processors, stopping_criteria = self._compose_stacks(
            steered_input_ids, runtime_kwargs, attention_mask, gen_kwargs
        )
        decoding_driver = self._resolve_decoding_driver()

        with contextlib.ExitStack() as stack:
            for state_control in self.state_controls:
                stack.enter_context(state_control)
            output_ids = decoding_driver.decode(
                input_ids=steered_input_ids,
                attention_mask=attention_mask,
                model=self.model,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                runtime_kwargs=runtime_kwargs,
                **gen_kwargs,
            )

        return output_ids

    def _resolve_decoding_driver(self):
        for control in self.output_controls:
            if isinstance(control, DecodingDriver) and getattr(control, "enabled", True):
                return control
        return self._default_driver

    def _collect_processors_and_criteria(self, input_ids, runtime_kwargs, attention_mask=None,
                                      for_scoring=False, **kwargs):
        from transformers import LogitsProcessorList, StoppingCriteriaList  # noqa: F401
        processors, criteria = [], []
        for control in self.output_controls:
            if not getattr(control, "enabled", True):
                continue
            if for_scoring and not getattr(control, "include_in_scoring", True):
                continue
            processors.extend(control.get_logits_processors(
                input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs) or [])
            if not for_scoring:
                criteria.extend(control.get_stopping_criteria(
                    input_ids, runtime_kwargs, attention_mask=attention_mask, **kwargs) or [])
        return processors, criteria

    def _compose_stacks(self, input_ids, runtime_kwargs, attention_mask, gen_kwargs):
        from transformers import LogitsProcessorList, StoppingCriteriaList
        processors, criteria = self._collect_processors_and_criteria(
            input_ids, runtime_kwargs, attention_mask=attention_mask, **gen_kwargs
        )
        user_processors = gen_kwargs.pop("logits_processor", None) or []
        user_criteria = gen_kwargs.pop("stopping_criteria", None) or []
        return (
            LogitsProcessorList([*processors, *user_processors]),
            StoppingCriteriaList([*criteria, *user_criteria]),
        )

    def generate_text(self, *args, **kwargs) -> str | list[str]:
        """Generate text and decode to string(s)."""
        ids = self.generate(*args, **kwargs)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def compute_logprobs(
            self,
            input_ids: list | torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            ref_output_ids: list | torch.Tensor = None,
            runtime_kwargs: dict | None = None,
            **forward_kwargs,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities of ref_output_ids."""
        if not self._is_steered:
            raise RuntimeError("Must call `.steer()` before `.compute_logprobs()`.")
        if ref_output_ids is None:
            raise ValueError("`ref_output_ids` is required for `compute_logprobs()`.")

        runtime_kwargs = runtime_kwargs or {}
        device = self.model.device

        # Input Control: adapt the prompt
        steered_input_ids, attention_mask = self._prepare_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            runtime_kwargs=runtime_kwargs,
        )

        # Normalize ref_output_ids
        if isinstance(ref_output_ids, list):
            ref_output_ids = torch.tensor(ref_output_ids, dtype=torch.long)
        if ref_output_ids.ndim == 1:
            ref_output_ids = ref_output_ids.unsqueeze(0)
        ref_output_ids = ref_output_ids.to(device)

        batch_size = steered_input_ids.size(0)
        ref_len = ref_output_ids.size(1)

        # Broadcast single ref sequence across batch
        if ref_output_ids.size(0) == 1 and batch_size > 1:
            ref_output_ids = ref_output_ids.expand(batch_size, -1)

        if ref_len == 0:
            return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)

        # State Control: register hooks
        self._setup_state_control(steered_input_ids, runtime_kwargs, **forward_kwargs)

        # Forward pass under state control context
        is_encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)

        with contextlib.ExitStack() as stack:
            for state_control in self.state_controls:
                stack.enter_context(state_control)
            with torch.no_grad():
                if is_encoder_decoder:
                    outputs = self.model(
                        input_ids=steered_input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=ref_output_ids,
                        **forward_kwargs,
                    )
                    logits = outputs.logits[:, :-1, :]
                    target_ids = ref_output_ids[:, 1:]
                else:
                    combined_ids = torch.cat([steered_input_ids, ref_output_ids], dim=1)
                    combined_mask = torch.cat([
                        attention_mask,
                        torch.ones(batch_size, ref_len, device=device, dtype=attention_mask.dtype),
                    ], dim=1)

                    outputs = self.model(
                        input_ids=combined_ids,
                        attention_mask=combined_mask,
                        **forward_kwargs,
                    )

                    input_len = steered_input_ids.size(1)
                    logits = outputs.logits[:, input_len - 1: input_len + ref_len - 1, :]
                    target_ids = ref_output_ids

            # Apply output-control scoring logits processors under the steered distribution
            from transformers import LogitsProcessorList
            processors, _ = self._collect_processors_and_criteria(
                steered_input_ids, runtime_kwargs, attention_mask=attention_mask,
                for_scoring=True, **forward_kwargs,
            )
            if processors:
                proc_stack = LogitsProcessorList(processors)
                for t in range(logits.size(1)):
                    prefix = (ref_output_ids[:, : t + 1] if is_encoder_decoder
                              else torch.cat([steered_input_ids, ref_output_ids[:, :t]], dim=1))
                    logits[:, t, :] = proc_stack(prefix, logits[:, t, :])

            # Compute logprobs via gather
            logprobs = torch.log_softmax(logits, dim=-1)
            token_logprobs = logprobs.gather(
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)

        return token_logprobs


# Pipeline Initialization Tests
class TestPipelineInitialization:
    """Tests for SteeringPipeline initialization."""

    def test_basic_initialization(self):
        """Test basic pipeline initialization."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )

        assert pipeline.model_name_or_path == "test-model"
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None
        assert not pipeline._is_steered

    def test_initialization_with_controls(self):
        """Test initialization with controls."""
        input_ctrl = MockInputControl()
        state_ctrl = MockStateControl()

        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl, state_ctrl],
        )

        assert pipeline.input_controls == [input_ctrl]
        assert pipeline.state_controls == [state_ctrl]
        assert isinstance(pipeline.structural_controls[0], NoStructuralControl)
        assert pipeline.output_controls == []

    def test_initialization_with_all_controls(self):
        """Test initialization with all four control types."""
        input_ctrl = MockInputControl()
        structural_ctrl = MockStructuralControl()
        state_ctrl = MockStateControl()
        output_ctrl = MockOutputControl()

        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl, structural_ctrl, state_ctrl, output_ctrl],
        )

        assert pipeline.input_controls == [input_ctrl]
        assert pipeline.structural_controls == [structural_ctrl]
        assert pipeline.state_controls == [state_ctrl]
        assert pipeline.output_controls == [output_ctrl]

    def test_lazy_initialization(self):
        """Test lazy initialization mode."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
            lazy_init=True,
        )

        assert pipeline.model is None
        assert pipeline.tokenizer is None

    def test_custom_hf_kwargs(self):
        """Test passing custom HuggingFace kwargs."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
            hf_model_kwargs={"torch_dtype": "float16"},
        )

        assert pipeline.hf_model_kwargs["torch_dtype"] == "float16"


# Pipeline Steer Tests
class TestPipelineSteer:
    """Tests for SteeringPipeline.steer() method."""

    def test_steer_marks_as_steered(self):
        """Test that steer() marks pipeline as steered."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )

        assert not pipeline._is_steered
        pipeline.steer()
        assert pipeline._is_steered

    def test_steer_called_once(self):
        """Test that steer() is effectively called once."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[MockInputControl()],
        )

        pipeline.steer()
        first_state = pipeline._is_steered

        pipeline.steer()  # Second call should be no-op

        assert first_state == pipeline._is_steered

    def test_steer_calls_control_steer_methods(self):
        """Test that steer() calls each control's steer method."""
        input_ctrl = MockInputControl()
        structural_ctrl = MockStructuralControl()
        state_ctrl = MockStateControl()

        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl, structural_ctrl, state_ctrl],
        )

        pipeline.steer()

        # Structural control should have been called
        assert structural_ctrl._steer_called

    def test_steer_order(self):
        """Test that controls are steered in correct order."""
        call_order = []

        class TrackingInputControl(MockInputControl):
            def steer(self, *args, **kwargs):
                call_order.append("input")
                super().steer(*args, **kwargs)

        class TrackingStructuralControl(MockStructuralControl):
            def steer(self, *args, **kwargs):
                call_order.append("structural")
                return super().steer(*args, **kwargs)

        class TrackingStateControl(MockStateControl):
            def steer(self, *args, **kwargs):
                call_order.append("state")
                super().steer(*args, **kwargs)

        class TrackingOutputControl(MockOutputControl):
            def steer(self, *args, **kwargs):
                call_order.append("output")
                super().steer(*args, **kwargs)

        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[
                TrackingInputControl(),
                TrackingStructuralControl(),
                TrackingStateControl(),
                TrackingOutputControl(),
            ],
        )

        pipeline.steer()

        # Order should be: structural -> input -> state -> output
        assert call_order == ["structural", "input", "state", "output"]

    def test_steer_passes_kwargs(self):
        """Test that steer() passes kwargs to controls."""
        received_kwargs = {}

        class KwargsCapturingControl(MockInputControl):
            def steer(self, model=None, tokenizer=None, **kwargs):
                received_kwargs.update(kwargs)
                super().steer(model, tokenizer, **kwargs)

        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[KwargsCapturingControl()],
        )

        pipeline.steer(custom_param="value")

        assert received_kwargs.get("custom_param") == "value"


# Pipeline Generate Tests
class TestPipelineGenerate:
    """Tests for SteeringPipeline.generate() method."""

    def test_generate_requires_steer(self):
        """Test that generate() fails without prior steer()."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )

        with pytest.raises(RuntimeError, match="steer"):
            pipeline.generate(torch.tensor([[1, 2, 3]]))

    def test_generate_basic(self):
        """Test basic generation."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        input_ids = torch.tensor([[1, 2, 3]])
        output = pipeline.generate(input_ids, max_new_tokens=5)

        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_generate_with_list_input(self):
        """Test generation with list input."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        output = pipeline.generate([1, 2, 3], max_new_tokens=5)

        assert output is not None

    def test_generate_with_1d_input(self):
        """Test generation with 1D tensor input."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        input_ids = torch.tensor([1, 2, 3])
        output = pipeline.generate(input_ids, max_new_tokens=5)

        assert output is not None

    def test_generate_passes_runtime_kwargs_to_state_control(self):
        """Test that runtime_kwargs are passed to state control."""
        state_ctrl = MockStateControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[state_ctrl],
        )
        pipeline.steer()

        runtime_kwargs = {"key": "value", "param": 123}
        pipeline.generate(torch.tensor([[1, 2, 3]]), runtime_kwargs=runtime_kwargs)

        assert state_ctrl._runtime_kwargs_received == runtime_kwargs

    def test_generate_passes_runtime_kwargs_to_output_control(self):
        """Test that runtime_kwargs are passed to output control."""
        output_ctrl = MockOutputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[output_ctrl],
        )
        pipeline.steer()

        runtime_kwargs = {"constraint": "test"}
        pipeline.generate(torch.tensor([[1, 2, 3]]), runtime_kwargs=runtime_kwargs)

        assert output_ctrl._runtime_kwargs_received == runtime_kwargs

    def test_generate_uses_input_control_adapter(self):
        """Test that generate uses input control's adapter."""
        input_ctrl = MockInputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl],
        )
        pipeline.steer()

        pipeline.generate(torch.tensor([[1, 2, 3]]))

        assert input_ctrl._adapter_call_count > 0

    def test_generate_creates_hooks(self):
        """Test that generate creates state control hooks."""
        state_ctrl = MockStateControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[state_ctrl],
        )
        pipeline.steer()

        pipeline.generate(torch.tensor([[1, 2, 3]]))

        assert state_ctrl._hooks_created

    def test_generate_requests_output_logits_processors(self):
        """Test that generate gathers the output control's logits processors."""
        output_ctrl = MockOutputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[output_ctrl],
        )
        pipeline.steer()

        pipeline.generate(torch.tensor([[1, 2, 3]]))

        assert output_ctrl._processors_requested


# Pipeline Compute Logprobs Tests
class TestPipelineComputeLogprobs:
    """Tests for SteeringPipeline.compute_logprobs() method."""

    def test_compute_logprobs_requires_steer(self):
        """Test that compute_logprobs() fails without prior steer()."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )

        with pytest.raises(RuntimeError, match="steer"):
            pipeline.compute_logprobs(
                input_ids=torch.tensor([[1, 2, 3]]),
                ref_output_ids=torch.tensor([[4, 5, 6]]),
            )

    def test_compute_logprobs_requires_ref_output_ids(self):
        """Test that compute_logprobs() fails without ref_output_ids."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        with pytest.raises(ValueError, match="ref_output_ids"):
            pipeline.compute_logprobs(
                input_ids=torch.tensor([[1, 2, 3]]),
                ref_output_ids=None,
            )

    def test_compute_logprobs_basic(self):
        """Test basic log probability computation."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        input_ids = torch.tensor([[1, 2, 3]])
        ref_output_ids = torch.tensor([[4, 5, 6]])

        logprobs = pipeline.compute_logprobs(
            input_ids=input_ids,
            ref_output_ids=ref_output_ids,
        )

        assert logprobs is not None
        assert isinstance(logprobs, torch.Tensor)
        assert logprobs.shape == (1, 3)  # batch=1, ref_len=3

    def test_compute_logprobs_with_list_input_ids(self):
        """Test compute_logprobs with list input_ids."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=[1, 2, 3],
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        assert logprobs is not None
        assert logprobs.shape == (1, 3)

    def test_compute_logprobs_with_list_ref_output_ids(self):
        """Test compute_logprobs with list ref_output_ids."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=[4, 5, 6],
        )

        assert logprobs is not None
        assert logprobs.shape == (1, 3)

    def test_compute_logprobs_with_1d_input_ids(self):
        """Test compute_logprobs with 1D input_ids tensor."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([1, 2, 3]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        assert logprobs is not None
        assert logprobs.shape == (1, 3)

    def test_compute_logprobs_with_1d_ref_output_ids(self):
        """Test compute_logprobs with 1D ref_output_ids tensor."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([4, 5, 6]),
        )

        assert logprobs is not None
        assert logprobs.shape == (1, 3)

    def test_compute_logprobs_empty_ref_output_ids(self):
        """Test compute_logprobs with empty ref_output_ids."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[]]),
        )

        assert logprobs.shape == (1, 0)

    def test_compute_logprobs_broadcasts_ref_output_ids(self):
        """Test that single ref_output_ids broadcasts across batch."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])  # batch=2
        ref_output_ids = torch.tensor([[7, 8]])  # batch=1

        logprobs = pipeline.compute_logprobs(
            input_ids=input_ids,
            ref_output_ids=ref_output_ids,
        )

        assert logprobs.shape == (2, 2)  # batch=2, ref_len=2

    def test_compute_logprobs_uses_input_control(self):
        """Test that compute_logprobs applies input control."""
        input_ctrl = MockInputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl],
        )
        pipeline.steer()

        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        assert input_ctrl._adapter_call_count > 0

    def test_compute_logprobs_uses_state_control(self):
        """Test that compute_logprobs applies state control hooks."""
        state_ctrl = MockStateControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[state_ctrl],
        )
        pipeline.steer()

        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        assert state_ctrl._hooks_created

    def test_compute_logprobs_passes_runtime_kwargs_to_input_control(self):
        """Test that runtime_kwargs are passed to input control."""
        input_ctrl = MockInputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl],
        )
        pipeline.steer()

        runtime_kwargs = {"key": "value"}
        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
            runtime_kwargs=runtime_kwargs,
        )

        assert input_ctrl._runtime_kwargs_received == runtime_kwargs

    def test_compute_logprobs_passes_runtime_kwargs_to_state_control(self):
        """Test that runtime_kwargs are passed to state control."""
        state_ctrl = MockStateControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[state_ctrl],
        )
        pipeline.steer()

        runtime_kwargs = {"substrings": ["test"]}
        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
            runtime_kwargs=runtime_kwargs,
        )

        assert state_ctrl._runtime_kwargs_received == runtime_kwargs

    def test_compute_logprobs_output_shape_matches_ref_length(self):
        """Test that output shape matches reference sequence length."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        # Various ref lengths
        for ref_len in [1, 5, 10, 20]:
            ref_output_ids = torch.tensor([[i for i in range(ref_len)]])
            logprobs = pipeline.compute_logprobs(
                input_ids=torch.tensor([[1, 2, 3]]),
                ref_output_ids=ref_output_ids,
            )
            assert logprobs.shape == (1, ref_len)

    def test_compute_logprobs_output_values_are_negative(self):
        """Test that log probabilities are negative (or zero for perfect predictions)."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        # logprobs should be <= 0
        assert (logprobs <= 0).all()

    def test_compute_logprobs_with_attention_mask(self):
        """Test compute_logprobs with explicit attention mask."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        input_ids = torch.tensor([[0, 1, 2, 3]])  # 0 might be pad
        attention_mask = torch.tensor([[0, 1, 1, 1]])  # mask first token

        logprobs = pipeline.compute_logprobs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ref_output_ids=torch.tensor([[4, 5]]),
        )

        assert logprobs is not None
        assert logprobs.shape == (1, 2)

    def test_compute_logprobs_with_batched_inputs(self):
        """Test compute_logprobs with batched inputs."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        batch_size = 4
        input_ids = torch.tensor([[1, 2, 3]] * batch_size)
        ref_output_ids = torch.tensor([[4, 5, 6]] * batch_size)

        logprobs = pipeline.compute_logprobs(
            input_ids=input_ids,
            ref_output_ids=ref_output_ids,
        )

        assert logprobs.shape == (batch_size, 3)

    def test_compute_logprobs_applies_scoring_logits_processors(self):
        """compute_logprobs gathers include_in_scoring logits processors (but never drives a loop)."""
        output_ctrl = MockOutputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[output_ctrl],
        )
        pipeline.steer()

        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        # scoring-time logits processors ARE requested from the output control
        assert output_ctrl._processors_requested

    def test_compute_logprobs_respects_include_in_scoring_opt_out(self):
        """A control with include_in_scoring=False does not contribute during scoring."""

        class OptOutControl(MockOutputControl):
            include_in_scoring = False

        output_ctrl = OptOutControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[output_ctrl],
        )
        pipeline.steer()

        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
        )

        assert not output_ctrl._processors_requested

    def test_compute_logprobs_passes_forward_kwargs(self):
        """Test that forward_kwargs are passed to model forward."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        # Track calls by wrapping the side_effect
        forward_calls = []
        original_side_effect = pipeline.model.side_effect

        def tracking_call(*args, **kwargs):
            forward_calls.append(kwargs)
            return original_side_effect(*args, **kwargs)

        pipeline.model.side_effect = tracking_call

        pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
            output_hidden_states=True,
        )

        assert len(forward_calls) > 0
        assert forward_calls[0].get("output_hidden_states") is True


# Supports Batching Property Tests
class TestPipelineSupportsBatching:
    """Tests for supports_batching property."""

    def test_default_controls_support_batching(self):
        """Test that default (null) controls support batching."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )

        assert pipeline.supports_batching

    def test_non_batching_control_disables_batching(self):
        """Test that non-batching control disables pipeline batching."""
        # MockInputControl has supports_batching = False
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[MockInputControl()],
        )

        assert not pipeline.supports_batching

    def test_all_batching_controls_enables_batching(self):
        """Test that all batching controls enables pipeline batching."""
        # MockStateControl has supports_batching = True
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[MockStateControl()],  # supports_batching = True
        )

        assert pipeline.supports_batching

    def test_mixed_batching_support(self):
        """Test mixed batching support (should be False)."""
        # MockStateControl supports batching, MockInputControl doesn't
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[MockStateControl(), MockInputControl()],
        )

        assert not pipeline.supports_batching

    def test_disabled_control_ignored_for_batching(self):
        """Test that disabled controls are ignored for batching check."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],  # All NoXXXControl which are disabled
        )

        # All default controls are disabled but support batching
        assert pipeline.supports_batching


# Generate Text Tests
class TestPipelineGenerateText:
    """Tests for generate_text convenience method."""

    def test_generate_text_returns_string(self):
        """Test that generate_text returns decoded string."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[],
        )
        pipeline.steer()

        result = pipeline.generate_text(torch.tensor([[1, 2, 3]]))

        # Mock tokenizer returns list of strings
        assert isinstance(result, list)


# Error Handling Tests
class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    def test_duplicate_control_type_composes_in_order(self):
        """Two controls of the same category are accepted and kept in list order."""
        ctrl1, ctrl2 = MockInputControl(), MockInputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[ctrl1, ctrl2],
        )
        assert pipeline.input_controls == [ctrl1, ctrl2]

    def test_unknown_control_type_raises(self):
        """Test that unknown control type raises error."""
        class UnknownControl:
            pass

        with pytest.raises(TypeError, match="Unknown"):
            MockSteeringPipeline(
                model_name_or_path="test-model",
                controls=[UnknownControl()],
            )


# Integration Tests
class TestPipelineIntegration:
    """Integration tests for pipeline workflows."""

    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow."""
        # Setup controls
        input_ctrl = MockInputControl(prefix="test_")
        state_ctrl = MockStateControl(target_layers=[0, 1])

        # Create pipeline
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[input_ctrl, state_ctrl],
        )

        # Steer
        pipeline.steer()

        # Generate
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        runtime_kwargs = {"key": "value"}

        output = pipeline.generate(
            input_ids,
            runtime_kwargs=runtime_kwargs,
            max_new_tokens=10,
        )

        # Verify workflow
        assert pipeline._is_steered
        assert input_ctrl._adapter_call_count > 0
        assert state_ctrl._hooks_created
        assert state_ctrl._runtime_kwargs_received == runtime_kwargs
        assert output is not None

    def test_multiple_generations(self):
        """Test multiple generations with same pipeline."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[MockStateControl()],
        )
        pipeline.steer()

        # Multiple generate calls
        for i in range(5):
            output = pipeline.generate(
                torch.tensor([[1, 2, 3]]),
                runtime_kwargs={"iteration": i},
            )
            assert output is not None

    def test_different_runtime_kwargs_per_call(self):
        """Test different runtime_kwargs for each generate call."""
        state_ctrl = MockStateControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[state_ctrl],
        )
        pipeline.steer()

        # First call
        pipeline.generate(torch.tensor([[1]]), runtime_kwargs={"call": 1})
        assert state_ctrl._runtime_kwargs_received == {"call": 1}

        # Second call with different kwargs
        pipeline.generate(torch.tensor([[2]]), runtime_kwargs={"call": 2})
        assert state_ctrl._runtime_kwargs_received == {"call": 2}

    def test_generate_and_compute_logprobs_same_pipeline(self):
        """Test using both generate and compute_logprobs on same pipeline."""
        state_ctrl = MockStateControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[state_ctrl],
        )
        pipeline.steer()

        # Generate first
        output = pipeline.generate(
            torch.tensor([[1, 2, 3]]),
            runtime_kwargs={"mode": "generate"},
        )
        assert output is not None

        # Then compute logprobs
        logprobs = pipeline.compute_logprobs(
            input_ids=torch.tensor([[1, 2, 3]]),
            ref_output_ids=torch.tensor([[4, 5, 6]]),
            runtime_kwargs={"mode": "logprobs"},
        )
        assert logprobs is not None
        assert state_ctrl._runtime_kwargs_received == {"mode": "logprobs"}

    def test_multiple_compute_logprobs_calls(self):
        """Test multiple compute_logprobs calls with same pipeline."""
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[MockStateControl()],
        )
        pipeline.steer()

        for i in range(5):
            logprobs = pipeline.compute_logprobs(
                input_ids=torch.tensor([[1, 2, 3]]),
                ref_output_ids=torch.tensor([[4, 5, 6]]),
                runtime_kwargs={"iteration": i},
            )
            assert logprobs is not None
            assert logprobs.shape == (1, 3)


# Duplicate-BOS guard (WS3): uses the REAL SteeringPipeline with a tiny hub-free model.
class TestDuplicateBosGuard:
    """`warn_if_duplicate_bos` warns once when a prompt starts with two BOS tokens."""

    def _steered_pipeline(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2)
        tokenizer = wordlevel_tokenizer()  # bos_token_id == 0
        pipeline = RealSteeringPipeline(lazy_init=True)
        pipeline.model = model
        pipeline.tokenizer = tokenizer
        pipeline.steer()
        return pipeline, tokenizer

    def test_double_bos_warns_once_across_two_calls(self, caplog):
        pipeline, tokenizer = self._steered_pipeline()
        bos = tokenizer.bos_token_id
        ids = torch.tensor([[bos, bos, 3, 4]])
        with caplog.at_level(logging.WARNING, logger="aisteer360.utils.tokenization"):
            pipeline.generate(ids, max_new_tokens=1)
            pipeline.generate(ids, max_new_tokens=1)
        dup_warnings = [r for r in caplog.records if "Duplicate BOS" in r.getMessage()]
        assert len(dup_warnings) == 1  # warn-once per pipeline lifecycle

    def test_single_bos_does_not_warn(self, caplog):
        pipeline, tokenizer = self._steered_pipeline()
        bos = tokenizer.bos_token_id
        ids = torch.tensor([[bos, 3, 4]])
        with caplog.at_level(logging.WARNING, logger="aisteer360.utils.tokenization"):
            pipeline.generate(ids, max_new_tokens=1)
        assert not [r for r in caplog.records if "Duplicate BOS" in r.getMessage()]

    def test_left_padded_double_bos_warns(self, caplog):
        # a left-padded batch [pad, pad, bos, bos, x] with the correct mask must still fire the guard,
        # proving the first-real-token (argmax) logic rather than a fixed position-0 check
        pipeline, tokenizer = self._steered_pipeline()
        bos = tokenizer.bos_token_id
        pad = tokenizer.pad_token_id
        ids = torch.tensor([[pad, pad, bos, bos, 3]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
        with caplog.at_level(logging.WARNING, logger="aisteer360.utils.tokenization"):
            pipeline.generate(ids, attention_mask=attention_mask, max_new_tokens=1)
        assert [r for r in caplog.records if "Duplicate BOS" in r.getMessage()]


from aisteer360.algorithms.output_control.base import OutputControl as RealOutputControl


class TestSameModelForwardsMetadata:
    """`same_model_forwards` is declarative component metadata on the declaring classes."""

    def test_declared_flags(self):
        from aisteer360.algorithms.output_control._common.logit_sources import PromptVariantSource
        from aisteer360.algorithms.output_control._common.values.subspace_margin import SubspaceMarginValue
        from aisteer360.algorithms.output_control.sasa.control import SASA

        assert SASA.same_model_forwards is True
        assert SubspaceMarginValue.same_model_forwards is True
        assert PromptVariantSource.same_model_forwards is True
        assert RealOutputControl.same_model_forwards is False

    def test_prompt_variant_source_construction_emits_no_warning(self):
        from aisteer360.algorithms.output_control._common.logit_sources import PromptVariantSource

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PromptVariantSource(lambda text: text)
