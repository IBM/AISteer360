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
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.conftest import (  # Base classes; Mock controls; Utilities
    InputControl,
    MockInputControl,
    MockOutputControl,
    MockStateControl,
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
        self.structural_control = controls_merged["structural_control"]
        self.input_control = controls_merged["input_control"]
        self.state_control = controls_merged["state_control"]
        self.output_control = controls_merged["output_control"]

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
            self.structural_control,
            self.input_control,
            self.state_control,
            self.output_control,
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
            self.structural_control,
            self.input_control,
            self.state_control,
            self.output_control
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

        # Apply input control adapter
        adapter = self.input_control.get_prompt_adapter(runtime_kwargs)
        steered_input_ids = adapter(input_ids, runtime_kwargs)

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
        hooks = self.state_control.get_hooks(steered_input_ids, runtime_kwargs, **kwargs)
        self.state_control.set_hooks(hooks)
        self.state_control._model_ref = self.model
        self.state_control.reset()

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

        # Output control: generate with hooks active
        with self.state_control:
            output_ids = self.output_control.generate(
                input_ids=steered_input_ids,
                attention_mask=attention_mask,
                runtime_kwargs=runtime_kwargs,
                model=self.model,
                **gen_kwargs
            )

        return output_ids

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

        with self.state_control:
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

        assert pipeline.input_control is input_ctrl
        assert pipeline.state_control is state_ctrl
        assert isinstance(pipeline.structural_control, NoStructuralControl)
        assert isinstance(pipeline.output_control, NoOutputControl)

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

        assert pipeline.input_control is input_ctrl
        assert pipeline.structural_control is structural_ctrl
        assert pipeline.state_control is state_ctrl
        assert pipeline.output_control is output_ctrl

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

    def test_generate_calls_output_control(self):
        """Test that generate calls output control's generate."""
        output_ctrl = MockOutputControl()
        pipeline = MockSteeringPipeline(
            model_name_or_path="test-model",
            controls=[output_ctrl],
        )
        pipeline.steer()

        pipeline.generate(torch.tensor([[1, 2, 3]]))

        assert output_ctrl._generate_called


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

    def test_compute_logprobs_does_not_call_output_control(self):
        """Test that compute_logprobs does NOT use output control."""
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

        # Output control's generate should NOT be called
        assert not output_ctrl._generate_called

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

    def test_duplicate_control_type_raises(self):
        """Test that duplicate control types raise error."""
        with pytest.raises(ValueError, match="Multiple"):
            MockSteeringPipeline(
                model_name_or_path="test-model",
                controls=[MockInputControl(), MockInputControl()],
            )

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
