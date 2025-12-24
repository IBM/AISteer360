"""
Shared fixtures and mock classes for tests.

This module provides:

- Device and model fixtures for integration tests
- Mock base classes
- Generic mock controls for each category (Input, Structural, State, Output)
- Common test fixtures for evaluation data, metrics, and use cases
- Utility functions used across test modules
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.utils.load_ci_models import get_models

# Real Model/Device Fixtures (for integration tests)
MODELS = get_models()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    """Parametrized device fixture for testing across CPU/CUDA/MPS."""
    name = request.param
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if name == "mps":
        has_mps = (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
        )
        if not has_mps:
            pytest.skip("MPS not available.")
    return torch.device(name)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(repo, id=tag)
        for tag, repo in MODELS.items()
    ],
)
def model_and_tokenizer(request):
    """
    Loads each model once per test session.
    """
    model_id: str = request.param
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load {model_id}: {exc}")

    # ensure padding token exists for batching
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:  # edge case
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return model, tokenizer


# Mock Base Classes
@dataclass
class BaseArgs:
    """Base class for all method's args classes."""

    @classmethod
    def validate(cls, data=None, **kwargs):
        """Create and validate an Args instance from dict, kwargs, or existing instance."""
        if isinstance(data, cls):
            return data
        if isinstance(data, Mapping):
            kwargs = {**data, **kwargs}
        return cls(**kwargs)


class Metric:
    """Base metric class for evaluation."""

    def __init__(self, **extras: Any) -> None:
        self.name: str = self.__class__.__name__
        self.extras: dict[str, Any] = extras

    def compute(self, responses: list[Any], prompts: list[str] | None = None, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


# Control Base Classes
class InputControl:
    """Base class for input control steering methods."""

    Args: type[BaseArgs] | None = None
    enabled: bool = True
    supports_batching: bool = False

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is not None:
            self.args = self.Args.validate(*args, **kwargs)
            for f in self.args.__dataclass_fields__:
                setattr(self, f, getattr(self.args, f))

    def get_prompt_adapter(self, runtime_kwargs: dict | None = None) -> Callable:
        """Return transformation function for input_ids."""
        return lambda ids, _: ids

    def steer(self, model=None, tokenizer=None, **kwargs) -> None:
        """Optional steering/preparation."""
        pass


class StructuralControl:
    """Base class for structural control steering methods."""

    Args: type[BaseArgs] | None = None
    enabled: bool = True
    supports_batching: bool = True

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is not None:
            self.args = self.Args.validate(*args, **kwargs)
            for f in self.args.__dataclass_fields__:
                setattr(self, f, getattr(self.args, f))

    def steer(self, model, tokenizer=None, **kwargs):
        """Required steering/preparation - returns modified model."""
        return model


class StateControl:
    """Base class for state control steering methods."""

    Args: type[BaseArgs] | None = None
    enabled: bool = True
    supports_batching: bool = False
    _model_ref = None

    def __init__(self, *args, **kwargs) -> None:
        self.hooks: dict[str, list] = {"pre": [], "forward": [], "backward": []}
        self.registered: list = []
        if self.Args is not None:
            self.args = self.Args.validate(*args, **kwargs)
            for f in self.args.__dataclass_fields__:
                setattr(self, f, getattr(self.args, f))

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs) -> dict[str, list]:
        """Create hook specifications for the current generation."""
        return {"pre": [], "forward": [], "backward": []}

    def steer(self, model, tokenizer=None, **kwargs) -> None:
        """Optional steering/preparation."""
        pass

    def set_hooks(self, hooks: dict[str, list]) -> None:
        self.hooks = hooks

    def register_hooks(self, model) -> None:
        pass

    def remove_hooks(self) -> None:
        self.registered.clear()

    def reset(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class OutputControl:
    """Base class for output control steering methods."""

    Args: type[BaseArgs] | None = None
    enabled: bool = True
    supports_batching: bool = False

    def __init__(self, *args, **kwargs) -> None:
        if self.Args is not None:
            self.args = self.Args.validate(*args, **kwargs)
            for f in self.args.__dataclass_fields__:
                setattr(self, f, getattr(self.args, f))

    def generate(self, input_ids, attention_mask, runtime_kwargs, model, **gen_kwargs) -> torch.Tensor:
        """Custom generation logic."""
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    def steer(self, model, tokenizer=None, **kwargs) -> None:
        """Optional steering/preparation."""
        pass


# Null Control Classes (identity/no-op implementations)
class NoInputControl(InputControl):
    """Identity input control - returns input unchanged."""
    enabled: bool = False
    supports_batching: bool = True


class NoStructuralControl(StructuralControl):
    """Identity structural control - returns model unchanged."""
    enabled: bool = False

    def steer(self, model, **__):
        return model


class NoStateControl(StateControl):
    """Identity state control - no hooks registered."""
    enabled: bool = False
    supports_batching: bool = True


class NoOutputControl(OutputControl):
    """Identity output control - uses default model.generate()."""
    enabled: bool = False
    supports_batching: bool = True


# Generic Mock Controls
@dataclass
class MockInputArgs(BaseArgs):
    """Args for a generic input control."""
    prefix: str = ""
    suffix: str = ""
    num_examples: int = 0


class MockInputControl(InputControl):
    """
    Generic mock input control for testing.

    Simulates prompt modification by optionally prepending/appending tokens.
    """
    Args = MockInputArgs
    supports_batching: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._adapter_call_count = 0

    def get_prompt_adapter(self, runtime_kwargs: dict | None = None):
        def adapter(input_ids, rt_kwargs):
            self._adapter_call_count += 1
            # In real implementation, would modify input_ids based on prefix/suffix
            return input_ids

        return adapter

    def steer(self, model=None, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer


@dataclass
class MockStructuralArgs(BaseArgs):
    """Args for a generic structural control."""
    learning_rate: float = 1e-4
    num_epochs: int = 1
    output_dir: str = "./output"


class MockStructuralControl(StructuralControl):
    """
    Generic mock structural control for testing.

    Simulates model modification (e.g., fine-tuning, adapter injection).
    """
    Args = MockStructuralArgs
    supports_batching: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._steer_called = False

    def steer(self, model, tokenizer=None, **kwargs):
        self._steer_called = True
        self.model = model
        self.tokenizer = tokenizer
        # In real implementation, would modify model weights
        return model


@dataclass
class MockStateArgs(BaseArgs):
    """Args for a generic state control."""
    target_layers: list = field(default_factory=lambda: [0, 1])
    scale_factor: float = 1.0
    mode: str = "add"


class MockStateControl(StateControl):
    """
    Generic mock state control for testing.

    Simulates activation steering via hooks.
    """
    Args = MockStateArgs
    supports_batching: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hooks_created = False
        self._runtime_kwargs_received = None

    def get_hooks(self, input_ids: torch.Tensor, runtime_kwargs: dict | None, **kwargs):
        self._hooks_created = True
        self._runtime_kwargs_received = runtime_kwargs

        # Simulate creating hooks for target layers
        hooks = {"pre": [], "forward": [], "backward": []}
        if hasattr(self, 'target_layers'):
            for layer in self.target_layers:
                hooks["pre"].append({
                    "module": f"model.layers.{layer}",
                    "hook_func": lambda *args, **kw: None,
                })
        return hooks

    def steer(self, model, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = getattr(model, 'device', torch.device('cpu'))


@dataclass
class MockOutputArgs(BaseArgs):
    """Args for a generic output control."""
    temperature: float = 1.0
    top_k: int = 50
    constraint_type: str = "none"


class MockOutputControl(OutputControl):
    """
    Generic mock output control for testing.

    Simulates custom decoding logic.
    """
    Args = MockOutputArgs
    supports_batching: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._generate_called = False
        self._runtime_kwargs_received = None

    def generate(self, input_ids, attention_mask, runtime_kwargs, model, **gen_kwargs):
        self._generate_called = True
        self._runtime_kwargs_received = runtime_kwargs
        # Delegate to model but track the call
        return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    def steer(self, model, tokenizer=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer


# Mock Metrics
class MockAccuracyMetric(Metric):
    """Simple accuracy metric for testing."""

    def compute(
            self,
            responses: list[str],
            reference_answers: list[str] = None,
            **kwargs
    ) -> dict[str, float]:
        if reference_answers is None:
            return {"accuracy": 0.0}

        correct = sum(1 for r, ref in zip(responses, reference_answers) if r == ref)
        accuracy = correct / len(responses) if responses else 0.0
        return {"accuracy": accuracy}


class MockScoreMetric(Metric):
    """Simple score metric that returns a fixed value for testing."""

    def __init__(self, fixed_score: float = 0.5, **extras):
        super().__init__(**extras)
        self.fixed_score = fixed_score

    def compute(self, responses: list[str], **kwargs) -> dict[str, float]:
        return {"score": self.fixed_score}


class MockPerSampleMetric(Metric):
    """Metric that returns per-sample scores for testing."""

    def compute(self, responses: list[str], **kwargs) -> dict[str, Any]:
        scores = [0.5 + 0.1 * i for i in range(len(responses))]
        return {
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "scores": scores,
        }


# Mock UseCase
class MockUseCase:
    """Mock UseCase for testing benchmark functionality."""

    def __init__(
            self,
            evaluation_data: list[dict],
            evaluation_metrics: list[Metric],
            num_samples: int = -1,
            **kwargs
    ):
        self.evaluation_data = list(evaluation_data)
        if num_samples > 0:
            self.evaluation_data = self.evaluation_data[:num_samples]
        self.evaluation_metrics = evaluation_metrics
        self._metrics_by_name = {m.name: m for m in evaluation_metrics}

        for key, value in kwargs.items():
            setattr(self, key, value)

        # Track calls for testing
        self._generate_calls = []
        self._evaluate_calls = []

    def generate(
            self,
            model_or_pipeline,
            tokenizer,
            gen_kwargs=None,
            runtime_overrides=None,
            **kwargs
    ) -> list[dict[str, Any]]:
        """Mock generation that returns predictable outputs."""
        self._generate_calls.append({
            "model_or_pipeline": model_or_pipeline,
            "tokenizer": tokenizer,
            "gen_kwargs": gen_kwargs,
            "runtime_overrides": runtime_overrides,
            "kwargs": kwargs,
        })

        generations = []
        for item in self.evaluation_data:
            generations.append({
                "response": "A",
                "prompt": item.get("question", item.get("prompt", "test prompt")),
                "question_id": item.get("id", "test_id"),
                "reference_answer": item.get("answer", "A"),
            })
        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Mock evaluation that computes metrics."""
        self._evaluate_calls.append(generations)

        eval_data = {
            "responses": [g["response"] for g in generations],
            "reference_answers": [g["reference_answer"] for g in generations],
            "question_ids": [g["question_id"] for g in generations],
        }

        scores = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)

        return scores

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Export profiles to JSON."""
        with open(Path(save_dir) / "profiles.json", "w") as f:
            json.dump(profiles, f, indent=4)


# Utility Functions
_DEFAULT_FACTORIES: dict[type, type] = {
    InputControl: NoInputControl,
    StructuralControl: NoStructuralControl,
    StateControl: NoStateControl,
    OutputControl: NoOutputControl,
}


def merge_controls(supplied) -> dict[str, object]:
    """Sort supplied controls by category and ensure at most one per category."""
    bucket: dict[type, list] = defaultdict(list)
    for control in supplied:
        for category in _DEFAULT_FACTORIES:
            if isinstance(control, category):
                bucket[category].append(control)
                break
        else:
            raise TypeError(f"Unknown control type: {type(control)}")

    for category, controls in bucket.items():
        if len(controls) > 1:
            names = [type(c).__name__ for c in controls]
            raise ValueError(f"Multiple {category.__name__}s supplied: {names}")

    out: dict[str, object] = {}
    for category, factory in _DEFAULT_FACTORIES.items():
        instance = bucket.get(category, [factory()])[0]
        out_key = (
            "input_control" if category is InputControl else
            "structural_control" if category is StructuralControl else
            "state_control" if category is StateControl else
            "output_control"
        )
        out[out_key] = instance
    return out


def ensure_pad_token(tokenizer):
    """Set pad token to eos token if not defined."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def create_mock_model(device: str = "cpu") -> MagicMock:
    """Create a mock model for testing."""
    model = MagicMock()
    model.device = torch.device(device)
    model.config = MagicMock()
    model.config.num_attention_heads = 8
    model.config.num_hidden_layers = 12

    def mock_generate(input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        new_tokens = kwargs.get("max_new_tokens", 10)
        return torch.randint(0, 1000, (batch_size, seq_len + new_tokens))

    model.generate = MagicMock(side_effect=mock_generate)
    model.parameters = MagicMock(return_value=iter([torch.tensor([1.0])]))
    return model


def create_mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.padding_side = "left"

    def mock_call(text, **kwargs):
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, 10)),
            "attention_mask": torch.ones(batch_size, 10),
        }

    tokenizer.side_effect = mock_call
    tokenizer.batch_decode = MagicMock(return_value=["decoded text"])
    tokenizer.decode = MagicMock(return_value="decoded text")
    return tokenizer


# Common Test Data Fixtures
@pytest.fixture
def sample_evaluation_data() -> list[dict]:
    """Sample evaluation data for testing."""
    return [
        {"id": "q1", "question": "What is 2+2?", "answer": "A", "choices": ["4", "5", "6", "7"]},
        {"id": "q2", "question": "Capital of France?", "answer": "B", "choices": ["London", "Paris", "Berlin"]},
        {"id": "q3", "question": "Closest planet to sun?", "answer": "A", "choices": ["Mercury", "Venus", "Earth"]},
    ]


@pytest.fixture
def large_evaluation_data() -> list[dict]:
    """Larger evaluation dataset for testing."""
    return [
        {"id": f"q{i}", "question": f"Question {i}?", "answer": "A", "choices": ["A", "B", "C", "D"]}
        for i in range(100)
    ]


@pytest.fixture
def evaluation_data_with_metadata() -> list[dict]:
    """Evaluation data with additional metadata fields."""
    return [
        {
            "id": "q1",
            "question": "Test question",
            "answer": "A",
            "instructions": ["instruction1", "instruction2"],
            "context": "Some context",
            "metadata": {"source": "test"},
        },
    ]


@pytest.fixture
def sample_metrics() -> list[Metric]:
    """Sample metrics for testing."""
    return [MockAccuracyMetric(), MockScoreMetric()]


@pytest.fixture
def sample_use_case(sample_evaluation_data, sample_metrics) -> MockUseCase:
    """Sample use case for testing."""
    return MockUseCase(
        evaluation_data=sample_evaluation_data,
        evaluation_metrics=sample_metrics,
    )


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock model fixture."""
    return create_mock_model()


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Mock tokenizer fixture."""
    return create_mock_tokenizer()


@pytest.fixture
def mock_input_control() -> MockInputControl:
    """Mock input control fixture."""
    return MockInputControl(prefix="test_", num_examples=3)


@pytest.fixture
def mock_structural_control() -> MockStructuralControl:
    """Mock structural control fixture."""
    return MockStructuralControl(learning_rate=1e-5, num_epochs=2)


@pytest.fixture
def mock_state_control() -> MockStateControl:
    """Mock state control fixture."""
    return MockStateControl(target_layers=[0, 1, 2], scale_factor=0.5)


@pytest.fixture
def mock_output_control() -> MockOutputControl:
    """Mock output control fixture."""
    return MockOutputControl(temperature=0.7, top_k=40)
