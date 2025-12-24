"""
Tests for Benchmark functionality.

Tests cover:

- Benchmark initialization and validation
- Running benchmarks with baseline (no controls)
- Running benchmarks with fixed controls
- Running benchmarks with runtime_overrides
- Multi-trial evaluation
- Export functionality
- ControlSpec for parameter sweeps
- Error handling
"""
import itertools
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock

import pytest
import torch

from tests.conftest import (  # Base classes; Mock controls; Mock metrics and use case; Utilities
    InputControl,
    Metric,
    MockAccuracyMetric,
    MockInputControl,
    MockOutputControl,
    MockPerSampleMetric,
    MockScoreMetric,
    MockStateControl,
    MockStructuralControl,
    MockUseCase,
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


# ControlSpec Mock Implementation
@dataclass
class ControlSpec:
    """
    Specification for a control with variable parameters.

    Allows defining parameter sweeps for benchmarking by specifying:
    - Fixed parameters (params)
    - Variable parameters (vars) in multiple formats

    Variable parameter formats:
    - dict[str, list]: Cartesian product of all values
    - list[dict]: Explicit list of parameter combinations
    - Callable: Function returning iterator of parameter dicts
    """
    control_cls: type
    params: dict[str, Any] = field(default_factory=dict)
    vars: dict[str, list] | list[dict] | Callable[[dict], Iterator[dict]] | None = None
    name: str | None = None

    def iter_points(self, context: dict | None = None) -> Iterator[dict[str, Any]]:
        """
        Iterate over all parameter combinations defined by vars.

        Args:
            context: Optional context dict passed to callable vars

        Yields:
            Dict of variable parameter values for each combination
        """
        context = context or {}

        if self.vars is None:
            yield {}
            return

        if callable(self.vars):
            # Callable that returns iterator
            yield from self.vars(context)
        elif isinstance(self.vars, list):
            # Explicit list of combinations
            yield from self.vars
        elif isinstance(self.vars, dict):
            # Cartesian product of all variable values
            if not self.vars:
                yield {}
                return
            keys = list(self.vars.keys())
            value_lists = [self.vars[k] for k in keys]
            for combo in itertools.product(*value_lists):
                yield dict(zip(keys, combo))
        else:
            raise TypeError(f"vars must be dict, list, or callable, got {type(self.vars)}")

    def resolve_params(
        self,
        chosen: dict[str, Any],
        context: dict | None = None
    ) -> dict[str, Any]:
        """
        Resolve full parameter dict by merging fixed params with chosen variable values.

        Args:
            chosen: Variable parameter values for this configuration
            context: Optional context (unused in basic implementation)

        Returns:
            Complete parameter dict for control instantiation
        """
        resolved = dict(self.params)
        resolved.update(chosen)
        return resolved


# Mock SteeringPipeline (for Benchmark testing)
@dataclass
class MockSteeringPipeline:
    """Mock SteeringPipeline for benchmark testing."""
    model_name_or_path: str | None = None
    controls: list = field(default_factory=list)
    tokenizer_name_or_path: str | None = None
    device_map: str = "auto"
    hf_model_kwargs: dict = field(default_factory=dict)
    lazy_init: bool = False

    def __post_init__(self):
        self._is_steered = False
        controls_merged = merge_controls(self.controls)
        self.structural_control = controls_merged["structural_control"]
        self.state_control = controls_merged["state_control"]
        self.input_control = controls_merged["input_control"]
        self.output_control = controls_merged["output_control"]

        self.model = create_mock_model()
        self.tokenizer = create_mock_tokenizer()
        self.device = self.model.device

    @property
    def supports_batching(self):
        controls = (self.structural_control, self.state_control,
                    self.input_control, self.output_control)
        return all(
            getattr(c, "supports_batching", False)
            for c in controls if getattr(c, "enabled", True)
        )

    def steer(self, **kwargs):
        if self._is_steered:
            return
        for control in (self.structural_control, self.state_control,
                        self.input_control, self.output_control):
            steer_fn = getattr(control, "steer", None)
            if callable(steer_fn):
                maybe_new_model = steer_fn(self.model, tokenizer=self.tokenizer, **kwargs)
                if maybe_new_model is not None:
                    self.model = maybe_new_model
        self._is_steered = True

    def generate(self, input_ids, attention_mask=None, runtime_kwargs=None, **gen_kwargs):
        if not self._is_steered:
            raise RuntimeError("Must call .steer() before .generate()")
        batch_size = input_ids.shape[0] if hasattr(input_ids, 'shape') else 1
        return torch.tensor([[1, 2, 3]] * batch_size)


# Mock Benchmark Class
class Benchmark:
    """Mock Benchmark class for testing."""

    def __init__(
        self,
        use_case,
        base_model_name_or_path: str,
        steering_pipelines: dict[str, list[Any]],
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        hf_model_kwargs: dict | None = None,
        gen_kwargs: dict | None = None,
        device_map: str = "auto",
        num_trials: int = 1,
        batch_size: int = 8,
    ):
        self.use_case = use_case
        self.base_model_name_or_path = base_model_name_or_path
        self.steering_pipelines = steering_pipelines
        self.runtime_overrides = runtime_overrides
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}
        self.device_map = device_map
        self.num_trials = int(num_trials)
        self.batch_size = int(batch_size)

        self._base_model = None
        self._base_tokenizer = None

    def _ensure_base_model(self):
        if self._base_model is not None:
            return
        self._base_model = create_mock_model()
        self._base_tokenizer = create_mock_tokenizer()

    @staticmethod
    def _has_structural_control(controls):
        return any(
            isinstance(c, StructuralControl) and getattr(c, "enabled", True)
            for c in controls
        )

    def run(self) -> dict[str, list[dict[str, Any]]]:
        profiles = {}
        for pipeline_name, pipeline in self.steering_pipelines.items():
            pipeline = pipeline or []

            # Check if pipeline contains ControlSpecs
            has_specs = any(isinstance(c, ControlSpec) for c in pipeline)

            if has_specs:
                # Validate: all must be ControlSpec or none
                if not all(isinstance(c, ControlSpec) for c in pipeline):
                    raise TypeError(
                        f"Pipeline '{pipeline_name}' mixes ControlSpec and concrete controls. "
                        "Use only ControlSpecs or only concrete controls."
                    )
                runs = self._run_spec_pipeline(pipeline_name, control_specs=pipeline)
            else:
                runs = self._run_pipeline(controls=pipeline, params=None)

            profiles[pipeline_name] = runs
        return profiles

    def _run_pipeline(
        self,
        controls: list[Any],
        params: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        runs = []
        try:
            self._ensure_base_model()

            if controls:
                pipeline = MockSteeringPipeline(
                    model_name_or_path=self.base_model_name_or_path,
                    controls=controls,
                    device_map=self.device_map,
                    hf_model_kwargs=self.hf_model_kwargs,
                )
                pipeline.steer()
                tokenizer = pipeline.tokenizer
                model_or_pipeline = pipeline
            else:
                model_or_pipeline = self._base_model
                tokenizer = self._base_tokenizer

            for trial_id in range(self.num_trials):
                generations = self.use_case.generate(
                    model_or_pipeline=model_or_pipeline,
                    tokenizer=tokenizer,
                    gen_kwargs=self.gen_kwargs,
                    runtime_overrides=self.runtime_overrides,
                    batch_size=self.batch_size,
                )
                scores = self.use_case.evaluate(generations)

                runs.append({
                    "trial_id": trial_id,
                    "generations": generations,
                    "evaluations": scores,
                    "params": params or {},
                })

            return runs
        finally:
            pass

    def export(self, profiles: dict[str, list[dict[str, Any]]], save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.use_case.export(profiles, save_dir)

    def _run_spec_pipeline(
        self,
        pipeline_name: str,
        control_specs: list[ControlSpec],
    ) -> list[dict[str, Any]]:
        """
        Run a pipeline whose controls are defined by ControlSpecs.

        Expands each ControlSpec into parameter combinations, takes cartesian product,
        and evaluates each configuration.
        """
        base_context = {
            "pipeline_name": pipeline_name,
            "base_model_name_or_path": self.base_model_name_or_path,
        }

        # Collect points per spec
        spec_points: list[tuple[ControlSpec, list[dict[str, Any]]]] = []
        for spec in control_specs:
            points = list(spec.iter_points(base_context))
            if not points:
                points = [{}]
            spec_points.append((spec, points))

        if not spec_points:
            return self._run_pipeline(controls=[], params=None)

        spec_list, points_lists = zip(*spec_points)
        combos = itertools.product(*points_lists)

        runs: list[dict[str, Any]] = []

        for combo_id, combo in enumerate(combos):
            params_by_spec: dict[str, dict[str, Any]] = {}
            controls_for_combo: list[Any] = []

            global_context = {
                "pipeline_name": pipeline_name,
                "base_model_name_or_path": self.base_model_name_or_path,
                "combo_id": combo_id,
            }

            for spec, local_point in zip(spec_list, combo):
                spec_name = spec.name or spec.control_cls.__name__
                kwargs = spec.resolve_params(chosen=local_point, context=global_context)
                control = spec.control_cls(**kwargs)
                controls_for_combo.append(control)
                params_by_spec[spec_name] = kwargs

            run = self._run_pipeline(
                controls=controls_for_combo,
                params=params_by_spec,
            )
            runs.extend(run)

        return runs


# Test Fixtures
@pytest.fixture
def basic_use_case(sample_evaluation_data, sample_metrics):
    """Basic use case for benchmark testing."""
    return MockUseCase(
        evaluation_data=sample_evaluation_data,
        evaluation_metrics=sample_metrics,
    )


@pytest.fixture
def use_case_with_tracking(sample_evaluation_data):
    """Use case that tracks calls for verification."""
    return MockUseCase(
        evaluation_data=sample_evaluation_data,
        evaluation_metrics=[MockAccuracyMetric()],
    )


# Benchmark Initialization Tests
class TestBenchmarkInitialization:
    """Tests for Benchmark initialization."""

    def test_basic_initialization(self, basic_use_case):
        """Test basic benchmark initialization."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        assert benchmark.use_case is basic_use_case
        assert benchmark.base_model_name_or_path == "test-model"
        assert "baseline" in benchmark.steering_pipelines
        assert benchmark.num_trials == 1
        assert benchmark.batch_size == 8

    def test_initialization_with_all_options(self, basic_use_case):
        """Test benchmark initialization with all optional arguments."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": [], "test": []},
            runtime_overrides={"TestControl": {"param": "column"}},
            hf_model_kwargs={"torch_dtype": "float16"},
            gen_kwargs={"max_new_tokens": 100},
            device_map="auto",
            num_trials=3,
            batch_size=16,
        )

        assert benchmark.num_trials == 3
        assert benchmark.batch_size == 16
        assert benchmark.gen_kwargs["max_new_tokens"] == 100
        assert benchmark.hf_model_kwargs["torch_dtype"] == "float16"
        assert benchmark.runtime_overrides is not None

    def test_initialization_with_multiple_pipelines(self, basic_use_case):
        """Test initialization with multiple steering pipelines."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "baseline": [],
                "input_steered": [MockInputControl()],
                "state_steered": [MockStateControl()],
            },
        )

        assert len(benchmark.steering_pipelines) == 3


# Benchmark Run Tests
class TestBenchmarkRun:
    """Tests for Benchmark.run() method."""

    def test_run_baseline_only(self, basic_use_case):
        """Test running benchmark with baseline (no steering)."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()

        assert "baseline" in profiles
        assert len(profiles["baseline"]) == 1
        assert "trial_id" in profiles["baseline"][0]
        assert "generations" in profiles["baseline"][0]
        assert "evaluations" in profiles["baseline"][0]
        assert "params" in profiles["baseline"][0]

    def test_run_with_input_control(self, basic_use_case):
        """Test running benchmark with an input control."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"input_steered": [MockInputControl()]},
        )

        profiles = benchmark.run()

        assert "input_steered" in profiles
        assert len(profiles["input_steered"]) == 1

    def test_run_with_state_control(self, basic_use_case):
        """Test running benchmark with a state control."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"state_steered": [MockStateControl()]},
        )

        profiles = benchmark.run()

        assert "state_steered" in profiles
        assert len(profiles["state_steered"]) == 1

    def test_run_with_structural_control(self, basic_use_case):
        """Test running benchmark with a structural control."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"structural_steered": [MockStructuralControl()]},
        )

        profiles = benchmark.run()

        assert "structural_steered" in profiles

    def test_run_with_output_control(self, basic_use_case):
        """Test running benchmark with an output control."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"output_steered": [MockOutputControl()]},
        )

        profiles = benchmark.run()

        assert "output_steered" in profiles

    def test_run_multiple_trials(self, basic_use_case):
        """Test running benchmark with multiple trials."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            num_trials=5,
        )

        profiles = benchmark.run()

        assert len(profiles["baseline"]) == 5
        for i, run in enumerate(profiles["baseline"]):
            assert run["trial_id"] == i

    def test_run_multiple_pipelines(self, basic_use_case):
        """Test running benchmark with multiple pipelines."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "baseline": [],
                "input_steered": [MockInputControl()],
                "state_steered": [MockStateControl()],
            },
            num_trials=2,
        )

        profiles = benchmark.run()

        assert len(profiles) == 3
        assert all(len(runs) == 2 for runs in profiles.values())

    def test_run_combined_controls(self, basic_use_case):
        """Test running benchmark with combined controls."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "combined": [MockInputControl(), MockStateControl()],
            },
        )

        profiles = benchmark.run()

        assert "combined" in profiles


# Benchmark Evaluation Tests
class TestBenchmarkEvaluation:
    """Tests for benchmark evaluation logic."""

    def test_evaluation_scores_structure(self, basic_use_case):
        """Test that evaluation scores have correct structure."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()
        evaluations = profiles["baseline"][0]["evaluations"]

        assert "MockAccuracyMetric" in evaluations
        assert "MockScoreMetric" in evaluations

    def test_generations_structure(self, sample_evaluation_data):
        """Test that generations have correct structure."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()
        generations = profiles["baseline"][0]["generations"]

        assert len(generations) == len(sample_evaluation_data)
        for gen in generations:
            assert "response" in gen
            assert "prompt" in gen
            assert "question_id" in gen
            assert "reference_answer" in gen

    def test_metrics_called_with_generations(self, sample_evaluation_data):
        """Test that metrics are called with generation data."""
        # Create a tracking metric
        class TrackingMetric(Metric):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.last_args = None

            def compute(self, responses, **kwargs):
                self.call_count += 1
                self.last_args = {"responses": responses, **kwargs}
                return {"tracked": 1.0}

        metric = TrackingMetric()
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[metric],
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        benchmark.run()

        assert metric.call_count == 1
        assert "responses" in metric.last_args


# Benchmark Export Tests
class TestBenchmarkExport:
    """Tests for Benchmark.export() method."""

    def test_export_creates_file(self, basic_use_case):
        """Test that export creates a profiles.json file."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark.export(profiles, tmpdir)
            output_file = Path(tmpdir) / "profiles.json"
            assert output_file.exists()

    def test_export_content_matches(self, basic_use_case):
        """Test that exported content matches profiles."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark.export(profiles, tmpdir)
            output_file = Path(tmpdir) / "profiles.json"

            with open(output_file) as f:
                loaded = json.load(f)

            assert "baseline" in loaded
            assert len(loaded["baseline"]) == 1

    def test_export_creates_directory(self, basic_use_case):
        """Test that export creates directory if it doesn't exist."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output"
            benchmark.export(profiles, str(nested_dir))
            assert (nested_dir / "profiles.json").exists()

    def test_export_multiple_pipelines(self, basic_use_case):
        """Test export with multiple pipelines."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "baseline": [],
                "steered": [MockInputControl()],
            },
            num_trials=2,
        )

        profiles = benchmark.run()

        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark.export(profiles, tmpdir)

            with open(Path(tmpdir) / "profiles.json") as f:
                loaded = json.load(f)

            assert "baseline" in loaded
            assert "steered" in loaded
            assert len(loaded["baseline"]) == 2
            assert len(loaded["steered"]) == 2


# Runtime Overrides Tests
class TestBenchmarkRuntimeOverrides:
    """Tests for runtime_overrides functionality."""

    def test_runtime_overrides_passed_to_generate(self, sample_evaluation_data):
        """Test that runtime_overrides are passed to use case generate."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            runtime_overrides={"MockStateControl": {"target": "column"}},
        )

        benchmark.run()

        # Check that generate was called with runtime_overrides
        assert len(use_case._generate_calls) == 1
        assert use_case._generate_calls[0]["runtime_overrides"] is not None

    def test_runtime_overrides_with_steered_pipeline(self, sample_evaluation_data):
        """Test runtime_overrides with a steered pipeline."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        state_ctrl = MockStateControl()

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"steered": [state_ctrl]},
            runtime_overrides={"MockStateControl": {"target_layers": "layers_column"}},
        )

        benchmark.run()

        assert use_case._generate_calls[0]["runtime_overrides"] is not None


# Edge Cases Tests
class TestBenchmarkEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_evaluation_data(self, sample_metrics):
        """Test benchmark with empty evaluation data."""
        use_case = MockUseCase(
            evaluation_data=[],
            evaluation_metrics=sample_metrics,
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()
        assert profiles["baseline"][0]["generations"] == []

    def test_single_sample_evaluation(self, sample_metrics):
        """Test benchmark with single evaluation sample."""
        use_case = MockUseCase(
            evaluation_data=[{"id": "1", "question": "test", "answer": "A"}],
            evaluation_metrics=sample_metrics,
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
        )

        profiles = benchmark.run()
        assert len(profiles["baseline"][0]["generations"]) == 1

    def test_num_samples_limiting(self, sample_evaluation_data, sample_metrics):
        """Test that num_samples limits evaluation data."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=sample_metrics,
            num_samples=2,
        )

        assert len(use_case.evaluation_data) == 2

    def test_none_pipeline_treated_as_baseline(self, basic_use_case):
        """Test that None pipeline is treated as empty list."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": None},
        )

        profiles = benchmark.run()
        assert "baseline" in profiles

    def test_zero_trials(self, basic_use_case):
        """Test benchmark with zero trials."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"baseline": []},
            num_trials=0,
        )

        profiles = benchmark.run()
        assert profiles["baseline"] == []


# Integration Tests
class TestBenchmarkIntegration:
    """Integration tests combining multiple components."""

    def test_full_benchmark_workflow(self, sample_evaluation_data):
        """Test complete benchmark workflow from setup to export."""
        # Setup
        metrics = [MockAccuracyMetric(), MockScoreMetric()]
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=metrics,
        )

        # Create controls
        input_ctrl = MockInputControl(prefix=">>")
        state_ctrl = MockStateControl(target_layers=[0, 1])

        # Create benchmark
        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "baseline": [],
                "input_steered": [input_ctrl],
                "state_steered": [state_ctrl],
            },
            gen_kwargs={"max_new_tokens": 100},
            num_trials=2,
        )

        # Run
        profiles = benchmark.run()

        # Verify structure
        assert len(profiles) == 3
        for pipeline_name, runs in profiles.items():
            assert len(runs) == 2
            for run in runs:
                assert "trial_id" in run
                assert "generations" in run
                assert "evaluations" in run
                assert len(run["generations"]) == len(sample_evaluation_data)

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            benchmark.export(profiles, tmpdir)
            assert (Path(tmpdir) / "profiles.json").exists()

    def test_benchmark_with_all_control_types(self, basic_use_case):
        """Test benchmark with all four control types."""
        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "all_controls": [
                    MockInputControl(),
                    MockStructuralControl(),
                    MockStateControl(),
                    MockOutputControl(),
                ],
            },
        )

        profiles = benchmark.run()

        assert "all_controls" in profiles
        assert len(profiles["all_controls"][0]["generations"]) > 0

    def test_benchmark_comparison_across_methods(self, sample_evaluation_data):
        """Test benchmark comparing multiple steering methods."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={
                "baseline": [],
                "method_a": [MockInputControl(prefix="a_")],
                "method_b": [MockStateControl(scale_factor=0.5)],
                "method_c": [MockInputControl(), MockStateControl()],
            },
            num_trials=3,
        )

        profiles = benchmark.run()

        # All methods should have results
        assert len(profiles) == 4
        for name in ["baseline", "method_a", "method_b", "method_c"]:
            assert name in profiles
            assert len(profiles[name]) == 3  # 3 trials each


# ControlSpec Tests
class TestControlSpecInitialization:
    """Tests for ControlSpec initialization."""

    def test_basic_initialization(self):
        """Test basic ControlSpec initialization."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "test_"},
        )

        assert spec.control_cls is MockInputControl
        assert spec.params == {"prefix": "test_"}
        assert spec.vars is None
        assert spec.name is None

    def test_initialization_with_name(self):
        """Test ControlSpec with custom name."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            name="my_input_control",
        )

        assert spec.name == "my_input_control"

    def test_initialization_with_dict_vars(self):
        """Test ControlSpec with dict-style vars."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "fixed_"},
            vars={"num_examples": [1, 2, 3]},
        )

        assert spec.vars == {"num_examples": [1, 2, 3]}

    def test_initialization_with_list_vars(self):
        """Test ControlSpec with list-style vars."""
        spec = ControlSpec(
            control_cls=MockStateControl,
            params={},
            vars=[
                {"scale_factor": 0.5, "mode": "add"},
                {"scale_factor": 1.0, "mode": "multiply"},
            ],
        )

        assert len(spec.vars) == 2


class TestControlSpecIterPoints:
    """Tests for ControlSpec.iter_points() method."""

    def test_iter_points_no_vars(self):
        """Test iter_points with no vars returns single empty dict."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "test_"},
            vars=None,
        )

        points = list(spec.iter_points())
        assert points == [{}]

    def test_iter_points_empty_dict_vars(self):
        """Test iter_points with empty dict vars."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars={},
        )

        points = list(spec.iter_points())
        assert points == [{}]

    def test_iter_points_single_var(self):
        """Test iter_points with single variable."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars={"num_examples": [1, 2, 3]},
        )

        points = list(spec.iter_points())
        assert points == [
            {"num_examples": 1},
            {"num_examples": 2},
            {"num_examples": 3},
        ]

    def test_iter_points_multiple_vars_cartesian(self):
        """Test iter_points produces cartesian product of multiple vars."""
        spec = ControlSpec(
            control_cls=MockStateControl,
            params={},
            vars={
                "scale_factor": [0.5, 1.0],
                "mode": ["add", "multiply"],
            },
        )

        points = list(spec.iter_points())

        # Should have 2 * 2 = 4 combinations
        assert len(points) == 4

        # Check all combinations present
        expected_scale_factors = {p["scale_factor"] for p in points}
        expected_modes = {p["mode"] for p in points}
        assert expected_scale_factors == {0.5, 1.0}
        assert expected_modes == {"add", "multiply"}

    def test_iter_points_list_vars(self):
        """Test iter_points with explicit list of combinations."""
        spec = ControlSpec(
            control_cls=MockStateControl,
            params={},
            vars=[
                {"scale_factor": 0.5, "target_layers": [0]},
                {"scale_factor": 1.0, "target_layers": [0, 1]},
                {"scale_factor": 2.0, "target_layers": [0, 1, 2]},
            ],
        )

        points = list(spec.iter_points())

        assert len(points) == 3
        assert points[0] == {"scale_factor": 0.5, "target_layers": [0]}
        assert points[2] == {"scale_factor": 2.0, "target_layers": [0, 1, 2]}

    def test_iter_points_callable_vars(self):
        """Test iter_points with callable vars."""
        def generate_points(context):
            base = context.get("base_value", 1.0)
            for multiplier in [0.5, 1.0, 2.0]:
                yield {"scale_factor": base * multiplier}

        spec = ControlSpec(
            control_cls=MockStateControl,
            params={},
            vars=generate_points,
        )

        points = list(spec.iter_points({"base_value": 2.0}))

        assert len(points) == 3
        assert points[0] == {"scale_factor": 1.0}
        assert points[1] == {"scale_factor": 2.0}
        assert points[2] == {"scale_factor": 4.0}

    def test_iter_points_callable_with_context(self):
        """Test that callable vars receive context."""
        received_context = {}

        def capture_context(context):
            received_context.update(context)
            yield {"param": "value"}

        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars=capture_context,
        )

        list(spec.iter_points({"pipeline_name": "test", "model": "gpt2"}))

        assert received_context["pipeline_name"] == "test"
        assert received_context["model"] == "gpt2"


class TestControlSpecResolveParams:
    """Tests for ControlSpec.resolve_params() method."""

    def test_resolve_params_no_vars(self):
        """Test resolve_params with only fixed params."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "test_", "num_examples": 5},
        )

        resolved = spec.resolve_params({})

        assert resolved == {"prefix": "test_", "num_examples": 5}

    def test_resolve_params_merges_chosen(self):
        """Test resolve_params merges chosen values."""
        spec = ControlSpec(
            control_cls=MockStateControl,
            params={"mode": "add"},
            vars={"scale_factor": [0.5, 1.0]},
        )

        resolved = spec.resolve_params({"scale_factor": 0.5})

        assert resolved == {"mode": "add", "scale_factor": 0.5}

    def test_resolve_params_chosen_overrides_params(self):
        """Test that chosen values override fixed params."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "default_", "num_examples": 3},
        )

        resolved = spec.resolve_params({"prefix": "override_"})

        assert resolved["prefix"] == "override_"
        assert resolved["num_examples"] == 3

    def test_resolve_params_preserves_original(self):
        """Test that resolve_params doesn't modify original params."""
        original_params = {"prefix": "test_"}
        spec = ControlSpec(
            control_cls=MockInputControl,
            params=original_params,
        )

        spec.resolve_params({"num_examples": 5})

        assert original_params == {"prefix": "test_"}


class TestControlSpecWithBenchmark:
    """Tests for ControlSpec integration with Benchmark."""

    def test_benchmark_with_single_spec(self, sample_evaluation_data):
        """Test benchmark with single ControlSpec."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"suffix": "_end"},
            vars={"num_examples": [1, 2, 3]},
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"variable_input": [spec]},
            num_trials=1,
        )

        profiles = benchmark.run()

        # Should have 3 runs (one per num_examples value)
        assert len(profiles["variable_input"]) == 3

    def test_benchmark_with_multiple_specs(self, sample_evaluation_data):
        """Test benchmark with multiple ControlSpecs."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        input_spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars={"num_examples": [1, 2]},
            name="input",
        )

        state_spec = ControlSpec(
            control_cls=MockStateControl,
            params={"mode": "add"},
            vars={"scale_factor": [0.5, 1.0]},
            name="state",
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"combined_specs": [input_spec, state_spec]},
            num_trials=1,
        )

        profiles = benchmark.run()

        # Should have 2 * 2 = 4 runs (cartesian product)
        assert len(profiles["combined_specs"]) == 4

    def test_benchmark_spec_params_recorded(self, sample_evaluation_data):
        """Test that benchmark records params for each spec configuration."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        spec = ControlSpec(
            control_cls=MockStateControl,
            params={"mode": "fixed"},
            vars={"scale_factor": [0.5, 1.0]},
            name="state_ctrl",
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"spec_pipeline": [spec]},
            num_trials=1,
        )

        profiles = benchmark.run()

        # Check params are recorded
        params_list = [run["params"] for run in profiles["spec_pipeline"]]
        assert len(params_list) == 2

        # Each should have the spec name as key
        for params in params_list:
            assert "state_ctrl" in params
            assert params["state_ctrl"]["mode"] == "fixed"
            assert params["state_ctrl"]["scale_factor"] in [0.5, 1.0]

    def test_benchmark_spec_with_trials(self, sample_evaluation_data):
        """Test benchmark with ControlSpec and multiple trials."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars={"num_examples": [1, 2]},
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"spec_pipeline": [spec]},
            num_trials=3,
        )

        profiles = benchmark.run()

        # Should have 2 configs * 3 trials = 6 runs
        assert len(profiles["spec_pipeline"]) == 6

    def test_benchmark_mixed_spec_and_concrete_raises(self, basic_use_case):
        """Test that mixing ControlSpec and concrete controls raises error."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars={"num_examples": [1, 2]},
        )
        concrete = MockStateControl()

        benchmark = Benchmark(
            use_case=basic_use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"mixed": [spec, concrete]},
        )

        with pytest.raises(TypeError, match="mixes ControlSpec"):
            benchmark.run()

    def test_benchmark_spec_no_vars_single_run(self, sample_evaluation_data):
        """Test ControlSpec with no vars produces single configuration."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "fixed_", "num_examples": 5},
            vars=None,  # No variable parameters
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"fixed_spec": [spec]},
            num_trials=1,
        )

        profiles = benchmark.run()

        assert len(profiles["fixed_spec"]) == 1

    def test_benchmark_spec_uses_class_name_if_no_name(self, sample_evaluation_data):
        """Test that spec uses control class name if name not provided."""
        use_case = MockUseCase(
            evaluation_data=sample_evaluation_data,
            evaluation_metrics=[MockAccuracyMetric()],
        )

        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "test_"},
            vars={"num_examples": [1]},
            name=None,  # No explicit name
        )

        benchmark = Benchmark(
            use_case=use_case,
            base_model_name_or_path="test-model",
            steering_pipelines={"unnamed_spec": [spec]},
            num_trials=1,
        )

        profiles = benchmark.run()

        # Should use class name as key
        params = profiles["unnamed_spec"][0]["params"]
        assert "MockInputControl" in params


class TestControlSpecEdgeCases:
    """Edge case tests for ControlSpec."""

    def test_spec_with_empty_list_vars(self):
        """Test ControlSpec with empty list vars."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars=[],  # Empty list
        )

        points = list(spec.iter_points())
        assert points == []

    def test_spec_with_single_value_var(self):
        """Test ControlSpec with single-value variable."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars={"num_examples": [5]},  # Single value
        )

        points = list(spec.iter_points())
        assert points == [{"num_examples": 5}]

    def test_spec_callable_generator(self):
        """Test ControlSpec with generator function that uses context."""
        def param_generator(context):
            # Use context to determine number of iterations
            num_iters = context.get("num_iterations", 3)
            base_value = context.get("base_value", 10)
            for i in range(num_iters):
                yield {"iteration": i, "value": i * base_value}

        spec = ControlSpec(
            control_cls=MockInputControl,
            params={"prefix": "gen_"},
            vars=param_generator,
        )

        # Test with custom context
        points = list(spec.iter_points({"num_iterations": 4, "base_value": 5}))

        assert len(points) == 4
        assert points[0] == {"iteration": 0, "value": 0}
        assert points[1] == {"iteration": 1, "value": 5}
        assert points[3] == {"iteration": 3, "value": 15}

    def test_spec_invalid_vars_type_raises(self):
        """Test that invalid vars type raises error."""
        spec = ControlSpec(
            control_cls=MockInputControl,
            params={},
            vars="invalid",  # type: ignore
        )

        with pytest.raises(TypeError, match="vars must be"):
            list(spec.iter_points())

    def test_spec_large_cartesian_product(self):
        """Test ControlSpec with larger cartesian product."""
        spec = ControlSpec(
            control_cls=MockStateControl,
            params={},
            vars={
                "scale_factor": [0.1, 0.5, 1.0, 2.0],
                "mode": ["add", "multiply", "replace"],
                "target_layers": [[0], [0, 1], [0, 1, 2]],
            },
        )

        points = list(spec.iter_points())

        # 4 * 3 * 3 = 36 combinations
        assert len(points) == 36
