import gc
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.data_utils import to_jsonable


class Benchmark:
    """Benchmark framework for comparing steering pipelines on a use case.

    A Benchmark runs one or more steering pipeline configurations on a given use case, optionally with multiple trials
    per configuration. Each trial reuses the same steered model and re-samples any generate-time randomness (e.g.,
    few-shot selection, sampling-based decoding, etc.).

    Attributes:
        use_case: Use case that defines prompt construction, generation logic, and evaluation metrics.
        base_model_name_or_path: Hugging Face model ID or local path for the base causal language model.
        steering_pipelines: Mapping from pipeline name to a list of controls or ControlSpec objects; empty list denotes
            a baseline (no steering).
        runtime_overrides: Optional overrides passed through to `UseCase.generate` for runtime control parameters.
        hf_model_kwargs: Extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained`.
        gen_kwargs: Generation kwargs forwarded to :meth:`UseCase.generate`.
        device_map: Device placement strategy used when loading models.
        num_trials: Number of evaluation trials to run per concrete pipeline configuration.
    """

    def __init__(
        self,
        use_case: UseCase,
        base_model_name_or_path: str | Path,
        steering_pipelines: dict[str, list[Any]],
        runtime_overrides: dict[str, dict[str, Any]] | None = None,
        hf_model_kwargs: dict | None = None,
        gen_kwargs: dict | None = None,
        device_map: str = "auto",
        num_trials: int = 1,
        batch_size: int = 8
    ) -> None:
        self.use_case = use_case
        self.base_model_name_or_path = base_model_name_or_path
        self.steering_pipelines = steering_pipelines
        self.runtime_overrides = runtime_overrides
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}
        self.device_map = device_map
        self.num_trials = int(num_trials)
        self.batch_size = int(batch_size)

        # lazy-init shared base model/tokenizer
        self._base_model: AutoModelForCausalLM | None = None
        self._base_tokenizer: AutoTokenizer | None = None

    def _ensure_base_model(self) -> None:
        """Load the base model/tokenizer once (for reuse across pipelines)."""
        if self._base_model is not None and self._base_tokenizer is not None:
            return

        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name_or_path,
            device_map=self.device_map,
            **self.hf_model_kwargs,
        )
        self._base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        self._base_tokenizer = ensure_pad_token(self._base_tokenizer)

    @staticmethod
    def _has_structural_control(controls: Sequence[Any]) -> bool:
        """Return True if any of the controls is a StructuralControl."""
        return any(
            isinstance(control, StructuralControl) and getattr(control, "enabled", True)
            for control in controls
        )

    def run(self) -> dict[str, list[dict[str, Any]]]:
        """Run the benchmark on all configured steering pipelines.

        Each pipeline configuration is expanded into one or more concrete control settings (via ControlSpecs when
        present). For each concrete configuration, the model is steered once and evaluated over `num_trials` trials.

        Returns:
            A mapping from pipeline name to a list of run dictionaries. Each run dictionary has keys:

                - `"trial_id"`: Integer trial index.
                - `"generations"`: Model generations returned by the use case.
                - `"evaluations"`: Metric results returned by the use case.
                - `"params"`: Mapping from spec name to constructor kwargs used for control, or an empty dict for
                    fixed/baseline pipelines.
        """
        profiles: dict[str, list[dict[str, Any]]] = {}

        for pipeline_name, pipeline in self.steering_pipelines.items():
            pipeline = pipeline or []

            print(f"Running pipeline: {pipeline_name}...", flush=True)

            has_specs = any(isinstance(control, ControlSpec) for control in pipeline)
            if has_specs and not all(isinstance(control, ControlSpec) for control in pipeline):
                raise TypeError(
                    f"Pipeline '{pipeline_name}' mixes ControlSpec and concrete controls. Either use only concrete "
                    "controls or only ControlSpecs. Wrap fixed configs in ControlSpec(vars=None) if needed."
                )

            if not pipeline:  # baseline (no steering)
                runs = self._run_pipeline(controls=[], params=None)
            elif has_specs:
                runs = self._run_spec_pipeline(pipeline_name, control_specs=pipeline)
            else:
                runs = self._run_pipeline(controls=pipeline, params=None)

            profiles[pipeline_name] = runs
            print("done.", flush=True)

        return profiles

    def _run_pipeline(
        self,
        controls: list[Any],
        params: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a concrete steering pipeline configuration for all trials.

        This helper handles both baseline (no controls) and fixed-control pipelines. Structural steering is applied
        once; the use case is evaluated `num_trials` times to capture generate-time variability.

        Args:
            controls: List of instantiated steering controls, or an empty list for the baseline (unsteered) model.
            params: Optional mapping from spec name to full constructor kwargs used to build the controls.

        Returns:
            A list of run dictionaries, one per trial. Each dict includes `"trial_id"`, `"generations"`, `"evaluations"`
             and `"params"`.
        """
        pipeline: SteeringPipeline | None = None
        tokenizer = None
        runs: list[dict[str, Any]] = []

        try:
            self._ensure_base_model()

            # build model or pipeline once
            if controls:
                if self._has_structural_control(controls):
                    # SteeringPipeline loads from the base checkpoint
                    pipeline = SteeringPipeline(
                        model_name_or_path=self.base_model_name_or_path,
                        controls=controls,
                        device_map=self.device_map,
                        hf_model_kwargs=self.hf_model_kwargs,
                    )
                    pipeline.steer()
                    tokenizer = pipeline.tokenizer
                    model_or_pipeline: Any = pipeline
                else:
                    # model not reloaded for non-structural controls (input/state/output only)
                    pipeline = SteeringPipeline(
                        model_name_or_path=None,
                        controls=controls,
                        tokenizer_name_or_path=None,
                        device_map=self.device_map,
                        hf_model_kwargs=self.hf_model_kwargs,
                        lazy_init=True,
                    )
                    # inject shared base model/tokenizer
                    pipeline.model = self._base_model
                    pipeline.tokenizer = self._base_tokenizer
                    if self._base_model is not None:
                        pipeline.device = self._base_model.device
                    pipeline.steer()
                    tokenizer = pipeline.tokenizer
                    model_or_pipeline = pipeline
            else:
                # baseline; use the shared base model directly
                model_or_pipeline = self._base_model
                tokenizer = self._base_tokenizer

            # run trials
            for trial_id in range(self.num_trials):
                generations = self.use_case.generate(
                    model_or_pipeline=model_or_pipeline,
                    tokenizer=tokenizer,
                    gen_kwargs=self.gen_kwargs,
                    runtime_overrides=self.runtime_overrides,
                    batch_size=self.batch_size
                )
                scores = self.use_case.evaluate(generations)

                runs.append(
                    {
                        "trial_id": trial_id,
                        "generations": generations,
                        "evaluations": scores,
                        "params": params or {},
                    }
                )

            return runs

        finally:
            if pipeline is not None:
                del pipeline
            if tokenizer is not None:
                del tokenizer

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_spec_pipeline(
        self,
        pipeline_name: str,
        control_specs: list[ControlSpec],
    ) -> list[dict[str, Any]]:
        """Run a pipeline whose controls are defined by ControlSpecs.

        This expands each ControlSpec into one or more local parameter choices, takes the cartesian product across specs
        to form concrete configurations, and evaluates each configuration using `_run_pipeline`.

        For each configuration, full constructor kwargs are resolved for every spec, controls are instantiated once to
        form a steering pipeline, and the steering pipeline is evaluated for `num_trials` trials

        Args:
            pipeline_name: Name of the pipeline being evaluated; passed into the context for ControlSpecs.
            control_specs: ControlSpec objects describing the controls used in
                this pipeline.

        Returns:
            A flat list of run dictionaries across all configurations and trials.
            Each run dictionary includes:

                - "trial_id": Integer trial index
                - "generations": Model outputs for that trial
                - "evaluations": Metric results for that trial
                - "params": Mapping from spec name to full constructor kwargs for that configuration
        """
        import itertools

        base_context = {
            "pipeline_name": pipeline_name,
            "base_model_name_or_path": self.base_model_name_or_path,
        }

        # collect points per spec
        spec_points: list[tuple[ControlSpec, list[dict[str, Any]]]] = []
        for spec in control_specs:
            points = list(spec.iter_points(base_context))
            if not points:
                points = [{}]
            spec_points.append((spec, points))

        if not spec_points:
            # behave as baseline
            return self._run_pipeline(controls=[], params=None)

        spec_list, points_lists = zip(*spec_points)
        combos = itertools.product(*points_lists)

        runs: list[dict[str, Any]] = []

        for combo_id, combo in enumerate(combos):
            print(
                f"Running configuration {combo_id + 1}...",
                flush=True,
            )
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

    def export(self, profiles: dict[str, list[dict[str, Any]]], save_dir: str) -> None:
        """Export benchmark results to disk.

        Sanitize to a JSON-friendly structure before delegating to the use case's export method.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        safe_profiles = to_jsonable(profiles)
        self.use_case.export(safe_profiles, save_dir)
