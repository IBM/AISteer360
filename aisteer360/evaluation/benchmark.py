"""Benchmark runner for steering pipelines.

Provides a `Benchmark` class for evaluating one or more steering pipeline configurations on a single `UseCase`.
"""
import gc
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.specs import ControlSpec
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.data_utils import _hash_params, to_jsonable

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = "checkpoint.json"


def _config_id_for(params: dict[str, Any] | None) -> str:
    """Derive a stable config identifier from a params dict (or None/empty for baselines)."""
    return _hash_params(params or {})


class Benchmark:
    """Benchmark functionality for comparing steering pipelines on a use case.

    A Benchmark runs one or more steering pipeline configurations on a given use case, optionally with multiple trials
    per configuration. Each trial reuses the same steered model and re-samples any generate-time randomness (e.g.,
    few-shot selection, sampling-based decoding, etc.).

    When ``save_dir`` is provided, results are checkpointed to disk after each completed configuration so that a run
    can be interrupted and resumed without re-generating completed work. On resume, configurations whose results are
    already present in the checkpoint are skipped entirely (no model loading or steering).

    Attributes:
        use_case: Use case that defines prompt construction, generation logic, and evaluation metrics.
        base_model_name_or_path: Hugging Face model ID or local path for the base causal language model.
        steering_pipelines: Mapping from pipeline name to a list of controls or `ControlSpec` objects; empty list
            denotes a baseline (no steering).
        runtime_overrides: Optional overrides passed through to `UseCase.generate` for runtime control parameters.
        hf_model_kwargs: Extra kwargs forwarded to `AutoModelForCausalLM.from_pretrained`.
        gen_kwargs: Generation kwargs forwarded to :meth:`UseCase.generate`.
        device_map: Device placement strategy used when loading models.
        num_trials: Number of evaluation trials to run per concrete pipeline configuration.
        save_dir: Optional directory for incremental checkpoints. When set, completed configurations are written to a
            ``checkpoint.json`` file and the use case's ``export()`` is called after each pipeline finishes. Subsequent
             calls on already-completed configurations are skipped.
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
        batch_size: int = 8,
        save_dir: str | Path | None = None,
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
        self.save_dir = Path(save_dir) if save_dir is not None else None

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

    def _load_checkpoint(self) -> dict[str, list[dict[str, Any]]]:
        """Load previously-saved profiles from disk, or return an empty dict."""
        if self.save_dir is None:
            return {}
        path = self.save_dir / _CHECKPOINT_FILENAME
        if not path.exists():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                profiles = json.load(f)
            n_runs = sum(len(runs) for runs in profiles.values())
            logger.info("Resumed from checkpoint: %d run(s) across %d pipeline(s)", n_runs, len(profiles))
            print(f"Resumed from checkpoint: {n_runs} run(s) across {len(profiles)} pipeline(s).", flush=True)
            return profiles
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read checkpoint file; starting fresh.", exc_info=True)
            return {}

    def _save_checkpoint(self, profiles: dict[str, list[dict[str, Any]]]) -> None:
        """Atomically write current profiles to the checkpoint file."""
        if self.save_dir is None:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        safe = to_jsonable(profiles)
        tmp = self.save_dir / f"{_CHECKPOINT_FILENAME}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(safe, f, ensure_ascii=False)
        tmp.rename(self.save_dir / _CHECKPOINT_FILENAME)

    @staticmethod
    def _runs_for_config(runs: list[dict[str, Any]], config_id: str) -> list[dict[str, Any]]:
        """Filter a list of runs to those matching a given config id."""
        return [r for r in runs if _config_id_for(r.get("params")) == config_id]

    def run(self) -> dict[str, list[dict[str, Any]]]:
        """Run the benchmark on all configured steering pipelines.

        Each pipeline configuration is expanded into one or more control settings (via `ControlSpecs` when present).
        For each configuration, the model is steered once and evaluated over `num_trials` trials.

        When ``save_dir`` was provided at construction time, completed configurations are persisted incrementally and
        the use case's ``export()`` method is called after each pipeline finishes. A subsequent call with the same
        ``save_dir`` automatically skips already-completed work.

        Returns:
            A mapping from pipeline name to a list of run dictionaries. Each run dictionary has keys:

                - `"trial_id"`: Integer trial index.
                - `"generations"`: Model generations returned by the use case.
                - `"evaluations"`: Metric results returned by the use case.
                - `"params"`: Mapping from spec name to constructor kwargs used for control, or an empty dict for
                    fixed/baseline pipelines.
        """
        profiles = self._load_checkpoint()

        for pipeline_name, pipeline in self.steering_pipelines.items():
            pipeline = pipeline or []

            print(f"Running pipeline: {pipeline_name}...", flush=True)
            logger.info("Running pipeline: %s", pipeline_name)

            has_specs = any(isinstance(control, ControlSpec) for control in pipeline)
            if has_specs and not all(isinstance(control, ControlSpec) for control in pipeline):
                raise TypeError(
                    f"Pipeline '{pipeline_name}' mixes ControlSpec and fixed controls. Either use only fixed controls "
                    "or only ControlSpecs. Wrap fixed configs in ControlSpec(vars=None) if needed."
                )

            existing_runs = profiles.get(pipeline_name, [])

            if not pipeline:  # baseline (no steering)
                runs = self._run_pipeline(controls=[], params=None, existing_runs=existing_runs)
            elif has_specs:
                runs = self._run_spec_pipeline(
                    pipeline_name, control_specs=pipeline, existing_runs=existing_runs, profiles=profiles,
                )
            else:
                runs = self._run_pipeline(controls=pipeline, params=None, existing_runs=existing_runs)

            profiles[pipeline_name] = runs
            logger.info("Pipeline %s complete", pipeline_name)
            print("done.", flush=True)

            self._save_checkpoint(profiles)
            self._try_export(profiles)

        return profiles

    def _run_pipeline(
        self,
        controls: list[Any],
        params: dict[str, dict[str, Any]] | None = None,
        existing_runs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a concrete steering pipeline configuration for all trials.

        This helper handles both baseline (no controls) and fixed-control pipelines. Structural steering is applied
        once; the use case is evaluated `num_trials` times (to capture generate-time variability).

        If the configuration is already present in *existing_runs* (from a prior checkpoint), its runs are returned
        immediately and the model is never loaded or steered.

        Args:
            controls: List of instantiated steering controls, or an empty list for the baseline (unsteered) model.
            params: Optional mapping from spec name to full constructor kwargs used to build the controls.
            existing_runs: Runs already loaded from a checkpoint for this pipeline.

        Returns:
            A list of run dictionaries, one per trial.
        """
        config_id = _config_id_for(params)

        # fast path: config already completed — skip model loading entirely
        cached = self._runs_for_config(existing_runs or [], config_id)
        if cached:
            logger.info("Skipping config=%s — already complete (%d run(s))", config_id, len(cached))
            return cached

        pipeline: SteeringPipeline | None = None
        tokenizer = None
        runs: list[dict[str, Any]] = []

        try:
            self._ensure_base_model()

            # build model or pipeline once
            if controls:
                if self._has_structural_control(controls):
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
                    pipeline = SteeringPipeline(
                        model_name_or_path=None,
                        controls=controls,
                        tokenizer_name_or_path=None,
                        device_map=self.device_map,
                        hf_model_kwargs=self.hf_model_kwargs,
                        lazy_init=True,
                    )

                    pipeline.model = self._base_model
                    pipeline.tokenizer = self._base_tokenizer
                    if self._base_model is not None:
                        pipeline.device = self._base_model.device

                    pipeline.steer()
                    tokenizer = pipeline.tokenizer
                    model_or_pipeline = pipeline
            else:
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

                runs.append({
                    "trial_id": trial_id,
                    "generations": generations,
                    "evaluations": scores,
                    "params": params or {},
                })

            return runs

        finally:
            # cleanup controls that may hold GPU resources (e.g., reward models)
            if pipeline is not None:
                for control in (pipeline.structural_control, pipeline.input_control,
                                pipeline.state_control, pipeline.output_control):
                    cleanup_fn = getattr(control, "cleanup", None)
                    if callable(cleanup_fn):
                        try:
                            cleanup_fn()
                        except Exception:
                            logger.warning("Control cleanup failed", exc_info=True)
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
        existing_runs: list[dict[str, Any]] | None = None,
        profiles: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a pipeline whose controls are defined by `ControlSpec`s.

        This method:

        - Expands each `ControlSpec` into one or more local parameter choices
        - Takes the cartesian product across specs to form pipeline configurations
        - Evaluates each configuration using `_run_pipeline`

        Configurations already present in the checkpoint are skipped entirely (no model loading or steering).

        Args:
            pipeline_name: Name of the pipeline being evaluated; passed into the context for `ControlSpec`s.
            control_specs: `ControlSpec` objects describing the controls used in the given pipeline.
            existing_runs: Runs already loaded from a checkpoint for this pipeline.
            profiles: The full profiles dict, passed through for incremental checkpointing after each config.

        Returns:
            A flat list of run dictionaries across all configurations and trials.
            Each run dictionary includes:

                - "trial_id": Integer trial index
                - "generations": Model outputs for the given trial
                - "evaluations": Metric results for the given trial
                - "params": Mapping from spec name to full constructor kwargs for the given configuration
        """
        existing_runs = existing_runs or []

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
            return self._run_pipeline(controls=[], params=None, existing_runs=existing_runs)

        spec_list, points_lists = zip(*spec_points)
        combos = itertools.product(*points_lists)

        runs: list[dict[str, Any]] = []

        for combo_id, combo in enumerate(combos):
            # pre-compute params so we can check the checkpoint before instantiating controls
            params: dict[str, dict[str, Any]] = {}
            global_context = {
                "pipeline_name": pipeline_name,
                "base_model_name_or_path": self.base_model_name_or_path,
                "combo_id": combo_id,
            }

            for spec, local_point in zip(spec_list, combo):
                spec_name = spec.name or spec.control_cls.__name__
                kwargs = spec.resolve_params(chosen=local_point, context=global_context)
                params[spec_name] = kwargs

            config_id = _config_id_for(params)

            # fast path: skip config entirely if already done
            cached = self._runs_for_config(existing_runs, config_id)
            if cached:
                logger.info("Skipping configuration %d (config=%s); already complete", combo_id + 1, config_id)
                print(f"  Skipping config {config_id}; restored {len(cached)} run(s) from checkpoint.", flush=True)
                runs.extend(cached)
                continue

            logger.info("Running configuration %d", combo_id + 1)
            print(f"Running configuration {combo_id + 1}...", flush=True)

            # instantiate controls only when we actually need to run
            controls: list[Any] = []
            for spec, local_point in zip(spec_list, combo):
                spec_name = spec.name or spec.control_cls.__name__
                control = spec.control_cls(**params[spec_name])
                controls.append(control)

            config_runs = self._run_pipeline(controls=controls, params=params, existing_runs=existing_runs)
            runs.extend(config_runs)

            # checkpoint after each config so partial spec sweeps survive interruption
            if profiles is not None:
                profiles[pipeline_name] = runs
                self._save_checkpoint(profiles)

        return runs

    def _try_export(self, profiles: dict[str, list[dict[str, Any]]]) -> None:
        """Call the use case's export method; log and swallow failures."""
        if self.save_dir is None:
            return
        try:
            self.export(profiles, str(self.save_dir))
        except Exception:
            logger.warning("Incremental export failed; checkpoint is still intact.", exc_info=True)

    def export(self, profiles: dict[str, list[dict[str, Any]]], save_dir: str) -> None:
        """Export benchmark results to disk.

        Sanitize to a JSON-friendly structure before delegating to the use case's export method.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        safe_profiles = to_jsonable(profiles)
        self.use_case.export(safe_profiles, save_dir)
