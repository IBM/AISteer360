import gc
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.evaluation.use_cases.base import UseCase


class Benchmark:
    """Benchmark framework for comparing steering pipelines on specific use cases.

    Provides a standardized way to compare different steering control configurations against a baseline model on a given
    evaluation task. Handles the complete benchmark workflow: model loading, generation, and evaluation.

    The benchmark runs each control pipeline configuration independently, allowing for fair comparison of controls on a
    common task.

    Args:
        use_case (UseCase): The evaluation task defining prompts, generation logic, and metrics.
            Must implement `generate()` and `evaluate()` methods.
        base_model_name_or_path (str | Path): HuggingFace model identifier or local path to the base model.
            Used for all pipeline configurations and baseline.
        steering_pipelines (dict[str, list[Any]]): Named configurations of steering pipelines.
            Keys are configuration names (e.g., "baseline", "with_activation_steering").
            Values are pipelines, e.g., lists of controls (StructuralControl, StateControl, etc.).
            Empty list or None creates a baseline configuration without steering.
        runtime_overrides (dict[str, dict[str, Any]], optional): Runtime parameters for specific pipeline
            configurations. Outer keys match `control_pipelines` keys,
            inner dicts contain runtime kwargs passed to controls during generation.
            Defaults to None.
        hf_model_kwargs (dict, optional): Additional arguments passed to `AutoModelForCausalLM.from_pretrained()`.
            Defaults to {}.
        gen_kwargs (dict, optional): Generation parameters passed to model.generate().
            Defaults to {}.
        device_map (str, optional): Device placement strategy for model loading.
            Defaults to "auto".
        """
    def __init__(
            self,
            use_case: UseCase,
            base_model_name_or_path: str | Path,
            steering_pipelines: dict[str, list[Any]],
            runtime_overrides: dict[str, dict[str, Any]] | None = None,
            hf_model_kwargs: dict | None = None,
            gen_kwargs: dict | None = None,
            device_map: str = "auto"
    ) -> None:
        self.use_case = use_case
        self.base_model_name_or_path = base_model_name_or_path
        self.steering_pipelines = steering_pipelines
        self.runtime_overrides = runtime_overrides
        self.hf_model_kwargs = hf_model_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}
        self.device_map = device_map

    def run(self) -> dict[str, Any]:
        """Run benchmark on all configured steering pipelines.

        Executes the benchmark by iterating through each pipeline configuration defined in `control_pipelines`. For each
        configuration, calls `_run_pipeline()` to handle model setup, generation, and evaluation. Results from all
        pipelines are collected for comparison.

        Returns:
            Benchmark profiles for all pipeline configurations. Keys are pipeline names from `control_pipelines`. Values are dicts containing:

                - "generations": Generated outputs from the model
                - "evaluations": Evaluation scores from the use case metrics
        """
        profiles = {}

        for steering_pipeline_name, steering_pipeline in self.steering_pipelines.items():

            print(f"Running pipeline: {steering_pipeline_name}...", flush=True)

            profile = self._run_pipeline(steering_pipeline)
            profiles[steering_pipeline_name] = profile

            print("done.")

        return profiles

    def _run_pipeline(self, steering_pipeline: list[Any]) -> dict[str, Any]:
        """Run steering pipeline."""

        model = None
        pipeline = None
        tokenizer = None

        try:

            if steering_pipeline:

                # todo: determine if lazy_init needed; raise warnings/errors according

                # build pipeline and steer
                pipeline = SteeringPipeline(
                    model_name_or_path=self.base_model_name_or_path,
                    controls=steering_pipeline,
                    device_map=self.device_map,
                    hf_model_kwargs=self.hf_model_kwargs,
                )

                # todo: check if steer_kwargs are necessary
                # steerer = steerer.steer(**steer_kwargs)
                pipeline.steer()

                tokenizer = pipeline.tokenizer
                model_or_pipeline = pipeline

            else:  # baseline

                model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name_or_path,
                    device_map=self.device_map,
                    **self.hf_model_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
                tokenizer = ensure_pad_token(tokenizer)
                model_or_pipeline = model

            # generate
            generations = self.use_case.generate(
                model_or_pipeline=model_or_pipeline,
                tokenizer=tokenizer,
                gen_kwargs=self.gen_kwargs,
                runtime_overrides=self.runtime_overrides
            )

            # evaluate
            scores = self.use_case.evaluate(generations)

            return {
                "generations": generations,
                "evaluations": scores
            }

        finally:  # cleanup

            if model is not None:
                del model

            if pipeline is not None:
                del pipeline

            if tokenizer is not None:
                del tokenizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def export(self, profiles: dict[str, Any], save_dir: str):
        """Export benchmark results to disk.

        Saves the benchmark profiles to the specified directory. Creates the directory if it doesn't exist. Delegates
        the actual export logic to the use case's export method, which handles format-specific serialization.

        Args:
            profiles (dict[str, Any]): Benchmark results from `run()` method.
                Contains generations and evaluations for each pipeline configuration.
            save_dir (str): Directory path where results will be saved.
                Will be created if it doesn't exist.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.use_case.export(profiles, save_dir)
