import json
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import batch_retry_generate

_EVALUATION_REQ_KEYS = [
    "prompt",
    "instructions",
    "instruction_id_list",
    "kwargs",
]


class InstructionFollowing(UseCase):
    """Instruction following evaluation use case using the IFEval dataset.

    Evaluates model ability to follow specific instructions by testing adherence to various formatting, content, and
    structural constraints. Uses the IFEval dataset which contains prompts with explicit instructions that models must
    follow precisely.

    The evaluation focuses on whether models can follow instructions like formatting requirements (e.g., "respond in
    exactly 3 sentences"), content constraints (e.g., "include the word 'fantastic' twice"), and structural
    requirements (e.g., "use bullet points", "write in JSON format").

    Attributes:
        evaluation_data: List of instances containing prompts and instruction metadata.
    """

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]) -> None:
        """Validates that evaluation data contains required fields for instruction following evaluation.

        Ensures each data instance has the necessary keys for the evaluation.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance with prompt, instructions, and metadata.

        Raises:
            ValueError: If required keys ('prompt', 'instructions', 'instruction_id_list', 'kwargs') are missing.
        """
        missing_keys = [key for key in _EVALUATION_REQ_KEYS if key not in evaluation_data]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs: dict | None = None,
        runtime_overrides: dict[tuple[str, str], str] | None = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Generates model responses for instruction following prompts.

        Processes evaluation data to create chat-formatted prompts and generates model responses.

        Args:
            model_or_pipeline: Either a HuggingFace model or SteeringPipeline instance to use for generation.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters passed to the model's generate method.
            runtime_overrides: Optional runtime parameter overrides for steering controls, structured as {(pipeline_name, param_name): value}.

        Returns:
            List of generation dictionaries, each containing:

                - "response": Generated text response from the model
                - "prompt": Original instruction following prompt
                - "instructions": List of specific instructions the model should follow
                - "instruction_id_list": Identifiers for each instruction type
                - "kwargs": Additional metadata for instruction evaluation
        """
        if not self.evaluation_data:
            print('No evaluation data provided.')
            return []
        gen_kwargs = dict(gen_kwargs or {})
        batch_size: int = int(kwargs["batch_size"])

        # form prompt data
        prompt_data = []
        for instance in self.evaluation_data:
            user_prompt = [{"role": "user", "content": instance["prompt"]}]
            prompt_data.append({"prompt": user_prompt})

        responses = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            evaluation_data=self.evaluation_data,
            batch_size=batch_size
        )

        generations = [
            {
                "response": response,
                "prompt": eval_data["prompt"],
                "instructions": eval_data["instructions"],
                "instruction_id_list": eval_data["instruction_id_list"],
                "kwargs": eval_data["kwargs"],
            }
            for eval_data, response in zip(self.evaluation_data, responses)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses against instruction requirements using configured metrics.

        Passes generation dictionaries to all evaluation metrics specified during initialization.

        Args:
            generations: List of generation dictionaries returned by the `generate()` method, each containing
                response, prompt, instructions, instruction_id_list, and kwargs fields.

        Returns:
            Dictionary of scores keyed by `metric_name`.
        """
        results = {}
        for metric in self.evaluation_metrics:
            results[metric.name] = metric(responses=generations)
        return results

    def export(self, profiles: dict[str, Any], save_dir: str) -> None:
        """Exports instruction following evaluation results to structured JSON files.

        Creates two output files:

            1. `responses.json`: Contains model responses for each steering method
            2. `scores.json`: Contains strict metric scores for each steering method

        Args:
            profiles: Dictionary containing evaluation results from all tested pipelines.
            save_dir: Directory path where results should be saved.
        """
        folder_path = Path(save_dir)
        folder_path.mkdir(parents=True, exist_ok=True)
        steering_methods, predictions, follow_instructions = [], {}, {}
        inputs = None

        for steering_method, results in profiles.items():
            generations = results.pop("generations")
            steering_methods.append(steering_method)
            predictions[steering_method] = [gen["response"] for gen in generations]

            # get instruction following details from the StrictInstruction metric
            if "StrictInstruction" in results["evaluations"]:
                follow_instructions[steering_method] = results["evaluations"][
                    "StrictInstruction"
                ].pop("follow_all_instructions")
            if not inputs:
                inputs = [gen["prompt"] for gen in generations]

        responses = []
        for idx, prompt in enumerate(inputs):
            response = {"prompt": prompt}
            for method in steering_methods:
                response[method] = predictions[method][idx]
                response[f"{method}_instr_follow"] = follow_instructions[method][idx]
            responses.append(response)

        with open(folder_path / "responses.json", "w") as f:
            json.dump(responses, f, indent=4)
        with open(folder_path / "scores.json", "w") as f:
            json.dump(profiles, f, indent=4)
