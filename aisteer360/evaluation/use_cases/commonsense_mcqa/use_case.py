import json
import math
import random
import re
from pathlib import Path
from typing import Any

from aisteer360.evaluation.use_cases.base import UseCase
from aisteer360.evaluation.utils.generation_utils import batch_retry_generate

_EVALUATION_REQ_KEYS = [
    "question",
    "answer",
    "choices"
]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CommonsenseMCQA(UseCase):
    """Commonsense MCQA evaluation use case.

    Evaluates model's ability to answer commonsense questions via accuracy on the CommonsenseMCQA dataset
    ([https://huggingface.co/datasets/tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa)). Supports
    answer choice shuffling across multiple runs to reduce position bias and improve evaluation robustness.

    The evaluation data should contain questions with multiple choice options where models are asked to respond with
    only the letter (A, B, C, etc.) corresponding to their chosen answer.

    Attributes:
        num_shuffling_runs: Number of times to shuffle answer choices for each question to mitigate position bias effects.
    """
    num_shuffling_runs: int

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]):
        """Validates that evaluation data contains required fields for MCQA evaluation.

        Ensures each data instance has the necessary keys and non-null values for the evaluation.

        Args:
            evaluation_data: Dictionary containing a single evaluation instance with question, answer choices, and correct answer information.

        Raises:
            ValueError: If required keys ('id', 'question', 'answer', 'choices') are missing or if any required fields contain null/NaN values.
        """
        if "id" not in evaluation_data.keys():
            raise ValueError("The evaluation data must include an 'id' key")

        missing_keys = [col for col in _EVALUATION_REQ_KEYS if col not in evaluation_data.keys()]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if any(
            key not in evaluation_data or evaluation_data[key] is None or
            (isinstance(evaluation_data[key], float) and math.isnan(evaluation_data[key]))
            for key in _EVALUATION_REQ_KEYS
        ):
            raise ValueError("Some required fields are missing or null.")

    def generate(
        self,
        model_or_pipeline,
        tokenizer,
        gen_kwargs: dict | None = None,
        runtime_overrides: dict[tuple[str, str], str] | None = None,
        **kwargs
    ) -> list[dict[str, Any]]:
        """Generates model responses for multiple-choice questions with shuffled answer orders.

        Creates prompts for each question with shuffled answer choices, generates model responses, and parses the
        outputs to extract letter choices. Repeats the process multiple times with different answer orderings to reduce
        positional bias.

        Args:
            model_or_pipeline: Either a HuggingFace model or SteeringPipeline instance to use for generation.
            tokenizer: Tokenizer for encoding/decoding text.
            gen_kwargs: Optional generation parameters.
            runtime_overrides: Optional runtime parameter overrides for steering controls, structured as {(pipeline_name, param_name): value}.

        Returns:
            List of generation dictionaries, each containing:

                - "response": Parsed letter choice (A, B, C, etc.) or None if not parseable
                - "prompt": Full prompt text sent to the model
                - "question_id": Identifier from the original evaluation data
                - "reference_answer": Correct letter choice for this shuffled ordering

        Note:

        - The number of returned generations will be `len(evaluation_data)` * `num_shuffling_runs` due to answer choice shuffling.
        """

        if not self.evaluation_data:
            print('No evaluation data provided.')
            return []
        gen_kwargs = dict(gen_kwargs or {})
        batch_size: int = int(kwargs["batch_size"])

        # form prompt data
        prompt_data = []
        for instance in self.evaluation_data:
            data_id = instance['id']
            question = instance['question']
            answer = instance['answer']
            choices = instance['choices']
            # shuffle order of choices for each shuffling run
            for _ in range(self.num_shuffling_runs):

                lines = ["You will be given a multiple-choice question and asked to select from a set of choices."]
                lines += [f"\nQuestion: {question}\n"]

                # shuffle
                choice_order = list(range(len(choices)))
                random.shuffle(choice_order)
                for i, old_idx in enumerate(choice_order):
                    lines.append(f"{_LETTERS[i]}. {choices[old_idx]}")

                lines += ["\nPlease only print the letter corresponding to your choice."]
                lines += ["\nAnswer:"]

                prompt_data.append(
                    {
                        "id": data_id,
                        "prompt": "\n".join(lines),
                        "reference_answer": _LETTERS[choice_order.index(choices.index(answer))]
                    }
                )

        # batch template/generate/decode
        choices = batch_retry_generate(
            prompt_data=prompt_data,
            model_or_pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            parse_fn=self._parse_letter,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            evaluation_data=self.evaluation_data,
            batch_size=batch_size
        )

        # store
        generations = [
            {
                "response": choice,
                "prompt": prompt_dict["prompt"],
                "question_id": prompt_dict["id"],
                "reference_answer": prompt_dict["reference_answer"],
            }
            for prompt_dict, choice in zip(prompt_data, choices)
        ]

        return generations

    def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Evaluates generated responses against reference answers using configured metrics.

        Extracts responses and reference answers from generations and computes scores using all evaluation metrics
        specified during initialization.

        Args:
            generations: List of generation dictionaries returned by the `generate()` method, each containing response,
                reference_answer, and question_id fields.

        Returns:
            Dictionary of scores keyed by `metric_name`
        """
        eval_data = {
            "responses": [generation["response"] for generation in generations],
            "reference_answers": [generation["reference_answer"] for generation in generations],
            "question_ids": [generation["question_id"] for generation in generations],
        }

        scores = {}
        for metric in self.evaluation_metrics:
            scores[metric.name] = metric(**eval_data)

        return scores

    def export(self, profiles: dict[str, Any], save_dir) -> None:
        """Exports evaluation profiles to (tabbed) JSON format."""

        with open(Path(save_dir) / "profiles.json", "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4, ensure_ascii=False)

    @staticmethod
    def _parse_letter(response) -> str:
        """Extracts the letter choice from model's generation.

        Parses model output to find the first valid letter (A-Z) that represents the model's choice.

        Args:
            response: Raw text response from the model.

        Returns:
            Single uppercase letter (A, B, C, etc.) representing the model's choice, or None if no valid letter choice could be parsed.
        """
        valid = _LETTERS
        text = re.sub(r"^\s*(assistant|system|user)[:\n ]*", "", response, flags=re.I).strip()
        match = re.search(rf"\b([{valid}])\b", text, flags=re.I)
        return match.group(1).upper() if match else None
