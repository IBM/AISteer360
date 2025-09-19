from typing import Any

from aisteer360.evaluation.metrics.base import Metric
from aisteer360.evaluation.metrics.custom.instruction_following.helpers.evaluation_main import (
    test_instruction_following_strict,
)


class StrictInstruction(Metric):
    """
    Evaluation wrapper around IFEval's official implementation from Google Research ([https://github.com/google-research/google-research/tree/master/instruction_following_eval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)).
    Measures how well models follow explicit instructions embedded within prompts, using strict binary evaluation criteria.
    """

    def _fix_kwargs(self, kwargs_list):
        """
        Fix kwargs list by removing None values and converting
        all-None dicts back to empty dicts
        """
        fixed_kwargs = []
        for kwarg_dict in kwargs_list:
            cleaned = {k: v for k, v in kwarg_dict.items() if v is not None}
            fixed_kwargs.append(cleaned)

        return fixed_kwargs

    def compute(
        self,
        responses: list[dict] | None = None,
        prompts: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Computes strict instruction-following metrics using IFEval evaluation.

        Evaluates model responses against structured instructions using the official IFEval framework. Each response is
        assessed both at the prompt level (whether ALL instructions were followed) and at the individual instruction
        level.

        Args:
            responses: List of response dictionaries, each containing:

                - "prompt": The input prompt with embedded instructions
                - "response": The model's generated response
                - "instruction_id_list": List of instruction IDs to evaluate
                - "kwargs": Additional parameters for instruction evaluation
            prompts: List of question prompts (unused, for interface compatibility).
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary of instruction-following metrics with values:

                - "strict_prompt_accuracy": Proportion of prompts where all instructions were followed correctly
                  (prompt-level accuracy)
                - "strict_instruction_accuracy": Proportion of individual instructions followed correctly across all
                  prompts (instruction-level accuracy)
                - "follow_all_instructions": List of boolean values indicating whether each prompt had all instructions
                  followed

        Note:

        - Returns zero accuracies and empty list if responses is None or empty.
        - The evaluation uses strict binary criteria (partial compliance counts as failure).
        """
        total_prompts = len(responses) if responses is not None else 0
        correct_prompts = 0
        total_instructions = 0
        correct_instructions = 0
        follow_all_instructions = []

        if responses is not None:
            for instance in responses:
                instance["instruction_id_list"] = instance["instruction_id_list"]
                instance["kwargs"] = self._fix_kwargs(instance["kwargs"])
                prompt = instance["prompt"]
                response = instance["response"]
                # test_instruction_following_strict expects an input with fields:
                # prompt, instruction_id_list, kwargs
                output_example = test_instruction_following_strict(
                    instance, {prompt: response}
                )

                # if all instructions followed
                if output_example.follow_all_instructions:
                    correct_prompts += 1
                    follow_all_instructions.append(True)
                else:
                    follow_all_instructions.append(False)

                num_instructions = len(output_example.follow_instruction_list)
                total_instructions += num_instructions
                correct_instructions += sum(output_example.follow_instruction_list)

        strict_prompt_accuracy = (
            correct_prompts / total_prompts if total_prompts > 0 else 0.0
        )
        strict_instruction_accuracy = (
            correct_instructions / total_instructions if total_instructions > 0 else 0.0
        )

        return {
            "strict_prompt_accuracy": strict_prompt_accuracy,
            "strict_instruction_accuracy": strict_instruction_accuracy,
            "follow_all_instructions": follow_all_instructions,
        }
