from collections import Counter, defaultdict

import numpy as np

from aisteer360.evaluation.metrics.base import Metric


class MCQAPositionalBias(Metric):
    """
    Positional bias metrics for multiple-choice QA.

    Measures whether the model exhibits bias toward selecting certain answer positions.
    """

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        question_ids: list[str] | None = None,
        **kwargs
    ) -> dict[str, float]:
        """Computes positional bias metrics for model predictions.

        Calculates how much the model's choice frequencies deviate from uniform distribution across answer positions.
        For K answer choices, each position should ideally be selected 1/K of the time.

        Args:
            responses: List of predicted answer choices (e.g., 'A', 'B', 'C', 'D').
            prompts: List of question prompts (unused, for interface compatibility).
            question_ids: Optional question IDs for computing per-question bias variance.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary of positional bias metrics with values:

                - "mean": Overall positional bias (mean |f_i - 1/K| across positions)
                - "std": Sample standard deviation of bias computed per question

        Note:

        - If question_ids is None, per-question analysis is skipped and std will be 0.0.
        """

        valid_responses = [r for r in responses if r is not None]

        position_counts = Counter(valid_responses)
        total_responses = len(valid_responses)
        positions = sorted(position_counts.keys())
        position_frequencies = [position_counts.get(pos, 0) / total_responses for pos in positions]
        expected_frequency = 1 / len(positions)

        # positional bias per question
        bias_per_question = []
        responses_by_question = defaultdict(list)

        for response, question_id in zip(responses, question_ids):
            if response is not None:
                responses_by_question[question_id].append(response)

        for question_id, question_responses in responses_by_question.items():
            if not question_responses:
                continue
            counts_for_question = Counter(question_responses)
            total_for_question = len(question_responses)
            frequencies_for_question = [counts_for_question.get(pos, 0) / total_for_question for pos in positions]
            bias_for_question = np.mean([abs(freq - expected_frequency) for freq in frequencies_for_question])
            bias_per_question.append(bias_for_question)

        return {
            "mean": np.mean([abs(freq - expected_frequency) for freq in position_frequencies]),
            "std": np.std(bias_per_question, ddof=1) if len(bias_per_question) > 1 else 0.0
        }
