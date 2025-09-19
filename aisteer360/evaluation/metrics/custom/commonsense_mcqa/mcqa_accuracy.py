from collections import defaultdict
from math import sqrt

from aisteer360.evaluation.metrics.base import Metric


class MCQAAccuracy(Metric):
    """
    Exact-match accuracy for multiple-choice QA.
    """

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        reference_answers: list[str] | None = None,
        question_ids: list[str] | None = None,
        **kwargs
    ) -> dict[str, float]:
        """Computes trial-level and question-level accuracy metrics.

        Args:
            responses: List of predicted answer choices (e.g., 'A', 'B', 'C', 'D').
            prompts: List of question prompts (unused, for interface compatibility).
            reference_answers: List of correct answer choices.
            question_ids: Optional question IDs for grouping responses by question.
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary of accuracy score statistics with values:

                - "trial_mean": micro (attempt-level accuracy)
                - "trial_std": sample std-dev over trials
                - "question_mean": macro (majority-vote accuracy)
                - "question_std": sample std-dev over questions

        Raises:
            ValueError: If reference_answers is None or length mismatches occur.
        """

        if reference_answers is None:
            raise ValueError("MCQAAccuracy needs `reference_answers`.")
        if len(responses) != len(reference_answers):
            raise ValueError("`responses` and `reference_answers` must be the same length.")
        if question_ids is not None and len(responses) != len(question_ids):
            raise ValueError("`question_ids` must match length of `responses`.")

        # micro
        attempt_correct = [
            choice.strip().upper() == answer.strip().upper()
            for choice, answer in zip(responses, reference_answers) if choice is not None
        ]
        attempt_accuracy = sum(attempt_correct) / len(attempt_correct) if attempt_correct else 0.0
        attempt_accuracy_std = self._sample_std(attempt_correct, attempt_accuracy)

        # macro
        if question_ids is None:
            question_accuracy = attempt_accuracy
        else:
            votes = defaultdict(list)
            for qid, is_correct in zip(question_ids, attempt_correct):
                votes[qid].append(is_correct)

            majority_outcomes = [int(sum(vote) > len(vote) / 2) for vote in votes.values()]
            question_accuracy = sum(majority_outcomes) / len(votes) if votes else 0.0
            question_accuracy_std = self._sample_std(majority_outcomes, question_accuracy)

        return {
            "trial_mean": attempt_accuracy,
            "trial_std": attempt_accuracy_std,
            "question_mean": question_accuracy,
            "question_std": question_accuracy_std,
        }

    @staticmethod
    def _sample_std(binary, mean):
        """Computes sample standard deviation for binary outcomes.

        Args:
            binary: List of binary values (0 or 1).
            mean: Pre-computed mean of the binary values.

        Returns:
            Sample standard deviation using Bessel's correction (n-1).
        """
        n = len(binary)
        if n < 2:
            return 0.0
        var = sum((x - mean) ** 2 for x in binary) / (n - 1)
        return sqrt(var)
