import numpy as np

from aisteer360.evaluation.metrics.base import Metric


class MCQACalibration(Metric):
    """
    Calibration metrics for multiple-choice QA.

    Measures how well model confidence scores align with actual performance using Expected Calibration Error (ECE)
    and related metrics.
    """

    def __init__(self, n_bins: int = 10):
        super().__init__()
        self.n_bins = n_bins

    def compute(
        self,
        responses: list[str],
        reference_answers: list[str] = None,
        confidence_scores: list[float] = None,
        question_ids: list[str] | None = None,
        **kwargs
    ) -> dict[str, float]:
        """Computes calibration metrics for model predictions.

        Args:
            responses: List of predicted answer choices (e.g., 'A', 'B', 'C', 'D').
            reference_answers: List of correct answer choices.
            confidence_scores: List of model confidence scores (0.0 to 1.0).
            question_ids: Optional question IDs (unused, for interface compatibility).
            **kwargs: Additional arguments (unused).

        Returns:
            Dictionary of calibration metrics with values:

                - "ece": Expected Calibration Error (lower is better, 0.0 is perfect)
                - "avg_confidence": Model's average confidence across all predictions
                - "overconfidence": avg_confidence - accuracy (positive means overconfident)

        Raises:
            ValueError: If reference_answers or confidence_scores is None.
        """

        if reference_answers is None:
            raise ValueError("MCQACalibration needs `reference_answers`.")
        if confidence_scores is None:
            raise ValueError("MCQACalibration needs `confidence_scores`.")

        # calculate ece
        valid_data = [
            (resp, ref, conf)
            for resp, ref, conf in zip(responses, reference_answers, confidence_scores)
            if conf is not None
        ]
        responses, answers, confidences = zip(*valid_data)
        confidences = np.array(confidences)
        accuracies = np.array([response == answer for response, answer in zip(responses, answers)], dtype=float)
        avg_confidence = float(np.mean(confidences))
        avg_accuracy = float(np.mean(accuracies))
        ece = self._calculate_ece(confidences, accuracies)

        return {
            "ece": ece,
            "avg_confidence": avg_confidence,
            "overconfidence": avg_confidence - avg_accuracy,
        }

    def _calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        """Calculates Expected Calibration Error using binned confidence scores.

        ECE measures the difference between confidence and accuracy across confidence bins. For each bin, it computes
        the absolute difference between average confidence and average accuracy, weighted by the proportion of samples
        in that bin.

        Args:
            confidences: Array of confidence scores (0.0 to 1.0).
            accuracies: Array of binary accuracy values (0.0 or 1.0).

        Returns:
            Expected Calibration Error as a float between 0.0 and 1.0.
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0

        for i in range(self.n_bins):
            if i == self.n_bins - 1:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            else:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                ece += prop_in_bin * abs(bin_confidence - bin_accuracy)

        return float(ece)
