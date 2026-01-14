"""
Base class for all use cases. Provides a framework for loading evaluation data, applying metrics, and running
standardized evaluations across different types of tasks. Subclasses must implement the `generate()` and `evaluate()`
methods to define task-specific evaluation logic.
"""
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aisteer360.evaluation.metrics.base import Metric


class UseCase(ABC):
    """
    Base use case class.
    """
    def __init__(
        self,
        evaluation_data: list[dict] | str | Path,
        evaluation_metrics: list[Metric],
        num_samples: int = -1,
        **kwargs
    ) -> None:

        self.evaluation_data = []
        if isinstance(evaluation_data, Sequence) and all(isinstance(item, Mapping) for item in evaluation_data):
            self.evaluation_data = list(evaluation_data)
        else:
            path = Path(evaluation_data) if isinstance(evaluation_data, str) else evaluation_data
            with open(path) as f:
                self.evaluation_data = [json.loads(line) for line in f] if path.suffix == '.jsonl' else json.load(f)
        if not self.evaluation_data:
            warnings.warn(
                "Either evaluation data was not provided, or was unable to be generated.",
                UserWarning
            )

        if num_samples > 0:
            self.evaluation_data = self.evaluation_data[:num_samples]

        self.evaluation_metrics = evaluation_metrics
        self._metrics_by_name = {metric.name: metric for metric in evaluation_metrics}

        # store kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # validation
        if not all(isinstance(metric, Metric) for metric in self.evaluation_metrics):
            raise TypeError("All items in `evaluation_metrics` must be of type `Metric`.")

    @abstractmethod
    def generate(
            self,
            model_or_pipeline,
            tokenizer,
            gen_kwargs=None,
            runtime_overrides: dict[tuple[str, str], str] | None = None,
            **kwargs
    ) -> list[dict[str, Any]]:
        """
        Required generation logic for the current use case.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
            self,
            generations: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """
        Required evaluation logic for model's generations via `evaluation_metrics`.
        """
        raise NotImplementedError

    def export(self,
               profiles: dict[str, dict[str, Any]],
               save_dir: str
    ) -> None:
        """
        Optional formatting and export of evaluation profiles.
        """
        raise NotImplementedError

    # def validate_steering_data(self, steering_data):
    #     pass

    def validate_evaluation_data(self, evaluation_data) -> None:
        """
        Optional validation of the evaluation dataset.
        """
        raise NotImplementedError
