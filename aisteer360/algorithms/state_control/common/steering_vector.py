"""Steering vector: per-layer direction vectors learned by an estimator."""
import json
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """Per-layer direction tensors for activation steering.

    Directions can be either:

        - Non-positional: shape [1, H]; a single direction broadcast to all
          steered positions (e.g., used by CAA, CAST).
        - Positional: shape [T, H]; a sequence of T directions, each applied
          at a specific aligned position (e.g., used by ActAdd).

    Attributes:
        model_type: HuggingFace model_type string (e.g., "llama").
        directions: Mapping from layer_id to direction tensor of shape [T, H].
        explained_variances: Optional mapping from layer_id to explained
            variance scalar. Only meaningful for estimators that produce a
            real variance (e.g., PCA-based). None when not applicable.
    """

    model_type: str
    directions: dict[int, torch.Tensor]
    explained_variances: dict[int, float] | None = None

    @property
    def num_tokens(self) -> int:
        """Number of token positions in the steering vector (T dimension)."""
        if not self.directions:
            return 0
        return next(iter(self.directions.values())).size(0)

    @property
    def is_positional(self) -> bool:
        """True if the vector carries per-token positional structure (T > 1)."""
        return self.num_tokens > 1

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SteeringVector":
        """Move all direction tensors to device/dtype. Returns self for chaining."""
        self.directions = {
            k: v.to(device=device, dtype=dtype) if dtype else v.to(device=device)
            for k, v in self.directions.items()
        }
        return self

    def validate(self) -> None:
        """Validate that required fields are populated.

        Raises:
            ValueError: If model_type or directions are empty.
        """
        if not self.model_type:
            raise ValueError("model_type must be provided.")
        if not self.directions:
            raise ValueError("directions must not be empty.")

    def save(self, file_path: str) -> None:
        """Save the SteeringVector to a JSON file.

        Args:
            file_path: Path to save to. ".svec" extension added if not present.
        """
        if not file_path.endswith(".svec"):
            file_path += ".svec"
        directory = os.path.dirname(file_path)
        
        if directory:
            os.makedirs(directory, exist_ok=True)
        data = {
            "model_type": self.model_type,
            "directions": {str(k): v.tolist() for k, v in self.directions.items()},
        }

        if self.explained_variances is not None:
            data["explained_variances"] = {str(k): v for k, v in self.explained_variances.items()}
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.debug("Saved SteeringVector to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        """Load a SteeringVector from a JSON file.

        Args:
            file_path: Path to load from. ".svec" extension added if not present.

        Returns:
            Loaded SteeringVector instance.
        """
        if not file_path.endswith(".svec"):
            file_path += ".svec"
        with open(file_path) as f:
            data = json.load(f)

        # load directions with backward compatibility: ensure at least 2D [T, H]
        directions = {}
        for k, v in data["directions"].items():
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)  # [H] -> [1, H]
            directions[int(k)] = t

        explained_variances = None
        if "explained_variances" in data:
            explained_variances = {int(k): float(v) for k, v in data["explained_variances"].items()}
        logger.debug("Loaded SteeringVector from %s with layers %s", file_path, list(directions.keys()))
        return cls(model_type=data["model_type"], directions=directions, explained_variances=explained_variances)
