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

    Directions are stored as [K, D] tensors per layer, where K and D are
    interpreted by the consuming transform:

        CAA, CAST: K=1, D=hidden_size (single broadcast direction)
        ActAdd: K=T, D=hidden_size (positional directions)
        Angular Steering: K=2, D=hidden_size (orthonormal basis pair)
        ITI: K=num_heads, D=head_dim (per-head directions)

    The container is agnostic to what K and D mean (varies depending on the method). 
    Semantics come from the consumer (transform).

    Attributes:
        model_type: HuggingFace model_type string (e.g., "llama").
        directions: Mapping from layer_id to direction tensor of shape [K, D].
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each head's output.
        explained_variances: Optional mapping from layer_id to explained
            variance scalar. Only meaningful for estimators that produce a
            real variance (e.g., PCA-based). None when not applicable.
        probe_accuracies: Optional mapping from (layer_id, head_id) to linear
            probe validation accuracy (used for head selection in ITI).
    """

    model_type: str
    directions: dict[int, torch.Tensor]
    num_heads: int | None = None
    head_dim: int | None = None
    explained_variances: dict[int, float] | None = None
    probe_accuracies: dict[tuple[int, int], float] | None = None

    @property
    def num_tokens(self) -> int:
        """Number of token positions in the steering vector (K dimension)."""
        if not self.directions:
            return 0
        return next(iter(self.directions.values())).size(0)

    @property
    def is_positional(self) -> bool:
        """True if the vector carries per-token positional structure (K > 1)."""
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
        data: dict = {
            "model_type": self.model_type,
            "directions": {str(k): v.tolist() for k, v in self.directions.items()},
        }

        if self.num_heads is not None:
            data["num_heads"] = self.num_heads
        if self.head_dim is not None:
            data["head_dim"] = self.head_dim
        if self.explained_variances is not None:
            data["explained_variances"] = {str(k): v for k, v in self.explained_variances.items()}
        if self.probe_accuracies is not None:
            data["probe_accuracies"] = {
                f"{layer}:{head}": acc for (layer, head), acc in self.probe_accuracies.items()
            }

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

        directions = {}
        for k, v in data["directions"].items():
            t = torch.tensor(v, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)  # [D] -> [1, D] backward compatibility
            directions[int(k)] = t

        explained_variances = None
        if "explained_variances" in data:
            explained_variances = {int(k): float(v) for k, v in data["explained_variances"].items()}

        num_heads = data.get("num_heads")
        head_dim = data.get("head_dim")

        probe_accuracies = None
        if "probe_accuracies" in data:
            probe_accuracies = {}
            for k, acc in data["probe_accuracies"].items():
                layer_str, head_str = k.split(":")
                probe_accuracies[(int(layer_str), int(head_str))] = float(acc)

        logger.debug("Loaded SteeringVector from %s with layers %s", file_path, list(directions.keys()))
        return cls(
            model_type=data["model_type"],
            directions=directions,
            num_heads=num_heads,
            head_dim=head_dim,
            explained_variances=explained_variances,
            probe_accuracies=probe_accuracies,
        )