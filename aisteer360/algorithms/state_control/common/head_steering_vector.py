"""Head steering vector: per-head direction vectors learned by an estimator."""
import json
import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class HeadSteeringVector:
    """Per-head direction tensors for head-level activation steering.

    Unlike SteeringVector (which stores per-layer directions), HeadSteeringVector
    stores directions keyed by (layer_id, head_id) tuples and vectors of shape [head_dim].

    The directions operate in pre-o_proj (pre-W_o) head space; each direction
    corresponds to an individual attention head's output before the output projection.

    Attributes:
        model_type: HuggingFace model_type string (e.g., "llama").
        num_heads: Number of attention heads per layer.
        head_dim: Dimension of each head's output.
        directions: Mapping from (layer_id, head_id) to direction tensor of shape [head_dim].
        probe_accuracies: Optional mapping from (layer_id, head_id) to linear probe
            validation accuracy. Used for head selection in ITI.
    """

    model_type: str
    num_heads: int
    head_dim: int
    directions: dict[tuple[int, int], torch.Tensor]
    probe_accuracies: dict[tuple[int, int], float] | None = None

    @property
    def layer_ids(self) -> list[int]:
        """Return sorted unique layer IDs present in directions."""
        return sorted({layer_id for layer_id, _ in self.directions.keys()})

    def to(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "HeadSteeringVector":
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
        """Save the HeadSteeringVector to a JSON file.

        Args:
            file_path: Path to save to. ".hsvec" extension added if not present.
        """
        if not file_path.endswith(".hsvec"):
            file_path += ".hsvec"
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # serialize (layer, head) tuple keys as "layer:head" strings
        data = {
            "model_type": self.model_type,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "directions": {f"{layer}:{head}": v.tolist() for (layer, head), v in self.directions.items()},
        }
        if self.probe_accuracies is not None:
            data["probe_accuracies"] = {
                f"{layer}:{head}": acc for (layer, head), acc in self.probe_accuracies.items()
            }
        with open(file_path, "w") as f:
            json.dump(data, f)
        logger.debug("Saved HeadSteeringVector to %s", file_path)

    @classmethod
    def load(cls, file_path: str) -> "HeadSteeringVector":
        """Load a HeadSteeringVector from a JSON file.

        Args:
            file_path: Path to load from. ".hsvec" extension added if not present.

        Returns:
            Loaded HeadSteeringVector instance.
        """
        if not file_path.endswith(".hsvec"):
            file_path += ".hsvec"
        with open(file_path) as f:
            data = json.load(f)

        # deserialize "layer:head" strings back to (layer, head) tuples
        directions = {}
        for k, v in data["directions"].items():
            layer_str, head_str = k.split(":")
            key = (int(layer_str), int(head_str))
            directions[key] = torch.tensor(v, dtype=torch.float32)

        probe_accuracies = None
        if "probe_accuracies" in data:
            probe_accuracies = {}
            for k, acc in data["probe_accuracies"].items():
                layer_str, head_str = k.split(":")
                key = (int(layer_str), int(head_str))
                probe_accuracies[key] = float(acc)

        logger.debug(
            "Loaded HeadSteeringVector from %s with %d heads across layers %s",
            file_path,
            len(directions),
            sorted({layer for layer, _ in directions.keys()}),
        )
        return cls(
            model_type=data["model_type"],
            num_heads=data["num_heads"],
            head_dim=data["head_dim"],
            directions=directions,
            probe_accuracies=probe_accuracies,
        )
