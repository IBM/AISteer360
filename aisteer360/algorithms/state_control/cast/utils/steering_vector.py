import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SteeringVector:
    """
    A dataclass representing a steering vector used for guiding the language model.
    This class is a light-weight version of the original steering vector class from
    the original implementation. It is load/save compatible w/ it.

    Attributes:
        model_type: The type of the model this vector is associated with.
        directions: A dictionary mapping layer IDs to numpy arrays of directions.
        explained_variances: A dictionary of explained variances.
    """
    model_type: str
    directions: dict[int, np.ndarray]
    explained_variances: dict

    def save(self, file_path: str) -> None:
        """
        Save the SteeringVector to a file.

        Args:
            file_path: The path to save the file to. If it doesn't end with '.svec',
                       this extension will be added.
        """
        if not file_path.endswith('.svec'):
            file_path += '.svec'

        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

        print(f"Saving SteeringVector to {file_path}")
        data = {
            "model_type": self.model_type,
            "directions": {k: v.tolist() for k, v in self.directions.items()},
            "explained_variances": self.explained_variances
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print("SteeringVector saved successfully")

    @classmethod
    def load(cls, file_path: str) -> "SteeringVector":
        """
        Load a SteeringVector from a file.

        Args:
            file_path: The path to load the file from. If it doesn't end with '.svec',
                       this extension will be added.

        Returns:
            A new SteeringVector instance loaded from the file.
        """
        if not file_path.endswith('.svec'):
            file_path += '.svec'

        print(f"Loading SteeringVector from {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)

        directions = {int(k): np.array(v) for k, v in data["directions"].items()}
        explained_variances = {int(k): v for k, v in data["explained_variances"].items()}

        print(f"Loaded directions for layers: {list(directions.keys())}")
        print(f"Shape of first direction vector: {next(iter(directions.values())).shape}")

        return cls(model_type=data["model_type"],
                   directions=directions,
                   explained_variances=explained_variances)

    def validate(self) -> None:
        if not self.model_type:
            raise ValueError("model_type must be provided")
        if not self.directions:
            raise ValueError("directions dict must not be empty")
        if not self.explained_variances:
            raise ValueError("explained_variances dict must not be empty")
