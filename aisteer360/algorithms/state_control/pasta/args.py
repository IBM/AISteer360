from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class PASTAArgs(BaseArgs):

    substrings: Optional[List[Union[str, List[str]]]] = field(
        default=None,
        metadata={"help": "List of substrings or groups of substrings to steer attention toward or away from."}
    )
    head_config: Union[Dict[int, List[int]], List[int]] = field(
        default_factory=lambda: [0, 1],
        metadata={"help": "Either a list of layer indices (to steer all heads), or a dict mapping layer index -> list of head indices."}
    )
    alpha: float = field(
        default=1.0,
        metadata={"help": "Scaling coefficient controlling the strength of attention modification."}
    )
    scale_position: Literal["include", "exclude", "generation"] = field(
        default="exclude",
        metadata={"help": (
            "'include' upweights the specified tokens, "
            "'exclude' downweights all others, "
            "'generation' applies scaling to the full sequence."
        )}
    )

    # validate
    def __post_init__(self):

        if self.substrings is not None:
            if not isinstance(self.substrings, list):
                raise ValueError("'substrings' must be a list of strings or lists of strings.")
            for item in self.substrings:
                if isinstance(item, str):
                    continue
                if isinstance(item, list):
                    if not all(isinstance(sub, str) for sub in item):
                        raise ValueError("All elements in substring groups must be strings.")
                else:
                    raise ValueError("Each substring must be a string or a list of strings.")

        if isinstance(self.head_config, dict):
            converted: Dict[int, List[int]] = {}
            for key, val in self.head_config.items():
                try:
                    layer_idx = int(key)
                except Exception:
                    raise ValueError("All head_config keys must be integers or convertible to integers.")
                if not isinstance(val, list) or not all(isinstance(h, int) for h in val):
                    raise ValueError("head_config values must be lists of integers.")
                converted[layer_idx] = val
            self.head_config = converted
        elif isinstance(self.head_config, list):
            if not all(isinstance(h, int) for h in self.head_config):
                raise ValueError("If head_config is a list, it must contain only integers.")
        else:
            raise ValueError("head_config must be either a dict mapping layer->heads or a list of head indices.")

        if not isinstance(self.alpha, (float, int)):
            raise ValueError("alpha must be a float or int.")
        if self.alpha <= 0:
            raise ValueError("alpha must be strictly positive.")

        allowed = {"include", "exclude", "generation"}
        if self.scale_position not in allowed:
            raise ValueError(f"scale_position must be one of {allowed}, got '{self.scale_position}'.")
