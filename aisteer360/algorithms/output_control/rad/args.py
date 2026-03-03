from dataclasses import dataclass, field

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class RADArgs(BaseArgs):
    """Arguments for RAD (Reward-Augmented Decoding)."""

    beta: float = field(
        default=0.0,
        metadata={"help": "Steering intensity."},
    )
    reward_path: str | None = field(
        default=None,
        metadata={"help": "Path to the trained reward model. See https://github.com/r-three/RAD for details."},
    )
    reward_model_id: str | None = field(
        default=None,
        metadata={
            "help": (
                "HuggingFace model ID or local path for an AutoModelForSequenceClassification "
                "reward model. When set, this is used instead of reward_path."
            )
        },
    )
    reward_model_kwargs: dict = field(
        default_factory=dict,
        metadata={"help": "Extra kwargs passed to AutoModelForSequenceClassification.from_pretrained()."},
    )

    def __post_init__(self):
        if self.beta < 0:
            raise ValueError("'beta' must be non-negative.")
        if self.reward_path is not None and self.reward_model_id is not None:
            raise ValueError("Cannot specify both 'reward_path' and 'reward_model_id'. Use one or the other.")
