from dataclasses import dataclass, field
from typing import Any

from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.args import DPOArgs


@dataclass
class SPPOArgs(DPOArgs):

    optim: str | None = field(default="rmsprop")
    logging_first_step: bool = field(default=True)
    beta: float = field(default=0.001)
    loss_type: str = field(default="sppo")

    start_iteration: int = field(default=1)
    end_iteration: int = field(default=1)
    max_input_length: int = field(default=2048)
    num_prompts: int = field(default=5)
    temp_dir: str = field(default="sppo_temp_dir")

    gen_max_new_tokens: int = field(default=128)
    ranking_batch_size: int = field(default=8)
    limit_num_examples: int | None = field(default=None)

    additional_train_datasets: list[Any] | None = field(default=None)

    def __post_init__(self) -> None:

        # keep DPO defaults but override for SPPO where needed
        super().__post_init__()
        self.training_args["optim"] = self.optim
        self.training_args["logging_first_step"] = self.logging_first_step
        self.training_args["beta"] = self.beta
        self.training_args["loss_type"] = self.loss_type

        # checks
        if self.start_iteration < 1 or self.end_iteration < self.start_iteration:
            raise ValueError("Iterations must satisfy 1 <= start_iteration <= end_iteration.")
        if self.max_input_length <= 0:
            raise ValueError("max_input_length must be > 0.")
        if self.num_prompts <= 0:
            raise ValueError("num_prompts must be > 0.")
        if self.gen_max_new_tokens <= 0:
            raise ValueError("gen_max_new_tokens must be > 0.")
        if self.ranking_batch_size <= 0:
            raise ValueError("ranking_batch_size must be > 0.")
