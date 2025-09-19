from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

#from peft.utils.peft_types import PeftType
from peft import PeftType, TaskType

from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class TRLArgs(BaseArgs):


    train_dataset: Any | None = None  # tokenized HF Dataset
    eval_dataset: Any | None = None
    data_collator: Any | None = None

    training_args: Dict[str, Any] = field(default_factory=dict)
    output_dir: str | Path | None = None
    learning_rate: float = field(default=2e-5)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    save_strategy: str = field(default='no')
    load_best_model_at_end: bool = field(default=True)
    bf16: bool = field(default=None)
    fp16: bool = field(default=False)
    logging_steps: float = field(default=10)



    use_peft: bool = field(default=False)   # set this to True in order to use PEFT
    peft_type: PeftType = field(default=PeftType.LORA) # set this ti type of PEFT if use_peft is True

    r: int = field(default=16)
    lora_alpha: int = field(default=32)
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    modules_to_save: Optional[List[str]] = field(default=None)
    lora_dropout: float = field(default=0.05)
    bias: str = field(default="none")  # "none" | "all" | "lora_only"
    task_type: TaskType = field(default=TaskType.CAUSAL_LM)

    # autofilled so PEFTMixin can jsut LoraConfig(**self.lora_kwargs)
    lora_kwargs: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.training_args = {
            'output_dir': self.output_dir,
            'learning_rate': self.learning_rate,
            'num_train_epochs': self.num_train_epochs,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'per_device_eval_batch_size': self.per_device_eval_batch_size,
            'load_best_model_at_end': self.load_best_model_at_end,
            'save_strategy': self.save_strategy,
            'bf16': self.bf16,
            'fp16': self.fp16,
            'logging_steps': self.logging_steps,
        }
        self.training_args.setdefault("remove_unused_columns", False)

        if self.r <= 0:
            raise ValueError("LoRA `r` must be > 0.")
        if self.lora_alpha <= 0:
            raise ValueError("`lora_alpha` must be > 0.")
        if self.lora_dropout < 0 or self.lora_dropout >= 1:
            raise ValueError("`lora_dropout` must be in [0, 1).")
        if self.bias not in {"none", "all", "lora_only"}:
            raise ValueError("bias must be 'none', 'all', or 'lora_only'; got {self.bias!r}")

        self.lora_kwargs = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "modules_to_save": self.modules_to_save,
            "task_type": self.task_type
        }
