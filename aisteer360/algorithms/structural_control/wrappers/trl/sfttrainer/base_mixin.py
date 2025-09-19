from typing import Any, Type

from peft import LoraConfig, PeftType, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer

from aisteer360.algorithms.structural_control.base import StructuralControl


class SFTTrainerMixin(StructuralControl):
    """

    """
    Config: Type  # set by subclass
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    data_collator:  Any | None = None

    train_dataset = None
    eval_dataset = None
    training_args: dict = None
    use_peft: bool = False
    peft_type = None
    lora_kwargs: dict = None

    def steer(self, model: PreTrainedModel, tokenizer=None, **_) -> PreTrainedModel:
        self.model = model
        self.tokenizer = tokenizer
        if self.train_dataset is not None:
            print("data_collator is ", self.data_collator)
            training_args = SFTConfig(
                **{
                    **self.training_args
                }
            )

            peft_config = None
            if self.use_peft and self.peft_type == PeftType.LORA:
                peft_config = LoraConfig(
                    **{
                        **self.lora_kwargs
                    }
                )

            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                data_collator=self.data_collator,
                processing_class=self.tokenizer,
                peft_config=peft_config
            )
            trainer.train()

            self.model = trainer.model

            if training_args.output_dir:
                trainer.save_model(training_args.output_dir)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self.model
