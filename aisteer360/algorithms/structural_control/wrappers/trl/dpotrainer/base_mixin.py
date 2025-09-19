from typing import Type

from peft import LoraConfig, PeftType, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig, DPOTrainer

from aisteer360.algorithms.structural_control.base import StructuralControl


class DPOTrainerMixin(StructuralControl):
    """

    """
    Config: Type  # set by subclass
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    ref_model: PreTrainedModel | None = None

    train_dataset = None
    eval_dataset = None
    training_args: dict = None
    use_peft: bool = False
    peft_type = None
    lora_kwargs: dict = None

    def steer(self, model: PreTrainedModel, tokenizer=None, ref_model=None, **_) -> PreTrainedModel:
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        if self.train_dataset is not None:
            training_args = DPOConfig(
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
                self.ref_model = None

            trainer = DPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
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
