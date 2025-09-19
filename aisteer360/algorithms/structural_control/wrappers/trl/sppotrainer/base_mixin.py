from typing import Type

from peft import LoraConfig, PeftType, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import DPOConfig

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.trainer import (
    SPPOTrainer,
)
from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.utils import (
    prepare_dataset_from_prompts,
)


class SPPOTrainerMixin(StructuralControl):
    """

    """
    Config: Type  # set by subclass
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    refModel: PreTrainedModel | None = None

    train_dataset = None
    eval_dataset = None
    training_args: dict = None
    use_peft: bool = False
    peft_type = None
    lora_kwargs: dict = None

    def steer(self, model: PreTrainedModel, tokenizer=None, refModel=None,
              maxlen=2048, num_prompts=5, start_iter_num=1, end_iter_num=1, additional_train_datasets=None,
              sppo_temp_dir="sppo_temp_dir", **_) -> PreTrainedModel:
        self.model = model
        self.tokenizer = tokenizer
        self.refModel = refModel
        if self.train_dataset is not None:
            training_args = DPOConfig(      #TrainingArguments(
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
                self.refModel = None

            checkpoints_path = ""
            steerer = None
            for i in range(start_iter_num, end_iter_num+1):

                checkpoints_path=f"{sppo_temp_dir}/checkpoints/SPPO-Iter{i}"  # steered model stored at each iteration

                if i == start_iter_num or additional_train_datasets is None:
                    dataset = self.train_dataset
                else:
                    dataset = additional_train_datasets[i-start_iter_num-1]
                processed_train = prepare_dataset_from_prompts(
                    self.model,
                    self.tokenizer,
                    dataset,
                    sppo_temp_dir=sppo_temp_dir,
                    iter_num=i,
                    maxlen = maxlen,
                    num_prompts=num_prompts
                )
                trainer = SPPOTrainer(
                    model=self.model,
                    ref_model=self.refModel,
                    args=training_args,
                    train_dataset=processed_train,
                    eval_dataset=self.eval_dataset,
                    processing_class=self.tokenizer,
                    peft_config = peft_config,
                    beta=training_args.beta,
                    max_length=training_args.max_length,
                    max_prompt_length=training_args.max_prompt_length,
                    loss_type=training_args.loss_type,
                )
                trainer.train()
                self.model = trainer.model

                trainer.save_model(checkpoints_path)
                if training_args.output_dir:
                    if i == end_iter_num:
                        trainer.save_model(training_args.output_dir)  ### this also needs to be changed

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self.model
