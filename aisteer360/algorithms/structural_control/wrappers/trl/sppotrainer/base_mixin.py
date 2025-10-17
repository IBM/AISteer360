from typing import Any

from peft import LoraConfig, PeftType
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import DPOConfig

from aisteer360.algorithms.structural_control.base import StructuralControl
from aisteer360.algorithms.core.steering_utils import ensure_pad_token
from aisteer360.algorithms.structural_control.wrappers.trl.base_mixin import TRLMixin
from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.trainer import SPPOTrainer
from aisteer360.algorithms.structural_control.wrappers.trl.sppotrainer.utils import prepare_dataset_from_prompts
from aisteer360.algorithms.structural_control.wrappers.trl.utils.prompt_schema import standardize_prompt_dataset


class SPPOTrainerMixin(TRLMixin, StructuralControl):
    """
    SPPO structural control (self-play preference optimization).

    Iterative loop:
      for i in [start_iteration, ..., end_iteration]:
        1) Build a prompt-only dataset for this iteration.
        2) Call prepare_dataset_from_prompts(...) to generate rollouts/pairs.
        3) Train with SPPOTrainer on the processed data.
        4) Save checkpoint for this iteration.
      After final iteration:
        - Optionally merge LoRA in place.
        - Save to output_dir (if set), freeze, and return the model.
    """

    train_dataset: Any | None = None
    eval_dataset: Any | None = None
    training_args: dict[str, Any] = {}

    use_peft: bool = False
    peft_type: Any = None
    lora_kwargs: dict[str, Any] = {}
    adapter_name: str | None = None

    # runtime
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    ref_model: PreTrainedModel | None = None

    def steer(
            self,
            model: PreTrainedModel | None,
            tokenizer: PreTrainedTokenizer | None = None,
            ref_model: PreTrainedModel | None = None,
            **_,
    ) -> PreTrainedModel:

        self.model = model
        self.tokenizer = tokenizer or (getattr(model, "tokenizer", None) if model is not None else None)
        self.ref_model = ref_model
        self._resolve_model_tokenizer(self.model, self.tokenizer)  # fills self.model/self.tokenizer/device
        self.tokenizer = ensure_pad_token(self.tokenizer)

        # filter training args to the TRL config we pass to SPPOTrainer
        config_kwargs = self._filter_kwargs_for_class_or_callable(DPOConfig, self.training_args)
        training_config = DPOConfig(**config_kwargs)

        # LoRA config if requested
        peft_config = None
        if self.use_peft and self.peft_type == PeftType.LORA:
            peft_config = LoraConfig(**self.lora_kwargs)
            self.ref_model = None

        # standardize datasets to prompt-only schema for SPPO
        train_dataset = None
        if self.train_dataset is not None:
            train_dataset = standardize_prompt_dataset(self.train_dataset)
        eval_dataset = None
        if self.eval_dataset is not None:
            # SPPO trainer usually does not require eval; keep prompt-only if provided
            try:
                eval_dataset = standardize_prompt_dataset(self.eval_dataset)
            except Exception:
                eval_dataset = None

        # iterative self-play loop
        start_iteration = getattr(self, "start_iteration", 1)
        end_iteration = getattr(self, "end_iteration", 1)
        max_input_length = getattr(self, "max_input_length", 2048)
        num_prompts = getattr(self, "num_prompts", 5)
        temp_dir = getattr(self, "temp_dir", "sppo_temp_dir")
        additional_train_datasets = getattr(self, "additional_train_datasets", None)

        for iteration in range(start_iteration, end_iteration + 1):
            # per-iteration output (checkpoint) path
            checkpoints_path = f"{temp_dir}/checkpoints/SPPO-Iter{iteration}"

            # pick the dataset for this iteration
            if iteration == start_iteration or not additional_train_datasets:
                iteration_source_dataset = train_dataset
            else:
                index = iteration - start_iteration - 1
                iteration_source_dataset = additional_train_datasets[index]
                iteration_source_dataset = standardize_prompt_dataset(iteration_source_dataset)

            # build processed training data via SPPO’s utility
            processed_train_dataset = prepare_dataset_from_prompts(
                self.model,
                self.tokenizer,
                iteration_source_dataset,
                sppo_temp_dir=temp_dir,
                iter_num=iteration,
                maxlen=max_input_length,
                num_prompts=num_prompts,
                gen_max_new_tokens=self.gen_max_new_tokens,
                ranking_batch_size=self.ranking_batch_size,
                limit_num_examples=self.limit_num_examples
            )

            # train one iteration
            trainer = SPPOTrainer(
                model=self.model,
                ref_model=self.ref_model,
                args=training_config,
                train_dataset=processed_train_dataset,
                eval_dataset=eval_dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config,

                beta=training_config.beta,
                max_length=training_config.max_length,
                max_prompt_length=training_config.max_prompt_length,
                loss_type=training_config.loss_type,
            )
            trainer.train()
            self.model = trainer.model

            # save iteration checkpoint
            trainer.save_model(checkpoints_path)

            # save final to output_dir if provided (only at last iteration)
            if iteration == end_iteration and training_config.output_dir:
                trainer.save_model(training_config.output_dir)
                try:
                    self.tokenizer.save_pretrained(training_config.output_dir)
                except Exception:
                    pass

        # optional in-place LoRA merge after the final iteration
        self._maybe_merge_lora_in_place()

        # freeze and return
        return self._post_train_freeze()
