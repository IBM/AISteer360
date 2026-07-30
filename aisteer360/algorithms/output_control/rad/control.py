from __future__ import annotations

import gc
import logging
import os

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from aisteer360.algorithms.output_control._common.candidates import rad_candidate_sizing
from aisteer360.algorithms.output_control._common.loading import load_sequence_classifier
from aisteer360.algorithms.output_control._common.processors.value_guided import ValueGuidedProcessor
from aisteer360.algorithms.output_control._common.values.reward_model import RewardModelValue
from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.rad.args import RADArgs
from aisteer360.algorithms.output_control.rad.utils import GPT2RewardModel

logger = logging.getLogger(__name__)


class RAD(OutputControl):
    """
    Implementation of RAD (Reward-Augmented Decoding) from Deng and Raffel, 2023.
    Integrated from the official implementation of RAD ([https://github.com/r-three/RAD?tab=readme-ov-file](https://github.com/r-three/RAD?tab=readme-ov-file)).

    RAD works in two phases:

    1. **Reward model training**: Train a reward model on a labeled dataset of texts and labels.
    For details about this step, please see [https://github.com/r-three/RAD?tab=readme-ov-file](https://github.com/r-three/RAD?tab=readme-ov-file). We skip this
    step in this implementation and re-use the open-source toxicity reward model trained by the authors via
    gdown [https://storage.googleapis.com/rad_release/saved_models.zip](https://storage.googleapis.com/rad_release/saved_models.zip)

    2. **Controlled decoding**: At every decoding step the candidate-token logits are shifted by `beta * reward`,
    where the `reward` is given by a trained reward model.

    RAD is a step-level control: `steer()` loads the reward model into a `RewardModelValue`, and
    `get_logits_processors()` returns a `ValueGuidedProcessor` that selects candidates (RAD's documented
    top-k/top-p precedence), scores them with the reward model, min-max normalizes within the candidate
    set (optionally inverted for the legacy toxicity head), and shifts the candidate logits by
    `beta * value` while masking non-candidates to `-inf`. As a step-level control, RAD composes with
    other output controls and with a decoding driver, and the sampling kwargs
    (`temperature`/`top_k`/`top_p`/`repetition_penalty`) are applied once by the driver's loop.

    Args:
        beta (float): Steering intensity. Defaults to 0.0.
        reward_path (str, optional): Path to the trained reward model. See [https://github.com/r-three/RAD](https://github.com/r-three/RAD) for details. Defaults to None.
        reward_model_id (str, optional): HuggingFace model ID or local path for an AutoModelForSequenceClassification
            reward model. When set, this is used instead of reward_path. Defaults to None.
        reward_model_kwargs (dict, optional): Extra kwargs passed to AutoModelForSequenceClassification.from_pretrained().
            Defaults to {}.

    Reference:

    - "Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model"
     Haikang Deng, Colin Raffel
     [https://arxiv.org/abs/2310.09520](https://arxiv.org/abs/2310.09520)
    """
    Args = RADArgs

    # placeholders (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None

    beta: float

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **__,
    ) -> PreTrainedModel:
        """Load and configure the reward model, then build the `RewardModelValue`.

        Supports two modes:

        1. **HuggingFace classifier**: When `reward_model_id` is set, loads any
           `AutoModelForSequenceClassification` compatible model from HuggingFace Hub.
        2. **Legacy toxicity model**: When `reward_path` is set (or neither is set),
           loads the GPT-2 based toxicity classifier from the original RAD paper.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for the base model.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model, unchanged.
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.device = next(model.parameters()).device

        # the legacy toxicity head is used iff no HF classifier id was supplied
        self._legacy = self.reward_model_id is None
        if self._legacy:
            self._load_legacy_toxicity_model()
            rm_score_fn = lambda output: output[:, 0]  # invert applied via _legacy
        else:
            self._load_hf_classifier()
            rm_score_fn = lambda output: output.logits[:, 0]  # general RM: higher = better

        self._value = RewardModelValue(
            reward_model=self.rm,
            rm_tokenizer=self.rm_tokenizer,
            rm_score_fn=rm_score_fn,
        )
        return model

    def _load_hf_classifier(self) -> None:
        """Load a HuggingFace AutoModelForSequenceClassification reward model."""
        logger.info("Loading reward model from HuggingFace: %s", self.reward_model_id)
        self.rm, self.rm_tokenizer = load_sequence_classifier(
            self.reward_model_id,
            device=self.device,
            hf_model_kwargs=self.reward_model_kwargs,
        )
        logger.info("HuggingFace reward model loaded successfully")

    def _load_legacy_toxicity_model(self) -> None:
        """Load the legacy GPT-2 toxicity reward model from the RAD paper."""
        self.rm_tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=self.reward_path)
        self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
        self.rm_tokenizer.padding_side = "right"
        self.rm_tokenizer.max_length = 1024

        if (self.reward_path is None) or not os.path.exists(os.path.join(self.reward_path, "pytorch_model.bin")):
            logger.info(
                "Reward model not found in: %s. Downloading from https://huggingface.co/hk/rad_rms/tree/main/gpt2_toxicity...",
                self.reward_path,
            )
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id="hk/rad_rms",
                filename="gpt2_toxicity/pytorch_model.bin",
                local_dir="./tmp/rad_saved_models/saved_models/",
            )
            logger.info(
                "Reward model downloaded. Please set reward_path='./tmp/rad_saved_models/saved_models/gpt2_toxicity' in the future."
            )
        else:
            logger.info("Reward model found in: %s", self.reward_path)

        if self.reward_path is None:
            self.reward_path = "./tmp/rad_saved_models/saved_models/gpt2_toxicity"

        state_dict = torch.load(os.path.join(self.reward_path, "pytorch_model.bin"), map_location="cpu")
        self.rm = GPT2RewardModel(reward_model_name="gpt2", out_features=7, cache_dir=self.reward_path)
        self.rm.load_state_dict(state_dict, strict=False)
        self.rm = self.rm.to(self.device)

        logger.info("Legacy toxicity reward model loaded successfully")

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """Return a fresh `ValueGuidedProcessor` implementing RAD's reward-augmented shift.

        The candidate policy follows RAD's documented top-k/top-p precedence (`rad_candidate_sizing`),
        which is total (no unassigned-variable path). Non-candidate tokens are masked to `-inf`;
        candidates are min-max normalized within the set (inverted for the legacy toxicity head) and
        shifted by `beta * value`.
        """
        if getattr(self, "_value", None) is None:
            raise RuntimeError("RAD.steer() must run before generation (reward model not loaded).")
        sizing = rad_candidate_sizing(kwargs)
        return [
            ValueGuidedProcessor(
                self._value,
                policy=sizing["policy"],
                k=sizing["k"],
                p=sizing["p"],
                beta=self.beta,
                normalize="minmax",
                invert=self._legacy,
                mask_non_candidates=True,
                lm_tokenizer=self.tokenizer,
            )
        ]

    def cleanup(self) -> None:
        """Release the reward model and tokenizer to free GPU memory."""
        if hasattr(self, "rm") and self.rm is not None:
            del self.rm
            self.rm = None
        if hasattr(self, "rm_tokenizer") and self.rm_tokenizer is not None:
            del self.rm_tokenizer
            self.rm_tokenizer = None
        self._value = None
        self.model = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("RAD cleanup completed")
