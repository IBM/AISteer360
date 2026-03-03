from __future__ import annotations

import gc
import logging
import os
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GPT2LMHeadModel,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.rad.args import RADArgs

logger = logging.getLogger(__name__)


class RAD(OutputControl):
    """
    Implementation of RAD (Reward-Augmented Decoding) from Deng and Raffel, 2023.
    Integrated from the official implementation of RAD ([https://github.com/r-three/RAD?tab=readme-ov-file](https://github.com/r-three/RAD?tab=readme-ov-file)).

    RAD works in two phases:

    1. **Reward model training**: Train a reward model with a lebeled dataset containing texts and labels.
    For detials about this step, please see [https://github.com/r-three/RAD?tab=readme-ov-file](https://github.com/r-three/RAD?tab=readme-ov-file). We skip this
    step in this implementation and re-use the open-source toxicity reward model trained by the authors via
    gdown [https://storage.googleapis.com/rad_release/saved_models.zip](https://storage.googleapis.com/rad_release/saved_models.zip)

    2. **Controlled decoding**: At every decoding step the candidate-token logits are shifted by **beta * reward**,
    where the *reward* is given by a trained reward model.

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
    base_generate: Callable | None = None
    _uses_hf_classifier: bool = False

    beta: float

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **__,
    ) -> PreTrainedModel:
        """Initialize RAD by loading and configuring the reward model.

        Sets up the reward model used for steering during generation. Supports two modes:

        1. **HuggingFace classifier** (new path): When `reward_model_id` is set, loads any
           `AutoModelForSequenceClassification` compatible model from HuggingFace Hub.
        2. **Legacy toxicity model** (original path): When `reward_path` is set (or neither is set),
           loads the GPT-2 based toxicity classifier from the original RAD paper.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for the base model.
                If None, attempts to retrieve from model attributes.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model, unchanged.

        Note:

        - For legacy path: Downloads ~500MB reward model on first use if not cached
        - Legacy reward model is GPT2-based with 7 toxicity classification heads
        - Model weights are loaded onto the same device as the base model
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.base_generate = model.generate
        self.device = next(model.parameters()).device

        if self.reward_model_id is not None:
            # new path: load any HuggingFace sequence classification model
            self._load_hf_classifier()
        else:
            # legacy path: load GPT-2 toxicity reward model
            self._load_legacy_toxicity_model()

        return model

    def _load_hf_classifier(self) -> None:
        """Load a HuggingFace AutoModelForSequenceClassification reward model."""
        logger.info("Loading reward model from HuggingFace: %s", self.reward_model_id)

        self.rm = AutoModelForSequenceClassification.from_pretrained(
            self.reward_model_id,
            **self.reward_model_kwargs,
        )
        self.rm = self.rm.to(self.device)
        self.rm.eval()

        self.rm_tokenizer = AutoTokenizer.from_pretrained(self.reward_model_id)
        if self.rm_tokenizer.pad_token is None:
            self.rm_tokenizer.pad_token = self.rm_tokenizer.eos_token
        self.rm_tokenizer.padding_side = "right"

        # set a reasonable max_length for the reward model tokenizer;
        # some tokenizers have model_max_length set to an absurdly large sentinel value
        max_len = getattr(self.rm_tokenizer, "model_max_length", None)
        if max_len is None or max_len > 100_000:
            max_len = 512  # reasonable default for reward models
        self.rm_tokenizer.max_length = max_len

        self._uses_hf_classifier = True
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

        self._uses_hf_classifier = False
        logger.info("Legacy toxicity reward model loaded successfully")

    def cleanup(self) -> None:
        """Release the reward model and tokenizer to free GPU memory."""
        if hasattr(self, "rm") and self.rm is not None:
            del self.rm
            self.rm = None
        if hasattr(self, "rm_tokenizer") and self.rm_tokenizer is not None:
            del self.rm_tokenizer
            self.rm_tokenizer = None

        self.model = None
        self.tokenizer = None
        self.base_generate = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("RAD cleanup completed")

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            runtime_kwargs: dict | None,
            model: PreTrainedModel,
            **gen_kwargs,
    ) -> torch.Tensor:
        """Execute RAD-guided generation with reward-augmented logits processing.

        Performs controlled generation by shifting token logits at each decoding step based on reward model scores.
        Returns generated text steered toward desired behavior.

        At each decoding step:

        1. Generate top-k candidate next tokens
        2. Score each candidate continuation with the reward model
        3. Adjust logits by beta * reward_score
        4. Sample from adjusted distribution

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask matching input_ids shape.
            runtime_kwargs (dict | None): Runtime parameters (currently unused).
            model (PreTrainedModel): The language model used for generation.
                Must match the model provided during steer().
            **gen_kwargs: Generation parameters passed to model.generate():

                - "temperature" (`float`, optional): Sampling temperature. Defaults to 1.0.
                - "top_k" (`int`, optional): Top-k filtering. Defaults to 0 (disabled).
                - "top_p" (`float`, optional): Nucleus sampling threshold. Defaults to 1.0.
                - "repetition_penalty" (`float`, optional): Penalty for repeated tokens. Defaults to 1.0.
                - Other standard generation arguments (max_length, pad_token_id, etc.)

        Returns:
            torch.Tensor: Generated token IDs with same batch dimension as input.

        Note:

        - Requires reward model to be loaded during steer() phase
        - When both top_k and top_p are specified, top_k takes precedence for RAD processing
        - Reward scores are clamped to [0, 1] and inverted (1 - score) for toxicity reduction
        - Non-top-k tokens are set to -inf to ensure selection from reward-adjusted candidates
        """

        runtime_kwargs = runtime_kwargs or {}
        beta = self.beta

        processors = LogitsProcessorList()
        temperature = gen_kwargs.get("temperature", 1.0)
        if temperature and temperature != 1.0:
            processors.append(TemperatureLogitsWarper(temperature))

        top_k = gen_kwargs.get("top_k", 0)
        if top_k and top_k > 0:
            processors.append(TopKLogitsWarper(top_k))
            rad_topk = top_k
            rad_topp = 1

        top_p = gen_kwargs.get("top_p", 1.0)
        if top_p and top_p < 1.0:
            processors.append(TopPLogitsWarper(top_p))
            rad_topp = top_p
            rad_topk = None

        repetition_penalty = gen_kwargs.get("repetition_penalty", 1.0)
        if repetition_penalty and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))

        # construct reward score extraction function based on model type
        if self._uses_hf_classifier:
            rm_score_fn = lambda output: output.logits[:, 0]
            inverse = False  # general reward models: higher = better
        else:
            rm_score_fn = lambda output: output[:, 0]
            inverse = True  # legacy toxicity model: invert scores

        processors.append(
            RewardAugmentedLogitsProcessorNoPkv(
                self.tokenizer,
                self.rm_tokenizer,
                self.rm,
                topk=rad_topk,
                topp=rad_topp,
                method="linear",
                beta=beta,
                inverse=inverse,
                rm_score_fn=rm_score_fn,
            )
        )

        # generate candidates
        output = self.base_generate(input_ids=input_ids, attention_mask=attention_mask, logits_processor=processors, **gen_kwargs)
        return output


class RewardAugmentedLogitsProcessorNoPkv(LogitsProcessor):
    """Logits processor that adjusts token probabilities based on reward model scores.

    Implements the core RAD algorithm by evaluating candidate tokens with a reward model and shifting their logits
    proportionally to the reward scores. Designed to work with transformers' generate() method as part of a
    `LogitsProcessorList`.

    Args:
        lm_tokenizer: Tokenizer for the language model being steered.
        rm_tokenizer: Tokenizer for the reward model (typically GPT-2).
        reward_model: Trained reward model that scores text for desired attributes.
        topk (int): Number of candidate tokens to evaluate. Defaults to 20.
        topp (float): Nucleus sampling threshold if using top-p instead of top-k. Defaults to 1.
        method (str): Reward application method. Currently only "linear" supported. Defaults to "linear".
        beta (float): Scaling factor for reward scores. Higher values = stronger steering. Defaults to 30.
        inverse (bool): Whether to invert reward scores (1 - score). Used for toxicity reduction. Defaults to False.
        rm_score_fn (Callable): Function to extract scalar reward from model output. Takes model output and returns
            a tensor of shape [batch]. Defaults to extracting the first column of raw output.
    """
    def __init__(self, lm_tokenizer, rm_tokenizer, reward_model, topk=20, topp=1,
                 method="linear", beta=30, inverse=False, rm_score_fn: Callable | None = None):
        self._lm_tokenizer = lm_tokenizer
        self._rm_tokenizer = rm_tokenizer
        self._reward_model = reward_model
        self._device = next(self._reward_model.parameters()).device
        self._reward_model.eval()
        self._topk = topk
        self._topp = topp
        self._method = method
        self._beta = beta
        self._inverse = inverse
        self._rm_score_fn = rm_score_fn if rm_score_fn is not None else (lambda output: output[:, 0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply reward-based adjustments to token logits.

        For each position in the batch, evaluates top-k candidate tokens by constructing full text sequences, scoring
        them with the reward model, and adjusting logits.

        Args:
            input_ids (torch.LongTensor): Current token sequence of shape [batch_size, seq_len].
            scores (torch.FloatTensor): Raw logits for next token of shape [batch_size, vocab_size].

        Returns:
            torch.FloatTensor: Adjusted logits with reward-based modifications.
                Non-candidate tokens are set to -inf to ensure sampling from evaluated tokens only.

        Note:
            - Dynamically switches between top-k and top-p candidate selection
            - Constructs full text for each candidate to enable proper reward model evaluation
            - Memory usage scales with batch_size * topk for candidate evaluation
        """
        if self._topp < 1:
            ## top p modification, batch=1
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_keep = cumulative_probs > (1 - self._topp)
            indices_to_keep = sorted_indices_to_keep.scatter(1, sorted_indices, sorted_indices_to_keep)
            topk_ids = torch.nonzero(indices_to_keep)[:,1].unsqueeze(0)
            self._topk = topk_ids.shape[1]
            del sorted_logits, sorted_indices, cumulative_probs, sorted_indices_to_keep, indices_to_keep
            torch.cuda.empty_cache()  # Ensure immediate deallocation
        else:
            _, topk_ids = torch.topk(scores, self._topk, dim=-1)                                    # (batch, topk,)
        input_ids_enflated = input_ids.unsqueeze(1).expand((-1, self._topk, -1))                # (batch, topk, seq_len)
        candidate_input_ids = torch.cat((input_ids_enflated, topk_ids.unsqueeze(-1)), dim=-1)   # (batch, topk, seq_len+1)
        candidate_input_ids_unroll = candidate_input_ids.reshape((
            candidate_input_ids.shape[0]*candidate_input_ids.shape[1], -1))         # (batch*topk, seq_len+1)
        candidate_input_texts = self._lm_tokenizer.batch_decode(candidate_input_ids_unroll, skip_special_tokens=True)

        # return reward scores
        reward_scores = self.get_reward(candidate_input_texts).reshape((input_ids.shape[0], -1))

        # apply function (topk_scores, logits)
        for score, id, rs in zip(scores, topk_ids, reward_scores):

            score[id] = self.apply_function(score[id], rs)
            inverse_id = torch.tensor(np.setdiff1d(range(len(score.cpu().numpy())), id.cpu().numpy()), device=self._device)
            score[inverse_id] = -float("Inf")  # set all other scores to -inf
        return scores

    def get_reward(self, candidate_texts):
        """Score candidate text sequences with the reward model.

        Args:
            candidate_texts: List of text strings to evaluate.

        Returns:
            torch.Tensor: Reward scores for each candidate, extracted via rm_score_fn.
        """
        with torch.inference_mode():
            # tokenizer should be configured in RAD
            inputs = self._rm_tokenizer.batch_encode_plus(
                candidate_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._rm_tokenizer.max_length,
            ).to(self._device)

            output = self._reward_model(**inputs)
            return self._rm_score_fn(output)

    def apply_function(self, original_score, reward_score):
        """Apply reward adjustment to original logits.

        Args:
            original_score: Original logit values for candidate tokens.
            reward_score: Reward model scores for candidates.

        Returns:
            torch.Tensor: Adjusted logits computed as original + (beta * reward).

        Raises:
            ValueError: If method is not "linear".

        Note:

        - Reward scores are normalized to [0, 1] via min-max scaling within the candidate set.
        """
        # normalize to [0, 1] within this candidate set
        r_min = reward_score.min()
        r_max = reward_score.max()
        if r_max - r_min > 1e-8:
            reward_score = (reward_score - r_min) / (r_max - r_min)
        else:
            reward_score = torch.ones_like(reward_score) * 0.5

        if self._inverse:
            reward_score = 1 - reward_score

        if self._method == "linear":
            return original_score + (reward_score * self._beta).to(original_score.dtype)
        else:
            raise ValueError(f"method {self._method} not supported")


class GPT2RewardModel(nn.Module):
    """GPT-2 based reward model for scoring text toxicity or other attributes.

    Modified GPT-2 architecture where the language modeling head is replaced with a classification head. Used to score
    text sequences for desired attributes during RAD-guided generation.

    Args:
        reward_model_name (str): Base GPT-2 model variant to use. Defaults to "gpt2".
        out_features (int): Number of output classes/attributes. Defaults to 1.
    """
    def __init__(self, reward_model_name="gpt2", out_features=1, cache_dir='./'):
        super(GPT2RewardModel, self).__init__()
        model = GPT2LMHeadModel.from_pretrained(reward_model_name, cache_dir=cache_dir)
        model.lm_head = nn.Linear(in_features=model.lm_head.in_features, out_features=out_features, bias=True)
        self.model = model
        self.pad_token_id = model.config.eos_token_id
        self.out_features = out_features

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass through reward model.

        Processes input through GPT-2 backbone and returns scores from the classification head at the last valid token
        position for each sequence.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len].
            past_key_values: Cached key-value pairs for efficient generation.
            attention_mask: Attention mask for padding.
            token_type_ids: Token type IDs (unused for GPT-2).
            position_ids: Position embeddings.
            head_mask: Attention head mask.

        Returns:
            torch.Tensor: Classification scores of shape [batch_size, out_features].
                Extracted from the last non-padding position of each sequence.
        """
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        logits = outputs['logits']
        # find the last valid token's ids
        sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(logits.device)
        # use the last valid token's representation: (batch, max_length, out_features) => (batch, out_features)
        scores = logits[torch.arange(input_ids.shape[0], device=logits.device), sequence_lengths]

        return scores
