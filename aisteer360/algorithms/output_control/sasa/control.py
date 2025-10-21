from __future__ import annotations

import copy
from typing import Callable, Optional

import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.cache_utils import DynamicCache
from transformers.generation.stopping_criteria import StoppingCriteriaList

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.sasa.args import SASAArgs


class SASA(OutputControl):
    """
    Implementation of SASA (Self-disciplined autoregressive sampling) from Ko et al., 2024.

    SASA works in two phases:

    1. **Subspace learning**: From a labelled toxic / non-toxic corpus, it fits a linear classifier in the model’s
    own sentence-embedding space; the weight vector defines a toxicity subspace.

    2. **Controlled decoding**: At every decoding step the candidate-token logits are shifted by **beta * margin**,
    where *margin* is the classifier distance of the updated context from the toxic side of the subspace.  Sampling
    from the soft-max of these adjusted logits (optionally with nucleus sampling) nudges generation away from
    toxic regions while staying close to the original distribution.

    Args:
        beta (float): Scaling coefficient for value redistribution. Defaults to 0.0.
        wv_path (str, optional): Path to a saved steering-vector tensor. Defaults to None.
        gen_wv_data_path (str, optional): Path to the value dataset, e.g. sentences with labeled toxicity. Defaults to "Jigsaw_data/".
        gen_wv_length (int, optional): The maximum number of samples used for preparing SASA steering if wv_path does not exist. Defaults to -1 (use all).
        gen_wv_batch_size (int, optional): The batch size used for preparing SASA steering if wv_path does not exist. Defaults to 4.

    Reference:

    - "Large Language Models can Become Strong Self-Detoxifiers"
      Ching-Yun Ko, Pin-Yu Chen, Payel Das, Youssef Mroueh, Soham Dan, Georgios Kollias, Subhajit Chaudhury,
      Tejaswini Pedapati, Luca Daniel
      [https://arxiv.org/abs/2410.03818](https://arxiv.org/abs/2410.03818)
    """
    Args = SASAArgs

    # placeholders (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    base_generate: Callable | None = None

    beta: float
    wv: torch.Tensor | None

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **__,
    ) -> PreTrainedModel:
        """Initialize SASA by loading or generating the toxicity steering vector.

        Sets up the linear classifier in the model's embedding space that defines the toxicity subspace. Either loads a
        pre-computed steering vector or generates one from labeled data.

        Args:
            model (PreTrainedModel): The base language model to be steered.
            tokenizer (PreTrainedTokenizer | None): Tokenizer for the base model.
                If None, attempts to retrieve from model attributes.
            **__: Additional arguments (unused).

        Returns:
            PreTrainedModel: The input model (unchanged).

        Raises:
            FileNotFoundError: If gen_wv_data_path doesn't contain required Jigsaw dataset

        Note:

        - If wv_path is provided, loads pre-computed steering vector
        - Otherwise generates steering vector from Jigsaw toxicity dataset
        - Steering vector generation uses closed-form Bayes optimal classifier
        - Saves generated steering vector to 'steer_wv.pt' for future use
        """
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        if self.tokenizer.pad_token_id is None:
            print("pad_token is absent. Setting it to eos_token or '<pad>'.")
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:  # edge case
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.pad_token_id = tokenizer.eos_token_id
        self.base_generate = model.generate
        self.device = next(model.parameters()).device
        if getattr(self, "wv_path", None):
            print("Loading SASA steer (wv)......")
            self.wv = torch.load(self.wv_path, map_location="cpu")
        else:
            print("Creating SASA steer (wv)......")
            self._setup_wv()
            # self.wv =  {k: v.cpu() for k, v in self.wv.item().items()}
            torch.save(self.wv, 'tmp/steer_wv.pt')
        self.wv = {key: value.to(self.device) for key, value in self.wv.items()}
        return model

    def _setup_wv(self):
        """Generate steering vector from labeled toxicity data.

        Loads the Jigsaw toxicity dataset and learns a linear classifier in the model's embedding space using a
        closed-form Bayes optimal solution. The resulting weight vector defines the toxicity subspace used during
        generation.

        Process:
        1. Load toxic and non-toxic sentences from Jigsaw dataset
        2. Generate sentence embeddings using the model's last hidden states
        3. Compute mean vectors and covariance matrix for both classes
        4. Apply SVD for dimensionality reduction and numerical stability
        5. Compute Bayes optimal linear classifier in reduced space
        6. Project back to original space and normalize

        Raises:
            FileNotFoundError: If Jigsaw dataset not found at gen_wv_data_path

        Note:

        - Uses pooled representation from last non-padding token
        - Handles NaN embeddings by filtering them out
        - Saves computed steering vector to 'steer_wv.pt'
        - Batch processing to manage memory usage
        """

        def batcher(sentences):
            """Generate sentence embeddings using the model's hidden states.

            Args:
                sentences: List of text strings to embed.

            Returns:
                torch.Tensor: Pooled embeddings from last hidden layer, shape [batch_size, hidden_dim].
                    Uses representation from last non-padding token position.
            """
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt', truncation=True, max_length=1024, padding=True,
            )
            for k in batch:
                batch[k] = batch[k].to(self.device)
            batch.pop('token_type_ids', None)

            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.hidden_states[-1]

            pooled_result = last_hidden[range(len(last_hidden)), batch['attention_mask'].sum(-1) - 1]
            return pooled_result.cpu()

        # Load dataset
        import os

        os.makedirs(self.gen_wv_data_path, exist_ok=True)
        if self.gen_wv_data is not None:
            print(f"Data found in: {self.gen_wv_data}")
            pos = self.gen_wv_data['pos']
            neg = self.gen_wv_data['neg']
        elif os.path.exists(os.path.join(self.gen_wv_data_path, "all_data.csv")):
            print(f"Dataset found in: {self.gen_wv_data_path}")
            dataset = pd.read_csv(os.path.join(self.gen_wv_data_path, "all_data.csv"))
            pos = [row for i, row in dataset['comment_text'].items() if isinstance(row, str) and dataset['toxicity'][i] == 0]
            neg = [row for i, row in dataset['comment_text'].items() if isinstance(row, str) and dataset['toxicity'][i] > 0]
        else:
            raise FileNotFoundError(
                f"""
                    Jigsaw dataset not found at: {self.gen_wv_data_path}
                    To use jigsaw_unintended_bias you have to download it manually from Kaggle: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
                    You can manually download the data from it's homepage or use the Kaggle CLI tool (follow the instructions here: https://www.kaggle.com/docs/api)
                    Please extract all files in one folder and then load the dataset with:
                    dataset = pd.read_csv('/tmp/Jigsaw_data/all_data.csv')
                    """
            )

        num = len(pos) + len(neg)
        print(f"There are overall {len(pos)} positive sentences and {len(neg)} negative sentences.")
        if self.gen_wv_length > 0 and self.gen_wv_length < num:
            num_pos = int(self.gen_wv_length / num * len(pos))
            num_neg = self.gen_wv_length - num_pos
            pos = pos[:num_pos]
            neg = neg[:num_neg]
        print(f"Generating wv via {len(pos)} positive sentences and {len(neg)} negative sentences.")

        sorted_pos = sorted(pos, key=lambda z: -len(z))
        sorted_neg = sorted(neg, key=lambda z: -len(z))

        # Gather embeddings
        embeddings_pos = []
        embeddings_neg = []
        for ii in tqdm(range(0, len(sorted_pos), self.gen_wv_batch_size), desc="Embedding POS"):
            batch = sorted_pos[ii:ii + self.gen_wv_batch_size]
            embeddings_pos.append(torch.tensor(batcher(batch)))
        for ii in tqdm(range(0, len(sorted_neg), self.gen_wv_batch_size), desc="Embedding NEG"):
            batch = sorted_neg[ii:ii + self.gen_wv_batch_size]
            embeddings_neg.append(torch.tensor(batcher(batch)))

        X1_train = torch.vstack(embeddings_pos)
        X2_train = torch.vstack(embeddings_neg)
        X1_train = X1_train[~torch.isnan(X1_train).any(dim=1)]
        X2_train = X2_train[~torch.isnan(X2_train).any(dim=1)]

        # Obtain closed-form Bayes optimal classifier
        mu_1 = torch.mean(X1_train, axis=0)
        cov = torch.cov(X1_train.T) * (X1_train.shape[0] - 1)
        mu_2 = torch.mean(X2_train, axis=0)
        cov += torch.cov(X2_train.T) * (X2_train.shape[0] - 1)
        cov = cov / (X1_train.shape[0] + X2_train.shape[0] - 2)

        torch.cuda.empty_cache()

        F, D, _ = torch.svd(cov, some=True)
        F = F[:, D > 1e-6].float()
        D = D[D > 1e-6].float()
        D_inv = torch.diag(D ** (-1))

        mu = torch.matmul(F.t(), (mu_1 - mu_2) / 2)
        mu_mu = (mu_1 + mu_2) / 2
        w_0 = torch.matmul(D_inv, mu)
        wv = torch.matmul(F, w_0)
        wv = wv / torch.norm(wv)

        self.wv = {'wv': wv, 'mu_mu': mu_mu}

    @staticmethod
    def repeat_kv_cache(cache, repeats: int):
        """Repeat KV cache entries for parallel candidate evaluation.

        Duplicates cache entries to enable efficient parallel processing of multiple candidate tokens without
        recomputing shared context.

        Args:
            cache: KV cache in various formats (DynamicCache, tuple, or custom).
            repeats (int): Number of times to repeat each cache entry.

        Returns:
            Repeated cache in same format as input.

        Raises:
            TypeError: If cache type is not supported.
        """
        if hasattr(cache, "batch_repeat_interleave"):
            cache.batch_repeat_interleave(repeats)
            return cache

        elif hasattr(cache, "to_legacy_cache"):
            raw = cache.to_legacy_cache()
            repeated = tuple(
                tuple(t.repeat(repeats, 1, 1, 1) for t in layer)
                for layer in raw
            )
            return DynamicCache.from_legacy_cache(repeated)

        elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i].repeat_interleave(repeats, dim=0)
                cache.value_cache[i] = cache.value_cache[i].repeat_interleave(repeats, dim=0)
            return cache

        elif isinstance(cache, tuple):
            return tuple(
                tuple(t.repeat_interleave(repeats, dim=0) for t in layer)
                for layer in cache
            )

        else:
            raise TypeError(f"Unsupported cache type: {type(cache).__name__}")

    @staticmethod
    def select_kv_cache(cache, select_idx: torch.Tensor):
        """Select specific entries from KV cache based on indices.

        Extracts cache entries corresponding to selected beam paths, used after evaluating multiple candidates to
        continue with the chosen token.

        Args:
            cache: KV cache in various formats.
            select_idx (torch.Tensor): 1D tensor of indices to select.

        Returns:
            Selected cache entries in same format as input.

        Raises:
            ValueError: If select_idx is not 1D.
            TypeError: If cache type is not supported.
        """
        if not torch.is_tensor(select_idx):
            select_idx = torch.as_tensor(select_idx)
        if select_idx.dtype != torch.long:
            select_idx = select_idx.long()
        if select_idx.dim() != 1:
            raise ValueError(f"select_idx must be 1D, got shape {tuple(select_idx.shape)}")

        if hasattr(cache, "batch_select"):
            cache.batch_select(select_idx)
            return cache

        elif hasattr(cache, "batch_gather"):
            cache.batch_gather(select_idx)
            return cache

        elif hasattr(cache, "to_legacy_cache"):
            raw = cache.to_legacy_cache()
            selected = tuple(
                tuple(t[select_idx, :, :, :] for t in layer)
                for layer in raw
            )
            return DynamicCache.from_legacy_cache(selected)

        elif hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
            for i in range(len(cache.key_cache)):
                if cache.key_cache[i] is not None:
                    select_idx_device = select_idx.to(cache.key_cache[i].device)
                    cache.key_cache[i] = cache.key_cache[i].index_select(dim=0, index=select_idx_device)
                if cache.value_cache[i] is not None:
                    select_idx_device = select_idx.to(cache.value_cache[i].device)
                    cache.value_cache[i] = cache.value_cache[i].index_select(dim=0, index=select_idx_device)
            return cache

        elif isinstance(cache, tuple):
            return tuple(
                tuple(t.index_select(dim=0, index=select_idx.to(t.device)) for t in layer)
                for layer in cache
            )

        else:
            raise TypeError(f"Unsupported cache type: {type(cache).__name__}")

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            runtime_kwargs: dict | None,
            model: PreTrainedModel,
            **gen_kwargs,
    ) -> torch.Tensor:
        """Execute SASA-guided generation with margin-based logit adjustment.

        Performs controlled generation by computing the distance from toxic subspace at each decoding step and adjusting
        token logits based on this margin. Returns text steered away from toxic regions while maintaining coherence.

        At each decoding step:

        1. Generate embeddings for all valid candidate tokens
        2. Compute margin (distance from toxic subspace) for each candidate
        3. Adjust logits by beta * softmax(margins)
        4. Sample from adjusted distribution

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [batch_size, seq_len].
            attention_mask (torch.Tensor): Attention mask matching input_ids shape.
            runtime_kwargs (dict | None): Runtime parameters (unused).
            model (PreTrainedModel): The language model used for generation.
                Must match the model provided during steer().
            **gen_kwargs: Generation parameters passed to model internals:

                - "generation_config" (`GenerationConfig`, optional): Generation configuration object
                - "logits_processor" (`LogitsProcessorList`, optional): Custom logit processors
                - "stopping_criteria" (`StoppingCriteriaList`, optional): Custom stopping criteria
                - "max_new_tokens" (`int`, optional): Maximum tokens to generate
                - Standard generation arguments (temperature, top_p, etc.)

        Returns:
            torch.Tensor: Generated token IDs including the input prompt.

        Note:

        - Computes full forward passes for all valid candidate tokens at each step
        - Uses custom KV cache manipulation for efficient candidate evaluation
        - Margins computed relative to learned toxic/non-toxic boundary
        - SASA is memory intensive; scales with vocabulary size at each generation step
        """

        runtime_kwargs = runtime_kwargs or {}
        beta = self.beta
        wv = self.wv

        # # If vanilla decoding, allow opt-out
        # if not runtime_kwargs.get("sasa_enabled", True):
        #     return self.base_generate(input_ids=input_ids, **gen_kwargs)

        inputs: torch.Tensor = input_ids

        generation_config: Optional[GenerationConfig] = gen_kwargs.pop("generation_config", None)
        logits_processor: Optional[LogitsProcessorList] = gen_kwargs.pop("logits_processor", None)
        stopping_criteria: Optional[StoppingCriteriaList] = gen_kwargs.pop("stopping_criteria", None)
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = gen_kwargs.pop(
            "prefix_allowed_tokens_fn", None)

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            generation_config = self.model.generation_config if hasattr(self.model,
                                                                        "generation_config") else GenerationConfig()
        else:
            generation_config = copy.deepcopy(generation_config)

        generation_config, model_kwargs = self.model._prepare_generation_config(
            generation_config,
            use_model_defaults=True,
            **gen_kwargs
        )
        generation_config.validate()

        # Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # Define model inputs
        # input_ids has to be defined
        # all model-specific keyword inputs are removed from `model_kwargs`
        input_ids, _, model_kwargs = self.model._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = input_ids.shape[0]  # todo: unused?
        device = input_ids.device
        self.model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        if generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        # Prepare logits processor, stopping criteria
        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
        )
        stopping_criteria = self.model._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, **gen_kwargs
        )

        # Expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self.model._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=False,
            **model_kwargs,
        )

        # Run sample
        # init values
        scores = ()
        mv = None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False  # used by synced_gpus only

        model_kwargs["cache_position"] = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1
        model_kwargs["attention_mask"] = attention_mask
        # model_kwargs = self.model._get_initial_cache_position(input_ids, model_kwargs)

        # auto-regressive generation
        while True:
            if mv is None:  # when generating the first token
                # prepare model inputs
                model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=True,
                )
            else:
                selected_index = (indices[:, -1] == next_tokens).nonzero(as_tuple=True)
                assert len(selected_index) == 1 and len(selected_index[0]) == 1
                outputs.logits = outputs.logits[selected_index, :, :]
                outputs.hidden_states = tuple(
                    [outputs.hidden_states[i][selected_index, :, :] for i in range(len(outputs.hidden_states))]
                )
                outputs.past_key_values = self.select_kv_cache(outputs.past_key_values, selected_index)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = logits_processor(input_ids, next_token_logits)

            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )

            # prepare the value margins
            with torch.no_grad():
                prev_hidden_states = outputs['hidden_states'][-1][:, -1, :].clone()
                indices = torch.nonzero(next_token_scores > -torch.inf)
                num = indices.shape[0]
                input_ids_temp = torch.cat([input_ids.repeat(num, 1), indices[:, -1].unsqueeze(1)], dim=-1)
                model_kwargs_temp = model_kwargs.copy()

                is_gemma = hasattr(self.model, 'config') and 'gemma' in str(type(self.model)).lower()
                if is_gemma:
                    if hasattr(model_kwargs['past_key_values'], 'get_seq_length'):
                        cache_length = model_kwargs['past_key_values'].get_seq_length()
                    else:
                        # fallback: use cache_position to infer length
                        cache_length = model_kwargs_temp['cache_position'][0].item()
                    # Trim attention_mask to match cache length for gemma
                    model_kwargs_temp['attention_mask'] = model_kwargs_temp['attention_mask'][:, :cache_length]

                    original_cache_pos = model_kwargs_temp['cache_position']
                    new_token_position = original_cache_pos[-1] + 1
                    model_kwargs_temp['cache_position'] = torch.tensor([new_token_position],
                                                                    dtype=original_cache_pos.dtype,
                                                                    device=original_cache_pos.device
                                                                    )
                model_kwargs_temp['attention_mask'] = model_kwargs_temp['attention_mask'].repeat(num, 1)
                model_kwargs_temp['past_key_values'] = self.repeat_kv_cache(model_kwargs['past_key_values'], num)

                model_inputs = self.model.prepare_inputs_for_generation(input_ids_temp, **model_kwargs_temp)
                outputs = self.model(**model_inputs, return_dict=True, output_attentions=True,
                                     output_hidden_states=True, )

                if wv is not None:
                    if isinstance(wv, dict) and len(wv) == 2:
                        mv = (wv['wv'] * (outputs['hidden_states'][-1][:, -1, :] - wv['mu_mu'])).sum(axis=1)
                    else:
                        mv = (wv * (outputs['hidden_states'][-1][:, -1, :] - prev_hidden_states)).sum(axis=1)

            # re-distribute weights
            if wv is not None and mv is not None:
                redistribute = next_token_scores[next_token_scores > -torch.inf] + (beta * mv.softmax(dim=-1)).to(
                    dtype=next_token_scores.dtype)
                next_token_scores[next_token_scores > -torch.inf] = redistribute

            probs = nn.functional.softmax(next_token_scores, dim=-1)
            assert probs.sum() > 0
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # store scores
            scores += (next_token_scores,)

            # finished sentences should have their next token be a padding token
            if generation_config.eos_token_id is not None:
                if generation_config.pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + generation_config.pad_token_id * (
                        1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            if this_peer_finished:
                break

        return input_ids
