from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.deal.args import DeALArgs


class DeAL(OutputControl):
    """
    Implementation of DeAL (Decoding-time Alignment) from Deng et al., 2024.

    DeAL performs controlled text generation through iterative lookahead search and reward-guided beam selection. Unlike
    training-time alignment methods, DeAL operates purely at inference time to steer language model outputs toward
    desired behaviors.

    The algorithm works in three phases:

    1. **Lookahead Generation**: Generate multiple candidate continuations using beam search from the current context.

    2. **Reward-based Scoring**: Evaluate each candidate continuation using a provided reward function that measures
    alignment with the desired objective (e.g., helpfulness, safety).

    3. **Iterative Refinement**: Select the top-k highest-scoring beams and repeat the process until termination
    conditions are met (EOS token, max length, or max iterations reached).

    This approach allows for flexible alignment with various objectives without requiring model retraining or
    fine-tuning.

    Args:
        reward_func (Callable): Function that scores generated continuations. Should accept
            (prompt: str, continuations: list[str], reward_params: dict) and return list[float].
        lookahead (int): Number of tokens to generate in each lookahead step. Defaults to 4.
        init_beams (int): Number of initial beams to generate at each iteration. Defaults to 8.
        topk (int): Number of top-scoring beams to retain for the next iteration. Defaults to 4.
        max_iterations (int): Maximum number of search iterations before termination. Defaults to 10.

    Reference:

    - "DeAL: Decoding-time Alignment for Large Language Models"
    James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour,
    Katrin Kirchhoff, Dan Roth
    https://arxiv.org/abs/2402.06147
    """

    Args = DeALArgs

    # placeholders
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    base_generate: Callable | None = None

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **_
    ) -> PreTrainedModel:
        """Lightweight preparation; attaches model, tokenizer, and generate to instance."""
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.base_generate = model.generate
        return model

    def _lookahead_generation(
        self,
        input_ids: torch.Tensor,
        reward_func: Callable[[str, list[str], dict], list[float]],
        reward_params: dict,
        base_generate: Callable,
        input_length: int,
        **gen_kwargs,
    ) -> tuple[list[float], torch.Tensor]:
        """Generate and score candidate continuations for one lookahead iteration.

        Generates multiple beam candidates using the base model's generation method, then evaluates each continuation
        with the reward function to guide selection.

        Args:
            input_ids (torch.Tensor): Current context tokens to continue from.
                Shape can vary based on number of active beams.
            reward_func (Callable[[str, list[str], dict], list[float]]): Function to score continuations.
                Receives (original_prompt, continuation_texts, params).
            reward_params (dict): Parameters passed to reward function, including algorithm
                settings (lookahead, init_beams, topk, max_iterations).
            base_generate (Callable): Generation function used to produce candidate continuations.
            input_length (int): Length of original input prompt, used to extract only the newly generated portion for
                scoring.
            **gen_kwargs: Generation parameters forwarded to base_generate (including num_beams, max_new_tokens, etc.)

        Returns:
            tuple[list[float], torch.Tensor]: Tuple containing:
                - Reward scores for each generated beam (list of floats)
                - Full token sequences including input and continuations (tensor)

        Raises:
            RuntimeError: If reward function returns wrong number of scores (must match number of generated beams).

        Note:

        - Continuations are decoded to text for reward evaluation
        - Special tokens are skipped when extracting continuation text
        - Stores original prompt in self.prompt for reward function access
        """
        lookaheads = base_generate(input_ids=input_ids, **gen_kwargs)
        continuations: list[str] = self.tokenizer.batch_decode(
            lookaheads[:, input_length:], skip_special_tokens=True
        )
        scores = reward_func(self.prompt, continuations, reward_params)
        if len(scores) != lookaheads.size(0):
            raise RuntimeError(f"Reward function returned {len(scores)} scores for {lookaheads.size(0)} beams.")
        return scores, lookaheads

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        runtime_kwargs: dict | None,
        model: PreTrainedModel,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Execute guided generation with iterative lookahead search and reward-based selection. Returns the
        highest-scoring generation.

        The generation process is as follows:

        1. Generate `init_beams` candidate continuations of `lookahead` tokens each
        2. Score all candidates using the provided reward function
        3. Select top-k highest scoring beams
        4. Check termination conditions (EOS, max length, max iterations)
        5. If not terminated, continue from the selected beams
        6. Return the highest-scoring complete generation

        Args:
            input_ids (torch.Tensor): Input token IDs of shape [1, seq_len].
                Currently only supports single prompts (batch size must be 1).
            attention_mask (torch.Tensor): Attention mask matching input_ids shape.
                Automatically recomputed during iteration based on padding tokens.
            runtime_kwargs (dict | None): Runtime parameters including:

                - "base_generate" (`Callable`, optional): Override the model's generate function
                - "reward_params" (`dict`, optional): Additional parameters passed to reward_func
            model (PreTrainedModel): The language model used for generation.
                Must match the model provided during steer().
            **gen_kwargs: Generation parameters passed to the underlying model.generate().
                Note: `max_new_tokens` is extracted and used as global limit; `num_beams` and `num_return_sequences` are
                overridden by DeAL parameters.

        Returns:
            torch.Tensor: Generated token IDs of shape [1, output_len] or [output_len].
                Contains the highest-scoring complete generation found during search.

        Raises:
            ValueError: If base_generate is not callable
            NotImplementedError: If input has batch size > 1 (multiple prompts not supported)
            RuntimeError: If reward function returns incorrect number of scores
        """
        runtime_kwargs = runtime_kwargs or {}

        reward_func = self.reward_func
        base_generate = runtime_kwargs.get("base_generate", self.base_generate)

        if not callable(base_generate):
            raise ValueError("'base_generate' must be callable; supplied or cached from steer().")

        # assert (
        #     self.model is not None and self.tokenizer is not None
        # ), "DeAL.steer() must run before generate()."

        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise NotImplementedError("Current DeAL implementation handles one prompt at a time.")

        # record call‑specific objects
        self.prompt: str = self.tokenizer.decode(
            input_ids[0], skip_special_tokens=True
        )
        input_length = input_ids.size(1)

        reward_params = {
            **runtime_kwargs.get("reward_params", {}),
            "lookahead": self.lookahead,
            "init_beams": self.init_beams,
            "topk": self.topk,
            "max_iterations": self.max_iterations,
        }

        original_max_tokens: Optional[int] = gen_kwargs.pop("max_new_tokens", None)

        # search loop
        best_beam: torch.Tensor | None = None
        best_score = float("-inf")
        current_input_ids = input_ids
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            attention_mask = (current_input_ids != self.tokenizer.pad_token_id).long()
            gen_args = copy.deepcopy(gen_kwargs)
            gen_args.update(
                {
                    "max_new_tokens": self.lookahead,
                    "num_beams": self.init_beams,
                    "num_return_sequences": self.init_beams,
                    "attention_mask": attention_mask,
                }
            )

            # rollout + scoring
            scores, beams = self._lookahead_generation(
                current_input_ids,
                reward_func=reward_func,
                reward_params=reward_params,
                base_generate=base_generate,
                input_length=input_length,
                **gen_args,
            )

            # select top-k
            score_tensor = torch.tensor(scores, device=beams.device)
            topk = min(self.topk, score_tensor.numel())
            top_idx = torch.topk(score_tensor, topk).indices
            beams = beams[top_idx]
            scores = score_tensor[top_idx].tolist()

            # termination mask
            finished_flags = []
            for beam in beams:
                eos_hit = beam[...,-1] == self.tokenizer.eos_token_id
                len_hit = (
                        original_max_tokens is not None
                        and beam.size(0) - input_length >= original_max_tokens
                )
                finished_flags.append(bool(eos_hit or len_hit))

            # update best-so-far
            best_local = int(torch.argmax(torch.tensor(scores)))
            if scores[best_local] > best_score:
                best_score = scores[best_local]
                best_beam = beams[best_local]

            if all(finished_flags):
                break

            # prune unfinished beams for next round
            current_input_ids = beams[
                [i for i, f in enumerate(finished_flags) if not f]
            ]

        final_ids = best_beam if best_beam is not None else beams[0]
        return final_ids.unsqueeze(0) if final_ids.dim() == 1 else final_ids
