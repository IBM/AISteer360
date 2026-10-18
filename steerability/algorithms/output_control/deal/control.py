from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.common.drivers.search import SearchDriver
from steerability.algorithms.output_control.deal.args import DeALArgs


class DeAL(SearchDriver):
    """Implementation of DeAL (Decoding-time Alignment) from Huang et al., 2024.

    DeAL aligns generation with an objective at inference time through a lookahead search guided by a
    reward function. Each search iteration has three steps:

    1. **Lookahead**: beam search generates `init_beams` continuations of up to `lookahead` tokens
       from each sequence in the frontier, which starts as the prompt.
    2. **Scoring**: `reward_func` scores each continuation for the objective (e.g., helpfulness or
       safety).
    3. **Selection**: the `topk` highest-scoring sequences are kept, and the unfinished ones form the
       next frontier.

    The search stops after `max_iterations` iterations, when every kept sequence is finished (it ends
    in an eos token or reaches `max_new_tokens`), or when no token budget is left. DeAL returns the
    highest-scoring sequence among those it stopped extending, i.e., the finished kept sequences of
    every iteration and the kept sequences of the last iteration. On equal scores the finished
    sequence found first is returned.

    The `reward_func` argument accepts a callable `(prompt, continuations, params) -> list[float]`
    that returns one score per continuation, with higher scores preferred. Each continuation is the
    decoded text after the prompt. The `lookahead` (default 10), `init_beams` (default 5), `topk`
    (default 3), and `max_iterations` (default 10) arguments must be positive integers, and `topk`
    must not exceed `init_beams`.

    The composed logits processors and stopping criteria apply in every lookahead rollout, and a
    step-level control such as RAD steers every rollout. Without sampling, beam search is
    deterministic. The `num_return_sequences` candidates of a prompt then come from a single search
    and are identical. Beam proposals require a backend with `Capability.BEAM_PROPOSALS`.

    The following `runtime_kwargs` are accepted:

    - `"reward_params"`: A mapping of entries added to the `params` passed to `reward_func` on every
      scoring call for the row. A single mapping applies to every row, and a sequence contains one
      mapping per row.

    Reference:

    - "DeAL: Decoding-time Alignment for Large Language Models"
    James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour,
    Katrin Kirchhoff, Dan Roth
    https://arxiv.org/abs/2402.06147
    """

    Args = DeALArgs

    tokenizer: PreTrainedTokenizerBase | None = None

    def _configure(self) -> None:
        """Map DeAL's mirrored args onto the generic `SearchDriver` fields."""
        self.scorer = self.reward_func
        self.segment_len = self.lookahead
        self.num_candidates = self.init_beams
        self.keep_k = self.topk
        # self.max_iterations is already mirrored from DeALArgs
        self.propose_mode = "beam"

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None, **_) -> PreTrainedModel:
        """Lightweight preparation; attach the tokenizer used to decode continuations."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model
