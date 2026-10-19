from __future__ import annotations

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.output_control.common.drivers.search import SearchDriver
from steerability.algorithms.output_control.common.resolve import resolve_scorer
from steerability.algorithms.output_control.search_decoding.args import SearchDecodingArgs


class SearchDecoding(SearchDriver):
    """Decoding driver that runs a configured segment search over continuations.

    `SearchDecoding` extends each prompt one segment at a time. Each iteration proposes
    `num_candidates` continuations of up to `segment_len` tokens from each kept sequence, scores them
    with the scorer, and keeps the `keep_k` highest-scoring sequences. The search runs for at most
    `max_iterations` iterations and returns the highest-scoring sequence among those it stopped
    extending, i.e., the finished kept sequences of every iteration and the kept sequences of the
    last iteration. On equal scores the finished sequence found first is returned.

    The `scorer` argument accepts a `SequenceScorer`, i.e., a callable
    `(prompt, continuations, params) -> list[float]` such as a function, a `MajorityVoteScorer`, or a
    `SampleSequenceScorer`. It also accepts a dict with a `"kind"` key, either `"reward_model"` (with
    a `model_id`) or `"majority_vote"`. `steer()` resolves a dict, and a `"reward_model"` dict loads
    its model onto the device of the pipeline's model. The `propose_mode` argument is `"sample"`
    (default) or `"beam"`. `segment_len=None` (default) uses the call's `max_new_tokens` as the
    segment length. The integer arguments must be positive, and `keep_k` must not exceed
    `num_candidates`.

    The defaults (`num_candidates=8`, `keep_k=1`, `max_iterations=1`, and sampled proposals) give
    best-of-N, which samples `num_candidates` full-length continuations once and returns the one with
    the highest score. Methods from the literature correspond to the following settings:

    - Best-of-N: the defaults with `scorer={"kind": "reward_model", "model_id": ...}` or any callable.
    - Self-consistency: the defaults with `scorer={"kind": "majority_vote"}`.
    - Blockwise controlled decoding: `segment_len=block` and `max_iterations=ceil(budget / block)`.
    - Scorer-guided reranking: the defaults with `scorer=SampleSequenceScorer(row_scorer)`.
    - DeAL: `propose_mode="beam"`, `segment_len=lookahead`, `num_candidates=init_beams`,
      `keep_k=topk`, and the DeAL `max_iterations`.

    The composed logits processors and stopping criteria apply in every rollout, and a step-level
    control such as `ValueGuidance` steers every proposed continuation. A call with
    `num_return_sequences=n` runs `n` searches per row. In beam mode without sampling, one search
    runs per row and its result is returned as each of the `n` candidates.

    The following `runtime_kwargs` are accepted:

    - `"reward_params"`: A mapping of entries added to the scorer's `params` on every scoring call
      for the row. A single mapping applies to every row, and a sequence contains one mapping per
      row.

    Reference:

    - "DeAL: Decoding-time Alignment for Large Language Models"
      James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas,
      Saab Mansour, Katrin Kirchhoff, Dan Roth
      [https://arxiv.org/abs/2402.06147](https://arxiv.org/abs/2402.06147)

    - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
      Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery,
      Denny Zhou
      [https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)
    """

    Args = SearchDecodingArgs

    tokenizer: PreTrainedTokenizerBase | None = None

    def _configure(self) -> None:
        """Map the mirrored args onto the generic `SearchDriver` fields (name-identical here)."""
        # self.scorer / segment_len / num_candidates / keep_k / max_iterations / propose_mode are
        # already mirrored from SearchDecodingArgs; the driver reads them under the same names
        self.tokenizer = None

    def steer(self, model: PreTrainedModel | None = None, tokenizer: PreTrainedTokenizerBase | None = None,
              **_) -> PreTrainedModel | None:
        """Attach the tokenizer and resolve the scorer spec (a device is needed for reward models)."""
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        device = next(model.parameters()).device if model is not None else None
        self.scorer = resolve_scorer(self.scorer, device=device)
        return model
