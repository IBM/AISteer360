"""Segment-search decoding driver: propose continuations, score them, keep the top k, iterate.

Every rollout receives the composed stacks, so step-level controls steer every lookahead of every
segment-search driver.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steerability.algorithms.core.execution.contracts import Capability, Requirements, needs
from steerability.algorithms.output_control.base import DecodingDriver, resolve_generate_callable
from steerability.algorithms.output_control.common.drivers.frontier import Frontier
from steerability.algorithms.output_control.common.drivers.proposer import SegmentProposer
from steerability.algorithms.output_control.common.scorers import SequenceScorer
from steerability.utils.tokenization import infer_attention_mask_from_ids, to_left_pad


def _resolve_reward_params(runtime_kwargs: Mapping[str, Any], batch_size: int = 1) -> list[dict[str, Any]]:
    """Return the `reward_params` mapping of each row for this call.

    A mapping applies to every row. A sequence contains one mapping per row, in row order, and is the
    form that a batched caller or the evaluation collator passes. A missing key, a None value, or a
    None element gives an empty mapping.

    Args:
        runtime_kwargs: The call's runtime kwargs.
        batch_size: The number of prompt rows in the call.

    Returns:
        One new dict per row, empty for a row without reward params.

    Raises:
        ValueError: If a sequence does not contain exactly one element per row.
        TypeError: If the value is neither a mapping nor a non-string sequence, or if a sequence
            element is neither a mapping nor None.
    """
    value = runtime_kwargs.get("reward_params")
    if value is None:
        return [{} for _ in range(batch_size)]
    if isinstance(value, Mapping):
        return [dict(value) for _ in range(batch_size)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != batch_size:
            raise ValueError(
                f"reward_params is row-scoped and takes one mapping per prompt row; got a sequence of "
                f"length {len(value)} for {batch_size} row(s)."
            )
        rows = []
        for row in value:
            if row is not None and not isinstance(row, Mapping):
                raise TypeError(f"reward_params rows must be mappings; got {type(row).__name__}.")
            rows.append(dict(row) if row is not None else {})
        return rows
    raise TypeError(
        f"reward_params must be a mapping or a sequence of mappings, one per row; got {type(value).__name__}."
    )


class SearchDriver(DecodingDriver):
    """Decoding driver that extends each prompt one segment at a time and keeps the highest-scoring continuations.

    `SearchDriver` is constructed with the arguments listed under `Args:`. Alternatively, a subclass
    sets an `Args` dataclass and overrides `_configure()` to set these fields, and the constructor
    then validates the subclass's `Args`. For each prompt row, with pad positions removed, a search
    repeats three steps:

    1. **Proposal**: `num_candidates` continuations of at most `segment_len` tokens are generated
       through the session from each sequence in the frontier, which starts as the prompt. With
       `propose_mode="beam"`, the continuations are the returned beams of a beam search with
       `num_candidates` beams. With `propose_mode="sample"`, they are sampled.
    2. **Scoring**: the scorer receives the decoded prompt, the decoded continuations (all text after
       the prompt), and the row's scoring parameters, and returns one score per continuation.
    3. **Selection**: the `keep_k` highest-scoring sequences are kept. The kept sequences that are
       finished leave the frontier, and the unfinished ones form the next frontier.

    The search stops after `max_iterations` iterations, when every kept sequence is finished, or when
    the token budget is used up. It returns the highest-scoring sequence seen in any iteration.

    The caller's `max_new_tokens` is the token budget of each search, and no returned continuation is
    longer than it. Each rollout generates at most `segment_len` tokens, and at most the budget minus
    the width of the frontier's continuations, which are right-padded to a common length. The
    caller's `min_new_tokens` applies to each rollout and is reduced to the rollout's length when
    larger. A kept sequence is finished when its continuation, with trailing pad tokens removed, ends
    in an eos token or reaches the budget. The eos ids are the tokenizer's eos id and the ids on the
    model's generation config. When the pad token is an eos id, a sequence with trailing pad tokens
    is also finished. A sequence that a caller's stop rule ended (a stopping criterion, stop string,
    or stop token id) is otherwise unfinished, and the search continues it.

    When the proposals sample, a call with `num_return_sequences=n` runs `n` searches per row. Each
    search is seeded differently, since each session call derives its own seeds. Proposals in sample
    mode always sample. In beam mode, they sample only when the call passes `do_sample=True`, or
    passes no `do_sample` and the model's generation config samples. Without sampling, the driver
    runs one search per row and returns its result as each of the row's `n` candidates. Each returned
    row is the padded prompt row followed by one search's best continuation, right-padded to a common
    length. Rows are ordered by prompt row and then by candidate.

    `whole_batch_rollouts` is False because each search runs on one row. With enabled in-process
    state controls, the pipeline accepts one prompt row per call.

    The following `runtime_kwargs` are accepted:

    - `"reward_params"`: A mapping of entries added to the scorer's `params` on every scoring call
      for the row. A single mapping applies to every row, and a sequence contains one mapping per
      row. The scorer's `params` also contain `segment_len`, `num_candidates`, `keep_k`, and
      `max_iterations`, which take precedence over entries of the same name.

    Args:
        scorer: A `SequenceScorer`, i.e., a callable `(prompt, continuations, params) -> list[float]`
            that returns one score per continuation, with higher scores preferred.
        segment_len: Maximum number of new tokens per rollout, or None to use the call's
            `max_new_tokens`.
        num_candidates: Number of continuations proposed from each frontier sequence per iteration.
        keep_k: Number of sequences kept after each iteration.
        max_iterations: Maximum number of search iterations.
        propose_mode: `"beam"` (default) or `"sample"`.

    Attributes:
        tokenizer: The tokenizer that decodes prompts and continuations, attached by the pipeline or
            by a subclass's `steer()`, or None before it is attached.
    """

    whole_batch_rollouts: bool = False
    tokenizer: PreTrainedTokenizerBase | None = None

    RUNTIME_KWARGS_SCHEMA = [
        {
            "name": "reward_params",
            "type": "dict",
            "scope": "row",
            "help": (
                "Entries merged into the scorer's params mapping on every scoring call for the "
                "row. The per-row form is one mapping; a batched delivery is a sequence containing "
                "one mapping per row, and a single mapping applies to every row. A per-sample "
                "mapping (for example a reference answer under 'reference') reaches the scorer's "
                "row through SampleSequenceScorer."
            ),
        },
    ]

    def __init__(self, *args, **kwargs):
        # a subclass with an `Args` dataclass validates it and sets its fields in `_configure()`
        if self.Args is not None:
            super().__init__(*args, **kwargs)
        else:
            self._init_fields(*args, **kwargs)

    def _init_fields(
        self,
        scorer: SequenceScorer,
        segment_len: int | None,
        num_candidates: int,
        keep_k: int,
        max_iterations: int,
        propose_mode: str = "beam",
    ) -> None:
        """Set the fields of a driver constructed directly with the arguments listed under `Args:`."""
        self.scorer = scorer
        self.segment_len = segment_len
        self.num_candidates = num_candidates
        self.keep_k = keep_k
        self.max_iterations = max_iterations
        self.propose_mode = propose_mode

    def max_rollouts_per_query(self) -> int:
        """Return the largest number of rollouts one search requests.

        The first iteration proposes `num_candidates` continuations from the prompt. Each later
        iteration proposes `num_candidates` continuations from each of at most `keep_k` kept
        sequences. A call with `num_return_sequences=n` runs at most `n` searches per row.

        Returns:
            `num_candidates * (1 + (max_iterations - 1) * keep_k)`.
        """
        return self.num_candidates * (1 + (self.max_iterations - 1) * self.keep_k)

    def requirements(self) -> Requirements:
        """Rollouts run through the session, so sampled proposals require nothing beyond the
        session contract; beam proposals require `Capability.BEAM_PROPOSALS`."""
        if getattr(self, "propose_mode", "sample") == "beam":
            return Requirements(generate=needs(
                Capability.BEAM_PROPOSALS,
                hint="use propose_mode='sample' or run this pipeline on the huggingface backend",
            ))
        return Requirements()

    def decode(self, input_ids, attention_mask, model: PreTrainedModel | None, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs) -> torch.Tensor:
        """Run the segment search from each prompt row and return `num_return_sequences` candidates per row.

        Args:
            input_ids: Prompt token ids of shape `[B, T]`, left-padded by the pipeline. A 1-D tensor
                is treated as one row.
            attention_mask: Prompt attention mask with the same shape as `input_ids`, or None when no
                row is padded.
            model: The pipeline's model, used only to read the eos ids and the `do_sample` default of
                its generation config, or None.
            logits_processors: The composed `LogitsProcessorList`, applied in every rollout.
            stopping_criteria: The composed `StoppingCriteriaList`, applied in every rollout.
            runtime_kwargs: Per-call parameters (see the class docstring).
            session: The `SteeringSession` on which every rollout runs.
            **gen_kwargs: Generation keyword arguments. `max_new_tokens` is the token budget of each
                search, and `num_return_sequences` sets the number of candidates per row.

        Returns:
            A tensor of shape `[B * n, T + L]`, where `n` is `num_return_sequences` and `L` is the
            length of the longest continuation (at most `max_new_tokens` when it is set). Rows are
            ordered by prompt row and then by candidate. Each row is the padded prompt followed by
            one search's best continuation, right-padded.

        Raises:
            RuntimeError: If no tokenizer is attached because `steer()` has not run, or if the scorer
                returns a number of scores that differs from the number of proposals.
            ValueError: If no session was provided, if neither `segment_len` nor `max_new_tokens` is
                set, if `propose_mode` is not `"beam"` or `"sample"`, or if `reward_params` is a
                sequence whose length differs from the number of rows.
            TypeError: If `reward_params` is not a mapping or a sequence of mappings.
        """
        if self.tokenizer is None:
            raise RuntimeError("SearchDriver requires a tokenizer; steer() must run first.")

        runtime_kwargs = runtime_kwargs or {}
        base_generate = resolve_generate_callable(model, runtime_kwargs, session=session)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        num_candidates = gen_kwargs.pop("num_return_sequences", None) or 1
        global_budget = gen_kwargs.pop("max_new_tokens", None)
        # deterministic proposals give every search of a row the same result
        num_searches = num_candidates if self._proposals_sample(gen_kwargs, model) else 1

        # segment_len=None means "one full-budget segment" (best-of-N as a config); each rollout then
        # spans the whole generation budget in a single iteration
        segment_len = self.segment_len if self.segment_len is not None else global_budget
        if segment_len is None:
            raise ValueError(
                "SearchDriver requires a segment length: set `segment_len` or pass `max_new_tokens` "
                "as the generation budget (segment_len=None uses the budget as the segment length)."
            )

        row_params = _resolve_reward_params(runtime_kwargs, input_ids.size(0))
        search_params = {
            "segment_len": segment_len,
            "num_candidates": self.num_candidates,
            "keep_k": self.keep_k,
            "max_iterations": self.max_iterations,
        }
        tokenizer_eos = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = {tokenizer_eos} if tokenizer_eos is not None else set()
        if model is not None:
            configured = getattr(model.generation_config, "eos_token_id", None)
            eos_ids.update([configured] if isinstance(configured, int) else (configured or []))

        full_rows = []
        for row in range(input_ids.size(0)):
            prompt_ids = input_ids[row] if attention_mask is None else input_ids[row][attention_mask[row].bool()]
            prompt_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
            for _ in range(num_searches):
                best = self._search(
                    prompt_ids.unsqueeze(0), prompt_text, {**row_params[row], **search_params}, segment_len,
                    global_budget, sorted(eos_ids) or None, model, base_generate, logits_processors,
                    stopping_criteria, gen_kwargs,
                )
                continuation = best[prompt_ids.numel():].to(device=input_ids.device, dtype=input_ids.dtype)
                full_rows.extend([torch.cat([input_ids[row], continuation])] * (num_candidates // num_searches))

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer_eos or 0
        width = max(full_row.numel() for full_row in full_rows)
        return torch.stack([
            torch.nn.functional.pad(full_row, (0, width - full_row.numel()), value=pad_token_id)
            for full_row in full_rows
        ])

    def _proposals_sample(self, gen_kwargs: dict, model: PreTrainedModel | None) -> bool:
        """Return whether the rollouts sample, which decides whether repeated searches from one prompt can differ.

        Rollouts in sample mode always sample. Rollouts in beam mode sample when `gen_kwargs` sets
        `do_sample=True`, or when it sets no `do_sample` and the model's generation config samples.

        Args:
            gen_kwargs: The call's generation keyword arguments.
            model: The pipeline's model, or None.

        Returns:
            True when the rollouts sample, otherwise False.
        """
        if self.propose_mode == "sample":
            return True
        do_sample = gen_kwargs.get("do_sample")
        if do_sample is None:
            do_sample = getattr(getattr(model, "generation_config", None), "do_sample", False)
        return bool(do_sample)

    def _search(self, prompt_ids, prompt_text, reward_params, segment_len, global_budget, eos_token_id, model,
                base_generate, logits_processors, stopping_criteria, gen_kwargs) -> torch.Tensor:
        """Run one segment search from a single unpadded prompt row.

        Each iteration proposes at most `segment_len` tokens, and at most the part of `global_budget`
        that the width of the frontier's continuations has not used. The search stops after
        `max_iterations` iterations, when every kept sequence is finished, or when no budget is left.

        Args:
            prompt_ids: The unpadded prompt token ids of shape `[1, P]`.
            prompt_text: The decoded prompt, passed to the scorer.
            reward_params: The scorer's `params` mapping for this row.
            segment_len: Maximum number of new tokens per rollout.
            global_budget: Maximum number of new tokens for the search, or None for no limit.
            eos_token_id: The eos ids that finish a sequence, or None.
            model: The pipeline's model, or None.
            base_generate: The generate callable the rollouts use.
            logits_processors: The composed `LogitsProcessorList`, applied in every rollout.
            stopping_criteria: The composed `StoppingCriteriaList`, applied in every rollout.
            gen_kwargs: The caller's generation keyword arguments. Each rollout receives a deep copy.

        Returns:
            The highest-scoring sequence seen in any iteration (the prompt followed by its
            continuation) as a 1-D tensor. When no score exceeds `-inf`, the top kept sequence of the
            last iteration is returned, and when no iteration runs, the prompt alone is returned.

        Raises:
            RuntimeError: If the scorer returns a number of scores that differs from the number of
                proposals.
            ValueError: If `propose_mode` is not `"beam"` or `"sample"`.
        """
        input_length = prompt_ids.size(1)
        proposer = SegmentProposer(mode=self.propose_mode)
        frontier = Frontier(
            keep_k=self.keep_k,
            eos_token_id=eos_token_id,
            input_length=input_length,
            max_new_tokens=global_budget,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
        )

        current_ids = prompt_ids
        kept = None
        for _ in range(self.max_iterations):
            proposal_len = segment_len
            if global_budget is not None:
                # frontier rows share one width, which bounds every row's continuation length
                remaining = global_budget - (current_ids.size(1) - input_length)
                if remaining <= 0:
                    break
                proposal_len = min(segment_len, remaining)
            # safe to deepcopy: the composed stacks travel as explicit decode() parameters, never inside gen_kwargs
            rollout_kwargs = copy.deepcopy(gen_kwargs)
            if rollout_kwargs.get("min_new_tokens") is not None and rollout_kwargs["min_new_tokens"] > proposal_len:
                rollout_kwargs["min_new_tokens"] = proposal_len
            # kept beams that finished early end in trailing pads; each row continues from its
            # left-padded form, which places the new tokens after the row's last real token
            frontier_mask = infer_attention_mask_from_ids(current_ids, self.tokenizer.pad_token_id)
            frontier_ids, frontier_mask = to_left_pad(current_ids, frontier_mask)
            beams = proposer.propose(
                frontier_ids,
                n=self.num_candidates,
                segment_len=proposal_len,
                processors=logits_processors,
                criteria=stopping_criteria,
                model=model,
                base_generate=base_generate,
                attention_mask=frontier_mask,
                **rollout_kwargs,
            )
            # rotate each beam's leading pads to its tail, restoring the prompt-first layout; the
            # rollouts of frontier row i are rows i * n to (i + 1) * n
            leading = (frontier_mask == 0).sum(dim=1).repeat_interleave(beams.size(0) // frontier_ids.size(0))
            width = beams.size(1)
            positions = torch.arange(width, device=beams.device).unsqueeze(0) + leading.to(beams.device).unsqueeze(1)
            beams = beams.gather(1, positions % width)
            continuations = self.tokenizer.decode(
                beams[:, input_length:], skip_special_tokens=True
            )
            scores = self.scorer(prompt_text, continuations, reward_params)
            if len(scores) != beams.size(0):
                raise RuntimeError(f"Scorer returned {len(scores)} scores for {beams.size(0)} beams.")

            step = frontier.keep(beams, scores)
            kept = step
            if all(step.finished_flags):
                break

            unfinished = [i for i, f in enumerate(step.finished_flags) if not f]
            if not unfinished:
                break
            current_ids = step.kept_ids[unfinished]

        if frontier.best_ids is not None:
            return frontier.best_ids
        return kept.kept_ids[0] if kept is not None else prompt_ids[0]
