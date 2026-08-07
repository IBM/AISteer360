"""Output control base classes.

Output controls participate in decoding through two mechanisms:

- Logits-processor composition: each control's `get_logits_processors()` results are gathered in
  pipeline `controls` list order and composed by `LogitsProcessorList`.
- Stopping-criteria composition: each control's `get_stopping_criteria()` results are gathered the
  same way; generation stops when any criterion fires.
- The decode loop is exclusive: it is implemented by exactly one `DecodingDriver`, which receives
  the composed stacks as explicit parameters and must apply them at every scoring step of every
  forward pass it issues.

Examples of output controls:

- Reward-augmented decoding (a step-level control)
- Self-disciplined autoregressive sampling (a step-level control)
- Decoding-time alignment / lookahead search (a decoding driver)
- Phase splicing / thinking intervention (a decoding driver)

See Also:

- `aisteer360.algorithms.output_control`: Implementations of output control methods
- `aisteer360.algorithms.output_control._common`: Shared component library
- `aisteer360.algorithms.core.steering_pipeline`: Integration with steering pipeline
"""
from abc import abstractmethod
from typing import Type

import torch
from transformers import LogitsProcessorList, PreTrainedModel, StoppingCriteriaList

from aisteer360.algorithms.core.base_args import BaseArgs
from aisteer360.algorithms.core.base_control import BaseControl


def stack_generate_kwargs(logits_processors, stopping_criteria) -> dict:
    """Build the `model.generate` kwargs for the composed stacks, each included only when non-empty.

    Shared by every driver that delegates to `model.generate` (the default driver, and the segment
    and phase drivers per rollout or per phase) so the "pass the stack only when non-empty" rule
    lives in one place.
    """
    extra: dict = {}
    if logits_processors is not None and len(logits_processors):
        extra["logits_processor"] = logits_processors
    if stopping_criteria is not None and len(stopping_criteria):
        extra["stopping_criteria"] = stopping_criteria
    return extra


class OutputControl(BaseControl):
    """Base class for output-control steering methods.

    An `OutputControl` participates in decoding through the composable mechanisms above.
    Controls that implement a decoding procedure subclass `DecodingDriver` instead.

    Class attributes:
        include_in_scoring: Whether this control's logits processors also apply during
            `SteeringPipeline.compute_logprobs()` (per-position, teacher-forced). Defaults
            to True. Set False when the processors are too expensive to evaluate per reference
            position (see `BaseCandidateValue.scoring_cost`).
        same_model_forwards: Whether this component issues additional forward passes through the
            pipeline's own model during decoding. Such passes must be wrapped in
            `auxiliary_pass()` (see `aisteer360.algorithms.core.utils.auxiliary_pass`), which
            keeps them out of state-control condition scoring, gate updates, and fallback
            position counting. Defaults to False; the flag is declarative metadata and is not
            read by the pipeline.
    """

    Args: Type[BaseArgs] | None = None
    RUNTIME_KWARGS_SCHEMA: list[dict] = []

    enabled: bool = True
    supports_batching: bool = False
    include_in_scoring: bool = True
    same_model_forwards: bool = False

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """The control's logits processors for the current generation.

        Called once per `generate()` / `compute_logprobs()` call, after input and state
        controls have prepared the prompt (mirrors `StateControl.get_hooks`). `**kwargs`
        carries `attention_mask` and the caller's generation kwargs. Returned objects
        follow the HF `LogitsProcessor` convention; in-list order is preserved by the composition.

        A processor must behave as a function of `(prefix_ids, scores)`. Internal state is
        permitted only as memoization keyed on the prefix and must re-derive on a prefix mismatch,
        since drivers may restart, rewind, or reorder sequences, and scoring replays prefixes
        teacher-forced (subclass `_common.processors.base.PrefixKeyedProcessor` to satisfy this
        mechanically). Return fresh processor instances from this hook; it is invoked once per call
        precisely so that per-generation state is isolated.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.
            **kwargs: Carries `attention_mask` and the caller's generation kwargs.

        Returns:
            A list of HF `LogitsProcessor`-style objects.
        """
        return []

    def get_stopping_criteria(self, input_ids, runtime_kwargs, **kwargs) -> list:
        """The control's stopping criteria.

        Not applied during scoring (there is no loop to stop). Same call convention as
        `get_logits_processors`.

        Args:
            input_ids: The steered prompt token ids `[batch, seq_len]`.
            runtime_kwargs: Per-call parameters supplied to `generate()`.
            **kwargs: Carries `attention_mask` and the caller's generation kwargs.

        Returns:
            A list of HF `StoppingCriteria`-style objects.
        """
        return []

    def steer(self, model: PreTrainedModel, tokenizer=None, **kwargs) -> None:
        """Optional one-time preparation (e.g., load a reward model, fit a probe)."""
        pass


class DecodingDriver(OutputControl):
    """An output control that implements the decoding procedure.

    Exactly one enabled driver may exist per pipeline (the decode loop does not compose).
    Driver contract: `logits_processors` and `stopping_criteria` are the composed,
    authoritative stacks for this generation; the driver applies them at every scoring
    step of every forward pass it issues. Delegating to `model.generate(...,
    logits_processor=..., stopping_criteria=...)` satisfies the contract; hand-rolled
    loops apply them explicitly.

    A driver is also an `OutputControl`: it may additionally contribute processors or
    criteria of its own via the `get_*` hooks, which the pipeline composes like any other
    control's.
    """

    @abstractmethod
    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: PreTrainedModel,
        logits_processors: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        runtime_kwargs: dict | None,
        **gen_kwargs,
    ) -> torch.Tensor:
        """Run the decoding procedure; return full sequence ids (prompt + continuation)."""


class HFGenerateDriver(DecodingDriver):
    """Default decoding driver: delegate the loop to the model's own `generate`."""

    supports_batching: bool = True

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, **gen_kwargs) -> torch.Tensor:
        extra = stack_generate_kwargs(logits_processors, stopping_criteria)
        return model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **extra, **gen_kwargs
        )
