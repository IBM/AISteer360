# Adding an output control method

Output control methods constrain or transform what leaves the decoder.

## Config first, subclass second

The first design decision is config first, subclass second. Before writing a class, check whether the method is an
assignment of a config of one of the [generic controls](../../concepts/controls.md#generic-controls). Most output
methods from the literature map onto one of them:

- a method that reshapes the next-token distribution from a per-candidate score is a [`ValueGuidance`](../../concepts/controls.md#generic-controls) config (FUDGE, ARGS, RAD, SASA)
- one that mixes weighted full-vocabulary log-prob sources is a [`ContrastiveGuidance`](../../concepts/controls.md#generic-controls) config (DExperts, contrastive decoding, proxy-tuning)
- one that changes the shape of the search (propose, score, keep, iterate) is a [`SearchDecoding`](../../concepts/controls.md#generic-controls) config (best-of-N, self-consistency, DeAL)
- one that splices forced and generated segments is a [`PhasedDecoding`](../../concepts/controls.md#generic-controls) config (budget forcing, response prefill, thinking intervention)
- one that stops on a substring, token, or budget is a [`StoppingRules`](../../concepts/controls.md#generic-controls) config

If so, implement the method as a config, not a class. When a config earns a name through use, promote it with a small preset
subclass that maps its named args onto the fields of the driver base the generic is built on. The named methods follow
this pattern, e.g., `BestOfN` subclasses `SearchDriver` (the base of `SearchDecoding`) and `BudgetForcing` subclasses
`PhasedDriver` (the base of `PhasedDecoding`):

```python
class BestOfN(SearchDriver):
    """Sample n continuations, return the scorer's argmax (rejection sampling)."""
    Args = BestOfNArgs          # fields: n, scorer

    def _configure(self):
        self.num_candidates = self.n
        self.keep_k = 1
        self.max_iterations = 1
        self.segment_len = None
        self.propose_mode = "sample"
        self.tokenizer = None
```

Write a full control class only when the method needs behavior no config expresses: a new candidate policy, a new
value/source/scorer component, or a bespoke decode loop.

## Contribute or drive?

If you are writing a class, output controls participate through one of two mechanisms, and the next design
decision is choosing which:

- **Contribute**: supply logits processors and/or stopping criteria. The pipeline composes every step-level control's
  processors in `controls`-list order into one list (and likewise for stopping criteria), then hands both lists to
  whichever driver owns the loop. A step-level control never runs the decode loop itself and therefore composes with
  other step-level controls and with a driver. **Override**: `get_logits_processors` and/or `get_stopping_criteria`.
- **Drive**: own the decode loop. A driver subclasses `DecodingDriver` and implements `decode(...)`, applying the
  composed processors and stopping criteria in every forward pass it issues. Since the loop does not compose, a pipeline admits at most one
  enabled driver. With none, decoding defaults to the model's own `generate`. **Override**: `decode`.

As a rule of thumb, if the method reshapes the next-token distribution one step at a time (reward shifts, contrastive
mixtures, constraint masks), it is a step-level control. If it changes the shape of the search (lookahead, re-ranking,
phased generation, best-of-N), it is a driver.

Both modes may also implement `steer()` (one-time preparation, e.g., loading a reward model) and `cleanup()` (release
those resources). Each method is a package directory with `args.py`, `control.py`, and a `STEERING_METHOD` export in
`__init__.py` that the registry discovers:

```python
from .control import KeywordBooster
from .args import KeywordBoosterArgs

STEERING_METHOD = {
    "category": "output_control",
    "name": "keyword_booster",
    "control": KeywordBooster,
    "args": KeywordBoosterArgs,
}
```

## Contribute: logits processors

`KeywordBooster` adds a fixed bias to the logits of a set of keyword tokens at every step, making those words more
likely. It edits the distribution one step at a time and is therefore a step-level control.

The args dataclass declares the hyperparameters. The keyword strings are tied to the prompt and supplied at inference
time, arriving via `runtime_kwargs` rather than the constructor. The control declares the name it consumes in
`RUNTIME_KWARGS_SCHEMA`. All controls read from the one `runtime_kwargs` dict, and the pipeline warns at `steer()`
time when two controls declare the same name.

```python
from dataclasses import dataclass, field
from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class KeywordBoosterArgs(BaseArgs):
    boost: float = field(
        default=5.0,
        metadata={"help": "Additive logit bias applied to each keyword token."},
    )

    def __post_init__(self):
        if self.boost < 0:
            raise ValueError("`boost` must be non-negative.")
```

The control returns a fresh processor from `get_logits_processors` on every call, since the hook is invoked once per
`generate()`/`compute_logprobs()` to isolate per-generation state. A processor is any callable
`(input_ids, scores) -> scores` following the Hugging Face `LogitsProcessor` convention:

```python
from transformers import PreTrainedModel, PreTrainedTokenizer

from steerability.algorithms.output_control.base import OutputControl
from steerability.algorithms.output_control.keyword_booster.args import KeywordBoosterArgs


class KeywordBooster(OutputControl):
    """Adds a fixed logit bias to a set of keyword tokens at every decoding step."""

    Args = KeywordBoosterArgs
    RUNTIME_KWARGS_SCHEMA = [{"name": "keywords"}]

    tokenizer: PreTrainedTokenizer | None = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **__) -> PreTrainedModel:
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs) -> list:
        runtime_kwargs = runtime_kwargs or {}
        keywords = runtime_kwargs.get("keywords", [])
        keyword_ids = [
            token_id
            for word in keywords
            for token_id in self.tokenizer.encode(word, add_special_tokens=False)
        ]

        def _boost(prefix_ids, scores):
            scores = scores.clone()
            for token_id in keyword_ids:
                scores[:, token_id] += self.boost
            return scores

        return [_boost]  # fresh instance per call
```

Because it only contributes, `KeywordBooster` composes freely. For instance, `controls=[KeywordBooster(...), DeAL(...)]`
applies the boost inside every DeAL rollout, and `controls=[KeywordBooster(...)]` alone runs under the default
`model.generate` loop.

!!! note "Processor purity"
    A processor must behave as a function of `(prefix_ids, scores)`. Drivers may restart, rewind, or reorder sequences
    (segment search re-enters from a shorter frontier and beam search permutes rows), and `compute_logprobs` replays
    prefixes teacher-forced. Any internal state must therefore be memoization keyed on the prefix. Subclass
    [`PrefixKeyedProcessor`](../../reference/algorithms/output_control/common.md) to get this contract mechanically.
    It calls your `reset_state(input_ids)` whenever the observed prefix no longer extends the last one.

By default a step-level control's logits edits also apply during `compute_logprobs`, and scoring therefore reflects
the steered distribution. Set `include_in_scoring = False` (a class attribute) to opt out when the per-position cost is prohibitive.

## Drive: a decoding driver

`ShortestOfN` samples several continuations and returns the shortest one. It changes the shape of the search and is
therefore a driver. A driver receives the composed `logits_processors` / `stopping_criteria` as explicit parameters
and must apply them in every forward pass it issues, i.e., pass them to each rollout it runs through the pipeline's
session. The helper `stack_generate_kwargs` builds the two kwargs, including each only when non-empty. The helper
`session_generate_items(session, rows, eos_token_ids=(), **gen_kwargs)` (from
`steerability.algorithms.core.execution.session_utils`) runs one session call over unpadded prompt rows, with one
candidate per row, and returns one `(continuation_ids, finish_reason)` pair per row. Each continuation has its trailing
pads removed. When the pad token is also an eos token, one pad is kept as the emitted eos unless the row already ends
on a terminal token, and `eos_token_ids` lists the further eos ids that count as terminal (e.g., the ids on the model's
generation config). For rollouts in the `model.generate` calling convention (padded input,
full sequences returned), `resolve_generate_callable(model, runtime_kwargs, session=session)` returns a callable that
generates through the session.

A driver also returns `n` candidates for every prompt row of the batch, where `n` is the caller's
`num_return_sequences` (passed to `pipeline.generate()` as `n` or `num_return_sequences`). `ShortestOfN` therefore
runs one selection per candidate, each over `num_samples` samples of the row's prompt with its left padding removed:

```python
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.session_utils import session_generate_items
from steerability.algorithms.output_control.base import DecodingDriver, stack_generate_kwargs


@dataclass
class ShortestOfNArgs(BaseArgs):
    num_samples: int = field(default=4, metadata={"help": "Number of continuations sampled per candidate."})

    def __post_init__(self):
        if self.num_samples < 1:
            raise ValueError("`num_samples` must be >= 1.")


class ShortestOfN(DecodingDriver):
    """Samples `num_samples` continuations per candidate and keeps the shortest."""

    Args = ShortestOfNArgs
    whole_batch_rollouts = False  # each session call covers the samples of a single row

    tokenizer: PreTrainedTokenizer | None = None

    def steer(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer | None = None, **__) -> PreTrainedModel:
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        return model

    def max_rollouts_per_query(self) -> int:
        return self.num_samples

    def decode(self, input_ids, attention_mask, model, logits_processors,
               stopping_criteria, runtime_kwargs, session=None, **gen_kwargs) -> torch.Tensor:
        stacks = stack_generate_kwargs(logits_processors, stopping_criteria)  # the composed processors and criteria
        num_candidates = gen_kwargs.pop("num_return_sequences", None) or 1
        kwargs = {**gen_kwargs, **stacks, "do_sample": True}
        eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos_token_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())

        rows = []
        for index, padded_prompt in enumerate(input_ids):
            prompt = padded_prompt if attention_mask is None else padded_prompt[attention_mask[index].bool()]
            samples = session_generate_items(
                session, [prompt] * (num_candidates * self.num_samples), eos_token_ids=eos_token_ids, **kwargs
            )
            for start in range(0, len(samples), self.num_samples):  # the samples of one candidate
                shortest = min((ids for ids, _ in samples[start:start + self.num_samples]), key=len)
                rows.append(torch.cat([padded_prompt, shortest.to(padded_prompt.device)]))

        width = max(row.numel() for row in rows)
        pad_id = self.tokenizer.pad_token_id
        return torch.stack([F.pad(row, (0, width - row.numel()), value=pad_id) for row in rows])
```

!!! note "The driver contract"
    `logits_processors` and `stopping_criteria` are the composed lists for this generation, and a driver must apply
    them in every forward pass. The `gen_kwargs` reaching `decode` never contain `logits_processor` or
    `stopping_criteria`, since the pipeline removes caller-supplied ones and composes them into these lists. The
    pipeline also passes `session=`, a `SteeredSession` that contains this generation's control entries, and rollouts
    issued through it run steered on any backend.

    The pipeline left-pads a batch of prompts before calling `decode`. For a batch of `B` rows of length `T` and
    `n = num_return_sequences`, `decode` returns `[B * n, T + L]` ids. Each row is a padded prompt row as received
    followed by one candidate's continuation, right-padded, in row-major, candidate-minor order (the layout of
    `model.generate`), and the pipeline slices every row at `T` to recover the continuations.

    In-process state hooks are built once per generation for the whole batch and index their gates and masks by
    row. A driver whose session calls cover a subset of the rows, or rows at different stream lengths, sets
    `whole_batch_rollouts = False`. The pipeline then accepts one prompt row per call when enabled state controls
    run in process. Under a seed with `seed_scope="item"`, the in-process session decodes each call one row at a
    time, and the pipeline then accepts one prompt row per call for any driver unless the call passes
    `seed_scope="dispatch"`. A driver can also override `max_rollouts_per_query()` to declare an upper bound on the
    continuations it requests per row and candidate (`ShortestOfN` returns `self.num_samples`). The default returns
    `None`, meaning no static bound.

    A decoding driver registered in the toolkit also needs an entry in `PRESET_KWARGS` in
    `tests/controls/test_driver_candidates.py`, which checks this output layout for every registered preset.

## Prefer the `common` library

Most methods do not start from scratch. The [`output_control.common`](../../reference/algorithms/output_control/common.md)
library factors the category into reusable components, and the methods in the toolkit are thin recipes over them:

- `ValueGuidedProcessor` (step-level candidate scoring): `RAD`, `SASA`.
- `ContrastiveMixtureProcessor` (mix full-vocabulary logit sources): `DExperts`, `ContrastiveDecoding`.
- `SearchDriver` (propose, score, keep top-k, iterate): `DeAL`, `BestOfN`.
- `PhasedDriver` (forced/generated segments with boundary rules): `BudgetForcing`.

A driver built on `SearchDriver` or `PhasedDriver` is a preset. It declares an `Args` dataclass and overrides
`_configure()` to set the fields the generic driver reads from its args, and it defines no `__init__` of its own.
When a subclass sets `Args`, the generic driver's constructor validates the args and copies them onto the instance
before it calls `_configure()`. See `deal/control.py` and `budget_forcing/control.py` for the pattern. An argument-free control (no hyperparameters) sets `Args = None` and takes no constructor arguments.

When adding a component to `common`, follow its naming convention. Within a `common/<family>/` folder, the primary
class in `<name>.py` is `<Name><FamilySingular>` (for example `values/classifier.py` defines `ClassifierValue`), and
the family base is in `base.py`. Top-level `common/*.py` modules (such as `candidates.py` and `criteria.py`) are
collection or helper modules exempt from the suffix rule.

## Running the control

Either mode is instantiated and added to a pipeline the same way:

```python
from steerability.algorithms.output_control.keyword_booster.control import KeywordBooster
from steerability.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

keyword_booster = KeywordBooster(boost=6.0)

pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[keyword_booster],
    device_map="auto",
)
pipeline.steer()

prompt = "Explain linear algebra in two sentences."
chat = pipeline.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = pipeline.tokenizer(chat, return_tensors="pt").to(pipeline.model.device)

output = pipeline.generate(
    input_ids=inputs.input_ids,
    runtime_kwargs={"keywords": ["matrix", "vector"]},
    max_new_tokens=50,
    do_sample=True,
)
print(pipeline.tokenizer.decode(output[0], skip_special_tokens=True))

# different keywords can be supplied at inference time, without re-steering
output = pipeline.generate(
    input_ids=inputs.input_ids,
    runtime_kwargs={"keywords": ["eigenvalue", "determinant"]},
    max_new_tokens=50,
    do_sample=True,
)
print(pipeline.tokenizer.decode(output[0], skip_special_tokens=True))
```
