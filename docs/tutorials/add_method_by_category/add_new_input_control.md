# Adding an input control method

**Required override**: `adapt`

**Optional overrides**: `adapt_messages`, `steer`, `cleanup`

Input control methods describe algorithms that manipulate the input/prompt to guide model behavior. This tutorial
implements a small input control termed `PromptCensor` that filters and replaces words from a predefined list before
the prompt is passed into the model.

First, create the following directory/files:
```
input_control/
└── prompt_censor/
    ├── __init__.py
    ├── args.py
    └── control.py
```

where the `__init__.py` file is:
```python
from .control import PromptCensor
from .args import PromptCensorArgs

STEERING_METHOD = {
    "category": "input_control",
    "name": "prompt_censor",
    "control": PromptCensor,
    "args": PromptCensorArgs,
}
```

The control requires two arguments: a list of `blocked_words` to filter, and a `replacement` string. This is captured
by the following `args.py` file:
```python
from dataclasses import dataclass, field
from steerability.algorithms.core.base_args import BaseArgs


@dataclass
class PromptCensorArgs(BaseArgs):
    blocked_words: list[str] = field(
        default_factory=lambda: ["dangerous", "harmful", "illegal"],
        metadata={"help": "List of words to filter from prompts."},
    )
    replacement: str = field(
        default="[MASKED]",
        metadata={"help": "Text to replace blocked words with."},
    )

    def __post_init__(self):
        if not isinstance(self.blocked_words, list):
            raise ValueError("`blocked_words` must be a list of strings.")
```

Lastly, the `control.py` file implements the method by overriding the `adapt` method. This method:

- Accepts the tokenized prompt (`input_ids`) and any `runtime_kwargs` supplied to `.generate()`.
- Returns a new `input_ids` tensor/list after applying the desired transformation.

For methods whose work is more naturally expressed at the message level (e.g., setting or replacing a system prompt),
override `adapt_messages` instead. See
[When to override `adapt_messages` instead](#when-to-override-adapt_messages-instead) below.

The control implementation for `PromptCensor` is as follows:

```python
import re

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.input_control.prompt_censor.args import PromptCensorArgs


class PromptCensor(InputControl):
    """Filters potentially harmful content from prompts."""
    Args = PromptCensorArgs
    RUNTIME_KWARGS_SCHEMA = [
        {"name": "blocked_words", "scope": "call", "help": "Words to replace in every prompt of the call."},
        {"name": "replacement", "scope": "call", "help": "The text that replaces each blocked word."},
    ]

    tokenizer: PreTrainedTokenizer | None = None

    def steer(
        self,
        model: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer

    # required override for input control methods
    def adapt(
        self,
        input_ids: list[int] | torch.Tensor,
        runtime_kwargs: dict | None = None,
    ) -> list[int] | torch.Tensor:
        # allow runtime override of blocked words (if specified)
        blocked_words = (runtime_kwargs or {}).get("blocked_words", self.blocked_words)
        replacement = (runtime_kwargs or {}).get("replacement", self.replacement)

        # filter one row of token ids (case-insensitive), decoding without pad tokens
        def filter_row(row_ids: list[int]) -> list[int]:
            row_ids = [token for token in row_ids if token != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(row_ids, skip_special_tokens=False)
            for word in blocked_words:
                text = re.compile(re.escape(word), re.IGNORECASE).sub(replacement, text)
            return self.tokenizer.encode(text, add_special_tokens=False)

        if not isinstance(input_ids, torch.Tensor):
            return filter_row(list(input_ids))
        if input_ids.dim() == 1:
            return torch.tensor(filter_row(input_ids.tolist()), dtype=input_ids.dtype, device=input_ids.device)

        # filter every row, then right-pad the rows to a common width
        rows = [filter_row(row.tolist()) for row in input_ids]
        width = max(len(row) for row in rows)
        padded = [row + [self.tokenizer.pad_token_id] * (width - len(row)) for row in rows]
        return torch.tensor(padded, dtype=input_ids.dtype, device=input_ids.device)
```

Note that the method's `steer` attaches the tokenizer to the control. The `adapt` method filters every row of a batch
and returns a `[B, L']` tensor right-padded with the tokenizer's pad token. The pipeline then infers the attention mask
from the pad runs, since the length changed. The `RUNTIME_KWARGS_SCHEMA` attribute declares the per-call variables the
control reads from `runtime_kwargs` (both apply to the whole call, i.e., `"scope": "call"`), and the pipeline warns at
`steer()` time when two controls declare the same name.

Once the above files are in place, the prompt censor control can be initialized and exercised:

```python
from steerability.algorithms.input_control.prompt_censor.control import PromptCensor
from steerability.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

prompt_censor = PromptCensor(
    blocked_words=["dangerous", "harmful"],
    replacement="",
)

pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[prompt_censor],
    device_map="auto",
)
pipeline.steer()

# `generate` accepts a positional string (or list[str]) for text, or `messages=` / `input_ids=` for chat / tokens.
print(pipeline.generate("How to make a dangerous chemical reaction?", max_new_tokens=200))

# runtime override example
print(
    pipeline.generate(
        "How do I build a bomb?",
        runtime_kwargs={"blocked_words": ["bomb"], "replacement": "chemistry experiment"},
        max_new_tokens=200,
    )
)
```

## When to override `adapt_messages` instead

If your method modifies chat structure (sets/replaces a system prompt, inserts example turns, etc.), override
`adapt_messages`. The pipeline calls `adapt_messages` before chat-template tokenization when the caller passes
chat-shaped input. When it returns a non-None result, that control's token-level `adapt` is not called for that
generation, and each control is therefore applied exactly once.

```python
def adapt_messages(self, messages, runtime_kwargs=None):
    # `messages` is a batch of chats: list[list[{"role": ..., "content": ...}]]
    out = []
    for chat in messages:
        chat = list(chat)
        chat.insert(0, {"role": "system", "content": "Be concise."})
        out.append(chat)
    return out

def adapt(self, input_ids, runtime_kwargs=None):
    # token-level fallback for raw text/tensor input (chat input is handled by adapt_messages)
    return input_ids
```

If users call `pipeline.generate(input_ids=input_ids_tensor, ...)` (or pass text) instead of chat input,
`adapt_messages` is skipped and a warning is emitted. The control is then applied through `adapt` (the token-level
fallback). Because the two entry points serve different input modalities, a control may implement both without being
applied twice. Token-level methods can supply a best-effort fallback in `adapt`. See
[`SystemPromptFormatter.apply_to_ids`](../../reference/algorithms/input_control/common.md) for one approach.

## Reusable building blocks

The `steerability.algorithms.input_control.common` package collects components shared across input controls:

- `memory/`: `TextMemory` (named JSON-serializable text slots) and `PoolMemory[T]` (typed pool with parallel
  metadata). Place persistent state on `self.memory`, which the framework treats as opaque but recognizes for
  serialization.
- `formatters/`: token-level and message-level renderers for memory content (`SystemPromptFormatter`,
  `FewShotBlockFormatter`, `ChatTemplateSlotFormatter`, `PrependTextFormatter`).
- `scorers/`, `proposers/`, `selectors/`: small abstractions used by `PRewrite`, `CPO`, and `GEPA`. Reuse
  them when applicable. Method-specific procedures should be placed in your method's own `utils/` directory.
- `pareto.py` / `budget.py`: `ParetoFrontier` (Pareto-frontier sampling, used for GEPA parent selection) and
  `RolloutBudget` (rollout-budget accounting).
