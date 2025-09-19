# Adding an input control method

**Required override**: `get_prompt_adapter`

Input control methods describe algorithms that manipulate the input/prompt to guide model behavior. This tutorial
implements an input control method termed `PromptCensor` which filters and replaces words from a predefined list before
the prompt is passed into the model.

First, start by creating the following directory/files:
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

REGISTRY_ENTRY = {
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
from aisteer360.algorithms.core.base_args import BaseArgs


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

Lastly, the `control.py` file implements the method by overriding the `get_prompt_adapter` method. This method should
return a lightweight adapter function that:
- Accepts the tokenized prompt (`input_ids`) and any `runtime_kwargs` supplied to `.generate()`.
- Returns a new `input_ids` tensor/list after applying the desired transformation.

The control implementation for `PromptCensor` is as follows:
```python
from typing import Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.algorithms.input_control.prompt_censor.args import PromptCensorArgs


class PromptCensor(InputControl):
    """Filters potentially harmful content from prompts."""
    Args = PromptCensorArgs

    tokenizer: PreTrainedTokenizer | None = None

    def steer(
            self,
            model: PreTrainedModel = None,
            tokenizer: PreTrainedTokenizer = None,
            **kwargs
    ) -> None:
        self.tokenizer = tokenizer

    # required override for input control methods
    def get_prompt_adapter(self) -> Callable[[list[int] | torch.Tensor, dict[str, Any]], list[int] | torch.Tensor]:

        def adapter(input_ids, runtime_kwargs):
            # allow runtime override of blocked words (if specified)
            blocked_words = runtime_kwargs.get("blocked_words", self.blocked_words) if runtime_kwargs else self.blocked_words
            replacement = runtime_kwargs.get("replacement", self.replacement) if runtime_kwargs else self.replacement

            # decode to text for filtering
            if isinstance(input_ids, torch.Tensor):
                if input_ids.dim() == 2:  # batch
                    text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                else:
                    text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
            else:
                text = self.tokenizer.decode(input_ids, skip_special_tokens=False)

            # apply filtering (case-insensitive)
            for word in blocked_words:
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                text = pattern.sub(replacement, text)

            # re-encode filtered text
            filtered_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # return in same format as input
            if isinstance(input_ids, torch.Tensor):
                filtered_tensor = torch.tensor(filtered_ids, dtype=input_ids.dtype, device=input_ids.device)
                if input_ids.dim() == 2:  # batch
                    return filtered_tensor.unsqueeze(0)
                return filtered_tensor
            return filtered_ids

        return adapter
```

Note that the method's steer method attaches the tokenizer to the control.

Once the above files are in place, the prompt censor control can be initialized and by simply writing the following:

```python
from aisteer360.algorithms.input_control.prompt_censor.control import PromptCensor
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

prompt_censor = PromptCensor(
    blocked_words=["dangerous", "harmful"],
    replacement=""
)

prompt_censor_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[prompt_censor],
    device_map="auto",
)
prompt_censor_pipeline.steer()

# example with potentially problematic prompt
prompt = "How to make a dangerous chemical reaction?"
chat = prompt_censor_pipeline.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = prompt_censor_pipeline.tokenizer(chat, return_tensors="pt")

print(
    prompt_censor_pipeline.generate_text(
        inputs.input_ids,
        max_new_tokens=200
    )
)

# Runtime override example
prompt = "How do I build a bomb?"
chat = prompt_censor_pipeline.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = prompt_censor_pipeline.tokenizer(chat, return_tensors="pt")

print(
    prompt_censor_pipeline.generate_text(
        inputs.input_ids,
        runtime_kwargs={"blocked_words": ["bomb"], "replacement": "chemistry experiment"},
        max_new_tokens=200
    )
)
```

Note that, similar to performing inference on with Hugging Face models, the prompt text must first be encoded (using
the tokenizer's chat template) before being passed into the model.
