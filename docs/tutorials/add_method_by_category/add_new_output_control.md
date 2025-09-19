# Adding an output control method

**Required override**: `generate`

Output control methods constrain or transform what leaves the decoder. In this tutorial we implement `KeywordReranker`,
an output control that:
- Generates multiple candidates by asking the base model for N continuations.
- Scores each candidate by counting occurrences of target keywords.
- Returns the best candidate (the one whose text contains the most keywords).

The registry entry is given by:
```python
from .control import KeywordReranker
from .args import KeywordRerankerArgs

REGISTRY_ENTRY = {
    "category": "output_control",
    "name": "keyword_reranker",
    "control": KeywordReranker,
    "args": KeywordRerankerArgs,
}
```

Next, the args dataclass defines three parameters for the method: `num_candidates` and `case_insensitive`. The target
keywords are passed in at inference time since they are tied to the specific prompt that is passed in to the model.


```python
from dataclasses import dataclass, field
from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class KeywordRerankerArgs(BaseArgs):
    num_candidates: int = field(
        default=5,
        metadata={"help": "How many beams / candidates to generate before reranking."},
    )
    case_insensitive: bool = field(
        default=True,
        metadata={"help": "Match keywords ignoring case."},
    )

    # validation
    def __post_init__(self):
        if self.num_candidates < 1:
            raise ValueError("`num_candidates` must be >= 1.")
```

Lastly, the control is implemented as follows:

```python
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.output_control.base import OutputControl
from aisteer360.algorithms.output_control.keyword_reranker.args import KeywordRerankerArgs


class KeywordReranker(OutputControl):
    """ Generates N continuations, keeps the one that mentions the most target keywords. """
    Args = KeywordRerankerArgs

    # class attributes (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    base_generate = None

    def steer(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer | None = None,
            **__,
    ) -> PreTrainedModel:
        self.model = model
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.base_generate = model.generate
        return model

    # required override for output control methods
    def generate(
            self,
            input_ids: torch.Tensor,
            runtime_kwargs: dict[str, Any] | None,
            model: PreTrainedModel,
            **gen_kwargs,
    ) -> torch.Tensor:
        """Generates multiple candidates and selects the one with the most keyword matches.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch size must be 1).
            runtime_kwargs (dict[str, Any] | None): Additional runtime configuration.
            model (PreTrainedModel): The language model used for generation.
            **gen_kwargs: Additional generation arguments.

        Returns:
            torch.Tensor: The selected continuation.
        """
        runtime_kwargs = runtime_kwargs or {}

        # get keywords from runtime_kwargs
        keywords = runtime_kwargs.get("keywords", [])
        if not keywords:
            raise ValueError("KeywordReranker requires 'keywords' in runtime_kwargs")

        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise NotImplementedError("KeywordReranker currently handles batch size 1.")

        # ensure we produce multiple candidates
        gen_kwargs.setdefault("num_beams", self.num_candidates)
        gen_kwargs.setdefault("num_return_sequences", self.num_candidates)

        # generate candidates
        candidates = self.base_generate(input_ids=input_ids, **gen_kwargs)

        # decode to text
        continuations: list[str] = self.tokenizer.batch_decode(candidates[:, input_ids.size(1):], skip_special_tokens=True)

        # simple keyword score
        keyset = [k.lower() if self.case_insensitive else k for k in keywords]

        def score(txt: str) -> int:
            txt_cmp = txt.lower() if self.case_insensitive else txt
            return sum(kw in txt_cmp for kw in keyset)

        scores = [score(t) for t in continuations]
        best_idx = int(torch.tensor(scores).argmax())

        return candidates[best_idx].unsqueeze(0)
```

The control can then be run as follows:

```python
from aisteer360.algorithms.output_control.keyword_reranker.control import KeywordReranker
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

keyword_reranker = KeywordReranker(num_candidates=4)

keyword_reranker_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[keyword_reranker],
    device_map="auto",
)

keyword_reranker_pipeline.steer()

# example prompt
prompt = "Explain linear algebra in two sentences."
chat = keyword_reranker_pipeline.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = keyword_reranker_pipeline.tokenizer(chat, return_tensors="pt")

output = keyword_reranker_pipeline.generate_text(
    inputs.input_ids,
    runtime_kwargs={"keywords": ["matrix", "vector"]},
    max_new_tokens=50,
    temperature=0.7
)
print(output)

# different keywords can be passed in at inference time (without resteering)
output = keyword_reranker_pipeline.generate_text(
    inputs.input_ids,
    runtime_kwargs={"keywords": ["eigenvalue", "determinant"], "case_insensitive": False},
    max_new_tokens=50,
    temperature=0.7
)
print(output)
```
