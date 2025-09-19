# Adding your own metric

Evaluation metrics are intended to be consumed by use cases. This guide illustrates how to add new metrics. Broadly,
metrics are of two categories:

- Generic metrics: metrics that can be called from any use case.
- Custom metrics:  metrics that are intended to be called from a specific use case (e.g., question answering)

Depending on the metric category, structure your files in `aisteer360/evaluation/metrics` as follows:
```
aisteer360/
└── evaluation/
    └── metrics/
        ├── custom/
        │   └── <my_use_case>/
        │       └── <custom_metric_name>.py
        └── generic/
            └── <generic_metric_name>.py
```

Implementation of a new metric is the same regardless of the metric's category. Both generic and custom metrics can be
one of two types:

- standard: subclasses `Metric` from `aisteer360.evaluation.metrics.base`
- LLM-as-a-judge: subclasses `LLMJudgeMetric` from `aisteer360.evaluation.metrics.base_judge`

All metrics compute scores using at minimum a `response`, with an optional field `prompt`. Any other necessary arguments
can be passed into the metric's `compute` method via `kwargs`.


## Implementing a standard metric

Standard metrics are any metric that require completely custom `compute` logic. Any unstructured computation can be
implemented as a function of `responses`, `prompts`, and `kwargs`. Any necessary parameter initialization should be
added to the metric’s constructor (`__init__`).

Below is an example implementation of a `DistinctN` metric (for computing unigrams, bigrams, etc.).

```python
from itertools import islice
from typing import Any

from aisteer360.evaluation.metrics.base import Metric


class DistinctN(Metric):
    """Corpus-level Distinct-n (Li et al., 2015).

    Distinct-n = (# unique n-grams) / (# total n-grams)

    Args:
        n (int, optional): Size of the n-gram.

    Li, J., Galley, M., Brockett, C., Gao, J. and Dolan, B., 2015.
    A diversity-promoting objective function for neural conversation models.
    arXiv preprint arXiv:1510.03055.
    """

    def __init__(self, n: int = 2):
        super().__init__()
        self.n = n

    def _ngrams(self, tokens: list[str]):
        return zip(*(islice(tokens, i, None) for i in range(self.n)))

    def compute(
        self,
        responses: list[str],
        prompts: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        total_ngrams = 0
        unique_ngrams: set[tuple[str, ...]] = set()

        for response in responses:
            response = response.lower()
            tokens = response.split()
            grams = list(self._ngrams(tokens))
            total_ngrams += len(grams)
            unique_ngrams.update(grams)

        score = len(unique_ngrams) / total_ngrams if total_ngrams else 0.0
        return {
            f"distinct_{self.n}": score
        }
```

The above metric is called as follows:

```python
from aisteer360.evaluation.metrics.generic.distinct_n import DistinctN

responses = [
    "I love exploring new places.",
    "I love exploring new places.",
    "Traveling is my passion."
]

unigram = DistinctN(n=1)

unigrams = unigram.compute(responses=responses)
```


## Implementing an LLM-as-a-judge metric


To facilitate evaluation of more complex quantities, we have implemented a base class for LLM-as-a-judge metrics
(`LLMJudgeMetric`) that extends the `Metric` class.

Implementation of LLM-as-a-judge metrics only requires specifying the prompt template `_PROMPT` and the scale interval
`scale`. The prompt template must contain a placeholder to `response` *and* the lower and upper bounds of the scale
interval (`lower_bound` and `upper_bound`). Optionally, the template can contain a placeholder for `prompt`. For
instance, the `Factuality` metric requires the `response` (the model's answer) and the `prompt` (the question).

```python
from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric


_PROMPT = """\
You are a careful fact-checker.

Considering only verifiable facts, rate the response’s factual accuracy with respect to the prompt on a scale from
{lower_bound} (completely incorrect) to {upper_bound} (fully correct).

PROMPT:
{prompt}

RESPONSE:
{response}

What is your score?
"""


class Factuality(LLMJudgeMetric):
    """
    Judge factual correctness of an answer to a question.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            prompt_template=_PROMPT,
            scale=(1, 5),
            **kwargs,
        )

```

LLM-as-a-judge metrics are initialized by specifying the judge model (via `model_or_id`) and any generation parameters
(via `gen_kwargs`). Note that we can run the judge multiple times on a given input as dictated by
`num_return_sequences`.

```python
from aisteer360.evaluation.metrics.generic.relevance import Relevance

# metric parameters
judge_model = "meta-llama/Llama-3.2-3B-Instruct"
judge_gen_kwargs = {
    "temperature": 0.8,
    "num_return_sequences": 3,
    "do_sample": True
}

# initialize metric
answer_relevance = Relevance(
    model_or_id=judge_model,
    gen_kwargs=judge_gen_kwargs
)

# run the metric
questions = ["What is the capital of Ireland?"]
answers = ["Dublin."]
factuality = answer_relevance(responses=answers, prompts=questions)
```

To call metrics, please see the tutorial on [adding your own use case](add_new_use_case.md).
