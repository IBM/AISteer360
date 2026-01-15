# Adding your own use case

Use cases define tasks for a model and specify how performance on that task (via the model's generations) is
measured. A use case instance is intended to be consumed by a benchmark. Please see the
[tutorial for adding your own benchmark](add_new_benchmark.md) for instructions on how to run a use case.

For the purposes of this tutorial, we will focus on a simple multiple-choice QA task, which we term `CommonsenseMCQA`,
based on the [CommonsenseQA dataset](https://huggingface.co/datasets/tau/commonsense_qa).

## Setup

The only required file to create a use case is `use_case.py`. This file must be placed in a new directory
`<custom_use_case>`, of your choosing, in `aisteer360/evaluation/use_cases`:
```
aisteer360/
└── evaluation/
    └── use_cases/
        └── <custom_use_case>/
            └── use_case.py
```

The `CommonsenseMCQA` use case is located at`commonsense_mcqa/use_case.py`. Every use case is instantiated by providing
`evaluation_data`, the data that the model uses to produce generations, and `evaluation_metrics`, the functions to
evaluate the model's behavior. Any number of additional keyword arguments specific to the use case (e.g.,
`num_shuffling_runs` for `CommonsenseMCQA`) can also be passed in to the class. For instance,

```python
from aisteer360.evaluation.use_cases.commonsense_mcqa.use_case import CommonsenseMCQA
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_accuracy import MCQAAccuracy
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_positional_bias import MCQAPositionalBias

commonsense_mcqa = CommonsenseMCQA(
    evaluation_data_path="./data/evaluation_qa.jsonl",
    evaluation_metrics=[
        MCQAAccuracy(),
        MCQAPositionalBias()
    ],
    num_shuffling_runs=20
)
```

Evaluation data should contain any information that is relevant for evaluating the model's performance. For our example
task, this data (stored as a `jsonl` file) contains the following information:

```python
{
    "id": "033b86ec-e7c1-40ac-8c9e-27ebfba41faf",
    "question": "Where would someone keep a grandfather clock?",
    "answer": "house",
    "choices": ["desk", "exhibition hall", "own bedroom", "house", "office building"]
}
```

We've implemented two custom metrics for our use case: `MCQAAccuracy` for evaluating the accuracy statistics of choices
with respect to the ground truth answers, and `MCQAPositionalBias` for measuring how much the model is biased toward
choices in a given position. This tutorial will not go into depth about these metrics; please see their implementations
at `aisteer360/evaluation/metrics/custom/commonsense_mcqa` for details. For details on contributing any new metrics
(either generic metrics or those custom to a use case), please see the
[tutorial on adding your own metric](./add_new_metric.md).


## Defining the use case class

Each use case subclasses the base `UseCase` class (`aisteer/evaluation/use_cases/base.py`), which contains all necessary
initialization logic. Please **do not** write an `__init__` for your custom use case. Any arguments specific to the use
case, like `num_shuffling_runs` above, are automatically saved as class attributes by the constructor of the base
`UseCase` class.  Optionally, you can add a placeholder (type hint), e.g., `num_shuffling_runs: int`, at the class level
to inform your IDE that your added argument(s) will exist at runtime. We additionally advise that contributors write
validation logic for their evaluation data (via `validate_evaluation_data`) based on the required columns
(`_EVALUATION_REQ_KEYS`). This is helpful for catching any errors early.

For our example use case:

```python
from aisteer360.evaluation.use_cases.base import UseCase

_EVALUATION_REQ_KEYS = [
    "id",
    "question",
    "answer",
    "choices"
]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CommonsenseMCQA(UseCase):
    """
    Commonsense multiple-choice question answering use case.

    """
    num_shuffling_runs: int

    def validate_evaluation_data(self, evaluation_data: dict[str, Any]):
        if "id" not in evaluation_data.keys():
            raise ValueError("The evaluation data must include an 'id' key")

        missing_keys = [col for col in _EVALUATION_REQ_KEYS if col not in evaluation_data.keys()]
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        if any(
            key not in evaluation_data or evaluation_data[key] is None or
            (isinstance(evaluation_data[key], float) and math.isnan(evaluation_data[key]))
            for key in _EVALUATION_REQ_KEYS
        ):
            raise ValueError("Some required fields are missing or null.")
```

!!! note
    We require that your evaluation data contains a column named `id`, serving to assign a unique identifier to each
    datapoint. This is required by the `Benchmark` class to ensure that any `runtime_kwargs` (any arguments that may be
    required by the controls at inference time; see the [tutorial on adding a benchmark](./add_new_benchmark.md) for
    details) are consistently populated.

Any use case class must define two required methods (`generate` and `evaluate`) and an optional method (`export`).
Implementation of these methods is outlined below.


### Generation via `generate`

The `generate` method produces outputs as a function of the evaluation data (accessible via `self.evaluation_data`). The
generate method must return `generations` as a list of dictionaries (i.e., `list[dict[str, Any]]`). Each dictionary must
contain at minimum a `response` key and can optionally contain a `prompt` key. The dictionary should also contain any
number of keyword args that may be necessary for later computation of metric scores. In other words, `generations`
should contain everything that the use case's evaluate method needs to run its evaluation.


The `generate` method for `CommonsenseMCQA` is defined as follows:
```python
def generate(
    self,
    model_or_pipeline,
    tokenizer,
    gen_kwargs: dict | None = None,
    runtime_overrides: dict[tuple[str, str], str] | None = None
) -> list[dict[str, Any]]:

    if not self.evaluation_data:
        print('No evaluation data provided.')
        return []
    gen_kwargs = dict(gen_kwargs or {})

    # form prompt data
    prompt_data = []
    for instance in self.evaluation_data:
        data_id = instance['id']
        question = instance['question']
        answer = instance['answer']
        choices = instance['choices']
        # shuffle order of choices for each shuffling run
        for _ in range(self.num_shuffling_runs):

            lines = ["You will be given a multiple-choice question and asked to select from a set of choices."]
            lines += [f"\nQuestion: {question}\n"]

            # shuffle
            choice_order = list(range(len(choices)))
            random.shuffle(choice_order)
            for i, old_idx in enumerate(choice_order):
                lines.append(f"{_LETTERS[i]}. {choices[old_idx]}")

            lines += ["\nPlease only print the letter corresponding to your choice."]
            lines += ["\nAnswer:"]

            prompt_data.append(
                {
                    "id": data_id,
                    "prompt": "\n".join(lines),
                    "reference_answer": _LETTERS[choice_order.index(choices.index(answer))]
                }
            )

    # batch template/generate/decode
    choices = batch_retry_generate(
        prompt_data=prompt_data,
        model_or_pipeline=model_or_pipeline,
        tokenizer=tokenizer,
        parse_fn=self._parse_letter,
        gen_kwargs=gen_kwargs,
        runtime_overrides=runtime_overrides,
        evaluation_data=self.evaluation_data
    )

    # store
    generations = [
        {
            "response": choice,
            "prompt": prompt_dict["prompt"],
            "question_id": prompt_dict["id"],
            "reference_answer": prompt_dict["reference_answer"],
        }
        for prompt_dict, choice in zip(prompt_data, choices)
    ]

    return generations

@staticmethod
def _parse_letter(response) -> str:
    valid = _LETTERS
    text = re.sub(r"^\s*(assistant|system|user)[:\n ]*", "", response, flags=re.I).strip()
    match = re.search(rf"\b([{valid}])\b", text, flags=re.I)
    return match.group(1).upper() if match else None
```

The `generate` method is designed to be called, via the benchmark class, on either a base (unsteered) model or a
steering pipeline, and thus the "model" object passed into `generate` is referenced via the required argument
`model_or_pipeline`. In addition, the `generate` method requires an associated `tokenizer` and
(optionally) any `gen_kwargs` and `runtime_overrides`. The current `CommonsenseMCQA` use case does not make use of any
`runtime_overrides` (since none of the studied controls in the associated benchmark require inference time arguments);
please see the [instruction following benchmark notebook](../notebooks/benchmarks/instruction_following.ipynb)
for an example of how these overrides are defined and used.

The first step in defining the `generate` method is to construct the prompt data. For the example MCQA task, our goal is
to (robustly) evaluate a model's ability to accurately answer (common sense) multiple choice questions, and thus we
present the same question to the model under various orderings/shufflings of the answers. This is implemented by defining
the prompt data as the question ID, the question (as the `prompt`), and the reference answer, under various shuffles of
the answer order.

Once the prompt data has been prepared for the use case, it then needs to be passed into the model (or steering
pipeline) to generate responses. We strongly advise that contributors make use of the `batch_retry_generate` helper
function to aid in this process. This function implements conversion to a model's chat template, batch encoding, batch
generation, batch decoding, and parsing (via `parse_fn`), and retry logic for a given list of prompts. For the example
use case, we define the parsing function as a custom `parse_letter` method, such that the model's choices can be
reliably extracted from its response (and stored as `choices`).

Lastly, we store each choice under the `response` key along with the prompt, question ID, and reference answer across
all elements of the prompt data.


### Evaluation via `evaluate`

The `evaluate` method defines how to process the model's generations (produced by the `generate` method) via evaluation
metrics. All evaluation metrics that were passed in as the use case's construction are used in the evaluation.

```python
def evaluate(self, generations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:

    eval_data = {
        "responses": [generation["response"] for generation in generations],
        "reference_answers": [generation["reference_answer"] for generation in generations],
        "question_ids": [generation["question_id"] for generation in generations],
    }

    scores = {}
    for metric in self.evaluation_metrics:
        scores[metric.name] = metric(**eval_data)

    return scores
```

A useful pattern for evaluation logic is to first define the necessary quantities across all generations (`eval_data`),
then simply pass these into each metric (via `**eval_data`). Note that for the example use case, the metrics make use of
the question IDs by computing statistics across the shuffled choice order for each question.


### Formatting and exporting via `export`

The `export` method (optional) is useful for storing benchmark evaluations for later plotting or analysis, e.g.,
comparing benchmark results across multiple base models. The `export` method allows the user to specify custom
processing before exporting. In the simplest case, the method can just save the profiles to a `json` file, as is done
in the example use case:

```python
def export(self, profiles: dict[str, Any], save_dir) -> None:

    with open(Path(save_dir) / "profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=4, ensure_ascii=False)
```


---


For a complete example of the `CommonsenseMCQA` use case, please see the implementation located at
`aisteer360/evaluation/use_cases/commonsense_mcqa/use_case.py`. For instructions on how to build an associated benchmark, please
see the [tutorial](./add_new_benchmark.md) and the [notebook](../notebooks/benchmarks/commonsense_mcqa.ipynb).
