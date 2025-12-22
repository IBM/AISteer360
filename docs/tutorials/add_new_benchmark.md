# Adding your own benchmark

Benchmarks facilitate comparison of steering pipelines on a given use case. This tutorial describes how to build a
benchmark for two cases: 1) A simple benchmark for the `CommonsenseMCQA` use case constructed in the
[tutorial for adding your own use case](add_new_use_case.md), and 2) A more complex benchmark for the
`InstructionFollowing` use case that contains steering methods which require specification of inference-time arguments
(via `runtime_overrides`).

Note that both of the described benchmarks use *fixed* controls, in the sense that all parameters are set upon
initialization and remain fixed for the duration of the benchmark. Oftentimes we want to investigate how behavior
changes as we sweep some subset of a control's variables over a range, i.e., *variable* controls. We describe how to
construct a benchmark with such controls at the end of this tutorial.

## Simple benchmark

The first step in building a benchmark is to initialize the use case of interest. For illustration purposes, we base our
benchmark on the evaluation dataset (`evaluation_qa.jsonl`) with elements of the form:

```python
{
    "id": "762d85c8-c891-46ac-907b-8f335d0d3be5",
    "question": "Sam ran out of clipboards. Where might he got for more?",
    "answer": "office supply store",
    "choices": ["windows 95", "school", "ammunition shop", "office supply store", "desk"]
}
```

Each question in the above evaluation data contains a unique `id`, a `question`, the ground-truth `answer`, and the
available `choices` presented to the model. As described in the previous tutorial, the `CommonsenseMCQA` use case is
instantiated by passing in the evaluation dataset, the metrics of interest, `MCQAAccuracy` and `MCQAPositionalBias`,
and a use case specific argument (`num_shuffling_runs`):

```python
from aisteer360.evaluation.use_cases.commonsense_mcqa.use_case import CommonsenseMCQA
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_accuracy import MCQAAccuracy
from aisteer360.evaluation.metrics.custom.commonsense_mcqa.mcqa_positional_bias import MCQAPositionalBias

commonsense_mcqa = CommonsenseMCQA(
    evaluation_data="data/evaluation_qa.jsonl",
    evaluation_metrics=[
        MCQAAccuracy(),
        MCQAPositionalBias(),
    ],
    num_shuffling_runs=20,
    num_samples=500  # optional
)
```
To decrease the execution time of the benchmark run, we additionally set `num_samples=500` which serves to limit the
evaluation to (the first) `500` elements of the evaluation dataset.

In this benchmark, we compare the model's base performance with two steering controls:
[`FewShot`](../notebooks/controls/few_shot.ipynb) and [`DPO (with LoRA)`](../notebooks/controls/trl_wrapper.ipynb). Both
of these controls require specification of steering data, i.e., the source data that a control uses to steer the base
model. Common steering data is used by both controls, forming the example pools for `FewShot` and the training dataset
for `DPO`. The steering dataset takes the following form:
```python
{
    "id": "11a7992e-7825-4263-8a22-a1fed72b5ecb",
    "question": "Where would you fire a projectile ball at a clown's mouth?",
    "answer_chosen": "arcade",
    "answer_rejected": "motion"
}
```
The steering dataset is loaded as follows:
```python
import json
steering_data_path = "data/steer_qa.jsonl"
with open(steering_data_path, "r") as f:
    steering_data = [json.loads(line) for line in f]
```
The steering data is defined as triples (`question`, `answer_chosen`, `answer_rejected`) where `answer_chosen` is the
correct answer and `answer_rejected` is one of the incorrect choices (sampled uniformly at random). The pairs
(`question`, `answer_chosen`) and (`question`, `answer_rejected`) are used to form the positive and negative example
pools, respectively, for `FewShot` as follows:

```python
positive_pool = []
negative_pool = []
for _, row in steering_data.iterrows():
    positive_pool.append({
        "question": row["question"],
        "answer": row["answer_chosen"]
    })
    negative_pool.append({
        "question": row["question"],
        "answer": row["answer_rejected"]
    })
```

The `DPO` control uses the triples as preference data. For DPO, the dataset must be injected into the control as a
Hugging Face `Dataset` object.

```python
from datasets import Dataset

train_examples = []
for row in steering_data:
    train_examples.append({
        "prompt": row['question'],
        "chosen": row['answer_chosen'],
        "rejected": row['answer_rejected']
    })
train_ds = Dataset.from_list(train_examples)
```

The controls can now be instantiated as follows:
```python
from aisteer360.algorithms.input_control.few_shot.control import FewShot

few_shot = FewShot(
    selector_name="random",
    positive_example_pool=positive_pool,
    negative_example_pool=negative_pool,
    k_positive=4,
    k_negative=4
)
```
and
```python
from peft import PeftType
from aisteer360.algorithms.structural_control.wrappers.trl.dpotrainer.control import DPO

dpo_lora = DPO(
    train_dataset=train_ds,

    # DPO / TRL config
    output_dir="trl_models/Qwen2.5-0.5B-DPO-Lora-Steer",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    learning_rate=1e-6,
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
    max_prompt_length=512,
    disable_dropout=True,
    logging_steps=100,
    save_strategy="no",
    report_to="none",
    seed=123,

    # LoRA config
    use_peft=True,
    peft_type=PeftType.LORA,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    adapter_name="dpo",
    merge_lora_after_train=False,
)
```

Now that the controls have been instantiated, we are now ready to construct the benchmark. Instantiation of a benchmark
requires specification of the following arguments:

- `use_case` (`UseCase`): The instantiated use case object.
- `base_model_name_or_path`: The base model to steer (as listed on Hugging Face).
- `steering_pipelines` (`dict[str, Any]`): The steering pipelines/methods that we want to compare in the benchmark.

A benchmark can also optionally accept

- `runtime_overrides`: A dictionary that indicates which how the evaluation data map to control variables (not used in this example).
- `hf_model_kwargs`: load-time options for configuration of the construction of the model.
- `gen_kwargs`: generation-time options for configuration of the behavior of the model.
- `device_map`: indicates how model layers are assigned to devices.

The benchmark for `CommonsenseMCQA` can now be constructed as follows:
```python
from aisteer360.evaluation.benchmark import Benchmark

benchmark = Benchmark(
    use_case=commonsense_mcqa,
    base_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    steering_pipelines={
        "baseline": [],  # no steering
        "few_shot": [few_shot],
        "dpo_lora": [dpo_lora],
    },
    gen_kwargs={
        "max_new_tokens": 300,
        "do_sample": True,
        "temperature": 0.7,
    },
    device_map="auto"
)
```
The benchmark is executed by calling the `run()` method, which generates the profiles:
```python
profiles = benchmark.run()
benchmark.export(profiles, save_dir="./profiles/")
```
A complete working example of the `CommonsenseMCQA` benchmark can be found in the
[example notebook](../notebooks/benchmarks/commonsense_mcqa/commonsense_mcqa.ipynb).


## Benchmark with inference-time arguments

The benchmark for the `CommonsenseMCQA` use case compares `FewShot` and `DPO` controls, neither of which require
additional inference-time arguments. In some cases, controls in a pipeline rely on information that is only available at
inference time, e.g., increasing attention weights on specific prompt tokens corresponding to instructions as in
[PASTA](../notebooks/controls/pasta.ipynb).

The `Benchmark` class allows these arguments to be passed in to each control via the specification of
`runtime_overrides`. We briefly illustrate how this is done for the `InstructionFollowing` use case.

As before, we initialize the use case and the controls that we wish to use. The `InstructionFollowing` use case is
initialized as follows:
```python
instruction_following = InstructionFollowing(
    evaluation_data=evaluation_data,
    evaluation_metrics=[StrictInstruction()],
    num_samples=50
)
```

The `PASTA` control is instantiated via:
```python
from aisteer360.algorithms.state_control.pasta.control import PASTA
pasta = PASTA(
    head_config=[8,9],
    alpha=0.01,
    scale_position="exclude",
)
```
The `ThinkingIntervention` control requires specification of an intervention function:
```python
def instruction_following_intervention(prompt: str, params: dict) -> str:
    intervention = (
        "I will first think using the  and  tags and then provide the final answer after that.\n"
        " I should ensure that the answer follows these instructions. "
    )
    modified_instr = [instr.replace("-", "") for instr in params["instructions"]]
    intervention += " and".join(modified_instr)
    return prompt + intervention + "\n"
```
which is then used when instantiating the control:
```python
from aisteer360.algorithms.output_control.thinking_intervention.control import ThinkingIntervention

thinking_intervention = ThinkingIntervention(
    intervention=instruction_following_intervention
)
```
Note that both `PASTA` and `ThinkingIntervention` require the specific instructions within a given prompt to be passed
to the control. This is facilitated through the `runtime_overrides` argument in the `Benchmark` class, i.e., a
dictionary of dictionaries each which is keyed by the control name and take values mapping the control's variable, e.g.,
`substrings` in `PASTA`, to the relevant column of the evaluation dataset, e.g., `instructions`. The full benchmark call
is as follows:
```python
benchmark = Benchmark(
    use_case=instruction_following,
    base_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    steering_pipelines={
        "baseline": [], # no steering
        "pasta": [pasta],
        "thinking_intervention": [thinking_intervention]
    },
    runtime_overrides={
        "PASTA": {"substrings": "instructions"},
        "ThinkingIntervention": {"params": {"instructions": "instructions"}},
    },
    gen_kwargs={
        "max_new_tokens": 100,
        "do_sample": False,
        "output_attentions": True,  # mandatory for PASTA
    },
)
```
The benchmark can then be run as usual to generate the profiles. We direct the reader to the
[notebook](../notebooks/benchmarks/instruction_following/instruction_following.ipynb) for the full implementation.

## Benchmark with variable controls

To study model behavior as control parameters change, the toolkit uses the `ControlSpec` class. Static parameters are
specified in the `params` dict, whereas variable parameters are specified in the `vars` dict.

```python
few_shot_spec = ControlSpec(
    control_cls=FewShot,
    params={
        "selector_name": "random",
        "positive_example_pool": positive_pool,
        "negative_example_pool": negative_pool,
        "k_negative": 0
    },
    vars={
        "k_positive": [5, 10, 20],
    },
    name="few_shot",
)
```

fixed pools, zero negative examples

Sweep positive examples over 5, 10, 20

The above can then be passed directly into the benchmark class and

Additionally, num_trials


```python
bench = Benchmark(
    use_case=commonsense_mcqa,
    base_model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    steering_pipelines={
        "baseline": [],
        "few_shot": [few_shot_spec],
        "dpo_lora": [dpo_lora_spec],
    },
    gen_kwargs={
        "max_new_tokens": 300,
        "do_sample": True,
        "temperature": 0.7
    },
    device_map="auto",
    num_trials=5
)

profiles = bench.run()
```

Both types of benchmarks can accpted control specs
