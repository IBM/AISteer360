# Steering Pipelines

![Steering pipeline](../assets/pipeline_darkmode.png#gh-dark-mode-only)
![Steering pipeline](../assets/pipeline_lightmode.png#gh-light-mode-only)
<p align="center">
  <em>The structure of a steering pipeline.</em>
</p>

Steering pipelines allow for the composition of multiple controls (across the [four control types](controls.md)) into a
single steering operation on a model. This allows for individual controls to be easily *mixed* to form novel steering
interventions.

Steering pipelines are created using the `SteeringPipeline` class. The most common pattern is to specify a Hugging Face
model name via `base_model_or_path` along with instantiated controls, e.g.,
[`few_shot`](../notebooks/controls/few_shot.ipynb) and [`dpo`](../notebooks/controls/trl_wrapper.ipynb), as follows:

```python
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

pipeline = SteeringPipeline(
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    controls=[few_shot, dpo]
)
```
The above chains the two controls into a single operation on the model.

!!! note
    Some structural controls (e.g., model merging methods) produce a model as output rather than modifying/tuning an
    existing model. In these cases, the steering pipeline must be initialized with the argument `lazy_init=True` ,
    rather than with the `model_name_or_path` argument. This defers loading of the base model until the steer step.

!!! note
    We currently impose a constraint that the pipeline consists of at most one control per category. Extending
    steering pipelines to contain more than one control per category is under active development.

## Steering the pipeline

Before a steering pipeline can be used for inference, all of the controls in the pipeline must be prepared and applied
to the model (e.g, training logic in a `DPO` control, or subspace learning in the `SASA` control). This step is referred
to as the *steer* step and is executed via:

```python
pipeline.steer()
```

Calling the `steer()` method on a pipeline instance invokes the steering logic for each control in the pipeline in a
*bottom-up* fashion (structural -> state -> input -> output). This ensures proper layering, e.g., we want to make sure
that activation (state) steering is done with respect to any updated structure of the model.

The `steer()` step can be resource-heavy, e.g., especially if any of the controls in the pipeline require any training.
Steering must be called exactly once before using the pipeline for inference.


## Running inference on the pipeline

Once the pipeline has been steered, inference can be run using the `generate()` method. AISteer360 has been built to be
tightly integrated with Hugging Face and thus running inference on a steering pipeline is operationally similar to
running inference on a Hugging Face model. As with Hugging Face models, prompts must first be encoded via the pipeline's
tokenizer. It is also recommended to apply the tokenizer's chat template if available:

```python
tokenizer = pipeline.tokenizer
chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(chat, return_tensors="pt")
```

Inference can then be run as usual, for instance:
```python
gen_params = {
    "max_new_tokens": 20,
    "temperature": 0.6,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

steered_output_ids = pipeline.generate(
    input_ids=inputs.input_ids,
    **gen_params,
)
```

Note that steering pipelines accept any of the generation parameters available in [Hugging Face's `GenerationConfig` class](https://huggingface.co/docs/transformers/en/main_classes/text_generation).
This includes any of the generation strategies for [custom decoding](https://huggingface.co/docs/transformers/en/generation_strategies).
