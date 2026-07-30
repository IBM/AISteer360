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
[`few_shot`](../examples/notebooks/algorithms/few_shot.ipynb) and [`dpo`](../examples/notebooks/algorithms/trl.ipynb), as follows:

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
    A pipeline may contain **any number of controls in every category**, each applied in list order. When multiple
    state controls are supplied, list order is the single, well-defined composition surface: list order = `steer()`
    order = hook registration order = execution order for hooks on the same module. PyTorch forward hooks chain (a
    later hook receives the previous hook's returned output; pre-hooks chain likewise on inputs), so a combination like
    "control A then control B at layer 12" is well-defined, and non-commuting pairs (e.g. ablation ∘ addition vs.
    addition ∘ ablation) are order-sensitive by design. An `ActivationAdapter` is the natural single-behavior atom
    here, i.e., steering with N behaviors is N adapters in the `controls` list.

!!! note "Input controls: two-phase chaining"
    Multiple input controls chain in list order across two phases. On chat input, every control's `adapt_messages`
    runs in list order over the message batch (each non-None return feeds the next control); the result is templated
    and tokenized once, then every control whose `adapt_messages` returned None runs its token-level `adapt` in list
    order over the token stream. On text/tensor input there is no message phase; every control's `adapt` runs in list
    order. Each control is applied exactly once per generation: at message level if its `adapt_messages` returned a
    non-None result for that call, else at token level. List order is authoritative within each phase, but the message
    phase structurally precedes the token phase: with `[TokenOnlyControl, MessageLevelControl]` on chat input, the
    message-level control's effect lands first even though it is listed second (tokens do not exist before
    templating). Recommended ordering: place semantic rewriters (`PRewrite`, `CPO`, `GEPA`) before surface formatting
    (`FewShot`), since a rewriter trained on bare instructions degrades on exemplar-prepended input.

!!! note "Structural controls: model threading"
    Multiple structural controls thread the model through `steer()` in list order: each control receives the previous
    control's returned model (and the possibly mutated tokenizer). Nothing implicit happens between stages, i.e., no
    adapter merging and no embedding-resize reconciliation; stage compatibility (a PEFT-wrapped model into a second
    trainer, resized embeddings, and the like) is the user's responsibility. Note that the TRL wrapper controls load
    their own base model when `base_model_name_or_path` is set in their args, silently discarding the threaded
    upstream model, so downstream structural controls should leave `base_model_name_or_path` unset to receive the
    threaded model.

!!! note "Output controls: step-level controls compose, the decode loop does not"
    Output controls participate through two mechanisms. Most are step-level controls supplying logits processors and/or
    stopping criteria, which the pipeline gathers in `controls`-list order, then appends any per-call
    `logits_processor` / `stopping_criteria` supplied in `generate()`, into one authoritative stack of each kind. The
    decode loop itself is exclusive: it is owned by at most one `DecodingDriver`, and supplying two enabled drivers
    raises at construction (two decoding procedures cannot both control generation). With no driver present, the loop
    defaults to the model's own `generate`, so a pipeline with no output controls decodes exactly as the base model
    does. Because the loop is a single owner while step-level controls compose, a step-level control (e.g. `RAD`)
    applies inside every rollout a driver issues (e.g. `DeAL`'s lookahead), a composition rather than a conflict.
    Step-level controls' logits processors also apply during `compute_logprobs`, so scoring reflects the steered
    next-token distribution; a control sets `include_in_scoring=False` to opt out (e.g. when the per-position cost is
    prohibitive).

## Steering the pipeline

Before a steering pipeline can be used for inference, all of the controls in the pipeline must be prepared and applied
to the model (e.g, training logic in a `DPO` control, or subspace learning in the `SASA` control). This step is referred
to as the *steer* step and is executed via:

```python
pipeline.steer()
```

Calling the `steer()` method on a pipeline instance invokes the steering logic for every control in the pipeline. Methods are 
steered independently; the effect of composing steered/trained controls is one of the main functionalities provided by the 
toolkit. Note that the `steer()` step can be resource-heavy, e.g., especially if any of the controls in the pipeline require any training.
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
