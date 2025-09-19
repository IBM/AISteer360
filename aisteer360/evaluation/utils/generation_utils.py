from typing import Any, Callable, Sequence

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

BATCH_SIZE = 64


def apply_chat_template(tokenizer, batch, **kwargs) -> list:
    """
    Constructs template prompts for each batch element based on following cases:
    1. If the model's tokenizer does not support chat_template, return the string as is.
    2. If it supports chat_template:
        Check each instance of the batch to construct chat messages if needed. Cases:
        - Plain string -> convert as 'content' of 'user'
        - List of dictionaries with 'role' and 'content'. Continue
        Then apply chat template and return
    """

    template_prompts = []
    for idx, item in enumerate(batch):
        prompt_obj = item["prompt"]
        if not hasattr(tokenizer, "apply_chat_template"):
            template_prompts.append(str(prompt_obj))
        else:
            if isinstance(prompt_obj, str):
                messages = [{"role": "user", "content": prompt_obj}]
            elif (
                isinstance(prompt_obj, list)
                and prompt_obj
                and isinstance(prompt_obj[0], dict)
            ):
                if not all("role" in m and "content" in m for m in prompt_obj):
                    raise ValueError(
                        f"Prompt {idx}: every chat message dict must have 'role' and 'content' keys."
                    )
                messages = prompt_obj
            else:
                raise TypeError(
                    f"Prompt {idx}: must be str or list of chat messages as list[dict[str, str]] "
                    f"(got {type(prompt_obj).__name__})."
                )

            chat_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            template_prompts.append(chat_str)
    return template_prompts


def chat_generate_model(
    batch: Sequence[dict[str, Any]],
    model,
    tokenizer,
    device: str | torch.device,
    gen_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """
    Batch generate on model with chunking to prevent OOM.
    Each instance of the batch must have a 'prompt' which could be:
    - A plain string , in which case we apply the chat template
    - Dict with the chat template already applied ('role' and 'content' keys)
    """

    prompts = apply_chat_template(tokenizer, batch)
    decoded_outputs = []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]

        try:
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **(gen_kwargs or {}),
                )
            start = inputs["input_ids"].shape[1]

            batch_decoded = tokenizer.batch_decode(outputs[:, start:], skip_special_tokens=True)
            decoded_outputs.extend(batch_decoded)

        except Exception as e:
            print(f"Issue with model generation at batch {i//BATCH_SIZE}: {e}")
            print("Hint - Do not apply chat template to your prompts.")
            raise

    return decoded_outputs


def chat_generate_pipeline(
    batch: Sequence[dict[str, Any]],
    pipeline,
    tokenizer,
    device: str | torch.device,
    gen_kwargs: dict[str, Any] | None = None,
    runtime_overrides: dict[tuple[str, str], str] | None = None,
    evaluation_data: list[dict] | None = None,
) -> list[str]:
    """Generate on pipeline."""

    if runtime_overrides is not None and evaluation_data is None:
        raise ValueError(
            "evaluation_data must be provided when runtime_overrides are supplied."
        )

    # create runtime_kwargs from runtime_overrides and evaluation_data
    runtime_kwargs_flat = {}
    if runtime_overrides:
        runtime_kwargs = {}
        for control in pipeline.controls:
            control_name = control.__class__.__name__
            if control_name in runtime_overrides:
                runtime_kwargs[control_name] = _map_runtime_overrides(
                    overrides=runtime_overrides[control_name],
                    data=evaluation_data,
                )

        # flatten and convert to list of dicts; todo: maintain control names to avoid possible collisions
        for kwargs in runtime_kwargs.values():
            for var, arg in kwargs.items():
                if var in runtime_kwargs_flat:
                    raise ValueError(
                        f"Duplicate runtime_kwargs for: {var!r}; ensure controls have distinct variables."
                    )
                runtime_kwargs_flat[var] = arg
        runtime_kwargs_flat = _runtime_kwargs_to_list(runtime_kwargs_flat)

    # Need to check for empty runtime_kwargs_flat since we may define runtime_overrides
    # for a subset of steering methods, but the current method may not have any overrides.
    # This will result in the above if block being executed with runtime_kwargs_flat = []
    if not runtime_overrides or not runtime_kwargs_flat:
        runtime_kwargs_flat = [None] * len(batch)

    prompts = apply_chat_template(tokenizer, batch)
    decoded_outputs = []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        batch_runtime_kwargs = runtime_kwargs_flat[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch_prompts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # generate
        # todo-future: run batch as dictated by availability of batch processing of controls in pipeline
        generations = []
        with torch.no_grad():
            for j in range(len(batch_prompts)):
                out = pipeline.generate(
                    input_ids=input_ids[j].unsqueeze(0),
                    attention_mask=attention_mask[j].unsqueeze(0),
                    runtime_kwargs=batch_runtime_kwargs[j],
                    **(gen_kwargs or {}),
                )
                generations.append(out)

        tokens = [generation.squeeze(0).tolist() for generation in generations]
        padded = tokenizer.pad({"input_ids": tokens}, padding=True, return_tensors="pt")
        outputs = padded["input_ids"]
        batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_outputs.extend(batch_decoded)

    return decoded_outputs


def batch_retry_generate(
    prompt_data: Sequence[dict[str, Any]],
    model_or_pipeline: PreTrainedModel | SteeringPipeline,
    tokenizer: PreTrainedTokenizerBase,
    gen_kwargs: dict[str, Any] | None = None,
    runtime_overrides: dict[tuple[str, str], str] | None = None,
    evaluation_data: dict | None = None,
    parse_fn: Callable[[str, dict[str, Any]], Any | None] | None = None,
    max_retries: int = 2,
    return_raw: bool = False,
) -> list[Any] | tuple[list[Any], list[str]]:
    """
    Generate chat completions with optional parsing/retry logic.

    Function keeps retrying only the prompts whose outputs fail parse_fn (up to max_retries); return value is a list
    of parsed objects (or None if parsing doesn't succeed).

    If return_raw is True the function instead returns a tuple (parsed_list, raw_list).
    """

    missing_prompt = [i for i, item in enumerate(prompt_data) if "prompt" not in item]
    if missing_prompt:
        raise ValueError(f"'prompt' key missing for {len(missing_prompt)} instances")

    gen_kwargs = dict(gen_kwargs or {})
    is_pipeline = isinstance(model_or_pipeline, SteeringPipeline)

    try:
        device_obj = model_or_pipeline.device
    except Exception as e:
        raise RuntimeError(f"Unable to identify model or pipeline device - {e}")

    if is_pipeline:
        responses = chat_generate_pipeline(
            batch=prompt_data,
            pipeline=model_or_pipeline,
            tokenizer=tokenizer,
            device=device_obj,
            gen_kwargs=gen_kwargs,
            runtime_overrides=runtime_overrides,
            evaluation_data=evaluation_data,
        )
    else:
        responses = chat_generate_model(
            batch=prompt_data,
            model=model_or_pipeline,
            tokenizer=tokenizer,
            device=device_obj,
            gen_kwargs=gen_kwargs,
        )

    if parse_fn is not None:
        # parse and retry
        parsed_responses = [parse_fn(response) for response in responses]
        retry_indices = [i for i, v in enumerate(parsed_responses) if v is None]
    else:
        parsed_responses = responses
        retry_indices = []

    tries = 0
    while retry_indices and tries < max_retries:
        retry_prompts = [prompt_data[i] for i in retry_indices]

        if is_pipeline:
            retry_raw = chat_generate_pipeline(
                batch=retry_prompts,
                pipeline=model_or_pipeline,
                tokenizer=tokenizer,
                device=device_obj,
                gen_kwargs=gen_kwargs,
                runtime_overrides=runtime_overrides,
                evaluation_data=evaluation_data,
            )
        else:
            retry_raw = chat_generate_model(
                batch=retry_prompts,
                model=model_or_pipeline,
                tokenizer=tokenizer,
                device=device_obj,
                gen_kwargs=gen_kwargs,
            )

        for local_i, global_i in enumerate(retry_indices):
            responses[global_i] = retry_raw[local_i]
            parsed_responses[global_i] = parse_fn(retry_raw[local_i])

        retry_indices = [i for i, v in enumerate(parsed_responses) if v is None]
        tries += 1

    return (parsed_responses, responses) if return_raw else parsed_responses


def _map_runtime_overrides(overrides, data):
    if isinstance(overrides, dict):
        result = {}
        for variable, column in overrides.items():
            result[variable] = _map_runtime_overrides(column, data)
        return result
    else:
        column_name = overrides
        values = []
        for item in data:
            value = item[column_name] if column_name in item else []
            values.append(value)
        return values


def _runtime_kwargs_to_list(flat_dict):
    def find_length(obj):
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, dict):
            return next(
                (
                    find_length(v)
                    for v in obj.values()
                    if (length := find_length(v)) is not None
                ),
                None,
            )
        return None

    def extract(obj, i):
        if isinstance(obj, list):
            return obj[i]
        if isinstance(obj, dict):
            return {k: extract(v, i) for k, v in obj.items()}
        return obj

    length = find_length(flat_dict)
    return [extract(flat_dict, i) for i in range(length)] if length else []
