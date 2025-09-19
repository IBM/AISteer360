import random

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.few_shot.control import FewShot
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Classify the sentiment of the following sentence as Positive or Negative.\n"
    "Sentence: I loved the cinematography but the plot was thin."
)

POS_POOL = [
    {"input": "The service was excellent and the food was great.", "label": "Positive"},
    {"input": "What an amazing performance; I had a wonderful time!", "label": "Positive"},
]
NEG_POOL = [
    {"input": "The device kept crashing and the battery died fast.", "label": "Negative"},
    {"input": "Terrible support; I regret this purchase.", "label": "Negative"},
]

FEWSHOT_GRID = {
    "mode": ["runtime", "pool", "none"],
    "k_positive": [1, 2],
    "k_negative": [0, 1],
    "selector_name": ["random"],
    "use_negative_runtime": [False, True],
}


def _runtime_kwargs_from_conf(conf):
    if conf["mode"] != "runtime":
        return {}
    pos = [{"input": "I am thrilled with the results.", "label": "Positive"}]
    neg = [{"input": "This was a waste of time.", "label": "Negative"}] if conf["use_negative_runtime"] else []
    runtime_kwargs = {}
    if pos:
        runtime_kwargs["positive_examples"] = pos
    if neg:
        runtime_kwargs["negative_examples"] = neg
    return runtime_kwargs


@pytest.mark.parametrize("conf", build_param_grid(FEWSHOT_GRID))
def test_few_shot(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that FewShot adapts prompts and generates on every model/device/param combo. Also sanity-check that the
    adapted prompt length increases when examples are provided.
    """
    # deterministic selector behavior
    random.seed(0)
    torch.manual_seed(0)

    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # build FewShot control based on the mode
    kwargs = {
        "directive": "Follow the schema. Classify correctly using the demonstrations",
        "selector_name": conf["selector_name"],
    }

    if conf["mode"] == "pool":
        kwargs.update(
            dict(
                positive_example_pool=POS_POOL,
                negative_example_pool=NEG_POOL if conf["k_negative"] > 0 else None,
                k_positive=conf["k_positive"],
                k_negative=conf["k_negative"],
            )
        )
    elif conf["mode"] == "runtime":
        # no pools; runtime examples will be provided via runtime_kwargs
        kwargs.update(dict(k_positive=None, k_negative=None))
    else:  # "none"; deliberately provide no pools and no runtime examples
        kwargs.update(dict(k_positive=None, k_negative=None))

    fewshot = FewShot(**kwargs)

    # pipeline
    pipeline = SteeringPipeline(controls=[fewshot], lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    # prepare inputs & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    runtime_kwargs = _runtime_kwargs_from_conf(conf)

    # sanity check
    adapter = fewshot.get_prompt_adapter()
    adapted = adapter(prompt_ids, runtime_kwargs)

    # handle tensor/list shapes consistently
    if isinstance(adapted, torch.Tensor):
        adapted_len = adapted.size(-1) if adapted.ndim > 1 else adapted.size(0)
        orig_len = prompt_ids.size(-1)
    else:
        adapted_len = len(adapted)
        orig_len = len(tokenizer.encode(PROMPT_TEXT, add_special_tokens=False))

    if conf["mode"] in ("runtime", "pool"):
        assert adapted_len > orig_len, "FewShot should prepend examples and increase prompt length"
    else:
        assert adapted_len == orig_len, "With no examples, prompt should be unchanged"

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        runtime_kwargs=runtime_kwargs,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"
