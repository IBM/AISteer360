import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.pasta.control import PASTA
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Answer truthfully. Therefore, when you respond: "
    "First, present your main point. "
    "Second, support it with evidence. "
    "Finally, conclude succinctly."
)

PASTA_GRID = {
    "substrings": [
        ["Therefore"],
        ["First,", "Second,", "Finally,"]
    ],
    "alpha": [0.25, 0.75],
    "scale_position": ["include", "exclude", "generation"],
    "head_config": [
        [0],
        [0, 1]
    ],
}


@pytest.mark.parametrize("conf", build_param_grid(PASTA_GRID))
def test_pasta(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that PASTA steers and generates on every model/device/param combo.
    """

    # move model to target device
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # build pipeline with PASTA control
    pasta = PASTA(
        head_config=conf["head_config"],
        alpha=conf["alpha"],
        scale_position=conf["scale_position"]
    )
    pipeline = SteeringPipeline(controls=[pasta], lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    # prepare prompt & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)
    runtime_kwargs = {"substrings": conf["substrings"]}

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
