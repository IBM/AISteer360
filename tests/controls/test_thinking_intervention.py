import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.output_control.thinking_intervention.control import (
    ThinkingIntervention,
)
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Solve briefly: What is the area of a 3x4 rectangle?"
)

THINKING_GRID = {
    "produces_tag": [True, False],  # whether stubbed generator will include a </think> tag in continuation
    "use_params": [True, False],  # whether to pass runtime params through to the intervention function
}


def simple_intervention(prompt: str, params: dict) -> str:
    """
    A minimal intervention that prepends a think block before the user prompt.
    """
    plan = params.get("plan", "List steps, then conclude.")
    return f"<think>{plan}</think>\n{prompt}"


@pytest.mark.parametrize("conf", build_param_grid(THINKING_GRID))
def test_thinking_intervention(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that ThinkingIntervention modifies the prompt, generates, and (when applicable) strips the thinking content up to the closing </think> tag.
    """
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    control = ThinkingIntervention(intervention=simple_intervention)

    pipeline = SteeringPipeline(controls=[control], lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    # prompt
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)

    # deterministic base_generate
    def fake_generate(**kwargs):
        """
        Mimics HF generate.
        """
        inputs = kwargs["input_ids"]
        if conf["produces_tag"]:
            continuation_text = " <think>intermediate steps</think> FINAL ANSWER."
        else:
            continuation_text = " <think>intermediate steps without closing tag FINAL ANSWER."
        contuation_ids = tokenizer(continuation_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(inputs.device)
        return torch.cat([inputs, contuation_ids], dim=1)

    # runtime kwargs
    runtime_kwargs = {
        "base_generate": fake_generate,
    }
    if conf["use_params"]:
        runtime_kwargs["params"] = {"plan": "Outline key steps concisely."}

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        runtime_kwargs=runtime_kwargs
    )

    # shape assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(0) == 1, "ThinkingIntervention test assumes batch size 1"

    # content assertions
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    if conf["produces_tag"]:
        assert "</think>" not in decoded, "Closing think tag should be stripped from final output"
        assert "<think>" not in decoded, "Thinking block should be stripped from final output"
        assert "FINAL ANSWER." in decoded, "Expected post-think content to remain"
    else:
        assert len(decoded) > 0, "Decoded output should be non-empty"
