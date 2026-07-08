"""Reject-path tests for LLMJudgeMetric construction-time validation."""
from __future__ import annotations

import warnings

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisteer360.evaluation.metrics.base_judge import LLMJudgeMetric


@pytest.fixture(scope="module")
def tiny_lm():
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_structured_output_false_requires_parser(tiny_lm):
    model, tokenizer = tiny_lm
    with pytest.raises(ValueError, match="parser"):
        LLMJudgeMetric(
            model_or_id=model,
            tokenizer=tokenizer,
            prompt_template="rate {response} from {lower_bound} to {upper_bound}",
            structured_output=False,
            parser=None,
        )


def test_structured_output_true_rejects_parser(tiny_lm):
    model, tokenizer = tiny_lm
    with pytest.raises(ValueError, match="not both"):
        LLMJudgeMetric(
            model_or_id=model,
            tokenizer=tokenizer,
            prompt_template="rate {response} from {lower_bound} to {upper_bound}",
            structured_output=True,
            parser=lambda text: 1.0,
        )


def test_num_return_sequences_requires_temperature(tiny_lm):
    model, tokenizer = tiny_lm
    with pytest.raises(ValueError, match="num_return_sequences"):
        LLMJudgeMetric(
            model_or_id=model,
            tokenizer=tokenizer,
            prompt_template="rate {response} from {lower_bound} to {upper_bound}",
            gen_kwargs={"temperature": 0.0, "num_return_sequences": 3},
        )


def test_device_with_preloaded_model_warns(tiny_lm):
    model, tokenizer = tiny_lm
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LLMJudgeMetric(
            model_or_id=model,
            tokenizer=tokenizer,
            prompt_template="rate {response} from {lower_bound} to {upper_bound}",
            device="cpu",
        )
    assert any(
        issubclass(w.category, UserWarning) and "ignoring `device`" in str(w.message)
        for w in caught
    )
