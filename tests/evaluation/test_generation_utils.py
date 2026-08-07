"""Tests for the evaluation generation utilities' `Output` adoption.

Covers `chat_generate_pipeline` / `chat_generate_model` returning aligned `(texts, outputs)` on both the
batched and per-example branches (including equivalence of the fallback decoding to the deleted
pad-then-decode path), the `batch_retry_generate` return-shape matrix and retry alignment of `outputs`,
and the use-case-level surfacing of `finish_reason` / `adapted_prompt` through generation dicts and exports.
"""
import json

import pytest
import torch

from aisteer360.algorithms.core.output import Output
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.input_control.base import InputControl
from aisteer360.evaluation.utils.generation_utils import (
    batch_retry_generate,
    chat_generate_model,
    chat_generate_pipeline,
    output_record_fields,
)

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
GEN_KWARGS = {"max_new_tokens": 4, "do_sample": False}


class _NonBatchingInputControl(InputControl):
    """Enabled, prompt-preserving input control that is not batch-safe (forces the fallback branch)."""

    supports_batching = False

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids


@pytest.fixture(scope="module")
def batching_pipeline():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL)
    pipeline.steer()
    return pipeline


@pytest.fixture(scope="module")
def fallback_pipeline():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[_NonBatchingInputControl()])
    pipeline.steer()
    return pipeline


@pytest.fixture(scope="module")
def tokenizer(batching_pipeline):
    return batching_pipeline.tokenizer


def _prompt_batch(n: int) -> list[dict]:
    return [{"prompt": f"question {i}"} for i in range(n)]


class TestChatGeneratePipeline:
    """Both branches return aligned `(texts, outputs)`; fallback decoding matches the old path."""

    def test_batched_branch_aligned(self, batching_pipeline, tokenizer):
        assert batching_pipeline.supports_batching
        batch = _prompt_batch(3)
        texts, outputs = chat_generate_pipeline(
            batch=batch,
            pipeline=batching_pipeline,
            tokenizer=tokenizer,
            device=batching_pipeline.device,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )
        assert len(texts) == len(outputs) == 3
        assert all(isinstance(text, str) for text in texts)
        assert all(isinstance(out, Output) for out in outputs)
        assert all(out.adapted_input_ids is not None for out in outputs)

    def test_fallback_branch_aligned(self, fallback_pipeline, tokenizer):
        assert not fallback_pipeline.supports_batching
        batch = _prompt_batch(3)
        texts, outputs = chat_generate_pipeline(
            batch=batch,
            pipeline=fallback_pipeline,
            tokenizer=tokenizer,
            device=fallback_pipeline.device,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )
        assert len(texts) == len(outputs) == 3
        assert all(isinstance(out, Output) for out in outputs)

    def test_fallback_decoding_matches_pad_then_decode(self, fallback_pipeline, tokenizer):
        # pin the equivalence of per-item decoding to the deleted rectangularize-then-batch_decode path
        batch = _prompt_batch(3)
        texts, outputs = chat_generate_pipeline(
            batch=batch,
            pipeline=fallback_pipeline,
            tokenizer=tokenizer,
            device=fallback_pipeline.device,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )

        token_lists = [out.output_ids.squeeze(0).tolist() for out in outputs]
        padded = tokenizer.pad({"input_ids": token_lists}, padding=True, return_tensors="pt")
        old_texts = tokenizer.batch_decode(padded["input_ids"], skip_special_tokens=True)

        assert texts == old_texts


class TestChatGenerateModel:
    """Raw-model wrapping populates outputs with `adapted_input_ids is None` and finish reasons."""

    def test_outputs_populated(self, batching_pipeline, tokenizer):
        model = batching_pipeline.model
        batch = _prompt_batch(3)
        texts, outputs = chat_generate_model(
            batch=batch,
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            gen_kwargs=GEN_KWARGS,
            batch_size=2,
        )
        assert len(texts) == len(outputs) == 3
        assert all(isinstance(out, Output) for out in outputs)
        assert all(out.adapted_input_ids is None for out in outputs)
        assert all(out.finish_reason in ("eos", "length", None) for out in outputs)


class _CountingParse:
    """parse_fn that returns None on the first call for a target text, then succeeds thereafter."""

    def __init__(self, fail_first_for: str):
        self.fail_first_for = fail_first_for
        self.seen: dict[str, int] = {}

    def __call__(self, text):
        self.seen[text] = self.seen.get(text, 0) + 1
        if text == self.fail_first_for and self.seen[text] == 1:
            return None
        return f"parsed:{text}"


class TestBatchRetryGenerate:
    """Return-shape matrix over (return_raw, return_outputs) and retry alignment of `outputs`."""

    @pytest.mark.parametrize(
        "return_raw,return_outputs,expected_len",
        [
            (False, False, None),  # plain list
            (True, False, 2),
            (False, True, 3),
            (True, True, 3),  # return_outputs wins regardless of return_raw
        ],
    )
    def test_return_shape_matrix(self, batching_pipeline, tokenizer, return_raw, return_outputs, expected_len):
        result = batch_retry_generate(
            prompt_data=_prompt_batch(2),
            model_or_pipeline=batching_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            return_raw=return_raw,
            return_outputs=return_outputs,
            batch_size=8,
        )
        if expected_len is None:
            assert isinstance(result, list)
            assert not isinstance(result, tuple)
        else:
            assert isinstance(result, tuple)
            assert len(result) == expected_len
            if return_outputs:
                parsed, raw, outputs = result
                assert len(parsed) == len(raw) == len(outputs) == 2
                assert all(out is None or isinstance(out, Output) for out in outputs)

    def test_retry_aligns_outputs_with_final_response(self, batching_pipeline, tokenizer):
        batch = _prompt_batch(3)
        # discover the first-attempt raw texts so the counting parser can target index 1
        first_texts, _ = chat_generate_pipeline(
            batch=batch,
            pipeline=batching_pipeline,
            tokenizer=tokenizer,
            device=batching_pipeline.device,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )
        parse_fn = _CountingParse(fail_first_for=first_texts[1])

        parsed, raw, outputs = batch_retry_generate(
            prompt_data=batch,
            model_or_pipeline=batching_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            parse_fn=parse_fn,
            max_retries=2,
            return_outputs=True,
            batch_size=8,
        )

        # index 1 was retried at least once; its final record must correspond to its final raw text
        assert parse_fn.seen[first_texts[1]] >= 2
        assert len(outputs) == 3
        for index in range(3):
            assert outputs[index] is None or isinstance(outputs[index], Output)
            assert parsed[index] == f"parsed:{raw[index]}"


class TestOutputRecordFields:
    """`output_record_fields` contributes finish_reason always and adapted_prompt only when present."""

    def test_none_output(self, tokenizer):
        fields = output_record_fields(None, tokenizer)
        assert fields == {"finish_reason": None}

    def test_pipeline_output_has_adapted_prompt(self, tokenizer):
        out = Output(
            output_ids=torch.tensor([[5, 6]]),
            adapted_input_ids=torch.tensor([[7, 8]]),
            finish_reason="length",
        )
        fields = output_record_fields(out, tokenizer)
        assert fields["finish_reason"] == "length"
        assert "adapted_prompt" in fields
        assert isinstance(fields["adapted_prompt"], str)

    def test_raw_model_output_omits_adapted_prompt(self, tokenizer):
        out = Output(output_ids=torch.tensor([[5, 6]]), adapted_input_ids=None, finish_reason="eos")
        fields = output_record_fields(out, tokenizer)
        assert fields["finish_reason"] == "eos"
        assert "adapted_prompt" not in fields


class TestUseCaseSurfacing:
    """End-to-end: instruction_following generation dicts and export carry the new keys."""

    @pytest.fixture
    def use_case_data(self):
        return [
            {
                "prompt": f"Write about topic {i}.",
                "instructions": ["be concise"],
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{}],
            }
            for i in range(2)
        ]

    def test_generation_dicts_have_new_keys(self, batching_pipeline, tokenizer, use_case_data):
        from aisteer360.evaluation.use_cases.instruction_following.use_case import InstructionFollowing

        use_case = InstructionFollowing.__new__(InstructionFollowing)
        use_case.evaluation_data = use_case_data
        use_case.evaluation_metrics = []

        generations = use_case.generate(
            model_or_pipeline=batching_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )
        assert len(generations) == 2
        for gen in generations:
            assert "finish_reason" in gen
            assert gen["finish_reason"] in ("eos", "length", None)
            assert "adapted_prompt" in gen  # pipeline path always carries the steered prompt
            assert isinstance(gen["adapted_prompt"], str)

    def test_export_round_trips_new_keys(self, tmp_path, batching_pipeline, tokenizer, use_case_data):
        from aisteer360.evaluation.use_cases.instruction_following.use_case import InstructionFollowing

        use_case = InstructionFollowing.__new__(InstructionFollowing)
        use_case.evaluation_data = use_case_data
        use_case.evaluation_metrics = []

        generations = use_case.generate(
            model_or_pipeline=batching_pipeline,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )
        # export reads follow_all_instructions from the StrictInstruction metric, one flag per generation
        evaluations = {"StrictInstruction": {"follow_all_instructions": [True] * len(generations)}}
        profiles = {"steered": [{"trial_id": 0, "generations": generations, "evaluations": evaluations, "params": {}}]}
        use_case.export(profiles, str(tmp_path))

        with open(tmp_path / "responses.json") as f:
            rows = json.load(f)
        assert len(rows) == 2
        for row in rows:
            assert "steered_finish_reason" in row
            assert "steered_adapted_prompt" in row

    def test_raw_model_path_omits_adapted_prompt(self, batching_pipeline, tokenizer, use_case_data):
        from aisteer360.evaluation.use_cases.instruction_following.use_case import InstructionFollowing

        use_case = InstructionFollowing.__new__(InstructionFollowing)
        use_case.evaluation_data = use_case_data
        use_case.evaluation_metrics = []

        generations = use_case.generate(
            model_or_pipeline=batching_pipeline.model,
            tokenizer=tokenizer,
            gen_kwargs=GEN_KWARGS,
            batch_size=8,
        )
        for gen in generations:
            assert "finish_reason" in gen
            assert "adapted_prompt" not in gen
