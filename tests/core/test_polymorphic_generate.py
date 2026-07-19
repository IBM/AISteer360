"""Tests for the polymorphic SteeringPipeline.generate dispatch table (design doc §3.3, §7)."""
import warnings

import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.core.types import Output
from aisteer360.algorithms.input_control.base import InputControl
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def pipeline():
    p = SteeringPipeline(model_name_or_path=TINY_MODEL)
    p.steer()
    return p


class TestClassifyInputs:
    """Pure unit tests for the input classifier."""

    def test_str(self):
        modality, single, norm = SteeringPipeline._classify_inputs("hello")
        assert modality == "text"
        assert single is True
        assert norm == ["hello"]

    def test_list_str(self):
        modality, single, _ = SteeringPipeline._classify_inputs(["a", "b"])
        assert modality == "text"
        assert single is False

    def test_single_chat(self):
        modality, single, norm = SteeringPipeline._classify_inputs(
            [{"role": "user", "content": "hi"}]
        )
        assert modality == "chat"
        assert single is True
        assert len(norm) == 1
        assert norm[0][0]["role"] == "user"

    def test_batch_chat(self):
        modality, single, _ = SteeringPipeline._classify_inputs(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
        )
        assert modality == "chat"
        assert single is False

    def test_tensor_1d(self):
        modality, single, norm = SteeringPipeline._classify_inputs(torch.tensor([1, 2, 3]))
        assert modality == "tensor"
        assert single is True
        assert norm.ndim == 2

    def test_tensor_2d(self):
        modality, single, _ = SteeringPipeline._classify_inputs(torch.tensor([[1, 2], [3, 4]]))
        assert modality == "tensor"
        assert single is False

    def test_list_int(self):
        modality, single, _ = SteeringPipeline._classify_inputs([1, 2, 3])
        assert modality == "tensor"
        assert single is True

    def test_list_list_int(self):
        modality, single, _ = SteeringPipeline._classify_inputs([[1, 2], [3, 4]])
        assert modality == "tensor"
        assert single is False

    def test_unsupported(self):
        with pytest.raises((TypeError, ValueError)):
            SteeringPipeline._classify_inputs(3.14)


class TestDispatchReturnTypes:
    """Each modality returns the correct default type per the dispatch table."""

    def test_str_returns_str(self, pipeline):
        out = pipeline.generate("hello", max_new_tokens=2)
        assert isinstance(out, str)

    def test_list_str_returns_list_str(self, pipeline):
        out = pipeline.generate(["a", "b"], max_new_tokens=2)
        assert isinstance(out, list)
        assert all(isinstance(x, str) for x in out)
        assert len(out) == 2

    def test_single_chat_returns_str(self, pipeline):
        out = pipeline.generate([{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert isinstance(out, str)

    def test_batch_chat_returns_list_str(self, pipeline):
        out = pipeline.generate(
            [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]],
            max_new_tokens=2,
        )
        assert isinstance(out, list)
        assert all(isinstance(x, str) for x in out)

    def test_tensor_returns_tensor(self, pipeline):
        out = pipeline.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert isinstance(out, torch.Tensor)

    def test_input_ids_kwarg_alias(self, pipeline):
        # legacy keyword still accepted
        out = pipeline.generate(input_ids=torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert isinstance(out, torch.Tensor)

    def test_inputs_and_input_ids_both_set_raises(self, pipeline):
        with pytest.raises(TypeError):
            pipeline.generate("hi", input_ids=torch.tensor([[1, 2]]))


class TestReturnOutputFlag:
    """`return_output=True` produces Output(s) regardless of input modality."""

    def test_single_returns_output(self, pipeline):
        out = pipeline.generate(torch.tensor([1, 2, 3]), max_new_tokens=2, return_output=True)
        assert isinstance(out, Output)
        assert out.output_ids.shape[0] == 1
        assert out.adapted_input_ids is not None

    def test_batched_returns_list_output(self, pipeline):
        out = pipeline.generate(["a", "b"], max_new_tokens=2, return_output=True)
        assert isinstance(out, list)
        assert all(isinstance(x, Output) for x in out)
        assert len(out) == 2

    def test_finish_reason_length(self, pipeline):
        out = pipeline.generate("hi", max_new_tokens=3, return_output=True)
        assert out.finish_reason in ("length", None)


class TestAttentionMaskWarning:
    def test_warns_on_text_with_attention_mask(self, pipeline):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pipeline.generate("hi", attention_mask=torch.ones(1, 3), max_new_tokens=1)
            assert any("attention_mask" in str(x.message).lower() for x in w)


class _BothEntryPointsControl(InputControl):
    """Implements BOTH adapt_messages (chat path) and adapt (text/tensor path), like CPO/GEPA/
    PRewrite/FewShot. Counts invocations so tests can assert exactly-once application."""

    supports_batching = True

    def __init__(self, handle_messages: bool = True):
        super().__init__()
        self.handle_messages = handle_messages
        self.adapt_calls = 0
        self.adapt_messages_calls = 0

    def adapt_messages(self, messages, runtime_kwargs=None):
        self.adapt_messages_calls += 1
        if not self.handle_messages:
            return None
        return [
            [{"role": "system", "content": "injected"}] + list(chat)
            for chat in messages
        ]

    def adapt(self, input_ids, runtime_kwargs=None):
        self.adapt_calls += 1
        return input_ids


class TestInputControlAppliedExactlyOnce:
    """Regression tests for the exactly-once input-control contract (design doc §3)."""

    def _make_pipeline(self, control):
        pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[control])
        pipeline.steer()
        return pipeline

    def test_chat_input_skips_token_level_adapt(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        pipeline.generate([{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert control.adapt_messages_calls == 1
        assert control.adapt_calls == 0

    def test_chat_input_falls_through_when_adapt_messages_returns_none(self):
        control = _BothEntryPointsControl(handle_messages=False)
        pipeline = self._make_pipeline(control)
        pipeline.generate([{"role": "user", "content": "hi"}], max_new_tokens=2)
        assert control.adapt_messages_calls == 1
        assert control.adapt_calls == 1

    def test_text_input_uses_token_level_adapt_only(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.generate("hi", max_new_tokens=2)
        assert control.adapt_messages_calls == 0
        assert control.adapt_calls == 1

    def test_tensor_input_uses_token_level_adapt_only(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=2)
        assert control.adapt_messages_calls == 0
        assert control.adapt_calls == 1

    def test_chat_system_prompt_injected_once(self):
        control = _BothEntryPointsControl(handle_messages=True)
        pipeline = self._make_pipeline(control)
        out = pipeline.generate(
            [{"role": "user", "content": "hi"}], max_new_tokens=2, return_output=True
        )
        prompt_text = pipeline.tokenizer.decode(
            out.adapted_input_ids[0], skip_special_tokens=True
        )
        assert prompt_text.count("injected") == 1


@pytest.fixture(scope="module")
def tiny_pipeline():
    """Hub-free steered pipeline for the return-semantics regression (CPU-only, offline)."""
    torch.manual_seed(0)
    model = tiny_llama(num_layers=2, hidden=16, heads=2)
    tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()
    return pipeline


class TestReturnSemantics:
    """WS5: tensor return is continuation-only by default; `return_full_sequence` includes the prompt.

    Guards against re-introducing the notebook bug of slicing a continuation-only result by prompt
    length (which discards generated tokens).
    """

    def test_default_is_continuation_only(self, tiny_pipeline):
        ids = torch.tensor([[3, 4, 5, 6]])
        prompt_len = ids.size(1)
        k = 5
        cont = tiny_pipeline.generate(ids, max_new_tokens=k, do_sample=False)
        full = tiny_pipeline.generate(
            ids, max_new_tokens=k, do_sample=False, return_full_sequence=True
        )
        # continuation-only excludes the prompt; full includes it
        assert full.shape[1] == prompt_len + cont.shape[1]
        assert cont.shape[1] == full.shape[1] - prompt_len
        # the two agree on the continuation tokens
        assert torch.equal(full[:, prompt_len:], cont)

    def test_continuation_length_matches_max_new_tokens(self, tiny_pipeline):
        ids = torch.tensor([[3, 4, 5]])
        for k in (1, 3, 6):
            cont = tiny_pipeline.generate(ids, max_new_tokens=k, do_sample=False)
            # tiny model won't emit EOS deterministically here, so the full budget is used
            assert cont.shape[1] == k
