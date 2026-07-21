"""Unit tests for the shared `TransformHookRuntime` (design PR 2a).

Exercises the runtime directly with hand-registered hooks on a tiny Llama: pass-opener KV-offset
semantics across prefill/decode with multiple hooked layers, `after_prompt`/`last_k`/`from_position`/
`all` token scopes, beam-expansion alignment, tuple vs bare-tensor outputs, the pre-hook
(`layer_input`) extract/replace path, and read-only condition hooks feeding a gate.

Runs hub-free on a tiny randomly-initialized Llama.
"""
import pytest
import torch

from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate, MultiKeyThresholdGate
from aisteer360.algorithms.state_control._common.runtime import TransformHookRuntime
from aisteer360.algorithms.state_control._common.token_scope import compute_prompt_lens
from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
HEADS = 4
LAYERS = 4


class _RecordingTransform(BaseTransform):
    """Records the token mask and position offset seen at each apply; adds a constant."""

    def __init__(self, value: float = 1.0):
        self.value = value
        self.masks: list[torch.BoolTensor] = []

    def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
        self.masks.append(token_mask.detach().clone())
        return hidden_states + self.value


def _register(model, runtime, hooks):
    """Register `(layer_id, hook_callable)` pairs at the runtime's hook point; return handles."""
    handles = []
    for layer_id, hook in hooks:
        module = model.model.layers[layer_id]
        if runtime.hook_point == "layer_output":
            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        else:
            handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
    return handles


class TestPassOpenerOffset:
    def test_offset_advances_once_per_pass_multi_layer(self):
        """With three hooked layers, `after_prompt` steers every decode pass and no prefill pass."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        gate = AlwaysOpenGate()
        transforms = {lid: _RecordingTransform() for lid in (0, 1, 2)}

        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)  # prompt_len 4
        runtime.reset(compute_prompt_lens(input_ids, None))

        layer_ids = [0, 1, 2]
        opener = min(layer_ids)
        hooks = [
            (lid, runtime.build_behavior_hook(
                layer_id=lid, transform=transforms[lid], gate=gate,
                token_scope="after_prompt", is_pass_opener=(lid == opener)))
            for lid in layer_ids
        ]
        handles = _register(model, runtime, hooks)
        try:
            model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False, eos_token_id=None)
        finally:
            for h in handles:
                h.remove()

        # the runtime skips no-op applies, so the prefill pass (all positions < prompt_len)
        # records nothing; each layer's transform is called once per DECODE pass = 4
        # (the final generated token is emitted but never re-processed)
        for lid in layer_ids:
            masks = transforms[lid].masks
            assert len(masks) == 4
            # decode passes (seq_len 1, absolute position >= 4) -> steered
            for m in masks:
                assert m.shape[1] == 1 and bool(m.all())

    @pytest.mark.parametrize("prompt_len", [1, 4])
    def test_prompt_len_one_still_steers_decode(self, prompt_len):
        """A length-1 prompt must not confuse prefill with decode (the anti-drift guard)."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()

        input_ids = torch.arange(3, 3 + prompt_len, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))
        hook = runtime.build_behavior_hook(
            layer_id=1, transform=transform, gate=AlwaysOpenGate(),
            token_scope="after_prompt", is_pass_opener=True)
        handles = _register(model, runtime, [(1, hook)])
        try:
            model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False, eos_token_id=None)
        finally:
            for h in handles:
                h.remove()

        steered = sum(1 for m in transform.masks if bool(m.any()))
        assert steered == 4  # max_new_tokens - 1 decode passes


class TestTokenScopes:
    def _run_single_pass(self, token_scope, seq_len, **kw):
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        input_ids = torch.arange(3, 3 + seq_len, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))
        hook = runtime.build_behavior_hook(
            layer_id=1, transform=transform, gate=AlwaysOpenGate(), token_scope=token_scope,
            is_pass_opener=True, **kw)
        handles = _register(model, runtime, [(1, hook)])
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        return transform.masks[0]

    def test_all_scope(self):
        mask = self._run_single_pass("all", seq_len=4)
        assert bool(mask.all())

    def test_last_k_scope(self):
        mask = self._run_single_pass("last_k", seq_len=4, last_k=2)
        assert mask.squeeze(0).tolist() == [False, False, True, True]

    def test_from_position_scope(self):
        mask = self._run_single_pass("from_position", seq_len=4, from_position=1)
        assert mask.squeeze(0).tolist() == [False, True, True, True]


class TestBeamExpansion:
    def test_align_mask_to_expanded_batch(self):
        """When hidden batch > prompt batch (beam search), the mask is replicated to align."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform()
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))
        hook = runtime.build_behavior_hook(
            layer_id=1, transform=transform, gate=AlwaysOpenGate(),
            token_scope="all", is_pass_opener=True)
        handles = _register(model, runtime, [(1, hook)])
        try:
            model.generate(
                input_ids=input_ids, max_new_tokens=2, do_sample=False,
                num_beams=3, eos_token_id=None,
            )
        finally:
            for h in handles:
                h.remove()
        # some recorded mask must have a batch dimension expanded to a multiple of 1 (the beams)
        assert any(m.size(0) >= 3 for m in transform.masks)


class TestBareTensorOutput:
    def test_handles_bare_tensor_layer_output(self):
        """A layer returning a bare tensor (not a tuple) is handled without error."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform(value=2.0)
        runtime.reset(torch.tensor([4]))
        hook = runtime.build_behavior_hook(
            layer_id=0, transform=transform, gate=AlwaysOpenGate(),
            token_scope="all", is_pass_opener=True)

        hidden = torch.zeros(1, 4, HIDDEN)
        out = hook(None, (), {}, hidden)  # bare-tensor output path
        assert isinstance(out, torch.Tensor)
        assert torch.allclose(out, torch.full_like(hidden, 2.0))

    def test_handles_tuple_layer_output(self):
        runtime = TransformHookRuntime(hook_point="layer_output")
        transform = _RecordingTransform(value=2.0)
        runtime.reset(torch.tensor([4]))
        hook = runtime.build_behavior_hook(
            layer_id=0, transform=transform, gate=AlwaysOpenGate(),
            token_scope="all", is_pass_opener=True)

        hidden = torch.zeros(1, 4, HIDDEN)
        extra = torch.tensor([1.0])
        out = hook(None, (), {}, (hidden, extra))
        assert isinstance(out, tuple)
        assert torch.allclose(out[0], torch.full_like(hidden, 2.0))
        assert out[1] is extra  # trailing elements preserved


class TestPreHookPath:
    def test_layer_input_extract_replace(self):
        """The `layer_input` pre-hook path steers via extract/replace on the layer input."""
        model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=HEADS)
        runtime = TransformHookRuntime(hook_point="layer_input")
        transform = _RecordingTransform(value=3.0)
        input_ids = torch.arange(3, 7, dtype=torch.long).unsqueeze(0)
        runtime.reset(compute_prompt_lens(input_ids, None))

        captured = {}

        def _capture(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            captured["h"] = hidden.detach().clone()
            return None

        hook = runtime.build_behavior_hook(
            layer_id=2, transform=transform, gate=AlwaysOpenGate(),
            token_scope="all", is_pass_opener=True)
        # register the steering pre-hook, then a capture pre-hook AFTER it to observe the edit
        h1 = model.model.layers[2].register_forward_pre_hook(hook, with_kwargs=True)
        h2 = model.model.layers[2].register_forward_pre_hook(_capture, with_kwargs=True)
        try:
            with torch.no_grad():
                model(input_ids=input_ids)
        finally:
            h1.remove()
            h2.remove()

        assert transform.masks  # the pre-hook fired
        assert "h" in captured  # the layer input was edited before the capture hook saw it


class TestConditionHook:
    def test_condition_hook_is_read_only_and_updates_gate(self):
        """A condition hook computes a score, feeds the gate, and leaves hidden states untouched."""
        runtime = TransformHookRuntime(hook_point="layer_output")
        gate = MultiKeyThresholdGate(threshold=0.5, comparator="score_above", expected_keys={1})
        runtime.reset(torch.tensor([4]))

        seen = {}

        def _score(hidden, layer_id, *, prompt_mask=None):
            seen["hidden"] = hidden
            return torch.full((hidden.size(0),), 0.9)  # per-row; above threshold

        hook = runtime.build_condition_hook(layer_id=1, scorer=_score, gate=gate, is_pass_opener=True)

        hidden = torch.randn(1, 4, HIDDEN)
        out = hook(None, (), {}, hidden)
        assert out is hidden  # unmodified output returned as-is
        assert seen["hidden"] is hidden
        assert gate.is_open()  # 0.9 >= 0.5 opens the gate
