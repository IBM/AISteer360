"""Probe-driven steering tests: ProbeSumGate semantics, adapter equivalence, and the guards.

Runs hub-free on a tiny randomly-initialized Llama. Probes are hand-built with fixed weights;
biases derived from a preliminary `ProbeSet.read` split a batch into open and closed rows, so
adapter decisions can be compared row-for-row against direct reads.
"""
import pytest
import torch

from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from aisteer360.algorithms.core.internals.probes.probe import Probe
from aisteer360.algorithms.core.internals.probes.probe_set import ProbeSet
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate
from aisteer360.algorithms.state_control._common.gates.cache_once import CacheOnceGate
from aisteer360.algorithms.state_control._common.condition_scorers import (
    ProbeContributionScorer,
    probe_condition,
)
from aisteer360.algorithms.state_control._common.gates.probe_sum import ProbeSumGate
from aisteer360.algorithms.state_control.activation_adapter.control import ActivationAdapter
from tests.utils.runtime_helpers import RecordingTransform
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4

PROMPTS = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 3, 5]])
GEN_KWARGS = {"do_sample": False, "eos_token_id": None}


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _probe(layer_ids, bias=0.0, seed=7, pooling="mean", meta=None):
    return Probe(
        model_type="llama",
        location="layer_input",
        pooling=pooling,
        layer_ids=list(layer_ids),
        weights={lid: _unit_vector(seed + lid) for lid in layer_ids},
        bias=bias,
        meta=meta or {},
    )


def _model_and_tokenizer(seed: int = 0):
    torch.manual_seed(seed)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    return model, tokenizer


def _steered_pipeline(model, tokenizer, controls) -> SteeringPipeline:
    pipeline = SteeringPipeline(controls=controls, lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()
    return pipeline


def _splitting_probe(model, layer_ids, seed=7):
    """A probe whose bias splits PROMPTS into open and closed rows, or None if inseparable."""
    scores = ProbeSet({"p": _probe(layer_ids, seed=seed)}).read(model, PROMPTS).scores["p"]
    ordered = scores.sort().values
    if ordered[1] - ordered[0] < 1e-5:
        return None
    midpoint = float((ordered[0] + ordered[1]) / 2)
    return _probe(layer_ids, bias=-midpoint, seed=seed)


class TestProbeSumGate:
    def _gate(self, layer_ids=(1, 2), bias=-1.0):
        return ProbeSumGate(_probe(layer_ids, bias=bias))

    def test_not_ready_until_every_layer_reports(self):
        gate = self._gate()
        gate.reset(2)
        assert not gate.is_ready()
        gate.update(torch.tensor([0.6, 0.1]), key=1)
        assert not gate.is_ready()
        gate.update(torch.tensor([0.5, 0.2]), key=2)
        assert gate.is_ready()

    def test_open_rows_sums_contributions_and_applies_bias(self):
        gate = self._gate(bias=-1.0)
        gate.reset(2)
        gate.update(torch.tensor([0.6, 0.1]), key=1)
        gate.update(torch.tensor([0.5, 0.2]), key=2)
        # row 0: 1.1 - 1.0 >= 0 opens; row 1: 0.3 - 1.0 stays closed
        assert gate.open_rows().tolist() == [True, False]

    def test_all_closed_before_any_evidence(self):
        gate = self._gate()
        gate.reset(3)
        assert gate.open_rows().tolist() == [False, False, False]

    def test_ties_open(self):
        gate = self._gate(layer_ids=(1,), bias=-1.0)
        gate.reset(1)
        gate.update(torch.tensor([1.0]), key=1)
        assert gate.open_rows().tolist() == [True]

    def test_reset_clears_evidence(self):
        gate = self._gate()
        gate.reset(1)
        gate.update(torch.tensor([5.0]), key=1)
        gate.update(torch.tensor([5.0]), key=2)
        assert gate.is_ready()
        gate.reset(2)
        assert not gate.is_ready()
        assert gate.open_rows().tolist() == [False, False]

    def test_scalar_score_allowed_single_row_only(self):
        gate = self._gate(layer_ids=(1,), bias=0.0)
        gate.reset(1)
        gate.update(0.5, key=1)
        assert gate.open_rows().tolist() == [True]
        gate.reset(2)
        with pytest.raises(ValueError, match="scalar"):
            gate.update(0.5, key=1)


class TestProbeCondition:
    def test_returns_adapter_ports(self):
        probe = _probe([1, 2], bias=0.5)
        ports = probe_condition(probe)
        assert set(ports) == {"score_fn", "gate", "condition_layer_ids"}
        assert isinstance(ports["score_fn"], ProbeContributionScorer)
        assert isinstance(ports["gate"], CacheOnceGate)
        assert isinstance(ports["gate"].inner, ProbeSumGate)
        assert ports["condition_layer_ids"] == [1, 2]

    def test_cache_once_false_returns_bare_gate(self):
        ports = probe_condition(_probe([1]), cache_once=False)
        assert isinstance(ports["gate"], ProbeSumGate)

    def test_allow_model_mismatch_disarms_scorer_fingerprint(self):
        probe = _probe([1], meta={"model_fingerprint": "abcd"})
        assert probe_condition(probe)["score_fn"].model_fingerprint == "abcd"
        assert probe_condition(probe, allow_model_mismatch=True)["score_fn"].model_fingerprint is None

    def test_as_condition_is_sugar_over_probe_condition(self):
        probe = _probe([1, 2])
        ports = probe.as_condition(cache_once=False)
        assert isinstance(ports["score_fn"], ProbeContributionScorer)
        assert isinstance(ports["gate"], ProbeSumGate)
        assert ports["condition_layer_ids"] == [1, 2]

    def test_scorer_zero_for_absent_layer(self):
        scorer = ProbeContributionScorer(_probe([1]))
        assert scorer(torch.randn(2, 3, HIDDEN), layer_id=3).tolist() == [0.0, 0.0]


class TestAdapterEquivalence:
    @pytest.mark.parametrize("layer_ids", [[1], [1, 2]], ids=["single_layer", "multi_layer"])
    def test_adapter_decisions_match_probe_set_read(self, layer_ids):
        model, tokenizer = _model_and_tokenizer()
        probe = _splitting_probe(model, layer_ids)
        if probe is None:
            pytest.skip("tiny-model probe scores not separable for this seed")

        expected = ProbeSet({"p": probe}).read(model, PROMPTS).decisions["p"]
        assert expected.any() and not expected.all()  # the bias genuinely splits the batch

        adapter = ActivationAdapter(
            transform=RecordingTransform(value=0.5),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(probe),
        )
        pipeline = _steered_pipeline(model, tokenizer, [adapter])
        pipeline.generate(input_ids=PROMPTS, max_new_tokens=2, **GEN_KWARGS)

        assert adapter._gate.open_rows().tolist() == expected.tolist()

    def test_cache_once_scores_prompt_once_and_freezes(self):
        model, tokenizer = _model_and_tokenizer()
        probe = _probe([1], bias=1e9)  # always open

        ports = probe_condition(probe)
        calls: list[tuple] = []
        inner_scorer = ports["score_fn"]

        class _SpyScorer:
            location = inner_scorer.location
            model_fingerprint = inner_scorer.model_fingerprint

            def __call__(self, hidden, layer_id, *, prompt_mask=None):
                calls.append(tuple(hidden.shape))
                return inner_scorer(hidden, layer_id, prompt_mask=prompt_mask)

        adapter = ActivationAdapter(
            transform=RecordingTransform(value=0.5),
            layer_ids=[3],
            hook_point="layer_input",
            score_fn=_SpyScorer(),
            gate=ports["gate"],
            condition_layer_ids=ports["condition_layer_ids"],
        )
        pipeline = _steered_pipeline(model, tokenizer, [adapter])
        pipeline.generate(input_ids=PROMPTS[:1], max_new_tokens=4, **GEN_KWARGS)

        assert len(calls) == 1  # prefill only; the frozen decision stops further scoring
        frozen = adapter._gate.open_rows().clone()
        pipeline.generate(input_ids=PROMPTS[:1], max_new_tokens=4, **GEN_KWARGS)
        assert torch.equal(adapter._gate.open_rows(), frozen)  # re-armed and re-frozen per call
        assert len(calls) == 2


class TestLocationGuard:
    def test_layer_input_probe_on_layer_output_adapter_raises(self):
        model, tokenizer = _model_and_tokenizer()
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            **probe_condition(_probe([1])),  # probe location "layer_input"; default hook_point "layer_output"
        )
        with pytest.raises(ValueError, match="expects features at 'layer_input'.*hooks 'layer_output'"):
            adapter.steer(model, tokenizer)

    def test_layer_output_probe_on_layer_input_adapter_raises(self):
        model, tokenizer = _model_and_tokenizer()
        probe = Probe(
            model_type="llama", location="layer_output", pooling="mean",
            layer_ids=[1], weights={1: _unit_vector(8)}, bias=0.0,
        )
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(probe),
        )
        with pytest.raises(ValueError, match="expects features at 'layer_output'.*hooks 'layer_input'"):
            adapter.steer(model, tokenizer)

    def test_matching_location_passes(self):
        model, tokenizer = _model_and_tokenizer()
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(_probe([1])),
        )
        adapter.steer(model, tokenizer)


class TestFingerprintGuard:
    def test_probe_from_other_model_raises_and_escape_disarms(self):
        model_a, _ = _model_and_tokenizer(seed=0)
        model_b, tokenizer = _model_and_tokenizer(seed=1)
        probe = _probe([1], meta={"model_fingerprint": model_fingerprint(model_a)})

        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(probe),
        )
        with pytest.raises(ValueError, match="different model"):
            adapter.steer(model_b, tokenizer)

        disarmed = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(probe, allow_model_mismatch=True),
        )
        disarmed.steer(model_b, tokenizer)

    def test_matching_fingerprint_passes(self):
        model, tokenizer = _model_and_tokenizer()
        probe = _probe([1], meta={"model_fingerprint": model_fingerprint(model)})
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(probe),
        )
        adapter.steer(model, tokenizer)

    def test_hand_built_probe_with_empty_meta_never_trips(self):
        model_a, _ = _model_and_tokenizer(seed=0)
        model_b, tokenizer = _model_and_tokenizer(seed=1)
        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            hook_point="layer_input",
            **probe_condition(_probe([1])),
        )
        adapter.steer(model_b, tokenizer)


class TestLegacyScorers:
    def test_scorer_without_optional_attributes_passes_steer(self):
        model, tokenizer = _model_and_tokenizer()

        def scorer(hidden, layer_id, *, prompt_mask=None):
            return torch.zeros(hidden.size(0))

        adapter = ActivationAdapter(
            transform=RecordingTransform(),
            layer_ids=[3],
            gate=MultiKeyThresholdGate(threshold=0.5, comparator="larger", expected_keys={1}),
            condition_layer_ids=[1],
            score_fn=scorer,
        )
        adapter.steer(model, tokenizer)
