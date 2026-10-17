"""Venue-matched pre-fitted artifact identity on engine layouts (model_ref and model_type)."""
import pytest
import torch

from steerability.algorithms.core.execution import ModelFacts
from steerability.algorithms.core.internals.probes.probe import Probe
from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
from steerability.algorithms.output_control.routed_decoding import P, Route, RoutedDecoding, Router
from steerability.algorithms.output_control.routed_decoding.actions import respond
from tests.utils.tiny_models import wordlevel_tokenizer

HIDDEN = 16


class _EngineSession:
    """Engine-session double: layout facts only, no live model."""

    def __init__(self, model_ref="org/served-model", model_type="llama"):
        self._model_ref = model_ref
        self._model_type = model_type

    @property
    def layout(self) -> ModelFacts:
        return ModelFacts(
            num_layers=2, hidden_size=HIDDEN, num_attention_heads=2, head_dim=HIDDEN // 2,
            dtype="float32", model_fingerprint="c" * 16,
            model_type=self._model_type, model_ref=self._model_ref, fingerprint_grade="config",
        )


def _probe(meta=None, model_type="llama") -> Probe:
    return Probe(
        model_type=model_type, location="layer_input", pooling="mean",
        layer_ids=[0], weights={0: torch.zeros(HIDDEN)}, bias=1e9, meta=dict(meta or {}),
    )


def _routed(probe) -> RoutedDecoding:
    return RoutedDecoding(
        probes=ProbeSet({"p": probe}),
        rules=Router(routes=[Route("r", when=P("p"), action=respond("x"))]),
    )


class TestEngineModelRefComparison:

    def test_matching_recorded_ref_accepts(self):
        control = _routed(_probe(meta={"model_ref": "org/served-model"}))
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_unrecorded_ref_is_exempt(self):
        control = _routed(_probe(meta={"model_fingerprint": "a" * 16}))
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_differing_recorded_ref_raises_naming_the_served_reference(self):
        control = _routed(_probe(meta={"model_ref": "org/other-model"}))
        with pytest.raises(ValueError, match="model reference.*'org/served-model'"):
            control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_escape_hatch_skips_the_ref_comparison(self):
        control = _routed(_probe(meta={"model_ref": "org/other-model"}))
        control.allow_model_mismatch = True
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_known_model_type_mismatch_raises(self):
        control = _routed(_probe(model_type="gpt2"))
        with pytest.raises(ValueError, match="model_type 'gpt2'.*serves 'llama'"):
            control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())

    def test_unknown_model_type_is_exempt_on_engines(self):
        control = _routed(_probe(model_type="unknown"))
        control.steer(model=None, tokenizer=wordlevel_tokenizer(), session=_EngineSession())


class TestFingerprintGrades:
    """Fingerprints are compared only at the weights grade; config-grade layouts compare `model_ref`."""

    def _vector(self):
        from steerability.algorithms.state_control.common.steering_vector import SteeringVector

        return SteeringVector(model_type="llama", directions={0: torch.ones(1, HIDDEN)})

    def _resolve(self, provenance, artifact_class, session, policy="strict"):
        from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed

        source = VerifiedPrecomputed(self._vector(), provenance=provenance, artifact_class=artifact_class,
                                     policy=policy)
        return source.resolve(None, None, session=session)

    def test_config_grade_layout_accepts_a_matching_model_ref(self):
        provenance = {"model_fingerprint": "a" * 16, "model_ref": "org/served-model"}
        assert self._resolve(provenance, "calibrated", _EngineSession()).directions[0].shape == (1, HIDDEN)

    def test_config_grade_layout_rejects_another_model_ref_for_a_calibrated_artifact(self):
        provenance = {"model_fingerprint": "a" * 16, "model_ref": "org/other-model"}
        with pytest.raises(ValueError, match="'org/other-model' but this backend serves 'org/served-model'"):
            self._resolve(provenance, "calibrated", _EngineSession())

    def test_config_grade_layout_warns_for_a_direction_artifact(self):
        provenance = {"model_ref": "org/other-model"}
        with pytest.warns(UserWarning, match="may transfer across fine-tunes"):
            self._resolve(provenance, "direction", _EngineSession())

    def test_weights_grade_layout_compares_fingerprints(self):
        class _InProcessLayoutSession(_EngineSession):
            @property
            def layout(self) -> ModelFacts:
                return ModelFacts(
                    num_layers=2, hidden_size=HIDDEN, num_attention_heads=2, head_dim=HIDDEN // 2,
                    dtype="float32", model_fingerprint="c" * 16, model_type="llama", model_ref="org/served-model",
                )

        with pytest.raises(ValueError, match="different model"):
            self._resolve({"model_fingerprint": "a" * 16, "model_ref": "org/served-model"}, "calibrated",
                          _InProcessLayoutSession())
        assert self._resolve({"model_fingerprint": "c" * 16}, "calibrated", _InProcessLayoutSession()) is not None

    def test_scoped_session_reads_in_process_from_the_grade(self):
        from steerability.algorithms.core.execution import ModelAccess
        from steerability.algorithms.core.execution.session_utils import ScopedSession

        assert ScopedSession(_EngineSession(), "C", ModelAccess.FACTS).in_process is False

    def test_gate_readout_fingerprint_is_not_compared_on_a_config_grade_layout(self):
        from steerability.algorithms.state_control.common.gating import Evidence, Gate, PerKeyThreshold
        from steerability.algorithms.state_control.common.gating import ProjectedCosineReadout
        from steerability.algorithms.state_control.common.specs import Intervention
        from steerability.algorithms.state_control.common.transforms import AdditiveTransform

        readout = ProjectedCosineReadout({0: torch.ones(1, HIDDEN)}, model_fingerprint="a" * 16)
        gate = Gate(Evidence((0,), readout), PerKeyThreshold(threshold=0.0, comparator="ge"))
        intervention = Intervention(layers=(1,), transform=AdditiveTransform({1: torch.ones(1, HIDDEN)}), gate=gate)
        layout = _EngineSession().layout
        intervention._validate_readout(readout, layout)
        with pytest.raises(ValueError, match="different model"):
            intervention._validate_readout(readout, ModelFacts(**{
                **{field: getattr(layout, field) for field in ModelFacts.__slots__}, "fingerprint_grade": "weights",
            }))
