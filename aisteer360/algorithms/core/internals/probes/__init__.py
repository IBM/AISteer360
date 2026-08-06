"""Probes: calibrated affine readouts over model internals, with routing rules.

`Probe` is model-free feature math with canonical polarity (a score at or above zero means
present). `fit_probe` and `calibrate_bias` fit and calibrate probes from contrastive pairs;
`ProbeSet` scores many probes in one read-only forward; `ProbeSetFit` defers fitting to the
model a pipeline provides at steer time. `P`, `Rule`, and `RoutingRules` route per-row probe
decisions to consumer-defined actions.
"""
from .fitting import CalibrationSpec, ProbeFitSpec, calibrate_bias, fit_probe
from .probe import Probe
from .probe_set import ProbeSet, ProbeSetFit, Readout
from .rules import P, ProbePredicate, RoutingRules, Rule

__all__ = [
    "CalibrationSpec",
    "P",
    "Probe",
    "ProbeFitSpec",
    "ProbePredicate",
    "ProbeSet",
    "ProbeSetFit",
    "Readout",
    "RoutingRules",
    "Rule",
    "calibrate_bias",
    "fit_probe",
]
