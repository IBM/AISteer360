"""Output control component library.

Factors the output category into reusable components: candidate policies, per-candidate value
functions, full-vocabulary logit sources, sequence scorers, a segment-search driver, a phased
driver, composable stopping criteria, a linear-probe estimator, KV-cache utilities, and the
`PrefixKeyedProcessor` base for stateful logits processors.
"""
from .candidate_forward import CandidateForward
from .candidates import CandidatePolicy, rad_candidate_sizing, select_candidates
from .criteria import BudgetTokens, StopOnSubstring, StopOnTokens
from .drivers import Fixed, Frontier, Generated, PhasedDriver, SearchDriver, SegmentProposer
from .estimators import LinearProbe, LinearProbeEstimator
from .logit_sources import AuxModelSource, BaseLogitSource, CallableSource, PromptVariantSource
from .processors import (
    ConstraintProcessor,
    ContrastiveMixtureProcessor,
    Normalize,
    PrefixKeyedProcessor,
    ValueGuidedProcessor,
)
from aisteer360.algorithms.core.internals.data import LabeledExamples, as_labeled_examples
from .scorers import MajorityVoteScorer, MetricScorer, RewardModelScorer, SequenceScorer
from .values import (
    BaseCandidateValue,
    CallableValue,
    ClassifierValue,
    RewardModelValue,
    StepContext,
    SubspaceMarginValue,
)
