"""Gate score-function tests: sign erasure and the signed cosine scorer.

Runs hub-free on deterministic tensors and a tiny randomly-initialized Llama. The sign-erasure
tests build their cluster geometry analytically (an on-axis component plus a unit vector
orthogonalized against the direction), so the characterization of the projected score as
`|cos(h, d)|` and its polarity inversion hold by construction rather than by model behavior.
"""
import pytest
import torch
import torch.nn.functional as F

from aisteer360.algorithms.state_control._common.condition_scorers import (
    CosineDirectionScorer,
    ProjectedCosineScorer,
    projected_cosine_similarity_tensor,
    rank_one_projector,
)
from aisteer360.algorithms.core.internals.pooling import masked_mean
from aisteer360.algorithms.state_control._common.selectors.condition_point import (
    ConditionPointSelector,
)
from aisteer360.algorithms.core.internals.data import ContrastivePairs
from aisteer360.algorithms.state_control._common.specs import (
    ConditionSearchSpec,
    VectorTrainSpec,
)
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

HIDDEN = 32
LAYERS = 4


def _unit_vector(seed: int, dim: int = HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _model_and_tokenizer(seed: int = 0):
    torch.manual_seed(seed)
    model = tiny_llama(num_layers=LAYERS, hidden=HIDDEN, heads=4)
    tokenizer = wordlevel_tokenizer()
    return model, tokenizer


def _orthonormal_rows(direction: torch.Tensor, num_rows: int, seed: int) -> torch.Tensor:
    """Unit rows orthogonal to `direction`, via Gram-Schmidt on seeded noise."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(num_rows, direction.numel(), generator=g)
    projected = noise - (noise @ direction).unsqueeze(-1) * direction
    return projected / projected.norm(dim=-1, keepdim=True)


class TestSignErasure:
    """The unsigned projected score erases direction sign; the signed cosine keeps it.

    Clusters mimic a mean-difference domain gate at the failing layer: positives moderately
    aligned with the direction (`0.3 * d`), negatives strongly anti-aligned (`-0.6 * d`), and
    unrelated content nearly orthogonal (`0.02 * d`), each plus a unit orthogonal component.
    The unrelated cluster keeps a small on-axis component on purpose: at exact orthogonality
    (`d @ h == 0`) the projected score is a 0/0 guarded only by the production epsilon and
    returns amplified float noise rather than 0, so the `|cos|` characterization below holds
    only away from exact orthogonality.
    """

    def setup_method(self):
        self.direction = _unit_vector(seed=7)
        self.positives = 0.3 * self.direction + _orthonormal_rows(self.direction, 4, seed=11)
        self.negatives = -0.6 * self.direction + _orthonormal_rows(self.direction, 4, seed=22)
        self.unrelated = 0.02 * self.direction + _orthonormal_rows(self.direction, 4, seed=33)
        self.projector = rank_one_projector(self.direction)

    def _projected(self, rows: torch.Tensor) -> torch.Tensor:
        return projected_cosine_similarity_tensor(rows, self.projector)

    def _signed(self, rows: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(rows, self.direction.unsqueeze(0), dim=-1)

    def test_projected_score_is_absolute_cosine(self):
        # tanh distortion is tiny at these magnitudes, so the projected score matches |cos|
        for rows in (self.positives, self.negatives, self.unrelated):
            assert torch.allclose(self._projected(rows), self._signed(rows).abs(), atol=0.005)

    def test_unsigned_score_inverts_polarity_and_opens_on_unrelated(self):
        proj_pos = self._projected(self.positives)
        proj_neg = self._projected(self.negatives)
        proj_unrel = self._projected(self.unrelated)
        # anti-aligned negatives outscore positives, so only "smaller" separates the classes
        assert proj_pos.max() < proj_neg.min()
        # every unrelated point lies below any separating threshold, i.e. opens the gate
        assert proj_unrel.max() < proj_pos.max() < proj_neg.min()

    def test_signed_score_fails_closed_on_unrelated(self):
        signed_pos = self._signed(self.positives)
        signed_neg = self._signed(self.negatives)
        signed_unrel = self._signed(self.unrelated)
        assert signed_pos.min() > 0 > signed_neg.max()
        assert signed_unrel.abs().max() < 0.05
        assert signed_pos.min() > signed_unrel.max()
        assert signed_pos.min() > signed_neg.max()


class TestCosineDirectionScorer:
    def setup_method(self):
        self.direction = _unit_vector(seed=5)
        g = torch.Generator().manual_seed(41)
        self.hidden = torch.randn(2, 4, HIDDEN, generator=g)
        self.mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]).bool()

    def _scorer(self, comparison_mode: str | None = None) -> CosineDirectionScorer:
        directions = {1: self.direction.unsqueeze(0)}
        if comparison_mode is None:
            return CosineDirectionScorer(directions)
        return CosineDirectionScorer(directions, comparison_mode=comparison_mode)

    def test_default_last_matches_manual_last_real_token(self):
        scores = self._scorer()(self.hidden, 1, prompt_mask=self.mask)
        last = torch.stack([self.hidden[0, 2], self.hidden[1, 3]])
        expected = F.cosine_similarity(last, self.direction.unsqueeze(0), dim=-1)
        assert torch.allclose(scores, expected, atol=1e-6)

    def test_mean_matches_masked_mean_pooling(self):
        scores = self._scorer("mean")(self.hidden, 1, prompt_mask=self.mask)
        pooled = masked_mean(self.hidden, self.mask)
        expected = F.cosine_similarity(pooled, self.direction.unsqueeze(0), dim=-1)
        assert torch.allclose(scores, expected, atol=1e-6)

    def test_missing_layer_returns_zeros(self):
        scores = self._scorer()(self.hidden, 3, prompt_mask=self.mask)
        assert torch.equal(scores, torch.zeros(2))

    def test_antiparallel_row_scores_negative_one(self):
        hidden = (-self.direction).view(1, 1, HIDDEN)
        scores = self._scorer()(hidden, 1)
        assert torch.allclose(scores, torch.tensor([-1.0]), atol=1e-5)


class TestSelectorScoreParam:
    def test_unknown_score_raises(self):
        model, tokenizer = _model_and_tokenizer()
        with pytest.raises(ValueError, match="projected_cosine"):
            ConditionPointSelector().select(
                model=model,
                tokenizer=tokenizer,
                condition_directions={0: torch.randn(HIDDEN)},
                data=ContrastivePairs(positives=["the cat"], negatives=["the dog"]),
                fit_spec=VectorTrainSpec(prompt_format="raw", location="layer_input"),
                search_spec=ConditionSearchSpec(),
                score="bogus",
            )
