"""Tests for state control common components.

Tests cover:
- SteeringVector save/load round-trip
- ContrastivePairs validation
- as_contrastive_pairs helper
- make_token_mask for all scopes
- get_model_layer_list against test model fixtures
- AlwaysOpenGate behavior
- MultiKeyThresholdGate behavior
- CacheOnceGate behavior
- projected_cosine_similarity
- AdditiveTransform
- NormPreservingTransform
"""
import tempfile
from pathlib import Path

import pytest
import torch

from aisteer360.algorithms.state_control._common import (
    ContrastivePairs,
    SteeringVector,
    VectorTrainSpec,
    as_contrastive_pairs,
)
from aisteer360.algorithms.state_control._common.gates import AlwaysOpenGate
from aisteer360.algorithms.state_control._common.hook_utils import (
    extract_hidden_states,
    get_model_layer_list,
    replace_hidden_states,
)
from aisteer360.algorithms.state_control._common.token_scope import (
    compute_prompt_lens,
    make_token_mask,
)


class TestSteeringVector:
    """Tests for SteeringVector dataclass."""

    def test_save_load_roundtrip(self, tmp_path):
        """Test that save/load preserves all data with torch.Tensor directions."""
        directions = {
            0: torch.randn(768),
            1: torch.randn(768),
            5: torch.randn(768),
        }
        variances = {0: 0.85, 1: 0.72, 5: 0.91}

        original = SteeringVector(
            model_type="llama",
            directions=directions,
            explained_variances=variances,
        )

        save_path = str(tmp_path / "test_vector")
        original.save(save_path)

        loaded = SteeringVector.load(save_path)

        assert loaded.model_type == original.model_type
        assert set(loaded.directions.keys()) == set(original.directions.keys())
        assert set(loaded.explained_variances.keys()) == set(original.explained_variances.keys())

        for k in original.directions:
            # loaded directions are at least 2D [T, H]; compare squeezed for 1D originals
            expected = original.directions[k].float()
            if expected.ndim == 1:
                expected = expected.unsqueeze(0)
            torch.testing.assert_close(
                loaded.directions[k],
                expected,
                rtol=1e-5,
                atol=1e-5,
            )

        for k in original.explained_variances:
            assert abs(loaded.explained_variances[k] - original.explained_variances[k]) < 1e-6

    def test_save_adds_svec_extension(self, tmp_path):
        """Test that .svec extension is added if not present."""
        vec = SteeringVector(
            model_type="gpt2",
            directions={0: torch.zeros(128)},
            explained_variances={0: 0.5},
        )

        save_path = str(tmp_path / "no_extension")
        vec.save(save_path)

        assert Path(save_path + ".svec").exists()

    def test_to_moves_tensors(self):
        """Test that to() moves all direction tensors."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(64), 1: torch.randn(64)},
            explained_variances={0: 0.5, 1: 0.5},
        )

        # move to float16
        vec.to("cpu", torch.float16)

        for d in vec.directions.values():
            assert d.dtype == torch.float16

    def test_validate_empty_model_type_raises(self):
        """Test that empty model_type raises ValueError."""
        vec = SteeringVector(
            model_type="",
            directions={0: torch.zeros(64)},
            explained_variances={0: 0.5},
        )
        with pytest.raises(ValueError, match="model_type must be provided"):
            vec.validate()

    def test_validate_empty_directions_raises(self):
        """Test that empty directions raises ValueError."""
        vec = SteeringVector(
            model_type="llama",
            directions={},
            explained_variances={0: 0.5},
        )
        with pytest.raises(ValueError, match="directions must not be empty"):
            vec.validate()

    def test_num_tokens_and_is_positional_1d(self):
        """Test num_tokens and is_positional with 1D (non-positional) directions."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(1, 64)},  # T=1, H=64
            explained_variances={0: 1.0},
        )
        assert vec.num_tokens == 1
        assert vec.is_positional is False

    def test_num_tokens_and_is_positional_2d(self):
        """Test num_tokens and is_positional with 2D (positional) directions."""
        vec = SteeringVector(
            model_type="llama",
            directions={0: torch.randn(5, 64)},  # T=5, H=64
            explained_variances={0: 1.0},
        )
        assert vec.num_tokens == 5
        assert vec.is_positional is True

    def test_num_tokens_empty_directions(self):
        """Test num_tokens returns 0 for empty directions."""
        vec = SteeringVector(
            model_type="llama",
            directions={},
            explained_variances={},
        )
        assert vec.num_tokens == 0
        assert vec.is_positional is False


class TestContrastivePairs:
    """Tests for ContrastivePairs dataclass."""

    def test_valid_pairs(self):
        """Test creation with valid data."""
        pairs = ContrastivePairs(
            positives=["positive example 1", "positive example 2"],
            negatives=["negative example 1", "negative example 2"],
        )
        assert len(pairs.positives) == 2
        assert len(pairs.negatives) == 2
        assert pairs.prompts is None

    def test_valid_pairs_with_prompts(self):
        """Test creation with prompts."""
        pairs = ContrastivePairs(
            positives=["yes", "yes"],
            negatives=["no", "no"],
            prompts=["Is this good? ", "Is this bad? "],
        )
        assert len(pairs.prompts) == 2

    def test_empty_positives_raises(self):
        """Test that empty positives raises ValueError."""
        with pytest.raises(ValueError, match="at least one entry"):
            ContrastivePairs(positives=[], negatives=["neg"])

    def test_empty_negatives_raises(self):
        """Test that empty negatives raises ValueError."""
        with pytest.raises(ValueError, match="at least one entry"):
            ContrastivePairs(positives=["pos"], negatives=[])

    def test_unequal_lengths_raises(self):
        """Test that unequal positive/negative lengths raises ValueError."""
        with pytest.raises(ValueError, match="must have equal length"):
            ContrastivePairs(
                positives=["a", "b"],
                negatives=["c"],
            )

    def test_prompts_length_mismatch_raises(self):
        """Test that prompts with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="prompts must have the same length"):
            ContrastivePairs(
                positives=["a", "b"],
                negatives=["c", "d"],
                prompts=["prompt1"],  # wrong length
            )


class TestAsContrastivePairs:
    """Tests for as_contrastive_pairs helper function."""

    def test_passthrough_instance(self):
        """Test that existing instance is returned as-is."""
        pairs = ContrastivePairs(positives=["a"], negatives=["b"])
        result = as_contrastive_pairs(pairs)
        assert result is pairs

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "positives": ["pos1", "pos2"],
            "negatives": ["neg1", "neg2"],
        }
        result = as_contrastive_pairs(data)
        assert isinstance(result, ContrastivePairs)
        assert result.positives == tuple(data["positives"]) or list(result.positives) == data["positives"]

    def test_from_dict_with_prompts(self):
        """Test creation from dict with prompts."""
        data = {
            "positives": ["p1"],
            "negatives": ["n1"],
            "prompts": ["prompt "],
        }
        result = as_contrastive_pairs(data)
        assert result.prompts is not None

    def test_invalid_type_raises(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError, match="Expected ContrastivePairs or dict"):
            as_contrastive_pairs("invalid")

        with pytest.raises(TypeError, match="Expected ContrastivePairs or dict"):
            as_contrastive_pairs(123)


class TestVectorTrainSpec:
    """Tests for VectorTrainSpec dataclass."""

    def test_defaults(self):
        """Test default values."""
        spec = VectorTrainSpec()
        assert spec.method == "pca_pairwise"
        assert spec.accumulate == "all"
        assert spec.batch_size == 8

    def test_invalid_batch_size_raises(self):
        """Test that batch_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            VectorTrainSpec(batch_size=0)


class TestComputePromptLens:
    """Tests for compute_prompt_lens function.

    compute_prompt_lens returns seq_len for all items, regardless of padding.
    This ensures that with "after_prompt" scope, generation starts after all
    input positions (including pads), which is correct for KV-cached generation.
    """

    def test_no_padding(self):
        """Test prompt lens with no padding."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        lens = compute_prompt_lens(input_ids, pad_token_id=None)
        assert lens.tolist() == [5]

    def test_with_padding(self):
        """Test prompt lens with left padding returns seq_len for all items."""
        input_ids = torch.tensor([
            [0, 0, 1, 2, 3],  # 3 non-pad tokens but seq_len=5
            [0, 1, 2, 3, 4],  # 4 non-pad tokens but seq_len=5
        ])
        lens = compute_prompt_lens(input_ids, pad_token_id=0)
        # returns seq_len (5) for all items regardless of pad count
        assert lens.tolist() == [5, 5]

    def test_1d_input(self):
        """Test that 1D input is handled correctly."""
        input_ids = torch.tensor([1, 2, 3, 4])
        lens = compute_prompt_lens(input_ids, pad_token_id=None)
        assert lens.tolist() == [4]


class TestMakeTokenMask:
    """Tests for make_token_mask function."""

    def test_scope_all(self):
        """Test 'all' scope returns all True."""
        prompt_lens = torch.tensor([3, 4])
        mask = make_token_mask("all", seq_len=5, prompt_lens=prompt_lens)
        assert mask.shape == (2, 5)
        assert mask.all()

    def test_scope_after_prompt(self):
        """Test 'after_prompt' scope returns True only after prompt."""
        prompt_lens = torch.tensor([3, 4])
        mask = make_token_mask("after_prompt", seq_len=6, prompt_lens=prompt_lens)

        # batch 0: prompt_len=3, so positions 3,4,5 should be True
        assert mask[0].tolist() == [False, False, False, True, True, True]
        # batch 1: prompt_len=4, so positions 4,5 should be True
        assert mask[1].tolist() == [False, False, False, False, True, True]

    def test_scope_after_prompt_with_position_offset(self):
        """Test 'after_prompt' with position_offset for KV-cached generation."""
        prompt_lens = torch.tensor([10])

        # simulate KV-cached generation: seq_len=1, but we're at position 10 (first generated token)
        mask = make_token_mask("after_prompt", seq_len=1, prompt_lens=prompt_lens, position_offset=10)
        assert mask[0].tolist() == [True]  # position 10 >= prompt_len 10

        # simulate being at position 9 (still in prompt)
        mask = make_token_mask("after_prompt", seq_len=1, prompt_lens=prompt_lens, position_offset=9)
        assert mask[0].tolist() == [False]  # position 9 < prompt_len 10

        # simulate being at position 15 (well into generation)
        mask = make_token_mask("after_prompt", seq_len=1, prompt_lens=prompt_lens, position_offset=15)
        assert mask[0].tolist() == [True]  # position 15 >= prompt_len 10

    def test_scope_after_prompt_initial_pass_all_prompt(self):
        """Test 'after_prompt' on initial pass where seq_len == prompt_len."""
        prompt_lens = torch.tensor([5])

        # initial forward pass: seq_len == prompt_len, all positions are prompt
        mask = make_token_mask("after_prompt", seq_len=5, prompt_lens=prompt_lens, position_offset=0)
        assert mask[0].tolist() == [False, False, False, False, False]

    def test_scope_last_k(self):
        """Test 'last_k' scope returns True only for last k tokens."""
        prompt_lens = torch.tensor([3])
        mask = make_token_mask("last_k", seq_len=5, prompt_lens=prompt_lens, last_k=2)

        # last 2 positions (3, 4) should be True
        assert mask[0].tolist() == [False, False, False, True, True]

    def test_scope_last_k_ignores_position_offset(self):
        """Test that 'last_k' uses local positions, ignoring position_offset."""
        prompt_lens = torch.tensor([3])

        # last_k should work on local seq_len regardless of position_offset
        mask = make_token_mask("last_k", seq_len=5, prompt_lens=prompt_lens, last_k=2, position_offset=100)
        assert mask[0].tolist() == [False, False, False, True, True]

        # with seq_len=1, last_k=1 should always be True
        mask = make_token_mask("last_k", seq_len=1, prompt_lens=prompt_lens, last_k=1, position_offset=100)
        assert mask[0].tolist() == [True]

    def test_last_k_none_raises(self):
        """Test that last_k=None with 'last_k' scope raises ValueError."""
        with pytest.raises(ValueError, match="last_k must be >= 1"):
            make_token_mask("last_k", seq_len=5, prompt_lens=torch.tensor([3]), last_k=None)

    def test_unknown_scope_raises(self):
        """Test that unknown scope raises ValueError."""
        with pytest.raises(ValueError, match="Unknown token scope"):
            make_token_mask("invalid", seq_len=5, prompt_lens=torch.tensor([3]))


class TestExtractHiddenStates:
    """Tests for extract_hidden_states function."""

    def test_from_positional_args(self):
        """Test extraction from positional args."""
        hidden = torch.randn(2, 10, 64)
        result = extract_hidden_states((hidden, "other"), {})
        assert result is hidden

    def test_from_kwargs(self):
        """Test extraction from kwargs."""
        hidden = torch.randn(2, 10, 64)
        result = extract_hidden_states((), {"hidden_states": hidden})
        assert result is hidden

    def test_not_found(self):
        """Test that None is returned if not found."""
        result = extract_hidden_states((), {"other_key": "value"})
        assert result is None


class TestReplaceHiddenStates:
    """Tests for replace_hidden_states function."""

    def test_replace_in_positional_args(self):
        """Test replacement in positional args."""
        old_hidden = torch.randn(2, 10, 64)
        new_hidden = torch.randn(2, 10, 64)
        other_arg = "other"

        new_args, new_kwargs = replace_hidden_states(
            (old_hidden, other_arg), {}, new_hidden
        )

        assert new_args[0] is new_hidden
        assert new_args[1] == other_arg
        assert new_kwargs == {}

    def test_replace_in_kwargs(self):
        """Test replacement in kwargs."""
        old_hidden = torch.randn(2, 10, 64)
        new_hidden = torch.randn(2, 10, 64)
        original_kwargs = {"hidden_states": old_hidden, "other": "value"}

        new_args, new_kwargs = replace_hidden_states(
            (), original_kwargs, new_hidden
        )

        assert new_args == ()
        assert new_kwargs["hidden_states"] is new_hidden
        assert new_kwargs["other"] == "value"


class TestGetModelLayerList:
    """Tests for get_model_layer_list function."""

    def test_llama_style_model(self, model_and_tokenizer):
        """Test layer extraction from llama-style model."""
        model, _ = model_and_tokenizer
        model_type = model.config.model_type

        # skip if not the right architecture
        if not (hasattr(model, "model") and hasattr(model.model, "layers")) and not (
            hasattr(model, "transformer") and hasattr(model.transformer, "h")
        ):
            pytest.skip(f"Model {model_type} has unknown architecture")

        modules, names = get_model_layer_list(model)

        assert len(modules) > 0
        assert len(names) == len(modules)

        # check naming convention
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            assert all(n.startswith("model.layers.") for n in names)
        else:
            assert all(n.startswith("transformer.h.") for n in names)


class TestAlwaysOpenGate:
    """Tests for AlwaysOpenGate."""

    def test_is_open_always_true(self):
        """Test that is_open() always returns True."""
        gate = AlwaysOpenGate()
        assert gate.is_open() is True

        gate.update(0.5)
        assert gate.is_open() is True

        gate.reset()
        assert gate.is_open() is True

    def test_is_ready_always_true(self):
        """Test that is_ready() returns True."""
        gate = AlwaysOpenGate()
        assert gate.is_ready() is True

    def test_update_does_nothing(self):
        """Test that update() is a no-op."""
        gate = AlwaysOpenGate()
        # should not raise
        gate.update(0.5, key=0)
        gate.update(-1.0, key=None)
        assert gate.is_open() is True

    def test_reset_does_nothing(self):
        """Test that reset() is a no-op."""
        gate = AlwaysOpenGate()
        # should not raise
        gate.reset()
        assert gate.is_open() is True


class TestMultiKeyThresholdGate:
    """Tests for MultiKeyThresholdGate."""

    def test_single_key_larger(self):
        """Test single key with 'larger' comparator."""
        from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate

        gate = MultiKeyThresholdGate(threshold=0.5, comparator="larger")

        gate.update(0.6, key=0)  # passes (0.6 >= 0.5)
        assert gate.is_open() is True

        gate.reset()
        gate.update(0.4, key=0)  # fails (0.4 < 0.5)
        assert gate.is_open() is False

    def test_single_key_smaller(self):
        """Test single key with 'smaller' comparator."""
        from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate

        gate = MultiKeyThresholdGate(threshold=0.5, comparator="smaller")

        gate.update(0.4, key=0)  # passes (0.4 <= 0.5)
        assert gate.is_open() is True

        gate.reset()
        gate.update(0.6, key=0)  # fails (0.6 > 0.5)
        assert gate.is_open() is False

    def test_multiple_keys_any(self):
        """Test multiple keys with 'any' aggregation."""
        from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate

        gate = MultiKeyThresholdGate(
            threshold=0.5,
            comparator="larger",
            expected_keys={0, 1},
            aggregate="any",
        )

        gate.update(0.3, key=0)  # fails
        assert gate.is_open() is False

        gate.update(0.7, key=1)  # passes
        assert gate.is_open() is True  # any(False, True) = True

    def test_multiple_keys_all(self):
        """Test multiple keys with 'all' aggregation."""
        from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate

        gate = MultiKeyThresholdGate(
            threshold=0.5,
            comparator="larger",
            expected_keys={0, 1},
            aggregate="all",
        )

        gate.update(0.7, key=0)  # passes
        gate.update(0.3, key=1)  # fails
        assert gate.is_open() is False  # all(True, False) = False

        gate.reset()
        gate.update(0.7, key=0)  # passes
        gate.update(0.6, key=1)  # passes
        assert gate.is_open() is True  # all(True, True) = True

    def test_is_ready_with_expected_keys(self):
        """Test is_ready() with expected_keys."""
        from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate

        gate = MultiKeyThresholdGate(
            threshold=0.5,
            comparator="larger",
            expected_keys={0, 1, 2},
        )

        assert gate.is_ready() is False
        gate.update(0.6, key=0)
        assert gate.is_ready() is False
        gate.update(0.6, key=1)
        assert gate.is_ready() is False
        gate.update(0.6, key=2)
        assert gate.is_ready() is True

    def test_empty_decisions_returns_false(self):
        """Test that is_open() returns False when no decisions made."""
        from aisteer360.algorithms.state_control._common.gates import MultiKeyThresholdGate

        gate = MultiKeyThresholdGate(threshold=0.5, comparator="larger")
        assert gate.is_open() is False


class TestCacheOnceGate:
    """Tests for CacheOnceGate."""

    def test_caches_decision_when_ready(self):
        """Test that decision is cached once inner gate is ready."""
        from aisteer360.algorithms.state_control._common.gates import CacheOnceGate, MultiKeyThresholdGate

        inner = MultiKeyThresholdGate(threshold=0.5, comparator="larger")
        gate = CacheOnceGate(inner)

        gate.update(0.6, key=0)  # inner becomes ready and passes
        assert gate._cached is True
        assert gate.is_open() is True

        # even after reset of inner (via update), cached stays
        gate.update(0.3, key=0)  # this would fail threshold, but cached
        assert gate.is_open() is True  # still True because cached

    def test_reset_clears_cache(self):
        """Test that reset() clears the cached decision."""
        from aisteer360.algorithms.state_control._common.gates import CacheOnceGate, MultiKeyThresholdGate

        inner = MultiKeyThresholdGate(threshold=0.5, comparator="larger")
        gate = CacheOnceGate(inner)

        gate.update(0.6, key=0)
        assert gate._cached is True

        gate.reset()
        assert gate._cached is None

    def test_freezes_on_first_ready(self):
        """Test that only first ready state is cached."""
        from aisteer360.algorithms.state_control._common.gates import CacheOnceGate, MultiKeyThresholdGate

        inner = MultiKeyThresholdGate(
            threshold=0.5,
            comparator="larger",
            expected_keys={0, 1},
        )
        gate = CacheOnceGate(inner)

        gate.update(0.3, key=0)  # fails
        assert gate._cached is None  # not ready yet

        gate.update(0.7, key=1)  # passes, now ready
        assert gate._cached is True  # any(False, True) = True
        assert gate.is_ready() is True


class TestProjectedCosineSimilarity:
    """Tests for projected_cosine_similarity function."""

    def test_known_values(self):
        """Test against known values."""
        from aisteer360.algorithms.state_control._common.gates.scores import projected_cosine_similarity

        # create a simple case
        hidden = torch.tensor([1.0, 0.0, 0.0])
        direction = torch.tensor([1.0, 0.0, 0.0])

        # projector = outer(d, d) / dot(d, d) = [[1,0,0],[0,0,0],[0,0,0]]
        projector = torch.outer(direction, direction) / (direction @ direction + 1e-8)

        # projection = tanh(P @ h) = tanh([1,0,0]) = [tanh(1), 0, 0]
        # sim = dot(h, proj) / (norm(h) * norm(proj))
        #     = tanh(1) / (1 * tanh(1)) = 1.0

        sim = projected_cosine_similarity(hidden, projector)
        assert abs(sim - 1.0) < 0.01  # should be close to 1

    def test_orthogonal_vectors(self):
        """Test with orthogonal vectors."""
        from aisteer360.algorithms.state_control._common.gates.scores import projected_cosine_similarity

        hidden = torch.tensor([1.0, 0.0, 0.0])
        direction = torch.tensor([0.0, 1.0, 0.0])
        projector = torch.outer(direction, direction) / (direction @ direction + 1e-8)

        # P @ h = [[0,0,0],[0,1,0],[0,0,0]] @ [1,0,0] = [0,0,0]
        # projection is zero, so tanh(0) = 0
        # sim with zero vector -> handle division by zero gracefully

        sim = projected_cosine_similarity(hidden, projector)
        # should be 0 or very small due to the 1e-8 epsilon
        assert abs(sim) < 0.1


class TestAdditiveTransform:
    """Tests for AdditiveTransform."""

    def test_applies_direction_with_mask(self):
        """Test that direction is added only where mask is True."""
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 4, 8)  # [B=1, T=4, H=8]
        directions = {0: torch.ones(8)}  # layer 0: all ones
        transform = AdditiveTransform(directions, strength=2.0)

        # mask only positions 1 and 3
        mask = torch.tensor([[False, True, False, True]])

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # positions 0, 2 should be zeros
        assert result[0, 0, :].sum().item() == 0
        assert result[0, 2, :].sum().item() == 0

        # positions 1, 3 should be 2.0 * ones = 2.0 per element, 8 elements
        assert result[0, 1, :].sum().item() == 16.0
        assert result[0, 3, :].sum().item() == 16.0

    def test_no_direction_returns_unchanged(self):
        """Test that missing layer direction returns hidden unchanged."""
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform

        hidden = torch.randn(2, 5, 16)
        transform = AdditiveTransform({0: torch.randn(16)}, strength=1.0)
        mask = torch.ones(2, 5, dtype=torch.bool)

        # layer 99 not in directions
        result = transform.apply(hidden, layer_id=99, token_mask=mask)
        assert torch.equal(result, hidden)

    def test_strength_scaling(self):
        """Test that strength parameter scales correctly."""
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 1, 4)
        directions = {0: torch.tensor([1.0, 2.0, 3.0, 4.0])}
        transform = AdditiveTransform(directions, strength=0.5)
        mask = torch.ones(1, 1, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)
        expected = torch.tensor([[[0.5, 1.0, 1.5, 2.0]]])
        torch.testing.assert_close(result, expected)

    def test_positional_mode_with_alignment(self):
        """Test positional mode (T>1) with alignment parameter."""
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 6, 4)  # [B=1, T=6, H=4]
        # positional steering vector with T=3 tokens
        directions = {0: torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # token 0
            [0.0, 2.0, 0.0, 0.0],  # token 1
            [0.0, 0.0, 3.0, 0.0],  # token 2
        ])}
        # inject starting at position 2
        transform = AdditiveTransform(directions, strength=1.0, alignment=2)
        mask = torch.ones(1, 6, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # positions 0, 1 should be unchanged (before alignment)
        assert result[0, 0, :].sum().item() == 0
        assert result[0, 1, :].sum().item() == 0
        # positions 2, 3, 4 should get the steering vectors
        torch.testing.assert_close(result[0, 2, :], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(result[0, 3, :], torch.tensor([0.0, 2.0, 0.0, 0.0]))
        torch.testing.assert_close(result[0, 4, :], torch.tensor([0.0, 0.0, 3.0, 0.0]))
        # position 5 should be unchanged (after steering vector ends)
        assert result[0, 5, :].sum().item() == 0

    def test_positional_mode_clips_at_seq_end(self):
        """Test that positional mode clips steering vectors at sequence end."""
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 4, 4)  # [B=1, T=4, H=4]
        # steering vector with T=3, but aligned at position 2 so only 2 fit
        directions = {0: torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],  # won't fit
        ])}
        transform = AdditiveTransform(directions, strength=1.0, alignment=2)
        mask = torch.ones(1, 4, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # only positions 2, 3 should get steering (position 4 is out of bounds)
        torch.testing.assert_close(result[0, 2, :], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(result[0, 3, :], torch.tensor([0.0, 2.0, 0.0, 0.0]))

    def test_positional_mode_skips_when_out_of_range(self):
        """Test that positional mode returns unchanged when alignment is beyond seq_len."""
        from aisteer360.algorithms.state_control._common.transforms import AdditiveTransform

        hidden = torch.zeros(1, 3, 4)  # [B=1, T=3, H=4]
        # use T=2 to trigger positional mode (T>1)
        directions = {0: torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])}
        # alignment at position 5, but seq_len is only 3
        transform = AdditiveTransform(directions, strength=1.0, alignment=5)
        mask = torch.ones(1, 3, dtype=torch.bool)

        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # should return unchanged since alignment is beyond sequence
        assert result.sum().item() == 0


class TestNormPreservingTransform:
    """Tests for NormPreservingTransform."""

    def test_preserves_norm_when_increased(self):
        """Test that norm is preserved when it would increase."""
        from aisteer360.algorithms.state_control._common.transforms import (
            AdditiveTransform,
            NormPreservingTransform,
        )

        # start with unit norm vectors
        hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # norm = 1
        directions = {0: torch.tensor([0.0, 2.0, 0.0, 0.0])}  # would add norm
        inner = AdditiveTransform(directions, strength=1.0)
        transform = NormPreservingTransform(inner)

        mask = torch.ones(1, 1, dtype=torch.bool)
        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # original norm = 1, after addition norm would be sqrt(1 + 4) = sqrt(5)
        # should be scaled back to norm = 1
        result_norm = result.norm(dim=-1)
        torch.testing.assert_close(result_norm, torch.tensor([[1.0]]), rtol=1e-5, atol=1e-5)

    def test_does_not_scale_when_norm_decreases(self):
        """Test that scaling doesn't happen when norm decreases."""
        from aisteer360.algorithms.state_control._common.transforms import (
            AdditiveTransform,
            NormPreservingTransform,
        )

        # large initial norm
        hidden = torch.tensor([[[3.0, 0.0, 0.0, 0.0]]])  # norm = 3
        directions = {0: torch.tensor([-2.0, 0.0, 0.0, 0.0])}  # subtracts
        inner = AdditiveTransform(directions, strength=1.0)
        transform = NormPreservingTransform(inner)

        mask = torch.ones(1, 1, dtype=torch.bool)
        result = transform.apply(hidden, layer_id=0, token_mask=mask)

        # after addition: [1, 0, 0, 0], norm = 1 < 3
        # should NOT be scaled (norm decreased)
        expected = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        torch.testing.assert_close(result, expected)

    def test_raises_on_nan(self):
        """Test that NaN detection raises ValueError."""
        from aisteer360.algorithms.state_control._common.transforms import NormPreservingTransform
        from aisteer360.algorithms.state_control._common.transforms.base import BaseTransform

        class NaNTransform(BaseTransform):
            def apply(self, hidden_states, *, layer_id, token_mask, **kwargs):
                return torch.tensor([[[float("nan")]]])

        transform = NormPreservingTransform(NaNTransform())
        mask = torch.ones(1, 1, dtype=torch.bool)

        with pytest.raises(ValueError, match="NaN or Inf detected"):
            transform.apply(torch.ones(1, 1, 1), layer_id=0, token_mask=mask)


class TestLayerHeuristics:
    """Tests for layer heuristics functions."""

    def test_late_third(self):
        """Test late_third returns correct layer range."""
        from aisteer360.algorithms.state_control._common.selectors import late_third

        # 12 layers -> last third is layers 8-11
        result = late_third(12)
        assert result == [8, 9, 10, 11]

        # 24 layers -> last third is layers 16-23
        result = late_third(24)
        assert result == list(range(16, 24))

        # 3 layers -> last third is layer 2
        result = late_third(3)
        assert result == [2]
