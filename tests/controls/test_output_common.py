"""Tests for the `output_control/_common` component library (output multiplicity design, P2).

Hub-free: uses tiny randomly-initialized models and scripted values/scorers/automata. Covers the
statefulness contract, candidate policies, the value-guided step shape, the contrastive-mixture
distribution shape, KV-cache round-trips, the linear-probe estimator, the segment and phase drivers,
composable criteria, and the constraint integration point.
"""
import math

import pytest
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList

from aisteer360.algorithms.output_control._common.candidates import (
    rad_candidate_sizing,
    select_candidates,
)
from aisteer360.algorithms.output_control._common.criteria import (
    BudgetTokens,
    StopOnSubstring,
    StopOnTokens,
)
from aisteer360.algorithms.output_control._common.estimators.linear_probe import (
    LinearProbe,
    LinearProbeEstimator,
)
from aisteer360.algorithms.output_control._common.kv_cache import repeat_cache, select_cache
from aisteer360.algorithms.output_control._common.logit_sources import BaseLogitSource
from aisteer360.algorithms.output_control._common.drivers.phased import (
    Fixed,
    Generated,
    PhasedDriver,
)
from aisteer360.algorithms.output_control._common.processors.base import PrefixKeyedProcessor
from aisteer360.algorithms.output_control._common.processors.constraint import ConstraintProcessor
from aisteer360.algorithms.output_control._common.processors.contrastive_mixture import (
    ContrastiveMixtureProcessor,
)
from aisteer360.algorithms.output_control._common.processors.value_guided import (
    ValueGuidedProcessor,
    _normalize,
)
from aisteer360.algorithms.output_control._common.drivers.search import SearchDriver
from aisteer360.algorithms.output_control._common.drivers.frontier import Frontier
from aisteer360.algorithms.core.internals.data import LabeledExamples
from aisteer360.algorithms.output_control._common.values.base import (
    BaseCandidateValue,
    StepContext,
)
from aisteer360.algorithms.output_control._common.candidate_forward import CandidateForward
from aisteer360.algorithms.output_control._common.values.subspace_margin import SubspaceMarginValue
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


# scripted value: returns a fixed per-candidate vector regardless of prefix
class _ScriptedValue(BaseCandidateValue):
    supports_batching = True

    def __init__(self, values):
        self._values = torch.tensor(values, dtype=torch.float32)

    def score(self, ctx: StepContext) -> torch.Tensor:
        return self._values.unsqueeze(0).to(ctx.candidate_ids.device)


# PrefixKeyedProcessor
class TestPrefixKeyedProcessor:
    def _proc(self, log):
        class _P(PrefixKeyedProcessor):
            def reset_state(self, input_ids):
                log.append(("reset", input_ids.shape[1]))

            def process(self, input_ids, scores):
                log.append(("process", input_ids.shape[1]))
                return scores

        return _P()

    def test_extends_keeps_state(self):
        log = []
        p = self._proc(log)
        p(torch.tensor([[1, 2]]), torch.zeros(1, VOCAB))
        p(torch.tensor([[1, 2, 3]]), torch.zeros(1, VOCAB))  # extends -> no reset
        resets = [e for e in log if e[0] == "reset"]
        assert len(resets) == 1  # only the first (no prior state)

    def test_rewind_triggers_reset(self):
        log = []
        p = self._proc(log)
        p(torch.tensor([[1, 2, 3]]), torch.zeros(1, VOCAB))
        p(torch.tensor([[1, 2]]), torch.zeros(1, VOCAB))  # shorter -> reset
        resets = [e for e in log if e[0] == "reset"]
        assert len(resets) == 2

    def test_row_permutation_triggers_reset(self):
        log = []
        p = self._proc(log)
        p(torch.tensor([[1, 2, 3]]), torch.zeros(1, VOCAB))
        p(torch.tensor([[9, 9, 9]]), torch.zeros(1, VOCAB))  # different prefix -> reset
        assert len([e for e in log if e[0] == "reset"]) == 2


# candidates
class TestCandidates:
    def test_top_k_batch_math(self):
        scores = torch.tensor([[0.0, 5.0, 1.0, 3.0], [2.0, 0.0, 4.0, 1.0]])
        ids, vals = select_candidates(scores, "top_k", k=2)
        assert ids.shape == (2, 2)
        assert set(ids[0].tolist()) == {1, 3}
        assert set(ids[1].tolist()) == {2, 0}

    def test_top_p_raises_on_batch(self):
        with pytest.raises(ValueError, match="batch size 1"):
            select_candidates(torch.zeros(2, VOCAB), "top_p", p=0.9)

    def test_surviving_raises_on_batch(self):
        with pytest.raises(ValueError, match="batch size 1"):
            select_candidates(torch.zeros(2, VOCAB), "surviving")

    def test_surviving_is_finite_set_after_mask(self):
        scores = torch.full((1, 6), float("-inf"))
        scores[0, [1, 4]] = torch.tensor([2.0, 3.0])
        ids, _ = select_candidates(scores, "surviving")
        assert sorted(ids[0].tolist()) == [1, 4]

    def test_rad_candidate_sizing_totality(self):
        assert rad_candidate_sizing({}) == {"policy": "top_k", "k": 20, "p": None}
        assert rad_candidate_sizing({"top_k": 5}) == {"policy": "top_k", "k": 5, "p": None}
        assert rad_candidate_sizing({"top_p": 0.9}) == {"policy": "top_p", "k": None, "p": 0.9}
        # both -> top_k precedence
        assert rad_candidate_sizing({"top_k": 5, "top_p": 0.9})["policy"] == "top_k"


# ValueGuidedProcessor
class TestValueGuidedProcessor:
    def test_minmax_mask_math(self):
        # scores favor tokens 0,1,2; value ranks them 0<1<2
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0, 0.0]])
        value = _ScriptedValue([0.0, 1.0, 2.0])  # for candidates (top-3 -> ids 0,1,2)
        proc = ValueGuidedProcessor(
            value, policy="top_k", k=3, beta=10.0, normalize="minmax",
            mask_non_candidates=True, lm_tokenizer=None,
        )
        out = proc(torch.tensor([[9]]), scores.clone())
        # non-candidates (ids 3,4) -> -inf
        assert out[0, 3] == float("-inf")
        assert out[0, 4] == float("-inf")
        # minmax of [0,1,2] = [0, 0.5, 1]; += beta*that
        assert out[0, 0] == pytest.approx(3.0 + 10.0 * 0.0, abs=1e-4)
        assert out[0, 1] == pytest.approx(2.0 + 10.0 * 0.5, abs=1e-4)
        assert out[0, 2] == pytest.approx(1.0 + 10.0 * 1.0, abs=1e-4)

    def test_degenerate_minmax_is_half(self):
        v = _normalize(torch.tensor([[4.0, 4.0, 4.0]]), "minmax", invert=False)
        assert torch.allclose(v, torch.full_like(v, 0.5))

    def test_invert(self):
        v = _normalize(torch.tensor([[0.0, 1.0]]), "minmax", invert=True)
        assert torch.allclose(v, torch.tensor([[1.0, 0.0]]))

    def test_surviving_forces_mask_off(self):
        value = _ScriptedValue([1.0, 1.0])
        proc = ValueGuidedProcessor(
            value, policy="surviving", normalize="softmax", mask_non_candidates=True,
        )
        assert proc.mask_non_candidates is False  # forced off for surviving


# ContrastiveMixtureProcessor
class _ConstLogitSource(BaseLogitSource):
    def __init__(self, logprobs_vec):
        self._lp = torch.tensor(logprobs_vec, dtype=torch.float32)

    def logprobs(self, prefix_ids):
        return self._lp.unsqueeze(0).to(prefix_ids.device)


class TestContrastiveMixture:
    def test_mixture_arithmetic_logspace(self):
        base = torch.tensor([[0.0, 0.0]])  # uniform base -> log p = -log 2 each
        src = _ConstLogitSource([math.log(0.9), math.log(0.1)])
        proc = ContrastiveMixtureProcessor([(src, 2.0)], base_weight=1.0)
        out = proc(torch.tensor([[7]]), base.clone())
        base_lp = math.log(0.5)
        assert out[0, 0] == pytest.approx(base_lp + 2.0 * math.log(0.9), abs=1e-4)
        assert out[0, 1] == pytest.approx(base_lp + 2.0 * math.log(0.1), abs=1e-4)

    def test_alpha_mask_keeps_plausible(self):
        # base strongly prefers token 0; alpha mask should kill token 1
        base = torch.tensor([[5.0, 0.0, 0.0]])
        src = _ConstLogitSource([0.0, 0.0, 0.0])
        proc = ContrastiveMixtureProcessor([(src, 1.0)], base_weight=1.0, alpha=0.5)
        out = proc(torch.tensor([[7]]), base.clone())
        assert out[0, 0] > float("-inf")
        assert out[0, 1] == float("-inf")
        assert out[0, 2] == float("-inf")


# kv_cache
class TestKVCache:
    def test_repeat_then_select_round_trip(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        ids = torch.tensor([[0, 3, 4]])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
        cache = out.past_key_values
        repeated = repeat_cache(cache, 3)
        # select the first repeated slice back -> matches a single-row cache length
        selected = select_cache(repeated, torch.tensor([0]))
        legacy = selected.to_legacy_cache() if hasattr(selected, "to_legacy_cache") else selected
        # first layer key tensor now has batch dim 1
        assert legacy[0][0].shape[0] == 1


# LinearProbeEstimator
class TestLinearProbeEstimator:
    def test_golden_probe(self, tmp_path):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        data = LabeledExamples(
            positives=["the cat sat", "the dog ran", "the cat ran on"],
            negatives=["mat on fast", "span attention", "fast mat sat"],
        )
        est = LinearProbeEstimator(pooling="last_token")
        probe = est.fit(model, tokenizer, data=data, batch_size=2, max_length=16)
        assert probe.direction.shape == probe.midpoint.shape
        # direction normalized
        assert torch.norm(probe.direction).item() == pytest.approx(1.0, abs=1e-4)

    def test_no_file_written_without_path(self, tmp_path):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        data = LabeledExamples(
            positives=["the cat sat", "the dog ran", "the cat ran"],
            negatives=["mat on fast", "span attention", "fast mat on"],
        )
        est = LinearProbeEstimator()
        est.fit(model, tokenizer, data=data, batch_size=2, max_length=16)
        assert list(tmp_path.iterdir()) == []  # nothing written

    def test_save_load_round_trip(self, tmp_path):
        probe = LinearProbe(direction=torch.tensor([1.0, 0.0]), midpoint=torch.tensor([0.5, 0.5]))
        path = str(tmp_path / "p")
        probe.save(path)
        loaded = LinearProbe.load(path)
        assert torch.allclose(loaded.direction, probe.direction)
        assert torch.allclose(loaded.midpoint, probe.midpoint)


# SearchDriver
def _scripted_scorer_favoring(target):
    def scorer(prompt, continuations, params):
        return [float(c.count(target)) for c in continuations]
    return scorer


class TestSearchDriver:
    def test_stacks_present_in_every_rollout(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        driver = SearchDriver(
            scorer=lambda p, c, params: [0.0] * len(c),
            segment_len=2, num_candidates=2, keep_k=1, max_iterations=2, propose_mode="beam",
        )
        driver.tokenizer = tokenizer

        seen_stacks = []
        real_generate = model.generate

        def spy_generate(**kwargs):
            seen_stacks.append("logits_processor" in kwargs)
            return real_generate(**kwargs)

        # force a logits processor into the stack
        def _p(prefix_ids, scores):
            return scores
        processors = LogitsProcessorList([_p])

        out = driver.decode(
            input_ids=torch.tensor([[0, 3, 4]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            model=model,
            logits_processors=processors,
            stopping_criteria=StoppingCriteriaList(),
            runtime_kwargs={"base_generate": spy_generate},
            max_new_tokens=4,
        )
        assert out.ndim == 2
        assert all(seen_stacks)  # every rollout received the processor stack

    def test_batch_gt_one_raises(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        driver = SearchDriver(
            scorer=lambda p, c, params: [0.0], segment_len=2, num_candidates=2, keep_k=1,
            max_iterations=1,
        )
        driver.tokenizer = wordlevel_tokenizer()
        with pytest.raises(NotImplementedError):
            driver.decode(
                input_ids=torch.zeros(2, 3, dtype=torch.long),
                attention_mask=torch.ones(2, 3, dtype=torch.long),
                model=model, logits_processors=LogitsProcessorList(),
                stopping_criteria=StoppingCriteriaList(), runtime_kwargs=None,
            )


class TestFrontier:
    def test_keeps_top_k_and_best_so_far(self):
        beams = torch.tensor([[1, 2, 3], [1, 2, 4], [1, 2, 5]])
        frontier = Frontier(keep_k=2, eos_token_id=None, input_length=2, max_new_tokens=None)
        step = frontier.keep(beams, [0.1, 0.9, 0.5])
        assert step.kept_ids.size(0) == 2
        assert frontier.best_score == pytest.approx(0.9)


# PhasedDriver
class _ScriptedPhasedDriver(PhasedDriver):
    def plan(self, prompt_text, params):
        return [Fixed("INJECT"), Generated()]


class TestPhasedDriver:
    def test_fixed_then_generated_splice(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        driver = _ScriptedPhasedDriver()
        driver.tokenizer = tokenizer
        ids = tokenizer("the cat", return_tensors="pt").input_ids
        out = driver.decode(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            model=model,
            logits_processors=LogitsProcessorList(),
            stopping_criteria=StoppingCriteriaList(),
            runtime_kwargs=None,
            max_new_tokens=3,
        )
        assert out.ndim == 2
        assert out.size(1) >= ids.size(1)

    def test_extract_after(self):
        tokenizer = wordlevel_tokenizer()

        class _Extract(PhasedDriver):
            def plan(self, prompt_text, params):
                return [Generated()]

        driver = _Extract(extract_after="</think>")
        driver.tokenizer = tokenizer

        prompt_ids = tokenizer("the cat", return_tensors="pt").input_ids

        def fake_generate(**kwargs):
            inp = kwargs["input_ids"]
            cont = tokenizer(" the </think> mat", return_tensors="pt", add_special_tokens=False).input_ids
            return torch.cat([inp, cont.to(inp.device)], dim=1)

        out = driver.decode(
            input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
            model=None, logits_processors=LogitsProcessorList(),
            stopping_criteria=StoppingCriteriaList(),
            runtime_kwargs={"base_generate": fake_generate},
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        assert "</think>" not in decoded
        assert "mat" in decoded

    def test_per_example_params_slicing(self):
        driver = _ScriptedPhasedDriver()
        params = driver._params_per_example({"params": {"x": [1, 2]}}, batch_size=2)
        assert params == [{"x": 1}, {"x": 2}]


# criteria
class TestCriteria:
    def test_stop_on_tokens(self):
        crit = StopOnTokens([5])
        fired = crit(torch.tensor([[1, 2, 5], [1, 2, 3]]), None)
        assert fired.tolist() == [True, False]

    def test_budget_tokens(self):
        crit = BudgetTokens(2, prompt_len=3)
        assert crit(torch.tensor([[1, 2, 3, 4]]), None).tolist() == [False]
        assert crit(torch.tensor([[1, 2, 3, 4, 5]]), None).tolist() == [True]

    def test_stop_on_substring(self):
        tokenizer = wordlevel_tokenizer()
        ids = tokenizer("the cat mat", return_tensors="pt").input_ids
        crit = StopOnSubstring(tokenizer, "mat", prompt_len=2)
        assert crit(ids, None).tolist() == [True]


# ConstraintProcessor
class TestConstraintProcessor:
    def test_two_state_automaton_constrains(self):
        class _Automaton:
            def __init__(self):
                self.step = 0

            def reset(self, prefix_ids):
                self.step = prefix_ids.size(1)

            def allowed(self, prefix_ids):
                # state 0 allows token 3; later allows token 4
                return torch.tensor([3]) if prefix_ids.size(1) % 2 == 0 else torch.tensor([4])

        proc = ConstraintProcessor(_Automaton())
        scores = torch.arange(6, dtype=torch.float32).unsqueeze(0)
        out = proc(torch.tensor([[0, 1]]), scores.clone())  # len 2 -> allows 3
        allowed_mask = out[0] > float("-inf")
        assert allowed_mask.nonzero().flatten().tolist() == [3]


# CandidateForward (incremental prefix cache — P3.5 F1)
class _ForwardCounter:
    """Wraps model.forward, recording each call's input_ids shape and cache state."""

    def __init__(self, model):
        self.model = model
        self._orig = model.forward
        self.calls = []  # list of (seq_len, had_past)

    def __enter__(self):
        def _counting_forward(*args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            self.calls.append((int(input_ids.size(1)), kwargs.get("past_key_values") is not None))
            return self._orig(*args, **kwargs)

        self.model.forward = _counting_forward
        return self

    def __exit__(self, *exc):
        self.model.forward = self._orig


class TestCandidateForward:
    def test_incremental_matches_fresh(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        cf = CandidateForward(model)
        prefixes = [torch.tensor([[0, 3, 4]]), torch.tensor([[0, 3, 4, 5]]), torch.tensor([[0, 3, 4, 5, 6]])]
        cands = torch.tensor([[7, 8, 9]])
        for prefix in prefixes:
            incremental = cf.last_hidden_states(prefix, cands)
            fresh = CandidateForward(model).last_hidden_states(prefix, cands)
            assert torch.allclose(incremental, fresh, rtol=1e-5, atol=1e-5)

    def test_forward_count_extending_then_rebuild(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        cf = CandidateForward(model)
        cands = torch.tensor([[7, 8]])

        with _ForwardCounter(model) as counter:
            cf.last_hidden_states(torch.tensor([[0, 3, 4]]), cands)
            first = list(counter.calls)
            counter.calls.clear()
            cf.last_hidden_states(torch.tensor([[0, 3, 4, 5]]), cands)  # extends
            extend = list(counter.calls)
            counter.calls.clear()
            cf.last_hidden_states(torch.tensor([[0, 1, 2]]), cands)  # diverges -> rebuild
            rebuild = list(counter.calls)

        # first call: full prefill (len 3, no past) + candidate batch (len 1, with past)
        assert first == [(3, False), (1, True)]
        # extending call: delta only (len 1, with past) + candidate batch (len 1, with past)
        assert extend == [(1, True), (1, True)]
        # diverging call: rebuild sees full prefix (len 3, no past) + candidate batch
        assert rebuild == [(3, False), (1, True)]

    def test_end_to_end_linear_in_length(self):
        # SASA e2e: total model forwards must be linear in N (pins the O(T^2) regression)
        from aisteer360.algorithms.output_control.sasa.control import SASA

        def _forward_count_for(n_new):
            torch.manual_seed(0)
            model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
            tokenizer = wordlevel_tokenizer()
            sasa = SASA(
                beta=2.0,
                gen_wv_data={
                    "pos": ["the cat sat", "the dog ran", "the cat ran on"],
                    "neg": ["mat on fast", "span attention", "fast mat sat"],
                },
            )
            from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
            pipeline = SteeringPipeline(controls=[sasa], lazy_init=True)
            pipeline.model = model
            pipeline.tokenizer = tokenizer
            pipeline.steer()
            prompt = tokenizer("the cat", return_tensors="pt").input_ids
            with _ForwardCounter(model) as counter:
                pipeline.generate(input_ids=prompt, max_new_tokens=n_new, do_sample=False, eos_token_id=None)
                return len(counter.calls)

        c3 = _forward_count_for(3)
        c6 = _forward_count_for(6)
        # each new step adds a constant number of forwards (one HF decode step + one CandidateForward
        # delta + one candidate batch); a full-prefill-per-step regression would make the per-step
        # increment grow with the prefix length. Assert the increment is exactly constant.
        per_step = (c6 - c3) / 3
        assert per_step == pytest.approx(3.0), f"expected 3 forwards/step, got {per_step}; O(T^2) regression?"

    def test_mask_extension_and_too_long_raises(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        prefix = torch.tensor([[0, 3, 4, 5]])
        cands = torch.tensor([[7, 8]])
        short_mask = torch.ones(1, 3, dtype=torch.long)  # prompt-length mask, prefix has grown

        # a too-short mask is right-extended with ones and matches the all-ones call (B=1)
        with_short = CandidateForward(model).last_hidden_states(prefix, cands, short_mask)
        with_ones = CandidateForward(model).last_hidden_states(prefix, cands, torch.ones_like(prefix))
        assert torch.allclose(with_short, with_ones, rtol=1e-5, atol=1e-5)

        long_mask = torch.ones(1, 5, dtype=torch.long)  # longer than the 4-token prefix
        with pytest.raises(ValueError, match="longer than"):
            CandidateForward(model).last_hidden_states(prefix, cands, long_mask)

    def test_preserve_input_does_not_mutate_cache(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        ids = torch.tensor([[0, 3, 4]])
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
        cache = out.past_key_values
        original_ptr = cache.to_legacy_cache()[0][0].data_ptr()
        original_batch = cache.to_legacy_cache()[0][0].shape[0]

        repeated = repeat_cache(cache, 4, preserve_input=True)
        # input cache unchanged: same batch size and same underlying storage
        assert cache.to_legacy_cache()[0][0].shape[0] == original_batch == 1
        assert cache.to_legacy_cache()[0][0].data_ptr() == original_ptr
        # repeated cache does not share storage with the input
        assert repeated.to_legacy_cache()[0][0].data_ptr() != original_ptr
        # the input cache is still usable for a subsequent 1-token forward
        with torch.no_grad():
            positions = torch.arange(3, 4)
            model(input_ids=torch.tensor([[5]]), past_key_values=cache,
                  use_cache=True, cache_position=positions, return_dict=True)

    def test_scoring_replay_takes_incremental_path(self):
        # position-by-position teacher-forced replay (mimics _apply_scoring_processors): each step
        # grows the prefix by one, so it hits the incremental path and matches fresh-per-call.
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        probe = LinearProbe(direction=torch.randn(16), midpoint=torch.randn(16))
        value = SubspaceMarginValue(probe)  # holds one CandidateForward across the replay
        tokenizer = wordlevel_tokenizer()

        ref = torch.tensor([[0, 3, 4, 5, 6]])
        cands = torch.tensor([[7, 8, 9]])
        incremental, fresh = [], []
        for t in range(2, ref.size(1)):
            prefix = ref[:, :t]
            incremental.append(value.score(StepContext(prefix, cands, tokenizer, model, None)))
            fresh_value = SubspaceMarginValue(probe)
            fresh.append(fresh_value.score(StepContext(prefix, cands, tokenizer, model, None)))
        for a, b in zip(incremental, fresh):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)


# ValueGuidedProcessor candidate-set bounding (P3.5 F2)
class _CheapScriptedValue(BaseCandidateValue):
    """Returns zeros for K candidates; records the K it was asked to score."""

    scoring_cost = "cheap"

    def __init__(self):
        self.seen_k = []

    def score(self, ctx):
        k = ctx.candidate_ids.size(1)
        self.seen_k.append(k)
        return torch.zeros(1, k)


class _ModelForwardScriptedValue(_CheapScriptedValue):
    scoring_cost = "model_forward"


class TestValueGuidedMaxCandidates:
    def test_clamp_surviving_by_score(self):
        value = _CheapScriptedValue()
        proc = ValueGuidedProcessor(
            value, policy="surviving", normalize="none", mask_non_candidates=False, max_candidates=3,
        )
        scores = torch.full((1, 10), float("-inf"))
        scores[0, [1, 2, 5, 7, 9]] = torch.tensor([1.0, 5.0, 2.0, 9.0, 3.0])  # 5 finite
        proc(torch.tensor([[0]]), scores.clone())
        assert value.seen_k[-1] == 3  # clamped to top-3 by score

    def test_clamp_top_k_by_score(self):
        value = _CheapScriptedValue()
        proc = ValueGuidedProcessor(
            value, policy="top_k", k=10, normalize="none", max_candidates=3,
        )
        proc(torch.tensor([[0]]), torch.randn(1, VOCAB))
        assert value.seen_k[-1] == 3

    def test_warn_once_for_model_forward_value(self, monkeypatch):
        import aisteer360.algorithms.output_control._common.processors.value_guided as vg
        monkeypatch.setattr(vg, "LARGE_CANDIDATE_SET_WARN_THRESHOLD", 8)
        value = _ModelForwardScriptedValue()
        proc = vg.ValueGuidedProcessor(
            value, policy="surviving", normalize="none", mask_non_candidates=False,
        )
        scores = torch.zeros(1, VOCAB)  # all 100 finite -> K=100 > threshold 8
        with pytest.warns(UserWarning, match="max_candidates"):
            proc(torch.tensor([[0]]), scores.clone())
        # second call does not warn again
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            proc(torch.tensor([[0]]), scores.clone())  # would raise if it warned

    def test_no_warn_for_aux_forward_value(self, monkeypatch):
        import aisteer360.algorithms.output_control._common.processors.value_guided as vg
        monkeypatch.setattr(vg, "LARGE_CANDIDATE_SET_WARN_THRESHOLD", 8)

        class _AuxValue(_CheapScriptedValue):
            scoring_cost = "aux_forward"

        proc = vg.ValueGuidedProcessor(
            _AuxValue(), policy="surviving", normalize="none", mask_non_candidates=False,
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            proc(torch.tensor([[0]]), torch.zeros(1, VOCAB))  # no warning despite large K

    def test_sasa_forwards_max_candidates(self):
        from aisteer360.algorithms.output_control.sasa.control import SASA

        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        sasa = SASA(beta=1.0, max_candidates=4, gen_wv_data={
            "pos": ["the cat sat", "the dog ran", "the cat ran on"],
            "neg": ["mat on fast", "span attention", "fast mat sat"],
        })
        from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
        pipeline = SteeringPipeline(controls=[sasa], lazy_init=True)
        pipeline.model = model
        pipeline.tokenizer = tokenizer
        pipeline.steer()
        proc = sasa.get_logits_processors(torch.tensor([[0, 3]]), {})[0]
        assert proc.max_candidates == 4


# AuxModelSource / PromptVariantSource mask correctness (P3.5 F4)
from aisteer360.algorithms.output_control._common.logit_sources import AuxModelSource, PromptVariantSource


class TestAuxSourceMaskCorrectness:
    def _aux(self, base_tokenizer, pad_id):
        aux_model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        src = AuxModelSource("unused", base_tokenizer=base_tokenizer, shared_vocab=False)
        src.set_model(aux_model, base_tokenizer)
        src._base_pad_id = pad_id
        return src

    def test_token_id_path_batched_equals_single(self):
        torch.manual_seed(0)
        tokenizer = wordlevel_tokenizer()
        pad_id = tokenizer.pad_token_id
        src = self._aux(tokenizer, pad_id)

        # two prefixes of different real lengths, left-padded into one batch
        row_a = torch.tensor([3, 4, 5])          # len 3
        row_b = torch.tensor([6, 7])             # len 2
        padded = torch.tensor([[3, 4, 5], [pad_id, 6, 7]])
        batched = src.logprobs(padded)

        single_a = src.logprobs(row_a.unsqueeze(0))
        single_b = src.logprobs(row_b.unsqueeze(0))
        assert torch.allclose(batched[0], single_a[0], atol=1e-5)
        assert torch.allclose(batched[1], single_b[0], atol=1e-5)

    def test_transform_path_batched_equals_single_and_left_pads(self):
        torch.manual_seed(0)
        tokenizer = wordlevel_tokenizer()
        src = self._aux(tokenizer, tokenizer.pad_token_id)
        src.prompt_transform = lambda t: f"the {t}"
        assert src.tokenizer.padding_side == "left"  # set by set_model

        prompts = torch.tensor([[3, 4, 5], [tokenizer.pad_token_id, 6, 7]])
        batched = src.logprobs(prompts)
        single_a = src.logprobs(torch.tensor([[3, 4, 5]]))
        single_b = src.logprobs(torch.tensor([[6, 7]]))
        assert torch.allclose(batched[0], single_a[0], atol=1e-5)
        assert torch.allclose(batched[1], single_b[0], atol=1e-5)

    def test_pad_id_none_degrades_to_all_ones(self):
        torch.manual_seed(0)
        tokenizer = wordlevel_tokenizer()
        src = self._aux(tokenizer, None)  # no pad id -> all-ones mask
        out = src.logprobs(torch.tensor([[3, 4, 5]]))  # unpadded, no exception
        assert out.shape == (1, VOCAB)


class TestPromptVariantSourceMask:
    def test_batched_equals_single(self):
        torch.manual_seed(0)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        src = PromptVariantSource(prompt_transform=lambda t: f"the {t}", base_tokenizer=tokenizer)
        src.prepare(model=model, tokenizer=tokenizer)
        assert tokenizer.padding_side == "left"

        prompts = torch.tensor([[3, 4, 5], [tokenizer.pad_token_id, 6, 7]])
        batched = src.logprobs(prompts)
        single_a = src.logprobs(torch.tensor([[3, 4, 5]]))
        single_b = src.logprobs(torch.tensor([[6, 7]]))
        assert torch.allclose(batched[0], single_a[0], atol=1e-5)
        assert torch.allclose(batched[1], single_b[0], atol=1e-5)
