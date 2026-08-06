"""Parity / behavior tests for the four ported output-control methods (output multiplicity design, P3).

Hub-free. RAD and SASA had no prior test coverage; their steering math is pinned here directly. DeAL
and ThinkingIntervention behavior is exercised against the port classes (the shape/content
assertions of the existing hub tests are covered separately in test_deal.py / test_thinking_intervention.py).
"""
import pytest
import torch
from transformers import LlamaConfig, LlamaForSequenceClassification

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.output_control._common.estimators.linear_probe import LinearProbe
from aisteer360.algorithms.output_control._common.values.base import StepContext
from aisteer360.algorithms.output_control._common.values.subspace_margin import SubspaceMarginValue
from aisteer360.algorithms.output_control.deal.control import DeAL
from aisteer360.algorithms.output_control.rad.control import RAD
from aisteer360.algorithms.output_control.sasa.control import SASA
from aisteer360.algorithms.output_control.thinking_intervention.control import ThinkingIntervention
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

VOCAB = 100


def _pipeline(controls, model=None, tokenizer=None):
    if model is None:
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
    if tokenizer is None:
        tokenizer = wordlevel_tokenizer()
    pipeline = SteeringPipeline(controls=controls, lazy_init=True)
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()
    return pipeline, model, tokenizer


# RAD
def _make_tiny_reward_model(tmp_path):
    """A hub-free tiny sequence classifier + wordlevel tokenizer saved to tmp_path."""
    cfg = LlamaConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=VOCAB,
        num_labels=2, pad_token_id=2,
    )
    rm = LlamaForSequenceClassification(cfg).eval()
    tokenizer = wordlevel_tokenizer()
    rm.save_pretrained(str(tmp_path))
    tokenizer.save_pretrained(str(tmp_path))
    return str(tmp_path)


class TestRADParity:
    def test_recipe_matches_reference_math(self, tmp_path):
        rm_path = _make_tiny_reward_model(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(beta=10.0, reward_model_id=rm_path)
        pipeline, model, tokenizer = _pipeline([rad], model=model)

        # build the processor exactly as get_logits_processors would (top-k default 20)
        processors = rad.get_logits_processors(torch.tensor([[0, 3, 4]]), {}, top_k=5)
        assert len(processors) == 1
        proc = processors[0]

        prefix = torch.tensor([[0, 3, 4]])
        scores = torch.randn(1, VOCAB)
        out = proc(prefix, scores.clone())

        # reference: top-5 candidates, min-max normalized reward (inverted=False for HF classifier),
        # non-candidates -> -inf
        cand_scores, cand_ids = torch.topk(scores, 5, dim=-1)
        # reward the candidates via the same value the processor uses
        v = proc.value.score(StepContext(prefix, cand_ids, tokenizer, model, None))
        r_min = v.min()
        r_max = v.max()
        norm = (v - r_min) / (r_max - r_min) if (r_max - r_min) > 1e-8 else torch.full_like(v, 0.5)
        expected = torch.full_like(scores, float("-inf"))
        expected.scatter_(1, cand_ids, cand_scores)
        expected.scatter_add_(1, cand_ids, 10.0 * norm.to(scores.dtype))

        assert torch.allclose(out, expected, atol=1e-4, equal_nan=True)

    def test_no_nameerror_without_sampling_kwargs(self, tmp_path):
        rm_path = _make_tiny_reward_model(tmp_path)
        rad = RAD(beta=5.0, reward_model_id=rm_path)
        _pipeline([rad], model=tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB))
        # no top_k/top_p -> default top-20, no NameError
        processors = rad.get_logits_processors(torch.tensor([[0, 3, 4]]), {})
        assert processors[0].policy == "top_k"
        assert processors[0].k == 20

    def test_unsteered_raises(self):
        rad = RAD(beta=1.0)
        with pytest.raises(RuntimeError, match="steer"):
            rad.get_logits_processors(torch.tensor([[0, 3, 4]]), {})

    def test_end_to_end_generate(self, tmp_path):
        rm_path = _make_tiny_reward_model(tmp_path)
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        rad = RAD(beta=5.0, reward_model_id=rm_path)
        pipeline, model, tokenizer = _pipeline([rad], model=model)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=4, do_sample=False, top_k=5, eos_token_id=None,
        )
        assert out.ndim == 2
        assert out.size(1) == 4


# SASA
class TestSASAParity:
    def test_margin_softmax_math(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()
        probe = LinearProbe(
            direction=torch.randn(16),
            midpoint=torch.randn(16),
        )
        sasa = SASA(beta=3.0, wv_path=None)
        # inject probe directly (skip fitting)
        sasa.model = model
        sasa.tokenizer = tokenizer
        sasa.probe = probe
        sasa.probe.to(next(model.parameters()).device)

        prefix = torch.tensor([[0, 3, 4]])
        attention_mask = torch.ones_like(prefix)
        # scores with a couple of -inf (surviving = finite set)
        scores = torch.randn(1, VOCAB)
        scores[0, 10:] = float("-inf")

        proc = sasa.get_logits_processors(prefix, {}, attention_mask=attention_mask)[0]
        out = proc(prefix, scores.clone())

        # reference: margins over the surviving set, softmax-normalized, += beta * softmax
        value = SubspaceMarginValue(probe)
        surviving = (scores > -torch.inf).nonzero()[:, 1].unsqueeze(0)
        margins = value.score(StepContext(prefix, surviving, tokenizer, model, attention_mask))
        soft = torch.softmax(margins, dim=-1)
        expected = scores.clone()
        expected.scatter_add_(1, surviving, 3.0 * soft.to(scores.dtype))

        assert torch.allclose(out, expected, atol=1e-4, equal_nan=True)

    def test_legacy_checkpoint_loads(self, tmp_path):
        wv = {"wv": torch.randn(16), "mu_mu": torch.randn(16)}
        path = str(tmp_path / "steer_wv.pt")
        torch.save(wv, path)
        probe = SASA._load_probe(path)
        assert isinstance(probe, LinearProbe)
        assert torch.allclose(probe.direction, wv["wv"])
        assert torch.allclose(probe.midpoint, wv["mu_mu"])

    def test_include_in_scoring_default_false(self):
        assert SASA.include_in_scoring is False

    def test_end_to_end_fit_and_generate(self):
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
        pipeline, model, tokenizer = _pipeline([sasa], model=model, tokenizer=tokenizer)
        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(input_ids=prompt, max_new_tokens=5, do_sample=False, eos_token_id=None)
        assert out.ndim == 2
        assert out.size(1) == 5  # continuation-only, prompt excluded
        assert abs(float(sasa.probe.direction.norm()) - 1.0) < 1e-4


# DeAL
class TestDeALPort:
    def test_base_generate_override_and_scorer_trajectory(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        # scorer favors continuations containing "mat"
        def scorer(prompt, continuations, params):
            return [float(c.count("mat")) for c in continuations]

        deal = DeAL(reward_func=scorer, lookahead=2, init_beams=4, topk=2, max_iterations=2)
        pipeline, model, tokenizer = _pipeline([deal], model=model, tokenizer=tokenizer)

        override_calls = []
        real_generate = model.generate

        def override_generate(**kwargs):
            override_calls.append("logits_processor" in kwargs)
            return real_generate(**kwargs)

        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={"base_generate": override_generate},
            max_new_tokens=4,
        )
        assert out.ndim == 2
        assert override_calls  # override was used for the rollouts

    def test_step_level_control_steers_every_rollout(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        # a step-level control forcing token 7, composed alongside the DeAL driver
        from aisteer360.algorithms.output_control.base import OutputControl

        class _ForceToken(OutputControl):
            Args = None

            def get_logits_processors(self, input_ids, runtime_kwargs, **kwargs):
                def _force(prefix_ids, scores):
                    out = torch.full_like(scores, float("-inf"))
                    out[:, 7] = 0.0
                    return out
                return [_force]

        deal = DeAL(reward_func=lambda p, c, params: [0.0] * len(c),
                    lookahead=3, init_beams=2, topk=1, max_iterations=1)
        pipeline, model, tokenizer = _pipeline([_ForceToken(), deal], model=model, tokenizer=tokenizer)

        prompt = tokenizer("the cat", return_tensors="pt").input_ids
        out = pipeline.generate(
            input_ids=prompt, max_new_tokens=3, return_full_sequence=True,
        )
        prompt_len = prompt.size(1)
        continuation = out[:, prompt_len:]
        # every rollout was steered to token 7
        assert torch.all(continuation == 7)


# ThinkingIntervention
class TestThinkingInterventionPort:
    def test_extract_after_and_prefix_splice(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        def intervention(prompt, params):
            plan = params.get("plan", "steps")
            return f"the {plan} </think> {prompt}"

        ti = ThinkingIntervention(intervention=intervention)
        pipeline, model, tokenizer = _pipeline([ti], model=model, tokenizer=tokenizer)

        prompt = tokenizer("the cat", return_tensors="pt").input_ids

        def fake_generate(**kwargs):
            inp = kwargs["input_ids"]
            cont = tokenizer(" </think> mat", return_tensors="pt", add_special_tokens=False).input_ids
            return torch.cat([inp, cont.to(inp.device)], dim=1)

        out = pipeline.generate(
            input_ids=prompt,
            runtime_kwargs={"base_generate": fake_generate, "params": {"plan": "list steps"}},
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=False)
        assert "</think>" not in decoded
        assert "mat" in decoded

    def test_batched_dict_of_lists_params(self):
        model = tiny_llama(num_layers=2, hidden=16, heads=2, vocab=VOCAB)
        tokenizer = wordlevel_tokenizer()

        seen_params = []

        def intervention(prompt, params):
            seen_params.append(params.get("tag"))
            return f"{params.get('tag', 'x')} </think> {prompt}"

        ti = ThinkingIntervention(intervention=intervention)
        pipeline, model, tokenizer = _pipeline([ti], model=model, tokenizer=tokenizer)

        prompts = tokenizer(["the cat", "the dog"], return_tensors="pt", padding=True).input_ids

        def fake_generate(**kwargs):
            inp = kwargs["input_ids"]
            cont = tokenizer(" </think> mat", return_tensors="pt", add_special_tokens=False).input_ids
            cont = cont.expand(inp.size(0), -1)
            return torch.cat([inp, cont.to(inp.device)], dim=1)

        pipeline.generate(
            input_ids=prompts,
            runtime_kwargs={"base_generate": fake_generate, "params": {"tag": ["the", "on"]}},
        )
        # per-example params sliced correctly
        assert seen_params == ["the", "on"]
