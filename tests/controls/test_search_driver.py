"""`SearchDriver` frontier layout: a kept beam that stopped early continues from its last real token.

A scripted session stands in for the model. In the first iteration one proposal stops one token
short of the segment (a stop that the frontier does not classify as finished), and its row is
right-padded among the kept beams. The second iteration continues every kept beam, and the scorer
prefers the continuation of the padded beam. Hub-free on a tiny Llama with a WordLevel tokenizer
whose pad id (2) differs from its eos id (1).
"""
import torch

from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.output_control.search_decoding.control import SearchDecoding
from tests.utils.runtime_helpers import script_session_generate
from tests.utils.tiny_models import tiny_llama, wordlevel_tokenizer

PROMPT = [0, 3, 4]  # "<s> the cat"
PAD = 2


def _scripted_generate(input_ids, attention_mask=None, num_return_sequences=1, **kwargs):
    """Continue each row after its last real token; the first rollout cuts one proposal short."""
    real = input_ids[0][attention_mask[0].bool()].tolist() if attention_mask is not None else input_ids[0].tolist()
    if real == PROMPT:
        continuations = [[5, 6], [7, PAD]]  # "sat on", and "mat" cut one token short
    elif real[-1] == 7:
        continuations = [[8, 9], [8, 10]]  # "dog ran", "dog fast"
    else:
        continuations = [[10, 10], [10, 10]]
    rows = [torch.cat([input_ids[0], torch.tensor(tokens)]) for tokens in continuations[:num_return_sequences]]
    return torch.stack(rows)


def _score(prompt, continuations, params):
    return [10.0 * text.count("dog") + text.count("ran") + text.count("mat") for text in continuations]


def test_continued_beam_has_no_interior_pads(monkeypatch):
    torch.manual_seed(0)
    tokenizer = wordlevel_tokenizer()
    driver = SearchDecoding(
        scorer=_score, propose_mode="sample", segment_len=2, max_iterations=2, keep_k=2, num_candidates=2,
    )
    model = tiny_llama()
    model.generation_config.eos_token_id = tokenizer.eos_token_id  # pad (2) and eos (1) are distinct
    pipeline = SteeringPipeline(controls=[driver], model=model, tokenizer=tokenizer)
    pipeline.steer()
    script_session_generate(monkeypatch, _scripted_generate)

    (output,) = pipeline.generate(input_ids=torch.tensor([PROMPT]), max_new_tokens=6, return_output=True)

    continuation = output.output_ids[0].tolist()
    real = [token for token in continuation if token != PAD]
    assert continuation[: len(real)] == real == [7, 8, 9]
    assert output.finish_reasons[0] is None
