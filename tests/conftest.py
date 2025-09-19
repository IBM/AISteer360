import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tests.utils.load_ci_models import get_models

MODELS = get_models()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def device(request):
    name = request.param
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available.")
    if name == "mps":
        has_mps = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
        if not has_mps:
            pytest.skip("MPS not available.")
    return torch.device(name)


@pytest.fixture(
    scope="session",
    params=[
        pytest.param(repo, id=tag)
        for tag, repo in MODELS.items()
    ],
)
def model_and_tokenizer(request):
    """
    Loads each model once per test session.
    """
    model_id: str = request.param
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load {model_id}: {exc}")

    # ensure padding token exists for batching
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:  # edge case
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return model, tokenizer
