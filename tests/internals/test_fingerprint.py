"""Tests for model-identity digests (`core/internals/fingerprint.py`)."""
import subprocess
import sys

import pytest
import torch
from transformers import LlamaForCausalLM

from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint
from tests.utils.tiny_models import tiny_llama


@pytest.fixture(scope="module")
def saved_model_dir(tmp_path_factory):
    torch.manual_seed(0)
    model = tiny_llama()
    path = tmp_path_factory.mktemp("fingerprint") / "tiny"
    model.save_pretrained(path)
    return path


class TestDeterminism:
    def test_same_checkpoint_loaded_twice_yields_equal_digests(self, saved_model_dir):
        a = LlamaForCausalLM.from_pretrained(saved_model_dir)
        b = LlamaForCausalLM.from_pretrained(saved_model_dir)
        digest_a = model_fingerprint(a)
        assert digest_a == model_fingerprint(b)
        assert len(digest_a) == 16
        assert digest_a == digest_a.lower()

    def test_digest_is_deterministic_across_processes(self, saved_model_dir):
        in_process = model_fingerprint(LlamaForCausalLM.from_pretrained(saved_model_dir))
        code = (
            "from transformers import LlamaForCausalLM\n"
            "from aisteer360.algorithms.core.internals.fingerprint import model_fingerprint\n"
            f"model = LlamaForCausalLM.from_pretrained({str(saved_model_dir)!r})\n"
            "print(model_fingerprint(model))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == in_process


class TestSensitivity:
    def test_weight_perturbation_changes_digest(self, saved_model_dir):
        model = LlamaForCausalLM.from_pretrained(saved_model_dir)
        before = model_fingerprint(model)
        first_param = next(model.parameters())
        with torch.no_grad():
            first_param.view(-1)[0] += 1.0
        assert model_fingerprint(model) != before

    def test_dtype_changes_digest(self, saved_model_dir):
        model = LlamaForCausalLM.from_pretrained(saved_model_dir)
        before = model_fingerprint(model)
        model.half()
        assert model_fingerprint(model) != before

    def test_config_changes_digest(self, saved_model_dir):
        torch.manual_seed(0)
        a = tiny_llama(num_layers=4)
        torch.manual_seed(0)
        b = tiny_llama(num_layers=2)
        assert model_fingerprint(a) != model_fingerprint(b)
