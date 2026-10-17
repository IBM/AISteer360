"""Recipe-only spipes: unsteered save/load/steer refits, thaw equivalence, verify report.

Also covers model-free freezing, i.e. `freeze=True` on an unsteered pipeline whose enabled
controls are all recipe-frozen (FACTS steer access, no fits, no exported state).
"""
from dataclasses import dataclass

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from steerability.algorithms.core.base_args import BaseArgs
from steerability.algorithms.core.execution.access import ModelAccess
from steerability.algorithms.core.registry import REGISTRY, register_method
from steerability.algorithms.core.steering_pipeline import SteeringPipeline
from steerability.algorithms.input_control.base import InputControl
from steerability.algorithms.input_control.few_shot.control import FewShot
from steerability.algorithms.input_control.system_prompt.control import SystemPrompt
from steerability.algorithms.input_control.user_prefix.control import UserPrefix
from steerability.algorithms.state_control.caa.control import CAA
from steerability.spipe import SPipe, SpipeSaveError

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def make_caa():
    return CAA(
        data={"positives": ["kind a", "kind b"], "negatives": ["mean a", "mean b"]},
        train_spec={"method": "mean_diff", "accumulate": "last_token"},
        layer_id=1,
        multiplier=2.0,
    )


@pytest.fixture(scope="module")
def model_and_tok():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def test_unsteered_pipeline_saves_recipe_only(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[make_caa()])
    spipe = pipeline.to_spipe()
    assert not spipe.is_frozen
    assert spipe.manifest["lock"] is None
    assert spipe.manifest["controls"][0]["resolved"] is None

    saved = spipe.save(tmp_path / "recipe.spipe")
    loaded = SPipe.load(saved)
    rebuilt = loaded.pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()
    plan = rebuilt._support_report.plan
    assert [(fit.control, fit.artifact) for fit in plan.fits] == [("CAA", "ContrastiveFit")]
    assert rebuilt.generate(text="hello", max_new_tokens=3, do_sample=False)


def test_freeze_requires_steered_pipeline():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[make_caa()])
    with pytest.raises(SpipeSaveError, match="steer") as excinfo:
        pipeline.to_spipe(freeze=True)
    message = str(excinfo.value)
    assert "CAA" in message
    assert "ContrastiveFit" in message


def test_thaw_equivalence(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_caa()],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    frozen = pipeline.to_spipe()
    recipe_only = pipeline.to_spipe(freeze=False)
    thawed = frozen.thaw()

    assert not thawed.is_frozen
    assert thawed.recipe_id == frozen.recipe_id == recipe_only.recipe_id
    thawed_controls = thawed.manifest["controls"]
    recipe_controls = recipe_only.manifest["controls"]
    assert [entry["args"] for entry in thawed_controls] == [entry["args"] for entry in recipe_controls]
    assert all(entry["resolved"] is None for entry in thawed_controls)


def test_verify_report_contents(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_caa()],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    spipe = pipeline.to_spipe()
    report = spipe.verify()
    assert report.ok
    assert not report.errors
    assert "verify: ok" in report.render()

    thin = spipe.save(tmp_path / "thin_dir", artifacts="thin")
    thin_report = SPipe.load(thin).verify()
    assert thin_report.ok
    assert any("thin" in message for message in thin_report.warnings)


def test_describe_lists_entries(model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[make_caa()],
                                model_name_or_path=TINY_MODEL)
    pipeline.steer()
    text = pipeline.to_spipe().describe()
    assert "state_control/caa" in text
    assert "steering_vector" in text


# model-free freezing
@dataclass
class QuietMarkerArgs(BaseArgs):
    """Arguments for the local recipe-frozen and blocking test controls."""
    marker: str = "[marker]"


class MarkerRolloutControl(InputControl):
    """A local input control declaring `ROLLOUTS` steer access with no fits."""
    Args = QuietMarkerArgs

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def steer_access(self) -> ModelAccess:
        return ModelAccess.ROLLOUTS


class MarkerStatefulControl(InputControl):
    """A local `FACTS` input control whose `export_state()` returns a non-empty mapping."""
    Args = QuietMarkerArgs

    def adapt(self, input_ids, runtime_kwargs=None):
        return input_ids

    def export_state(self) -> dict:
        return {"marker_memory": self.marker}


def make_prompt_only_pipeline():
    return SteeringPipeline(
        model_name_or_path=TINY_MODEL,
        controls=[
            SystemPrompt(text="Answer in one sentence."),
            UserPrefix(text="[concise]"),
        ],
    )


def test_prompt_only_pipeline_freezes_without_a_model(tmp_path, model_and_tok):
    model, tokenizer = model_and_tok
    pipeline = make_prompt_only_pipeline()
    spipe = pipeline.to_spipe(freeze=True)

    assert spipe.is_frozen
    assert not pipeline._is_steered
    assert pipeline.model is None
    lock = spipe.manifest["lock"]
    assert lock["model_fingerprint"] is None
    assert lock["tokenizer_fingerprint"] is None
    assert lock["torch_dtype"] is None
    assert lock["recipe_id"] and lock["config_id"] and lock["versions"]
    assert all(entry["resolved"] is None for entry in spipe.manifest["controls"])

    loaded = SPipe.load(spipe.save(tmp_path / "prompt_only.spipe"))
    assert loaded.verify().ok
    rebuilt = loaded.pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()
    assert rebuilt.generate(text="hello", max_new_tokens=3, do_sample=False)


def test_empty_pipeline_freezes_without_a_model():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[])
    spipe = pipeline.to_spipe(freeze=True)
    assert spipe.is_frozen
    assert spipe.manifest["controls"] == []


def test_few_shot_with_fixed_pools_freezes_without_a_model():
    few_shot = FewShot(
        positive_example_pool=[{"input": "hi", "output": "Hello."}],
        negative_example_pool=[{"input": "hi", "output": "yo"}],
        k_positive=1,
        k_negative=1,
    )
    spipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[few_shot]).to_spipe(freeze=True)
    assert spipe.is_frozen
    assert spipe.manifest["controls"][0]["resolved"] is None


def test_model_access_above_facts_blocks_model_free_freeze():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[MarkerRolloutControl()])
    with pytest.raises(SpipeSaveError, match="declares steer access ROLLOUTS"):
        pipeline.to_spipe(freeze=True)


def test_exported_state_blocks_model_free_freeze():
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[MarkerStatefulControl()])
    with pytest.raises(SpipeSaveError, match="exports state marker_memory"):
        pipeline.to_spipe(freeze=True)


def test_disabled_blocking_control_does_not_block_model_free_freeze():
    """A disabled control is not consulted (it is registered here only so the entry serializes)."""
    register_method("input", "marker_rollout", MarkerRolloutControl, QuietMarkerArgs)
    try:
        blocking = MarkerRolloutControl()
        blocking.enabled = False
        pipeline = SteeringPipeline(
            model_name_or_path=TINY_MODEL,
            controls=[SystemPrompt(text="Answer in one sentence."), blocking],
        )
        spipe = pipeline.to_spipe(freeze=True)
        assert spipe.is_frozen
    finally:
        REGISTRY["input_control"].pop("marker_rollout", None)


# auxiliary models given by reference
def _recipe_roundtrip(control, tmp_path):
    pipeline = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[control])
    spipe = pipeline.to_spipe()
    loaded = SPipe.load(spipe.save(tmp_path / "recipe.spipe"))
    return spipe, loaded


def test_gepa_with_a_named_reflection_model_saves_and_loads(tmp_path):
    from steerability.algorithms.input_control.common.memory.text import TextMemory
    from steerability.algorithms.input_control.gepa.control import GEPA

    gepa = GEPA(reflection_lm_name_or_path=TINY_MODEL, memory=TextMemory(slots={"instruction": "Be terse."}))
    spipe, loaded = _recipe_roundtrip(gepa, tmp_path)
    assert loaded.config_id == spipe.config_id
    (rebuilt,) = loaded.pipeline().input_controls
    assert rebuilt.reflection_lm_name_or_path == TINY_MODEL
    assert rebuilt.memory["instruction"] == "Be terse."


def test_cpo_with_a_named_prompt_model_saves_and_loads(tmp_path):
    from steerability.algorithms.input_control.cpo.control import CPO

    cpo = CPO(
        seed_prompt="be helpful",
        offline_data=[{"query": "q", "prompt": "p", "score": 1.0}],
        prompt_lm_name_or_path=TINY_MODEL,
        memory=str(tmp_path / "cpo_memory"),
    )
    spipe, loaded = _recipe_roundtrip(cpo, tmp_path)
    assert loaded.config_id == spipe.config_id
    (rebuilt,) = loaded.pipeline().input_controls
    assert rebuilt.prompt_lm_name_or_path == TINY_MODEL
    assert rebuilt.steer_access() is ModelAccess.FACTS


def test_gepa_with_a_loaded_reflection_model_does_not_save(model_and_tok):
    from steerability.algorithms.input_control.common.memory.text import TextMemory
    from steerability.algorithms.input_control.gepa.control import GEPA

    model, tokenizer = model_and_tok
    gepa = GEPA(reflection_lm=model, reflection_tokenizer=tokenizer, memory=TextMemory(slots={"instruction": "x"}))
    with pytest.raises(SpipeSaveError, match="_name_or_path"):
        SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[gepa]).to_spipe()


def test_recipe_with_a_memory_freezes_without_a_model(tmp_path, model_and_tok):
    from steerability.algorithms.input_control.prewrite.control import PRewrite

    model, tokenizer = model_and_tok
    prewrite = PRewrite(memory={"slots": {"instruction": "Be formal."}})
    assert prewrite.steer_access() is ModelAccess.FACTS
    spipe = SteeringPipeline(model_name_or_path=TINY_MODEL, controls=[prewrite]).to_spipe(freeze=True)
    assert spipe.is_frozen
    assert spipe.manifest["controls"][0]["resolved"] is None

    rebuilt = SPipe.load(spipe.save(tmp_path / "prewrite.spipe")).pipeline()
    rebuilt.model, rebuilt.tokenizer = model, tokenizer
    rebuilt.steer()
    assert rebuilt.input_controls[0].memory["instruction"] == "Be formal."


def test_steered_recipe_with_a_memory_exports_nothing(model_and_tok):
    from steerability.algorithms.input_control.common.memory.text import TextMemory
    from steerability.algorithms.input_control.gepa.control import GEPA

    model, tokenizer = model_and_tok
    gepa = GEPA(memory=TextMemory(slots={"instruction": "Be terse."}))
    pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[gepa], model_name_or_path=TINY_MODEL)
    pipeline.steer()
    assert gepa.memory["instruction"] == "Be terse."
    assert gepa.export_state() == {}
    assert pipeline.to_spipe().manifest["controls"][0]["resolved"] is None
