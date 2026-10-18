# Steerability v0.5.5 — fix design for the 36 validated review findings

Implementation brief for Claude Code. Each item states the defect (with the current file:line), the
chosen fix, the alternatives that were rejected as over-engineered, and the tests/docs that must change
with it. Line numbers are those of the reviewed `steerability-main` tree and will drift as earlier items
land; the surrounding code is quoted so each site stays findable.

## 0. Ground rules

Repository conventions (from `AGENTS.md`) that every change below must respect:

- **No compatibility shims.** No aliases, re-exports, deprecated parameters, or `try/except TypeError`
  fallbacks. When a signature changes, change every caller and test in the same commit.
- **One declarative statement per control.** A capability or fit is declared once (on the class that
  owns the behavior), not restated in subclasses.
- **Google-style docstrings**, updated whenever behavior changes. A docstring that no longer describes
  the code is a bug.
- **Tests per control** live under `tests/controls/`; core/execution/spipe tests under `tests/core/`.
  Golden regeneration is `STEERABILITY_CAPTURE_GOLDENS=1 pytest <file> -s` (then paste the printed
  values).
- **Docs**: a behavior change that users can observe updates `docs/concepts/controls.md` (the control's
  bullet), the reference page if the public signature changed, and the notebook that demonstrates the
  control if it would now error.
- Run `uv sync --all-extras` once, then `uv run pytest -q tests/` after each phase. Model-backed tests use
  the tiny HF checkpoints already in `tests/conftest.py`; nothing new needs a GPU.

Suggested ordering: four PRs (phases A–D below). Phase A changes runtime behavior and should land and
be tested first; Phase B changes identities/serialization (one-time fingerprint change — see A5);
Phase C is docs/style; Phase D is test hygiene and follow-ups.

Over-engineering guardrails applied throughout:

1. Prefer deleting a wrong code path to adding a flag that selects between the wrong and right path.
2. Reuse an existing helper (`to_left_pad`, `session_artifact_identity`, `runtime_kwargs_schema`,
   `strip_leading_pads`) before writing a new one.
3. New constructor arguments only where a user has no other way to express the intent (A14, A16).
4. No new abstractions (no "padding strategy" classes, no fingerprint registry, no store transaction
   layer). Every fix below is local to the file(s) named.

---

## Phase A — behavior bugs

### A1. CAST: a partial condition point silently degrades to unconditional steering

**Defect.** `steerability/algorithms/state_control/cast/args.py:255-271` validates "layer ids without
threshold" and "threshold without layer ids" only inside `if not self.search.auto_find:`. `auto_find`
defaults to `True` (`state_control/common/fit_specs.py:94`). In `common/sources.py`
`ConditionPointSearch.resolve_gate` runs the search only when `layer_ids is None and condition_data is
not None` (:524), then `conditional = bool(layer_set) and threshold is not None` (:540) and returns
`None` — an *unconditional* gate — for every partial configuration: manual layers without a threshold,
a threshold with a `condition_vector` but no `condition_data`, or a `condition_vector` alone.
`ConditionPointSearch.__post_init__` (:466) only coerces data and checks the comparator although its
docstring says `threshold` is "required with `layer_ids`".

**Fix.** Three small guards, no new fields.

1. `cast/args.py`: hoist the two partial-point checks out of `if not self.search.auto_find:` so they
   run unconditionally; keep only the `condition_data`-without-point check under `not auto_find`. Change
   the `condition_vector`-without-layers check to also require data for the search:
   `if self.condition_vector is not None and self.condition_layer_ids is None and (not self.search.auto_find or self.condition_data is None): raise ValueError(...)`.
2. `sources.py` `ConditionPointSearch.__post_init__`: add
   `if (self.layer_ids is None) != (self.threshold is None): raise ValueError("layer_ids and threshold must be given together ...")`
   and the same vector-without-search-data guard as above.
3. `sources.py` `resolve_gate`: replace the silent return with
   ```python
   if not conditional:
       if condition_supplied:
           raise ValueError(
               "A condition was supplied but no condition point resolved; give layer_ids and "
               "threshold together, or enable search.auto_find with condition_data."
           )
       self.resolved_point = None
       return None
   ```
   This is the guard that cannot be bypassed by constructing the dataclass another way.

**Rejected.** A `strict: bool` on `ConditionSearchSpec` (keeps the degrade path alive); a warning
instead of an error (the user asked for conditional steering and gets unconditional steering — that is
not a warning-grade outcome).

**Tests.** `tests/controls/test_cast_conditional.py`: three construction-time `pytest.raises(ValueError)`
cases (layers only, threshold only, vector only with `auto_find=True` and no data), one `resolve_gate`
case built directly from `ConditionPointSearch(condition_vector=..., layer_ids=[2])` to hit guard 3.
Any existing test that relied on silent fallback is wrong and must be updated.

**Docs.** `ConditionPointSearch` and `CASTArgs` docstrings: "An unconditional configuration (no
condition at all) resolves to None; a partial one is an error."

### A2. `ActivationStats.estimate` requires a live model; `ProbeSetFit(stats=StatsSpec(...))` crashes below MODULE access

**Defect.** `core/internals/stats.py:171-173` does `next(model.parameters()).device`,
`model.config.model_type`, `model_fingerprint(model)` before the capture at :184 (which already accepts
`model=None, session=`). `probe_set.py:104-105` calls `stats.estimate(model, tokenizer, session=session)`
and `routed_decoding/control.py:146-148,236` declares `CAPTURE` access for `ProbeSetFit`, so the pipeline
passes `model=None` (`steering_pipeline.py:844`) and `estimate` raises `AttributeError` on `None`.
`probes/fitting.py:229,371-377` already shows the model-free pattern.

**Fix.** In `ActivationStats.estimate` mirror `fit_probe`:
```python
if model is not None:
    device = next(model.parameters()).device
    model_type = getattr(model.config, "model_type", "unknown")
    fingerprint = model_fingerprint(model)
else:
    device = torch.device("cpu")
    model_type, session_meta = session_artifact_identity(session)
    fingerprint = session_meta.get("model_fingerprint")
```
Type the parameter `model: PreTrainedModel | None` and the dataclass field
`model_fingerprint: str | None` (JSON round-trips `None`; `to_dict`/`from_dict` need no change). In
`fit_probe`'s stats check (`fitting.py:378-383`) add `and stats.model_fingerprint is not None` to the
condition so an unfingerprinted stats artifact is not reported as a mismatch.

**Rejected.** A sentinel string `"unknown"` for the fingerprint (it would compare unequal to a real
fingerprint and raise a false mismatch); making `StatsSpec` escalate `RoutedDecoding` to MODULE access
(wrong direction — capture is the declared access and it is sufficient).

**Tests.** `tests/controls/test_routed_decoding.py`: `RoutedDecoding(probes=ProbeSetFit(stats=StatsSpec(...)))`
steered through the HF backend session (`model=None` path). `tests/core/test_stats.py` (or the existing
stats test file): `estimate(None, tokenizer, texts, session=session)` returns an `ActivationStats` whose
`model_fingerprint` equals the session layout's fingerprint.

### A3. `HeadProfile._prepare_batches` is not left-padded although its docstring and consumers assume it

**Defect.** `state_control/pasta/profiling.py:422-458` renders prompts with `prepare_inputs` and never
calls `to_left_pad`; the docstring claims left padding, and `:479-494` slices
`batch.input_ids[i:i+1]` into per-row `GenerationItem`s and locates spans with
`control.locate_spans(input_ids, substrings)` in the tensor's coordinates. A tokenizer with
`padding_side="right"` yields right-padded rows, so the shared hook's token ranges and the per-item
prompts disagree.

**Fix.** After `prepare_inputs(...)` in `_prepare_batches`:
`input_ids, attention_mask = to_left_pad(input_ids, attention_mask)` (from `steerability.utils.tokenization`).
One line; the docstring becomes true.

**Rejected.** Making `prepare_inputs` left-pad globally (it is shared by generation paths that rely on
the tokenizer's own side); passing `padding_side` through the profiling API.

**Tests.** `tests/controls/test_pasta_profiling.py` (or the existing profiling test): tokenizer copy with
`padding_side="right"`, two prompts of different length, assert the located spans are identical to the
`padding_side="left"` run.

### A4. `per_item_state_entries` hands every row clone the whole-batch row-scoped runtime kwargs

**Defect.** `core/utils/assembly.py:108-158` builds one clone per row and calls
`clone.get_hooks(row_ids, runtime_kwargs, attention_mask=row_mask, ...)` with the *unsliced*
`runtime_kwargs`. PASTA declares `substrings` with `scope: "row"` (`pasta/control.py:82`) and
`_normalize_substrings` (`:452-472`) raises `Need {batch_size} substring groups` when a batch-1 clone
receives `batch` groups. Trigger: `generate(..., seed=..., seed_scope="item")` with batch > 1 and PASTA
(`steering_pipeline.py:1727-1736`).

**Fix.** Slice row-scoped entries per row using the schema the controls already declare:
```python
schema = runtime_kwargs_schema(state_controls)   # core.utils.controls
batch = input_ids.size(0)

def _row_kwargs(index):
    if not runtime_kwargs:
        return runtime_kwargs
    sliced = dict(runtime_kwargs)
    for name, entry in schema.items():
        value = runtime_kwargs.get(name)
        if entry["scope"] == "row" and isinstance(value, (list, tuple)) and len(value) == batch:
            sliced[name] = [value[index]]
    return sliced
```
and pass `_row_kwargs(index)` to each clone. The `[value[index]]` form keeps the schema contract ("a
sequence with one element per prompt row"), which PASTA already accepts at batch 1. `str` values are not
sequences here (`isinstance(value, (list, tuple))`), so a broadcast string passes through unchanged.

**Rejected.** A new `slice_rows()` method on `StateControl` (every control would have to implement it
for one call site); turning the per-item path off when row-scoped kwargs are present (loses the
documented per-item seeding).

**Tests.** `tests/core/test_seed_scope.py` (or `tests/controls/test_pasta.py`): batched PASTA with
`substrings=[["a"], ["b"]]`, `seed=0`, `seed_scope="item"`, batch 2 — generates without error and the
per-row output equals the batch-1 output for each prompt with the same seed.

### A5. `model_fingerprint` is brittle to `transformers` upgrades and can miss q/v-only LoRA merges

**Defect.** `core/internals/fingerprint.py:33-62` hashes `model.config.to_json_string()`, whose
`to_diff_dict()` keeps `transformers_version` (verified against upstream `configuration_utils.py`); every
transformers upgrade changes every fingerprint, which under `verify="strict"` is a hard error for
calibrated artifacts (`sources.py:289-299`, `specs.py:339-348`). The weight sample is 8 evenly spaced
parameters × 64 elements; for the Llama-3.1-8B layout none of the 8 is a `q_proj`/`v_proj`, so a merged
q/v-only LoRA is deterministically invisible (the docstring claims "any weight edit that touches the
sampled slices" and lists "LoRA merge").

**Fix.**
1. Config part: `config_dict = model.config.to_diff_dict(); config_dict.pop("transformers_version", None)`
   and hash `json.dumps(config_dict, sort_keys=True, default=str)`. Make
   `backends/vllm/backend.py:_config_layout` (:112-118) build its digest from the same
   `to_diff_dict()`-minus-`transformers_version` form instead of `to_dict()` so the two config digests
   drift under the same rules.
2. Weight part: sample one parameter per *parameter kind*, where the kind is the name with numeric path
   components replaced by `*` (`model.layers.3.self_attn.q_proj.weight` → `model.layers.*.self_attn.q_proj.weight`).
   Take the first parameter of each kind in `named_parameters()` order, hash `name` + first 64 elements
   as today. Typical decoder: ~12 kinds, so cost stays trivial and every projection kind is covered.
3. Docstring: describe the new scheme and state the honest limit ("an adapter applied to a subset of
   layers that excludes the sampled layer of a kind is not detected").
4. Changelog entry: fingerprints computed by ≤0.5.5 do not match 0.5.6; frozen bundles with calibrated
   artifacts must be re-frozen (`thaw()` + `steer()` + `to_spipe()`), or loaded with `verify="warn"`.

**Rejected.** Hashing all parameters (cost); a fingerprint version field with dual verification (keeps
both algorithms alive — the changelog note is the migration); hashing only the config (loses LoRA/SFT
detection, which the lock exists for).

**Tests.** `tests/core/test_fingerprint.py`: (a) fingerprint unchanged when `config.transformers_version`
is mutated; (b) fingerprint changes when the tiny model's layer-0 `q_proj.weight` is perturbed in place
(layer 0 is the sampled instance of that kind; name the test so the limit is explicit); (c) equal across
two loads of the same checkpoint.

### A6. Estimator-path fits lose `meta["location"]` and `ContrastiveFit.access` misreports estimator fits

**Defect.** `sources.py:134-156` `ContrastiveFit._fit`: the `estimator=` path calls
`estimator.fit(model, tokenizer, data=..., session=session, **kwargs)` inside `try/except TypeError`
and never stamps `meta["location"]`; the spec path stamps `master.meta["location"] = self.location` at
:152. `ContrastiveFit.access` (:108-112) returns `MODULE` whenever an estimator is given, although every
built-in estimator supports `model=None` capture. `directional_ablation/control.py:50-58` and
`angular_steering/control.py:63-71` use the estimator path (through `LayerFilteredFit`, whose `access`
delegates), so their artifacts have no location stamp and force MODULE access.

**Fix.**
1. Each built-in estimator stamps `meta["location"] = spec.location` next to where it builds `meta`
   (`mean_difference.py` ≈:150, `contrastive_direction.py`, `steering_plane.py` copies `feature_sv.meta`
   so it inherits it; `single_pair.py:112` already stamps `"layer_input"`). Remove the now-redundant
   stamp at `sources.py:152`.
2. `estimators/base.py`: `access: ClassVar[ModelAccess] = ModelAccess.CAPTURE` on `BaseEstimator`;
   `SinglePairEstimator.access = ModelAccess.MODULE` only if it truly needs the module tree (it uses
   `capture_hidden` with `session`, so it does not — leave it at CAPTURE). `ContrastiveFit.access`
   returns `self.estimator.access if self.estimator is not None else ModelAccess.CAPTURE`.
3. Delete the `except TypeError` fallback at `sources.py:137-142`; the estimator protocol takes
   `session=`. Document the protocol in `BaseEstimator.fit`'s docstring.

**Rejected.** A `location` kwarg threaded through `ContrastiveFit` to the estimator (the spec already
carries it); keeping the fallback "for user estimators" (AGENTS.md forbids shims; a user estimator that
ignores `session` just declares `**kwargs`).

**Tests.** `tests/controls/test_directional_ablation.py` / `test_angular_steering.py`: fitted
`SteeringVector.meta["location"] == train_spec.location`; `control.steer_access()` is CAPTURE.
`tests/controls/test_state_common.py`: `ContrastiveFit(estimator=MeanDifferenceEstimator()).access is CAPTURE`.

### A7. `TRLArgs.__post_init__` allocates a temp dir, which lands in `config_id`; dead capability arms

**Defect.** `structural_control/wrappers/trl/args.py:76-80` sets
`self.output_dir = self.output_dir or training_args.get("output_dir") or tempfile.mkdtemp(prefix="steerability-trainer-")`
and copies it into `training_args["output_dir"]`. `config_descriptor_from_controls`
(`core/identity.py:187-190`) canonicalizes every `init` field of `args`, so two identical TRL
configurations get different `config_id`s (and therefore different eval identities and
`derive_trial_seed`s). `fit_identity` (`base_mixin.py:228-250`) excludes it, so the *fit* digest is
fine — only `config_id` is unstable. Because the dir is always set, the `else None` arms in
`artifact_capability` (`base_mixin.py:176-182`) are dead and its docstring ("a configuration without
an output directory produces no artifact") is false; `docs/concepts/controls.md:109` repeats the false
claim ("requires a configured output directory").

**Fix.** Move the temp-dir default from construction to steer time.
1. `trl/args.py`: `self.output_dir = self.output_dir or self.training_args.get("output_dir")` (no
   `mkdtemp`); keep `"output_dir": self.output_dir` in `base_training_args` — `resolve_config_kwargs`
   drops `None` values (`base_mixin.py:19`), so an unset dir no longer reaches the config.
2. `base_mixin.py`:
   ```python
   def _resolved_output_dir(self) -> str:
       configured = self.training_args.get("output_dir") or self.output_dir
       if configured:
           return str(configured)
       if getattr(self, "_transient_output_dir", None) is None:
           self._transient_output_dir = tempfile.mkdtemp(prefix="steerability-trainer-")
       return self._transient_output_dir
   ```
   `_maybe_save_trained_artifacts` (:155) uses `self._resolved_output_dir()`.
3. Each trainer `steer` injects the resolved dir into the config kwargs it builds
   (`sfttrainer/base_mixin.py:28`, `dpotrainer/base_mixin.py:64-70`, `grpotrainer/base_mixin.py:48`,
   `ppotrainer/base_mixin.py:92`): `config_kwargs.setdefault("output_dir", self._resolved_output_dir())`
   before `XConfig(**config_kwargs)`. (APO reuses the DPO mixin.)
4. `artifact_capability`: drop the two `if self._resolved_output_dir() else None` arms (the dir is always
   resolvable); keep `SERVE_CHECKPOINT if self.merged_output_dir else None` for merged LoRA. Rewrite the
   docstring accordingly. `export_artifact` unchanged.
5. `docs/concepts/controls.md:109`: "serves the steer-time artifact, a checkpoint or LoRA adapter; a
   merged LoRA run needs `merged_output_dir`".

**Rejected.** An `exclude_from_identity` hook in `canonical_value` (generic machinery for one field);
making `output_dir` `init=False` (users set it).

**Tests.** `tests/controls/test_trl_wrapper.py`: two `TRL(...)` constructions with identical args have
equal `config_id`s and `training_args` contains no `output_dir`; after `steer()` with the smoke dataset
(`RUN_TRL_SMOKE`), the model is saved under the transient dir; `artifact_capability()` for a LoRA run
without dirs is `SERVE_LORA`.

### A8. `SearchDriver._search` proposes from a right-padded frontier, producing interior pads

**Defect.** `output_control/common/drivers/search.py:357-371`: after iteration 1 the frontier
`current_ids = step.kept_ids[unfinished]` is right-padded (HF pads early-finished beams on the right).
`infer_attention_mask_from_ids` masks the trailing pads, but `propose` appends new tokens *after* them,
so beams become `[prompt, cont, PAD…, new]`. `Frontier.keep` (`frontier.py:70-75`) strips only trailing
pads, `decode(skip_special_tokens=True)` hides the pads from the scorer, and the returned `best_ids`
carries interior pads into the pipeline's output (wrong finish reasons, wrong token counts, pads in
`skip_special_tokens=False` text). HF also warns about right padding for decoder-only generation.

**Fix.** Left-pad the frontier for the rollout and restore the prompt-first layout afterwards:
```python
frontier_mask = infer_attention_mask_from_ids(current_ids, self.tokenizer.pad_token_id)
frontier_ids, frontier_mask = to_left_pad(current_ids, frontier_mask)
beams = proposer.propose(frontier_ids, ..., attention_mask=frontier_mask, **rollout_kwargs)
# rotate each beam row's leading pads to its tail; HF groups the n rollouts of frontier row i at rows i*n:(i+1)*n
lead = (frontier_mask == 0).sum(dim=1).repeat_interleave(beams.size(0) // frontier_ids.size(0))
width = beams.size(1)
positions = (torch.arange(width, device=beams.device).unsqueeze(0) + lead.unsqueeze(1)) % width
beams = beams.gather(1, positions)
```
`Frontier.keep`, the `[:, input_length:]` slices, and the final `best[prompt_len:]` slice are then valid
without change. When `pad_token_id` is None the mask is all ones and both steps are no-ops.

**Rejected.** Carrying an explicit `(ids, mask, lengths)` frontier structure through `Frontier` (a
refactor of a working class to fix one call site); re-stripping pads inside `Frontier.keep` (does not fix
the model seeing right-padded context).

**Tests.** `tests/controls/test_search_driver.py`: `SearchDecoding(propose_mode="sample", segment_len=2,
max_iterations=2, keep_k=2, num_candidates=2)` on the tiny Gemma fixture (pad ≠ eos) with a scorer that
prefers the shortest continuation so one beam finishes early; assert no `pad_token_id` appears before the
last non-pad token of any returned row.

### A9. `BestOfN.decode` injects a hidden `max_new_tokens=256`

**Defect.** `output_control/best_of_n/control.py:63` `gen_kwargs.setdefault("max_new_tokens", 256)`.
The pipeline infers finish reasons from the caller's `params.max_new_tokens`
(`steering_pipeline.py:1788-1796`), which is None in that case, and `SearchDriver.decode` already raises
a clear `ValueError` when neither `segment_len` nor `max_new_tokens` is given (`search.py:238-242`).

**Fix.** Delete the `decode` override in `BestOfN` (and its docstring sentence about 256). The base
class error is the behavior. The notebook (`examples/notebooks/algorithms/best_of_n.ipynb`) and
`tests/controls/test_best_of_n.py` already pass `max_new_tokens`; any other test that relied on the
hidden default should pass it explicitly.

**Rejected.** Raising a BestOfN-specific error (duplicate of the base one).

**Tests.** `tests/controls/test_best_of_n.py`: `generate(...)` without `max_new_tokens` raises
`ValueError` mentioning `max_new_tokens`.

### A10. `SinglePairEstimator` co-pads on the tokenizer's side, masks the pads, and mutates the shared tokenizer

**Defect.** `state_control/common/estimators/single_pair.py:72-85` swaps `tokenizer.pad_token_id` to the
space id (a mutation of the pipeline's shared tokenizer, restored in `finally`) and calls
`tokenizer([...], padding=True)` without a side. Chat tokenizers default to `padding_side="left"`, so the
shorter prompt gets spaces *prepended* and the two prompts no longer align token by token (ActAdd pads
the shorter prompt on the right). The pads also come back with `attention_mask=0`, and `capture_hidden`
zeroes masked positions, so `h_pos - h_neg` at those positions is `h - 0` rather than the paper's
difference against attended spaces.

**Fix.** Build the pair manually, right-padded with attended space tokens, and stop touching the
tokenizer:
```python
pos = tokenizer.encode(positive_prompt, add_special_tokens=False, truncation=True)
neg = tokenizer.encode(negative_prompt, add_special_tokens=False, truncation=True)
space = (tokenizer.encode(" ", add_special_tokens=False) or [tokenizer.pad_token_id])[:1]
width = max(len(pos), len(neg))
rows = [ids + space * (width - len(ids)) for ids in (pos, neg)]
enc = {
    "input_ids": torch.tensor(rows, device=device),
    "attention_mask": torch.ones(2, width, dtype=torch.long, device=device),
}
```
Docstring: "the shorter prompt is right-padded with attended space tokens (ActAdd §3)".

**Rejected.** `padding_side="right"` on the tokenizer call while keeping the pad-id swap (still mutates
shared state and still masks the spaces); a `pad_strategy` argument.

**Tests.** `tests/controls/test_act_add.py`: with a tokenizer copy set to `padding_side="left"`, the
fitted directions equal those from a `padding_side="right"` copy. Regenerate the ActAdd goldens in
`tests/controls/test_position_tracking_goldens.py` (fidelity change, expected).

### A11. `RewardModelScorer` truncates without `max_length`

**Defect.** `output_control/common/scorers/reward_model.py:37-39` tokenizes with `truncation=True` and no
`max_length`; `loading.py:53-56` stores the reward model's limit on `tokenizer.max_length`, and
`values/reward_model.py:121-128` reads it. The scorer therefore truncates at the tokenizer's
`model_max_length` (often the `int(1e30)` sentinel → no truncation → reward-model position overflow).

**Fix.** `max_length=getattr(self.tokenizer, "max_length", None)` in the scorer's tokenizer call, same
as the value class.

**Tests.** `tests/controls/test_reward_model_scorer.py`: a tokenizer with `max_length=8` yields
`input_ids.shape[1] <= 8` for a long continuation.

### A13. DPO mixin overwrites `self.train_dataset` / `self.eval_dataset` with the standardized copies

**Defect.** `wrappers/trl/dpotrainer/base_mixin.py:54-61` assigns the standardized datasets back to the
instance, so a second `steer()` (re-fit after `thaw()`, or an eval sweep reusing the control) re-renders
already-rendered prompts (`preference_schema.py:181-183` re-renders when `prompt_format != "raw"`), and
`fit_identity`/`config_id` now digest the derived dataset rather than the user's. The SFT, GRPO and PPO
mixins already use locals.

**Fix.** Local variables `train_dataset`/`eval_dataset` in the DPO mixin, passed to `DPOTrainer(...)`.
Four lines.

**Tests.** `tests/controls/test_dpo_wrapper.py`: after `steer()` (smoke-gated) `control.train_dataset is
original_dataset`; `fit_identity()` unchanged before/after steer.

### A21. PASTA's constructor `substrings` is validated and serialized but never applied

**Defect.** `pasta/args.py:13-16,40-50` declares and validates `substrings`; `pasta/control.py:251-252`
raises unless `runtime_kwargs["substrings"]` is present, and only `frozen_form` (:412) reads
`self.substrings`. The documented knob is dead.

**Fix.** Fall back to the constructor value:
```python
substrings = (runtime_kwargs or {}).get("substrings", self.substrings)
if substrings is None:
    raise ValueError("PASTA requires 'substrings' at construction or inside runtime_kwargs.")
token_ranges, input_len = self.locate_spans(input_ids, substrings)
```
`_normalize_substrings` already interprets `list[str]` (one group, batch 1) and `list[list[str]]` (one
group per row), so the constructor form has the same meaning as the runtime form. Update the
`get_hooks` docstring and the `substrings` help text ("default spans, overridden per call by
`runtime_kwargs['substrings']`").

**Rejected.** Deleting the argument (touches `frozen_form`, the reference page, and the eval-side
`RUNTIME_KWARGS_SCHEMA` consumers for no user benefit).

**Tests.** `tests/controls/test_pasta.py`: `PASTA(substrings=["x"], ...)` generates at batch 1 without
runtime kwargs; a runtime `substrings` overrides it; neither → `ValueError`.

### A24. vLLM capture rejects `last_token` when the plugin advertises only the wire mode it will use

**Defect.** `backends/vllm/session.py:236-247` builds `required = CaptureKinds(modes={mode})` and raises
when the backend does not advertise `mode`, but the request always goes out as
`wire_mode = "all_tokens" if mode == "last_token" else mode` (:263).

**Fix.** Compute `wire_mode` before the capability check and require `modes=frozenset({wire_mode})`.
Move the two-line comment with it.

**Tests.** `tests/core/test_vllm_capabilities.py` (fake backend): capture kinds advertising only
`{"all_tokens"}` accept `mode="last_token"`.

---

## Phase B — freeze, identity, and declaration correctness

### B14. GEPA and CPO can only take a *live* auxiliary LM, so their recipes can never be saved

**Defect.** `codec.encode` raises `SpipeSaveError("live model/tokenizer objects do not serialize; pass
the corresponding *_name_or_path reference instead.")` (`codec.py:556-560`), and `_recipe_args_encoded`
(`freeze.py:71-80`) encodes every init arg — frozen or not. `PRewrite` offers
`rewriter_model_name_or_path` (`prewrite/args.py:26-51`, loaded in `_resolve_rewriter`,
`control.py:195-210`); `GEPA` (`gepa/args.py:82-89`: `reflection_lm`, `reflection_tokenizer`) and `CPO`
(`cpo/args.py:67-77`: `prompt_lm`, `prompt_tokenizer`) have no reference form, so the codec's advice
cannot be followed and a pipeline containing either control with an auxiliary model is unsaveable.
(`config_id`/`fit_identity` of a live object canonicalize value-blind to its type name — acceptable
for session-local objects, and moot once a name is used.)

**Fix.** Mirror PRewrite exactly; no change to `frozen_form`/`fit_identity`.
1. `gepa/args.py`: `reflection_lm_name_or_path: str | None = None` (help: "Hub id or path of the
   reflection model, loaded at steer; mutually exclusive with `reflection_lm`"). Validation: error when
   both a name and a live object are given. Keep the existing both-or-neither rule for
   `reflection_lm`/`reflection_tokenizer`.
2. `gepa/control.py` `steer` (≈:149-152): resolve the reflection pair through a `_resolve_reflection`
   helper copied from PRewrite's `_resolve_rewriter`: live object → use it; name → `AutoModelForCausalLM`
   + `AutoTokenizer` (`device_map="auto"`, `trust_remote_code=self.trust_remote_code`, pad=eos fallback);
   neither → task model.
3. `cpo/args.py` / `cpo/control.py`: `prompt_lm_name_or_path` the same way; the `__post_init__` rule
   "`prompt_lm` is required when offline_data is generated from train_dataset" accepts either form.
   `requirements()` (:118-129) and `steer_access()` (:132-140) both dispatch on
   `self.prompt_lm is not None`; make them dispatch on "a proposer was supplied" (live object *or* name)
   so a named proposer is prompt-only / ROLLOUTS exactly like a live one. CPO's proposer is also used at
   generate time, so the model loaded from the name stays on the instance as a live `prompt_lm` does.
4. `fit_identity` for both includes the new field automatically (it iterates init fields).

**Rejected.** Dropping live objects from `frozen_form` (the recipe section still fails to encode, so it
buys nothing); a generic "model reference" codec tag that serializes a live model by its `name_or_path`
(hides a device/dtype decision in a save path).

**Tests.** `tests/core/test_spipe_recipe_only.py`: `GEPA(reflection_lm_name_or_path="tiny", memory=...)`
and `CPO(prompt_lm_name_or_path="tiny", memory=...)` round-trip through `to_spipe()`/`SPipe.load()` with
equal `config_id`. `tests/controls/test_gepa.py`: live object + name → `ValueError`.

### B15. A recipe constructed with `memory=` re-exports that memory and blocks model-free freezing

**Defect.** `gepa/control.py:100-103`, `prewrite/control.py:90-92`, `cpo/control.py:144` return
`{"memory": self.memory}` whenever `self.memory` is set — including when it came from `args.memory`.
`_recipe_frozen_blockers` (`freeze.py:179-216`) treats a non-empty `export_state()` as a blocker, so a
recipe that is already its frozen form (`steer_access()` is FACTS, `fit_identity()` is None) cannot be
frozen without a model, unlike `SystemPrompt`/`FewShot`.

**Fix.** In all three: `return {} if self.args.memory is not None else ({"memory": self.memory} if self.memory is not None else {})`.
Docstring: "nothing when the recipe already carries the memory".

**Tests.** Extend `tests/core/test_spipe_recipe_only.py` with a `PRewrite(memory=...)` pipeline frozen
with `to_spipe(freeze=True)` and no model.

### B16. SASA declares no fit and does not verify a loaded probe's model identity

**Defect.** `output_control/sasa/control.py` has `export_state`/`frozen_form`/`fit_identity`
(:121-141) but no `steer_fits()`, so the `SteerPlan` lists no calibrated fit for it
(`steering_pipeline.py:674-688` — the cross-venue "calibrated thresholds may shift" notice can never
fire). `steer` (:180-220) loads a probe from `wv_path` and `_validate_probe_space` (:26-48) checks only
location/pooling/layer, so a probe fitted on another model of the same depth is accepted silently,
whereas `RoutedDecoding` checks fingerprint and `model_type` with an `allow_model_mismatch` escape
(`routed_decoding/control.py:237-285`).

**Fix.**
1. `steer_fits`: `return () if self.wv_path else (("ProbeFitSpec", "calibrated"),)`.
2. `sasa/args.py`: `allow_model_mismatch: bool = False` (same help text as RoutedDecoding's).
3. In `steer`, after loading a probe and before `_validate_probe_space`:
   ```python
   if not self.allow_model_mismatch:
       live = model_fingerprint(model)
       recorded = self.probe.meta.get("model_fingerprint")
       if recorded not in (None, live):
           raise ValueError("SASA probe was fitted on a different model (fingerprint ...); refit, or set allow_model_mismatch=True.")
       live_type = getattr(model.config, "model_type", "unknown")
       if self.probe.model_type not in ("unknown", live_type):
           raise ValueError(...)
   ```
   SASA is MODULE access, so `model` is always live here; legacy `.pt` probes carry no fingerprint and
   `model_type="unknown"`, and pass.
4. `frozen_form` carries `allow_model_mismatch` forward like the other decode-time settings.

**Rejected.** Wrapping the loaded probe in `VerifiedPrecomputed` (that wrapper is for steering vectors
resolved through the state-control path, not for probes).

**Tests.** `tests/controls/test_sasa.py`: `steer_fits()` shape; a probe with a foreign
`meta["model_fingerprint"]` raises unless `allow_model_mismatch=True`.

### B17. `ArtifactStore` partial writes leave a payload without a sidecar that is never repaired

**Defect.** `spipe/store.py:380-421` `put_tensors`/`put_tree` return early when `TENSOR_FILE` /
`PAYLOAD_DIR` exists, but `has()` (:362-364) and `record_for` key on `SIDECAR_FILE`. A crash between
payload and sidecar leaves an entry that `has()` reports missing and `put_*` refuses to rewrite.

**Fix.** Key the early return on the sidecar (the completion marker): `if (entry / SIDECAR_FILE).exists(): return`.
Before writing, `shutil.rmtree(entry, ignore_errors=True)` if the directory exists (it is then a partial
write). Keep writing the sidecar last, and write it atomically: `_write_sidecar` dumps to
`artifact.json.tmp` in the same directory and `os.replace`s it onto `artifact.json`, so a reader never
sees a half-written marker even with a concurrent writer (two writers of one id produce identical bytes,
so the only hazard was the marker itself). Document in the module docstring: "the sidecar is the commit
marker; it is written last and atomically".

**Rejected.** Staging the whole artifact directory under a temp name and renaming it into place (a
directory-level transaction; the sidecar-last protocol plus the atomic marker write gives the same
crash safety for a content-addressed, idempotent store at two lines of code).

**Tests.** `tests/core/test_spipe_store.py`: create a payload without a sidecar, call `put_tensors`
again, assert `has()` is True and the tensors load.

### B18. The recorded `model.revision` is never applied on load

**Defect.** `spipe/freeze.py:247,348` records `hf_model_kwargs["revision"]`; `spipe.py:868-873`
`pipeline()` forwards only `model_name_or_path`, so a bundle frozen from a pinned revision loads the
default branch silently.

**Fix.** In `SPipe.pipeline()`: `hf_model_kwargs = dict(hf_model_kwargs or {}); if revision and "revision" not in hf_model_kwargs: hf_model_kwargs["revision"] = revision` where `revision` is the manifest's
recorded value. Three lines; the caller's explicit kwargs still win, which is the documented contract
("backend, device, and dtype stay the caller's choice"). Add one sentence to `docs/concepts/spipe.md`
under *Saving and loading*.

**Tests.** `tests/core/test_spipe_identity.py`: a manifest with `model.revision="abc"` produces
`pipeline(...).hf_model_kwargs["revision"] == "abc"` (model loading can be monkeypatched).

### B23. Batching and schema declarations: `BudgetForcing` undeclared; `"description"` key; dead arms

**Defect.** `phased_decoding/control.py:112` and `routed_decoding/control.py:74` each declare
`supports_batching = True`; `BudgetForcing(PhasedDriver)` declares nothing and inherits `False`
(`output_control/base.py:116`), so the evaluation provider (`provider.py:247`) serializes arms containing
it although the driver is batch-capable. `routed_decoding/control.py:82` uses `"description"` where the
schema contract uses `"help"` (`base_control.py:33`). The dead `else None` arms are handled by A7.

**Fix.** Declare `supports_batching: bool = True` once on `PhasedDriver` (`drivers/phased.py`) and
delete the two subclass declarations (one declarative statement). Rename the key to `"help"`. Update
`docs/concepts/controls.md:273` ("Among the drivers, only `PhasedDecoding` and `RoutedDecoding`
declare…" → "The phased drivers declare `supports_batching`; the search drivers do not").

**Tests.** `tests/controls/test_budget_forcing.py`: `BudgetForcing(...).supports_batching is True`;
`runtime_kwargs_schema([RoutedDecoding(...)])["canned_responses"]["help"]` exists.

### B36. Weights-grade and config-grade fingerprints are compared as if they were the same thing

**Defect.** `ModelFacts.model_fingerprint` (`core/execution/payloads.py:109`) is a 16-hex digest in both
venues, but in process it is `model_fingerprint(model)` (weights-grade) and on vLLM it is
`_config_layout`'s config digest (`backends/vllm/backend.py:112-118`). `VerifiedPrecomputed._check`
(`sources.py:289-299`) and `Intervention._validate_readout` (`specs.py:339-348`) compare the recorded
fingerprint with `layout.model_fingerprint` regardless of grade, so an artifact fitted in process and
installed on an engine (or the reverse) always mismatches — a hard error for calibrated artifacts under
`verify="strict"`. `RoutedDecoding` already dispatches on `ScopedSession.in_process`
(`session_utils.py:215-220`) and falls back to `model_ref` comparison on engines.

**Fix.**
1. `ModelFacts`: `fingerprint_grade: Literal["weights", "config"] = "weights"`. `_config_layout` passes
   `fingerprint_grade="config"`. (`core/internals/facts.py:46` keeps the default.)
2. `VerifiedPrecomputed._check` and `_validate_readout`: compare fingerprints only when
   `layout.fingerprint_grade == "weights"`; otherwise compare `provenance.get("model_ref")` /
   `readout.model_ref` against `layout.model_ref` when both are present (hard for calibrated, warn for
   direction — the same policy the fingerprint path uses).
3. `ScopedSession.in_process` becomes `self.layout.fingerprint_grade == "weights"` (False when the
   venue has no layout) — one source of truth for the dispatch RoutedDecoding already performs.
4. `docs/concepts/spipe.md` *Verification*: one sentence — on engine backends the check compares the
   served model reference, since weights are not readable.

**Rejected.** Making the vLLM layout's fingerprint `None` (then engine-fitted artifacts have no identity
at all); encoding the grade in the digest string (parsing).

**Tests.** `tests/core/test_identity_checks.py`: a `ModelFacts(fingerprint_grade="config", model_ref="m")`
layout accepts an artifact with a weights-grade fingerprint and `model_ref="m"`, rejects `model_ref="n"`
for a calibrated artifact.

---

## Phase C — documentation and style

### C12. `InputControl.adapt` docstring does not say what layout it returns

**Fix.** State in `input_control/base.py` (`adapt` docstring): "Returns the batch in the padding
layout it was received in; the pipeline re-infers the attention mask from pad runs when the length
changes." No code change (the behavior is correct; it was only undocumented).

### C19. Recipe-only `config_id` divergence — defer

`docs/concepts/spipe.md:149-153` already documents this as a `spipe/2` change. No action in this round.

### C20. Unreachable duplicate `PoolMemory` code check

`spipe/codec.py:759-764` repeats the `allow_code` check already performed at :707. Delete the second
block.

### C22. `PER_STEP_LOGIT_SPECS` / `ProcessorSpecEntry` have no producer or advertiser

Forward-looking seam used by the vLLM-Hook roadmap. No change; add one sentence to the `Capability`
docstring: "`PER_STEP_LOGIT_SPECS` is reserved; no shipped backend advertises it."

### C25–C32. Documentation corrections

| # | File | Change |
|---|------|--------|
| C25 | `output_control/common/drivers/phased.py:82-93` (`PlanState.last_stop`) | Reorder to match `_stop_cause` (:509-514): `until_token`, `until`, then `eos` (an eos token or stop rule, checked before the length tests), then `budget`, then `length`. |
| C26 | `docs/tutorials/add_method_by_category/add_new_output_control.md:19-22` | `class BestOfN(SearchDriver)` (the real base; `SearchDecoding` is a sibling preset). Mention `BudgetForcing(PhasedDriver)` likewise. |
| C27 | `docs/tutorials/add_method_by_category/add_new_input_control.md:75-125` | Make `PromptCensor.adapt` batch-correct: decode and re-encode every row, right-pad to a common width with `tokenizer.pad_token_id`, return `[B, L']`; add `"scope": "call"` (and `"help"`) to its two schema entries. |
| C28 | `docs/reference/algorithms/structural_control/trl_wrapper.md` | Remove the `!apotrainer`, `!dpotrainer`, `!sfttrainer` filters so all five trainers render (GRPO/PPO already do). |
| C29 | `docs/concepts/controls.md:264` | `RoutedDecoding` backends: "HF, vLLM with the hook plugin (the probe pass needs hidden-state capture)". `:109` per A7. `:273` per B23. |
| C30 | `NOTICE` | Toolkit version 0.5.5; MergeKit pin is a git commit (`pyproject.toml:69`), not `>=0.1.4,<0.2.0`; Inspect pin `>=0.3.262` (`:60`). `mkdocs.yml:55-58`: remove the "Aug 2025 initial release" announcement. `docs/.nav.yml:27,79-82`: delete the commented `benchmarks.md`/`dev_notes` lines (the files do not exist). `examples/index.md:193` "desribe" → "describe". |
| C31 | `AGENTS.md:304` | `artifact_dir` is optional on serve: without it, payloads are `PUT` to `/v1/hook/artifacts/{id}` (`backends/vllm/backend.py:414-437`); with it, the directory must be the server's registry on a shared filesystem. `:478-479`: `summarize_runs` also emits `{metric}_sem` (`evaluation/runner.py:800`). |
| C32 | `examples/notebooks/studies/instruction_following/artifacts/pasta_profiled.spipe` | Regenerate under the pinned toolchain (`requires-python>=3.12`; the shipped lock says python 3.11.13, toolkit 0.5.0) once A5 lands, so the lock's fingerprint matches the new algorithm. Follow-up task; the bundle loads today because its PASTA head map is an opaque artifact that no fingerprint check covers. |

### C33. Style: shims and dead code to delete

| Site | Action |
|------|--------|
| `state_control/common/model_layout.py` (compat re-export) | Delete; point `tests/controls/test_intervention_ir.py:518` at `steerability.algorithms.core.internals.model_layout`. |
| `state_control/common/token_scope.py:7-33` `compute_prompt_lens(input_ids, pad_token_id=None)` | Drop the "Unused. Kept for API compatibility" parameter; update the caller `state_control/base.py:311` and `tests/controls/test_state_common.py:360,369`. |
| `backends/vllm/rendering.py:234-240` `GuidedDecodingParams` ImportError fallback | Delete (unreachable under the `vllm>=0.26.0` pin). |
| `input_control/few_shot/selectors/epr/utils/train_encoder.py:168-184` `train()` alias | Delete; call the real function. |
| `state_control/common/transforms/context.py:109-112` and `sources.py:137-142` `except TypeError` | Delete (A6 covers `sources.py`); `ArtifactSource.resolve` takes `session=` (`sources.py:51-52`). Update the custom source in `tests/controls/test_activation_adapter.py:62` to `def resolve(self, model, tokenizer, *, session=None)`. |
| `output_control/deal/args.py:28` `metadata={"help": ""},  # todo: write description` | Write the `reward_func` help ("`(prompt: str, continuations: list[str], params: dict) -> list[float]`, higher is better") and a `DeALArgs` class docstring. |
| `steering_pipeline.py:130,195` `lazy_init` ("Deprecated and inert") | Delete the parameter, its docstring entry, and `tests/core/test_construction_semantics.py::test_lazy_init_is_inert`. |

---

## Phase D — tests and follow-ups

### D34. Test hygiene

1. `tests/core/test_spec_hook_equivalence.py:13` and `tests/controls/test_intervention_ir.py:45`
   `importorskip("vllm_hook_plugins")`: add one test in `tests/core/test_environment.py` that *fails*
   (rather than skips) when `STEERABILITY_REQUIRE_PLUGINS=1` is set and `vllm_hook_plugins` is not
   importable, so a CI job installed with the `vllm` extra cannot silently skip the spec/hook parity
   suite. Same pattern for the MergeKit construction test (`merging` extra).
2. `tests/controls/test_position_tracking_goldens.py:88-97`: the ITI and CAA goldens are identical for
   prompt lengths 1 and 4, which means the fixture saturates and the test cannot detect a
   position-tracking regression. Pick a non-saturating config (smaller multiplier or a longer prompt),
   regenerate with `STEERABILITY_CAPTURE_GOLDENS=1`, and assert the two lengths differ.
3. Document the smoke gates `RUN_TRL_SMOKE` (`tests/core/test_staged_steer.py:613`) and
   `RUN_GRPO_SMOKE` (`tests/controls/test_grpo_wrapper.py:133`) and the seed-dependent skips
   (`test_cast_conditional.py:182,200,360`, `test_routed_decoding.py:165`, `test_probe_condition.py:101`)
   in `tests/README.md` (create it; one table).
4. There is no `.github/workflows/`; add a minimal `ci.yml` running `uv sync --extra eval` and
   `uv run pytest -q tests/ -m "not slow"` on 3.12. Out of scope for the fixes above but it is what keeps
   them fixed.

---

## Acceptance checklist

- [ ] Phase A: `uv run pytest -q tests/controls tests/core` green; new tests listed in A1–A24 present.
- [ ] A5 changelog entry written; `pasta_profiled.spipe` regenerated (C32) or its README notes the
      `verify="warn"` requirement.
- [ ] Phase B: `tests/core/test_spipe_*` green, including the new recipe-only freeze cases (B14, B15).
- [ ] `grep -rn "except TypeError" steerability/algorithms/state_control` returns nothing (A6, C33).
- [ ] `grep -rn "supports_batching" steerability/algorithms/output_control` shows the declaration on
      `PhasedDriver` only among the phased drivers (B23).
- [ ] `grep -rn "mkdtemp" steerability/algorithms/structural_control` matches only
      `base_mixin._resolved_output_dir` (A7).
- [ ] `mkdocs build --strict` passes after C25–C31 (removing the filters in C28 exposes any broken
      docstring in the DPO/APO/SFT mixins — fix those docstrings rather than restoring the filters).
- [ ] No new public parameter other than: `GEPA.reflection_lm_name_or_path`, `CPO.prompt_lm_name_or_path`,
      `SASA.allow_model_mismatch`, `ModelFacts.fingerprint_grade`.
