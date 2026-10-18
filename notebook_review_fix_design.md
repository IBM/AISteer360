# Steerability v0.5.6 — fix design for the notebook output review

Implementation brief for Claude Code, following `steerability_fix_design.md`. It resolves every item of
`notebook_output_review.md` and the conditions found while tracing those items to the code. Line numbers are
those of the `fixes-0.5.6` branch at `09ae9b5`; the surrounding code is quoted so each site stays findable.

Findings the items rest on (details in `notebook_output_investigation.md`):

- One difference is caused by the branch: 1.4 (ActAdd), through A10. The new padding is the paper's Algorithm 1,
  so the method is right and the notebook's `multiplier` was tuned on the previous vector. Phase B1 re-tunes it
  inside the notebook.
- One pre-existing code behavior produces a reviewed difference: 2.5, where `Frontier.keep` updates the
  best-so-far only on a strictly higher score, so a search whose scores tie returns its first segment. Phase A1
  replaces the return rule.
- One edge case in the new A8 code: the frontier mask is inferred from the pad id over the whole row, prompt
  included, and the rotation then moves a pad-valued prompt token to the tail. Phase A2 guards it.
- Everything else is sampling, environment numerics (torch 2.13.0 / transformers 5.12.1 against the September
  runs), or notebook text that outran its outputs. Phase B edits the parameters and the narrative.

## 0. Ground rules

The conventions of `AGENTS.md` apply unchanged (no shims, one declarative statement, Google docstrings kept true,
tests per control under `tests/controls/`, `uv run pytest -q tests/` after each phase). Two rules for Phase B:

- A notebook's narrative is written after the notebook has been re-executed end to end on the GPU box that
  produced `c73881f` (A100, the pinned environment). Every sentence that asserts an effect ("the steered model
  mentions weddings", "the grade falls") is backed by a value the notebook prints. Where the outcome cannot be
  known before execution, this brief gives the text as a rule ("state X if the table shows Y") and the executor
  writes the sentence from the output.
- Notebook prose follows the `AGENTS.md` notebook conventions: "we", neutral register, connectives, paragraphs
  rather than bulleted claims, imports in the setup cell, explanation in markdown cells.

No new Python files are added anywhere in this brief. No new public parameters.

Suggested ordering: Phase A (two small code changes, tests, docstrings), then Phase B notebook by notebook,
re-executing each one after its source edits, then Phase C (documentation lines and the review file).

---

## Phase A — code

### A1. `SearchDriver` returns the first-segment prefix whenever scores tie

**Defect.** `output_control/common/drivers/frontier.py:78-81`:

```python
best_local = int(torch.argmax(torch.tensor(kept_scores)))
if kept_scores[best_local] > self.best_score:
    self.best_score = kept_scores[best_local]
    self.best_ids = kept[best_local]
```

`best_ids` is updated on a strictly higher score, and every kept beam of every iteration is a candidate, finished
or not. `search.py:394-396` returns `frontier.best_ids` when set. With a scorer that returns equal scores
(the blockwise cell of `generics/search_decoding.ipynb`, where no candidate mentions the sea), the iteration-0
segment is never replaced and the driver returns 16 of the 64 budgeted tokens. More generally, the rule returns an
intermediate, unfinished prefix whenever a later segment scores no higher, which neither blockwise controlled
decoding (Mudgal et al.) nor DeAL (Huang et al.) does: both continue every kept beam and return a sequence the
search stopped extending. The docstrings (`search.py:81`, `search.py:323-325`, `search_decoding/control.py:16`,
`deal/control.py:23-24`) state "the highest-scoring sequence seen in any iteration" and no tie rule.

**Fix.** A kept sequence becomes a candidate for the result when the search stops extending it: when it is
finished (eos, or the budget, as `Frontier.keep` already classifies), or when it is kept in the last iteration.
The result is the highest-scoring candidate; on equal scores the finished sequence found first wins.

1. `frontier.py` `Frontier.keep`: track finished beams only. Replace lines 78-81 with

   ```python
   for index, finished in enumerate(finished_flags):
       # kept beams arrive in descending score order, so the first finished beam at a score wins the tie
       if finished and kept_scores[index] > self.best_score:
           self.best_score = kept_scores[index]
           self.best_ids = kept[index]
   ```

   Class docstring (`:27-33`): "`best_ids` and `best_score` track the highest-scoring finished beam seen so far.
   Unfinished beams are not candidates, since the search continues them. On equal scores the earlier finished
   beam is kept." `keep` docstring (`:55`): "update the finished best-so-far".
2. `search.py` `_search`: replace lines 394-396 with

   ```python
   if kept is None:
       return prompt_ids[0]
   # the last iteration's kept sequences are also terminal; a finished sequence wins a tie
   if frontier.best_ids is not None and frontier.best_score >= kept.kept_scores[0]:
       return frontier.best_ids
   return kept.kept_ids[0]
   ```

   `kept.kept_scores` is in descending order (`torch.topk` sorts), so index 0 is the top kept sequence.
3. Docstrings. `search.py:81`: "It returns the highest-scoring sequence among those it stopped extending: the
   finished kept sequences of every iteration and the kept sequences of the last iteration. On equal scores the
   finished sequence found first is returned." `search.py:322-325` (`_search` Returns): the same rule, then "When
   no iteration runs, the prompt alone is returned." `search_decoding/control.py:15-16` and
   `deal/control.py:23-24`: replace "returns the highest-scoring sequence seen in any iteration" with the same
   sentence. `BestOfN` runs one iteration, so its "keeping the argmax" wording stays true and is unchanged.

Behavior under the presets: `BestOfN` and single-iteration `SearchDecoding` return the argmax of the one kept set
whether the candidates finished or were cut by a stop rule (all of them are last-iteration sequences). Blockwise
configurations with `keep_k=1` return the sequence after the last segment. DeAL configurations return the best
finished beam, or the best beam of the last iteration when that one scores higher.

**Rejected.** `>=` alone (prefers the later sequence on ties but keeps unfinished intermediate prefixes as
candidates, which no referenced algorithm does, and still returns a shorter prefix whenever a later segment scores
lower); returning only the last iteration's top beam (loses a finished beam that scored higher than the beams
that ran out of iterations, which DeAL keeps as a completed hypothesis); a `return_rule` argument (a flag that
keeps the wrong path alive).

**Tests.**

- `tests/controls/test_search_frontier.py:43-50` `test_best_ids_tracks_best_score_across_calls` becomes
  `test_best_ids_tracks_finished_beams_only`: `Frontier(keep_k=1, eos_token_id=9, input_length=2,
  max_new_tokens=None, pad_token_id=0)`; `keep` on an unfinished beam with score 0.9 leaves `best_ids` None;
  `keep` on a beam ending in 9 with score 0.4 sets `best_ids` to it and `best_score` to 0.4; a later eos beam with
  0.4 does not replace it (tie keeps the earlier one).
- `tests/controls/test_output_common.py::TestSearchDriver`, with the `ScriptedSession` pattern already used at
  `:380-404`: (a) `test_tied_scores_return_the_last_iteration_sequence`: constant scorer, `segment_len=2,
  num_candidates=1, keep_k=1, max_iterations=2, propose_mode="sample"`, a fake generate that appends token 5
  `max_new_tokens` times, call with `max_new_tokens=4`; expect `requested == [2, 2]` and
  `out.tolist() == [[0, 3, 4, 5, 5, 5, 5]]`. (b) `test_finished_sequence_wins_a_tie_with_the_last_iteration`
  and `test_last_iteration_sequence_wins_on_a_higher_score`: `segment_len=2, num_candidates=2, keep_k=2,
  max_iterations=2, propose_mode="sample"`, `model=None` so the eos set is the tokenizer's `</s>` (id 1), call with
  `max_new_tokens=6`. The fake generate dispatches on the frontier width and returns `num_return_sequences` rows
  per frontier row: from the prompt (width 3) it returns `[0, 3, 4, 5, 1]` ("sat", finished by eos) and
  `[0, 3, 4, 6, 6]` ("on on", unfinished); from the width-5 frontier it returns `[0, 3, 4, 6, 6, 7, 7]` ("on on
  mat mat") twice. With a text scorer `{"sat": 2.0, "on on": 1.0, "on on mat mat": 2.0}` the finished beam is
  returned, `out.tolist() == [[0, 3, 4, 5, 1]]`; with `"on on mat mat": 3.0` the last-iteration beam is returned,
  `out.tolist() == [[0, 3, 4, 6, 6, 7, 7]]`. The existing `test_last_segment_is_clamped_to_the_remaining_budget`
  (`:380`) and `test_frontier_at_the_budget_width_ends_the_search` (`:437`) keep their expected outputs (the first
  returns a budget-finished sequence; in the second the budget-finished candidate scores 0.0 and the unfinished
  last kept candidate 1.0, so the fallback returns the latter) and serve as regression checks.
- `tests/controls/test_search_driver.py` (strictly increasing scores) and the DeAL equivalence tests
  (`test_generic_output_controls.py:591`, `test_output_ports.py`) are unaffected; run them.

**Docs.** `AGENTS.md:348-351` (the search-driver sentence in Composition rules): append "Each search returns the
highest-scoring sequence among those it stopped extending (finished, or kept in the last iteration)."
`docs/concepts/controls.md:274-277`: the same clause after "run one independent search per candidate on the row's
unpadded prompt".

### A2. The A8 frontier mask is inferred over the prompt, and the rotation can move a prompt token

**Defect.** `search.py:356-359`:

```python
frontier_mask = infer_attention_mask_from_ids(current_ids, self.tokenizer.pad_token_id)
frontier_ids, frontier_mask = to_left_pad(current_ids, frontier_mask)
```

`infer_attention_mask_from_ids` marks leading and trailing runs of the pad id over the whole row. The prompt that
reaches `_search` has no padding (`decode` slices it by the caller's mask at `:261`), so inference over the prompt
positions can only be wrong: a prompt whose first or last token equals the pad id (a tokenizer whose pad is set to
an eos that is also its bos, with a prompt that starts with it; a text prompt ending in the eos string) is marked
as padding, `to_left_pad` moves a trailing one to the front, and the rotation at `:371-376` then moves it to the
tail, so `best[prompt_ids.numel():]` is off by one and the continued context lost a real token. Before A8 the
inference only hid such a token from attention; A8 added the layout change.

**Fix.** Force the prompt columns on, after the inference and before `to_left_pad`:

```python
frontier_mask = infer_attention_mask_from_ids(current_ids, self.tokenizer.pad_token_id)
frontier_mask[:, :input_length] = 1  # the prompt carries no padding; only the continuations' trailing pads are inferred
frontier_ids, frontier_mask = to_left_pad(current_ids, frontier_mask)
```

On the first iteration `current_ids` is the prompt, so the mask is all ones and A8 is a no-op, which is the
intended behavior. On later iterations the trailing pads of the continuations are the ones HF wrote. Update the
comment at `:356-357` to say so.

**Rejected.** Carrying explicit lengths through `Frontier` (rejected in A8 already); resolving the rollout pad id
from `gen_kwargs` / `generation_config` for the `Frontier` and the inference (a caller pad id that is neither the
tokenizer's pad nor an eos is the only case where it matters, and a pad equal to an eos classifies the row as
finished under rule 4 of `infer_finish_reasons` either way).

**Tests.** `tests/controls/test_output_common.py::TestSearchDriver`: `test_prompt_tokens_equal_to_the_pad_id_are_
not_treated_as_padding`: tokenizer `wordlevel_tokenizer()` (pad id 2), `driver.decode(input_ids=torch.tensor([[0,
3, 2]]), attention_mask=None, ...)` with a fake generate that records the `attention_mask` it receives and appends
`[5, 5]`, constant scorer, `segment_len=2, max_iterations=1`, `max_new_tokens=2`; expect the recorded mask to be
all ones and `out.tolist() == [[0, 3, 2, 5, 5]]`.

### A3. The `norm_input` exemption of the artifact-location check is undocumented on the control

**Defect.** `d91997e` (A6) added `if self.resolved_site() == "norm_input": return` to
`Intervention._validate_boundary` (`state_control/common/specs.py:321`) because the plane
`SteeringPlaneEstimator` fits at `train_spec.location` (layer outputs) now records that location, while
`AngularSteering` applies it at the `norm_input` site with `boundary="layer_input"`
(`angular_steering/control.py:85-88`). The exemption is stated on `_validate_boundary` only.

**Fix.** Documentation only. In the `AngularSteering` class docstring (`angular_steering/control.py:43-47`, the
"declarative" paragraph), add: "The plane is fit at `train_spec.location` (layer outputs by default) and applied
to the stream entering each layer and its mid-layer stream; the artifact-location check does not apply at the
`norm_input` site, since the stream entering layer `l` is the output of layer `l - 1`." No code change; the
exemption is correct for that site.

---

## Phase B — notebooks

Procedure for every notebook below: apply the source edits, re-execute the notebook in place on the GPU box, read
the outputs, then write the narrative cells. Keep `examples/index.md` entries as they are unless a notebook's
headline claim changes. Regenerate `.secrets.baseline` if the executed widget ids move it (as in `c73881f`).

### B1. `algorithms/act_add.ipynb` (review 1.4; follow-up to A10)

The vector rows beyond the shorter prompt are now differences against attended space tokens (Algorithm 1 of the
paper; Appendix C). Under `multiplier=8` at layer 9 those rows dominate the first prompt positions, and the
executed outputs show the signature (doubled spaces, full-width punctuation) with 1 of 4 completions on topic. The
multiplier is re-tuned inside the notebook; no helper module.

1. New markdown and code cell after cell 11 (the baseline), titled "Token alignment of the contrast pairs". The
   code prints the token count of each prompt of both pairs:

   ```python
   for positive, negative in [("Love", "Hate"), ("I talk about weddings constantly", "I do not talk about weddings constantly")]:
       pos = tokenizer.encode(positive, add_special_tokens=False)
       neg = tokenizer.encode(negative, add_special_tokens=False)
       print(f"{positive!r}: {len(pos)} tokens; {negative!r}: {len(neg)} tokens")
   ```

   The markdown states, in one paragraph: ActAdd takes the per-position difference of the two prompts' residual
   streams, so the prompts have to be aligned token by token; following Algorithm 1 of the paper, the estimator
   right-pads the shorter prompt with attended space tokens, and the rows of the vector beyond the shorter prompt
   are therefore differences between a trailing space and the longer prompt's remaining tokens. Then the counts
   from the printed output ("under Qwen's tokenizer `Hate` is two tokens and `Love` one, so the sentiment vector
   has two rows, the second against a space; the wedding pair spans five and seven tokens, so rows 5 and 6 are
   against spaces"). Write the counts after execution.
2. Cell 12 (sentiment markdown): keep layer 8 and multiplier 8; replace "Qwen's tokenizer prepends no special
   tokens to the prompt, so `alignment=0` places the steering vector on the first prompt tokens." with the same
   sentence followed by "The vector has two rows (see the token counts above), so it covers the first two prompt
   tokens." If the executed sentiment table shows the full-width punctuation or doubled spaces described under
   item 4, lower `multiplier` to 6 and say so.
3. Cell 15 (topic markdown): replace "intervening at layer 9 with a multiplier of 8 and `alignment=0` as before.
   Note that this contrast pair spans seven token positions, so the injection window covers the first seven
   tokens of the sequence." with: "intervening at layer 9 with `alignment=0` as before. The pair spans five and
   seven tokens, so the vector has seven rows and the injection window covers the first seven tokens of the
   sequence; rows 5 and 6 are differences against trailing spaces. The multiplier is chosen by the sweep in the
   next section." Keep the sentence on prompts of at least seven tokens.
4. New section between cell 16 (topic baselines) and cell 17, titled "Choosing the multiplier", one markdown
   cell and one code cell. Markdown: "The multiplier trades the strength of the topic shift against damage to the
   prompt representation at the padded positions, which shows up as doubled spaces or full-width punctuation at
   the start of the continuation. We sweep the multiplier at layer 9, sampling eight completions per prompt
   (seeds 0 to 7) for each value, and count the completions that mention a wedding-related word and the
   completions that show those artifacts; the unsteered model gives the reference rate." Code:

   ```python
   import re

   MULTIPLIERS = [2, 4, 6, 8]
   N_SEEDS = 8
   WEDDING_WORDS = re.compile(r"\b(wedding|weddings|bride|groom|marr(?:y|ied|iage)|honeymoon|vows?)\b", re.IGNORECASE)
   ARTIFACTS = re.compile(r"[\u3000-\u303F\uFF00-\uFFEF]|  ")  # full-width punctuation or a doubled space

   def topic_rates(pipeline):
       mentions = artifacts = 0
       for prompt in topic_prompts:
           enc = tokenizer(prompt, return_tensors="pt").to(device)
           for seed in range(N_SEEDS):
               torch.manual_seed(seed)
               output_ids = pipeline.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, **gen_params)
               text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
               mentions += bool(WEDDING_WORDS.search(text))
               artifacts += bool(ARTIFACTS.search(text))
       total = len(topic_prompts) * N_SEEDS
       return mentions / total, artifacts / total

   unsteered = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[])
   unsteered.steer()
   sweep_rows = [["unsteered", *topic_rates(unsteered)]]
   for multiplier in MULTIPLIERS:
       control = ActAdd(
           positive_prompt="I talk about weddings constantly",
           negative_prompt="I do not talk about weddings constantly",
           layer_id=9, multiplier=multiplier, alignment=0,
       )
       pipeline = SteeringPipeline(model=model, tokenizer=tokenizer, controls=[control])
       pipeline.steer()
       sweep_rows.append([f"multiplier = {multiplier}", *topic_rates(pipeline)])

   print(tabulate(sweep_rows, headers=["configuration", "wedding mentions", "artifacts"], floatfmt=".2f", tablefmt="grid"))
   ```

   `pipeline.generate` returns continuation-only ids, which is what the two regexes read. The fit is one forward
   pass over the pair, so refitting per multiplier costs nothing; the sweep is 160 generations of 30 tokens.
   Selection rule for the executor: `TOPIC_MULTIPLIER` is the smallest multiplier whose wedding rate is at least
   twice the unsteered rate and whose artifact rate is at most 0.10. If no multiplier qualifies, take the one with
   the highest wedding rate and describe the artifacts in the narrative. If no multiplier separates from the
   unsteered rate at all, repeat the sweep with `layer_id=6` (the paper places this family of vectors early) before
   writing anything. A markdown cell after the table states the rule and the value it selected in this run, and
   the next code cell is `TOPIC_MULTIPLIER = <value>`.
5. Cell 17: `multiplier=TOPIC_MULTIPLIER`. Cell 18 unchanged (the table of one completion per prompt remains the
   qualitative view; the rates are the quantitative one).
6. Cell 19 (summary) item 3: replace "The topic example demonstrated steering toward wedding-related content."
   with a sentence carrying the two rates: "The topic example raised the share of completions that mention
   weddings from <unsteered rate> to <rate at TOPIC_MULTIPLIER> at multiplier <value>, with the padded rows of the
   vector setting the ceiling on the usable strength." Add one sentence to the closing paragraph: "The estimator
   right-pads the shorter prompt of a pair with attended space tokens, so pairs of unequal token length carry rows
   that compare a trailing space with the longer prompt's tokens; equal-length pairs avoid this."

### B2. `algorithms/sasa.ipynb` (review 1.1)

1. Cell 6: `BETAS = [10, 20, 40]  # spans the margin scale (about one logit unit; see the mechanism section)`.
   `beta=BETAS[1]` in cell 16 and `middle = BETAS[1]` in cells 30 to 37 follow automatically (20).
2. Cell 24: aggregate `["mean", "median", "std"]` instead of `["mean", "std"]`; the column renaming already
   handles the extra statistic.
3. Cells 25 and 27 (narrative), after execution. Replace "Increasing `beta` lowers the grade of responses to
   prompts that carry no style instruction, moving them toward the plain-language target without changing the
   prompt." with a sentence that reads the `fk_grade_median` column ("The median grade falls from <baseline> to
   <beta=40> as `beta` rises, moving the responses toward the plain-language target without changing the prompt")
   plus, when the `beta=40` row has a `words_per_sentence_std` above 10, one sentence naming the cause: "At
   `beta=40` the shift is large enough to force low-probability tokens, and a few responses lose their sentence
   breaks; Flesch-Kincaid scores a run-on response in the forties, which is why the mean and the median diverge
   there." If the `beta=40` row is unusable (median grade above the `beta=20` median), set `BETAS = [10, 20, 30]`,
   re-execute, and say in the narrative that larger values derail responses. In cell 27 replace "Rising `beta`
   lowers the grade and raises perplexity, the same shape as the paper's Figure 3." with the same claim on the
   median, keeping the Figure 3 comparison only if the scatter shows it.
4. Cell 36 (reusing the probe): if the printed response shows a word cut before a newline (as "Well-construc" in
   the current run), add one sentence: "The cut words show the shift favoring a newline over a word continuation,
   which is the method's failure mode as `beta` grows."

### B3. `algorithms/pasta.ipynb` (review 1.2)

1. Cell 13 (baseline): add `do_sample=False` to `model.generate(...)`.
2. Cell 19 (steered): add `do_sample=False`; delete `output_attentions=True` and its comment. PASTA edits the 4D
   attention mask in a pre-hook on the attention module (`pasta/control.py:745-828`) and reads no attention
   weights; transformers 5 warns that the flag is not a valid generation flag.
3. Cell 20: rewrite the observation from the executed steered output. If the greedy steered response satisfies
   the four instructions, say which ones the baseline broke and that the steered response satisfies them; if it
   satisfies some, name those. Do not reuse the current sentence.
4. Cell 21: "simplicitly" to "simplicity".

### B4. `algorithms/gepa.ipynb` (review 1.3 and 2.6)

1. Cell 25: add `comparison["optimized_correct"] = eval_default["correct"]` and extend the filter to
   `& comparison["optimized_correct"]`, so the printed rows satisfy the parenthetical of cell 24 ("while keeping
   the answer correct").
2. Cell 27 (markdown before the strong reflector): after "the same `seed` (so minibatch sampling is the same)"
   add "; the reflections themselves are sampled at temperature 1.0 by the proposer, so the search trace differs
   between runs even under one seed".
3. Cell 34: replace "Generally, `gepa_strong` shows a marginal improvment over `gepa_default` althought the gains
   are small in this particular demo due to the simplicity of the constraints." with a sentence written from the
   comparison table in cell 33: name the two held-out mean scores and the two accept counts (in the current run,
   "Both reflectors reach a held-out mean score of 0.958 and follow the rule on every held-out question; the
   stronger reflector needed one accepted revision where the default reflector needed two.").
4. Cell 37 item 3: replace "gains were marginal in this notebook due to the simplicity of the constraints" with
   "in this notebook the two reflectors reach the same held-out score, since the rule is simple enough for the 4B
   model to discover; the difference shows in the search trace rather than the result" (adjust to the table).
5. The sentence in cell 30 (markdown before the strong reflector's trace, "The stronger reflector tends to reach
   `best_mean` in fewer accepts, often discovering the rule in a single revision rather than refining it over
   several.") stays only if the executed accept counts support it.

### B5. `algorithms/generics/search_decoding.ipynb` (review 1.5 and 2.5)

1. Cell 11 (self-consistency): import `extract_numeric_answer` from `steerability.utils.answers` in the setup
   cell 5; change the question's last sentence to "Think step by step and end with 'Answer: N'."; replace
   `last_number` with `extract_numeric_answer` in `recording_majority_vote` and in the histogram and the final
   print; set `max_new_tokens=256`. Delete the `last_number` definition.
2. Cell 10 (markdown): the built-in spec sentence becomes "The built-in scorer spec is `{"kind":
   "majority_vote", "answer_extractor": extract_numeric_answer}`, whose extractor anchors on an `Answer:` label
   with a last-number fallback." Keep the rest. After execution, if the majority is 19, add "The majority answer
   is the correct one, 19." If it is not, raise `max_new_tokens` to 384 once; if the majority is still wrong,
   report the histogram as it is and say that the vote is over sampled chains and can miss.
3. Cell 13 (blockwise): `bw_prompt = "Write a few sentences about a walk along the coast."`.
4. Cell 12 (markdown): append "The driver returns the highest-scoring sequence among the finished sequences and
   those kept in the last iteration, so with a monotone scorer such as this count the final output is the
   sequence kept after the last iteration." After execution, state that the final output equals the last row of
   the table.
5. Cell 18 (summary): "Blockwise controlled decoding proposed, scored, and kept short segments iteratively" is
   fine; add "and returned the sequence kept after the last segment".

### B6. `algorithms/angular_steering.ipynb` (review 2.1)

Narrative only; the prompts stay (replacing the keylogger prompt would add a new harmful prompt for no gain).

1. Cell 13: replace "The safety-tuned model refuses these without steering." with the executed behavior: in the
   current run, "Without steering the model refuses the first and third prompt and answers the keylogger request
   with a disclaimer."
2. Cell 25: replace "At 0 degrees the model keeps refusing the held-out prompts, matching the baseline." with
   "At 0 degrees the rotation aligns the in-plane component with the refusal axis, and the model refuses all three
   held-out prompts, including the keylogger request that the unsteered model answered." (confirm against the
   executed table; `mode="target"` at 0 degrees enforces refusal rather than leaving the model unchanged).

### B7. `recipes/routed_decoding/routed_decoding.ipynb` (review 2.2)

Cell 11 (markdown), the paragraph beginning "Note that the `advice` score on the informational coffee query sits
close to zero": replace with "Note that the layer of each probe is chosen among the layers that reach the best
calibrated F1 by the standardized margin on the calibration pools (`fit_probe`), which on pools of this size can
move between runs; the raw scores and the bias move with it, while the decisions do not. The next section shows how
to move the operating point." Keep the final sentence about the rules. No code change; pinning `layer_ids` would
hide the behavior the sentence now explains.

### B8. `algorithms/directional_ablation.ipynb` (review 2.3)

1. Cell 26: `ALPHAS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]`.
2. Cell 25 (markdown): replace "Sweeping `alpha` shows the refusal fading in a controlled way rather than
   switching off all at once." with "The projection runs at every layer of the band, and each layer removes
   `alpha` of the component that reaches it, so the fraction left after fourteen layers is far below `1 - alpha`;
   the sweep therefore moves quickly at small `alpha`." After execution, state the first `alpha` at which the
   response complies.
3. Cell 29 bullet: replace "`alpha` gives graded control from no change at `0.0` to full removal at `1.0`." with
   "`alpha` scales the removal per layer; across a band of layers the effect compounds, and the sweep locates the
   value at which refusal gives way."

### B9. `algorithms/generics/stopping_rules.ipynb` (review 2.4)

1. Cell 15: `compose_prompt = "Write a few sentences about a walk through the city on a rainy afternoon."`;
   keep `beta=4.0`.
2. After execution, if the composed completion differs from the first 32 tokens of the unsteered one, write cell
   14's claim from the two texts ("the steered text differs from the unsteered one from <word> on"). If the two
   are identical, set `beta=8.0`, re-execute, and say so. If they are still identical, replace "both effects show
   in one output, i.e., the text is steered positive and the generation is cut at thirty-two new tokens" with
   "the budget cuts the generation at thirty-two new tokens, and the sentiment shift at this `beta` is too small
   to change the greedy path on this prompt"; the point of the section is composition, not the size of the shift.

### B10. `algorithms/best_of_n.ipynb` (review 3.1)

Cell 15: after "Note that one pair of seeded runs is a noisy comparison (a single sample can get lucky)." add
"; the single sample is not one of the eight, since eight samples drawn in one batch consume the seeded generator
differently from one sample, and it can score higher, as it does here" (keep the last clause only if the executed
scores show it).

---

## Phase C — documentation and the review file

- `AGENTS.md:348-351` and `docs/concepts/controls.md:274-277` per A1.
- `notebook_output_review.md`: once Phase B has been executed, either remove the file or append a "Resolution"
  line to each item naming the cell edit that resolved it; it describes outputs that no longer exist.
- `mkdocs build --strict` after the notebook edits (mkdocs-jupyter renders the notebooks).

---

## Considered, no change

- `>=` in `Frontier.keep` alone: covered under A1 (rejected).
- Resolving the rollout pad id from `gen_kwargs` / `generation_config` for the frontier bookkeeping: covered
  under A2 (rejected); the vLLM arm also rejects `pad_token_id` as an unmapped parameter, so the driver must not
  inject one.
- A4 slicing of a flat `list[str]` of batch length under `seed_scope="item"` (`core/utils/assembly.py:142-160`):
  the batched path raises for that form and the per-item path reinterprets it as one substring per row. Only
  invalid input reaches it, and distinguishing the forms generically would bake PASTA's grouping into the slicer.
  No change.
- Extending `extract_numeric_answer` to anchor on "The answer is": the notebook asks for the format the extractor
  documents instead (B5).
- Persisting the SASA elicitations next to the probe: the probe is the artifact the notebook reuses; the
  elicitation is the demonstration.
- Pinning the `advice` probe's layer in the routed-decoding notebook (B7): the moving layer is the behavior the
  rewritten note explains.
- Swapping the keylogger prompt in the angular-steering notebook (B6).

## Acceptance checklist

- [ ] A1: `uv run pytest -q tests/controls/test_search_frontier.py tests/controls/test_output_common.py
      tests/controls/test_search_driver.py tests/controls/test_generic_output_controls.py
      tests/controls/test_output_ports.py tests/controls/test_deal.py tests/controls/test_constrained_decoding.py
      tests/core/test_generated_tokens.py` green, with the four new tests present.
- [ ] A2: the mask test present; `grep -n "input_length\] = 1" steerability/algorithms/output_control/common/drivers/search.py` matches once.
- [ ] `grep -rn "seen in any iteration" steerability/ docs/ AGENTS.md` returns nothing.
- [ ] Every notebook under B1 to B10 re-executed on the pinned environment, outputs committed, and no narrative
      sentence contradicts a printed value (re-read each edited markdown cell against the cell output below it).
- [ ] B1: `TOPIC_MULTIPLIER` set from the printed sweep table by the stated rule, and the summary carries the two
      rates.
- [ ] `uv run pytest -q tests/` green; `pre-commit run --all-files` clean; `mkdocs build --strict` passes.
