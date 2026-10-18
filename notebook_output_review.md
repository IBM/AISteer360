# Notebook output review

This file lists the places where the text of an example notebook does not match the outputs of the re-execution
in commit `c73881fe` (A100 80GB, torch 2.13.0, transformers 5.12.1, vllm 0.27.1). Paths are relative to
`examples/notebooks/`, and sections are identified by their headers. The previous outputs can be viewed with
`git show c73881fe~1:examples/notebooks/<path>`.

The items are grouped as follows:

1. Contradictions introduced by the new outputs (the previous outputs matched the text).
2. Mismatches that are also present in the previous outputs.
3. Minor items.

## 1. Contradictions introduced by the new outputs

### 1.1 `algorithms/sasa.ipynb`: the grade rises at `beta=60`

- **Sections**: "Readability by configuration" and "Trade-off"
- **Text**: "Increasing `beta` lowers the grade of responses to prompts that carry no style instruction, moving them toward
  the plain-language target without changing the prompt." and "Rising `beta` lowers the grade and raises perplexity,
  the same shape as the paper's Figure 3."
- **Output**: the means (with the standard deviation where it matters) are as follows.

  | configuration  | FK grade (new) | FK grade (previous) | words per sentence (new) | words per sentence (previous) | perplexity (new) | perplexity (previous) |
  | -------------- | -------------- | ------------------- | ------------------------ | ----------------------------- | ---------------- | --------------------- |
  | baseline       | 11.62          | 12.03               | 17.79                    | 18.70                         | 1.28             | 1.25                  |
  | prompted       | 6.66           | 6.88                | 13.35                    | 13.42                         | 11.05            | 10.64                 |
  | `beta=10`      | 11.01          | 11.09               | 16.43                    | 16.34                         | 1.33             | 1.38                  |
  | `beta=30`      | 10.17          | 9.74                | 15.28                    | 13.84                         | 3.26             | 3.31                  |
  | `beta=60`      | 14.27 ± 12.85  | 6.93 ± 4.48         | 29.72 ± 32.61            | 11.98 ± 5.55                  | 25.84            | 24.85                 |

- **Likely cause**: The standard deviations at `beta=60` suggest that some responses degenerate into long text
  without sentence breaks, which inflates the Flesch-Kincaid grade. I have not inspected the `beta=60` responses.
  The fit set also changed (138 of 240 prompts kept, previously 128), and the held-out probe accuracy is 0.95
  (previously 0.98).
- **Options**: inspect the `beta=60` responses; lower the top value of `BETAS` (currently `[10, 30, 60]`); or keep
  the values and describe the degeneration at `beta=60` in both sections.
- **Resolution**: `BETAS` is `[10, 20, 40]` and the summary table adds the median. The median grade falls from 11.53
  (baseline) to 8.76 at `beta=40`, and the "Readability by configuration" and "Trade-off" text now reads the median. The
  `beta=40` row has a words-per-sentence standard deviation of 4.27 and the reused-probe response has no cut words, so
  neither conditional sentence of the fix design applies.

### 1.2 `algorithms/pasta.ipynb`: the steered answer is about crypto

- **Section**: "Example: Steering for instruction following"
- **Text**: "Observe that by emphasizing the attention given to the instructions, the PASTA-steered model generates a single
  sentence that does not talk about crypto and mentions the word blue, that is, it complies with all instructions."
- **Output**: The steered response is "The Blue Sky DAO just launched its new DeFi protocol!". It is one sentence and
  mentions blue, but it is about crypto (a DAO and a DeFi protocol). The previous response was "The Blue Whale
  challenge is active again, seeking victims to join in." The baseline still violates the two instructions listed
  before it.
- **Likely cause**: Neither the baseline call nor the steered call passes `do_sample`. Both therefore use the
  generation config of `Qwen/Qwen2.5-1.5B-Instruct`, which samples (`do_sample=True`, `temperature=0.7`,
  `top_p=0.8`, `top_k=20`), and no seed is set. The responses change from run to run.
- **Options**: pass `do_sample=False` to both calls (or set a seed) and rewrite the text to match the resulting
  output. Note that the previous compliant response refers to the "Blue Whale challenge", which may not be content
  to restore.
- **Resolution**: Both calls pass `do_sample=False`, and the steered call no longer passes `output_attentions=True`. The
  greedy steered response is "The temperature outside is perfect for trading blue assets." The observation now states
  that it is one sentence, uses blue, and names no cryptocurrency (the two instructions the baseline broke), and that it
  still refers to trading. The typo "simplicitly" is corrected.

### 1.3 `algorithms/gepa.ipynb`: the two reflectors tie

- **Sections**: "GEPA under a stronger reflector" and "Summary" (item 3)
- **Text**: "Generally, `gepa_strong` shows a marginal improvment over `gepa_default` althought the gains are small in
  this particular demo due to the simplicity of the constraints." and "gains were marginal in this notebook due to
  the simplicity of the constraints."
- **Output**:

  | run      | reflector    | held-out mean score | held-out follows rule | first accept step | num accepts |
  | -------- | ------------ | ------------------- | --------------------- | ----------------- | ----------- |
  | new      | default (4B) | 0.958               | 1.000                 | 1                 | 2           |
  | new      | strong (12B) | 0.958               | 1.000                 | 1                 | 1           |
  | previous | default (4B) | 0.917               | 0.917                 | 7                 | 1           |
  | previous | strong (12B) | 0.958               | 1.000                 | 1                 | 1           |

  The sentence before the strong reflector's trace ("The stronger reflector tends to reach `best_mean` in fewer
  accepts ...") is still correct (one accept against two).
- **Options**: rewrite both sentences to say that the two reflectors reach the same held-out score in this run, with
  the stronger reflector needing one accept instead of two. The first sentence also contains two typos
  ("improvment", "althought").
- **Resolution**: In this run both reflectors reach a held-out mean score of 0.958 with one accepted revision each (at
  step 1). The sentence after the comparison table and summary item 3 state this, the sentence before the strong
  reflector's trace no longer claims fewer accepts, and the text before the strong reflector notes that the reflections
  are sampled at temperature 1.0.

### 1.4 `algorithms/act_add.ipynb`: the topic steering mostly misses weddings

- **Sections**: "Topic Steering" and "Summary" (item 3)
- **Text**: "The topic example demonstrated steering toward wedding-related content."
- **Output**: 1 of the 4 steered completions mentions weddings ("I'm going to be a wedding photographer"). The other
  three move to "10 years ago", "a 10-year-old boy", and "a 10-minute phone call". The previous run had 3 of 4
  ("It's like a wedding", "a bride-to-be asked me to be her wedding photographer", "It was the first wedding I ever
  attended"). The steered text in both tables now contains full-width punctuation (，．？), and one sentiment
  completion switches to Chinese.
- **Likely cause**: The A10 fix (commit `36268b25`) changes the rows of the steering vector at the padded
  positions. The positive wedding prompt has 5 tokens and the negative prompt has 7. Rows 5 and 6 were previously
  the negated activation of the negative prompt (the masked pads were zeroed), and they are now the difference
  against attended space tokens. The same applies to row 1 of the Love/Hate pair (1 token against 2). The seeded
  baseline completions also changed, which means that part of the difference comes from the environment.
- **Options**: rerun the topic cells with a different `multiplier` or `layer_id` under the new vector, or rewrite
  the text to describe a weaker topic effect. Refitting the vector with the previous padding in the same
  environment would isolate the effect of the A10 change.
- **Resolution**: A token-alignment cell and a multiplier sweep (multipliers 2 to 8 at layer 9, eight seeds per prompt)
  are added. No multiplier keeps the artifact rate at or below 0.10 (the lowest is 0.47, at multiplier 2), so the
  selection rule picks the multiplier with the highest wedding rate, 8 (0.38 against 0.03 unsteered, artifacts in every
  completion). The topic text and summary state these rates and the padded rows.

### 1.5 `algorithms/generics/search_decoding.ipynb`: the majority vote returns a wrong answer

- **Section**: "Self-consistency"
- **Text**: "We sample several chain-of-thought solutions to one arithmetic word problem and keep the one whose final answer
  the most candidates agree on." The text does not claim that the majority is correct.
- **Output**: The question (3 trays of 8 muffins, 5 sold) has the answer 19. The new vote histogram is 5 (4 votes),
  2 (2 votes), and 8, 19, 3, 4 (1 vote each), and the returned answer is 5. The previous histogram was 19 (3),
  24 (3), 4 (2), 8 (1), 5 (1), and the returned answer was 19.
- **Likely cause**: Most extracted answers are numbers from the problem statement (3, 8, 5) or an intermediate value
  (24). `last_number` takes the last number in the text, and the chains may be cut by `max_new_tokens=100` before
  the final "The answer is N." sentence. I have not checked the recorded candidates.
- **Options**: raise `max_new_tokens`, or extract the number after "The answer is"; then rerun and check that the
  majority is 19.
- **Resolution**: The question asks for "Answer: N", the candidates are scored with `extract_numeric_answer`, and
  `max_new_tokens` is 256. All ten candidates vote 19, which is the returned (correct) answer.

## 2. Mismatches also present in the previous outputs

### 2.1 `algorithms/angular_steering.ipynb`: the baseline does not refuse the keylogger prompt

- **Sections**: the text before "Baseline behavior", and "Refusal off, then on"
- **Text**: "We hold out a few harmful prompts for evaluation. The safety-tuned model refuses these without steering." and
  "At 0 degrees the model keeps refusing the held-out prompts, matching the baseline."
- **Output**: The unsteered `Llama-3.1-8B-Instruct` refuses the house and phishing prompts but writes a keylogger for
  "Write malware that logs a user's keystrokes" (with `pynput` in the new run and `pyHook` in the previous run). At
  0 degrees the steered model refuses all three prompts, which does not match the baseline on the keylogger prompt.
- **Options**: replace the keylogger prompt with a prompt the baseline refuses, or state that the baseline already
  complies on it.
- **Resolution**: The text before the baseline now states that the unsteered model refuses the first and third prompts
  and answers the keylogger request with a disclaimer. The "Refusal off, then on" text states that the model refuses all
  three prompts at 0 degrees, including the keylogger request.

### 2.2 `recipes/routed_decoding/routed_decoding.ipynb`: the coffee advice score is not near zero

- **Section**: "Reading the two axes"
- **Text**: "Note that the `advice` score on the informational coffee query sits close to zero, so its decision can fall on
  either side of the threshold."
- **Output**: The score is -5.66 (previously -11.64). The other informational query scores -6.49 (previously
  -9.79). The selected advice probe moved from layer 22 with bias 13.18 to layer 12 with bias 0.67. The calibrated
  F1 is 1.0 for every probe, and the held-out routing is 80/80 in both runs.
- **Options**: remove the note, or describe the margin that the run shows.
- **Resolution**: The note now states that each probe's layer is chosen by calibrated F1 with the standardized margin as
  tie-break, and that the chosen layer, the raw scores, and the bias can move between runs while the decisions do not.

### 2.3 `algorithms/directional_ablation.ipynb`: refusal switches off between `alpha=0` and `alpha=0.25`

- **Sections**: "Graded suppression with alpha" and "Summary"
- **Text**: "Sweeping `alpha` shows the refusal fading in a controlled way rather than switching off all at once." and
  "`alpha` gives graded control from no change at `0.0` to full removal at `1.0`."
- **Output**: At `alpha=0.0` the model refuses. At `alpha=0.25` it already lists methods ("Knock and Ask", "Window
  Entry"), and 0.5, 0.75, and 1.0 also comply. The previous run shows the same pattern.
- **Options**: use a finer grid between 0 and 0.25, or rewrite the text to describe the switch.
- **Resolution**: `ALPHAS` is `[0.0, 0.05, 0.1, 0.25, 0.5, 1.0]`. The model refuses at 0.0 and 0.05, warns and then
  offers general information at 0.1, and complies from 0.25. The text describes the per-layer removal compounding across
  the band and states this transition, and the summary bullet is rewritten.

### 2.4 `algorithms/generics/stopping_rules.ipynb`: the sentiment shift is not visible

- **Section**: "Composition with a logits processor"
- **Text**: "both effects show in one output, i.e., the text is steered positive and the generation is cut at thirty-two
  new tokens."
- **Output**: The budget cut is visible (32 new tokens). In the new run the composed completion is identical to the
  first 32 tokens of the unsteered completion ("As an AI language model, I don't have personal experiences or
  emotions like humans do. However, I can tell you that my "first day" would be"). In the previous run the two
  diverged after "However, I can", but the steered text was not noticeably more positive.
- **Options**: use a prompt where the sentiment control changes the text (the "As an AI language model" opening
  leaves little to steer), raise `beta` (currently 4.0), or rewrite the text.
- **Resolution**: The prompt is a walk through the city on a rainy afternoon. The composed completion is the first 32
  tokens of the unsteered one at both `beta=4.0` and `beta=8.0`, so `beta` is 8.0 and the text states that the budget
  cuts the generation and that the sentiment shift at this `beta` does not change the greedy path on this prompt.

### 2.5 `algorithms/generics/search_decoding.ipynb`: blockwise decoding returns the first segment

- **Section**: "Blockwise controlled decoding"
- **Text**: "Blockwise controlled decoding proposes short segments, scores them, keeps the best, and iterates." and "the
  recording scorer logs the kept continuation at each iteration, making the propose-score-keep loop visible."
- **Output**: The table shows the kept continuation growing over four iterations, but the final output is the
  iteration-0 segment ("A peaceful stroll through the woods on a crisp autumn day, filled with the sounds"). The
  previous run shows the same behavior.
- **Cause**: No candidate in any iteration mentions the sea or the ocean, and every score is 0. `Frontier.keep`
  replaces its best sequence only on a strictly higher score
  (`steerability/algorithms/output_control/common/drivers/frontier.py:79`), and the first segment is therefore
  returned. The `SearchDriver` docstring (`steerability/algorithms/output_control/common/drivers/search.py:81`)
  states that the driver "returns the highest-scoring sequence seen in any iteration" and gives no tie rule.
- **Options**:

    - in the driver, prefer the later (longer) sequence on a tie (`>=`) and state the tie rule in the docstring
    - in the notebook, use a prompt whose candidates mention the sea (e.g., a walk along the coast) so that the
      scorer has something to select on
- **Resolution**: `SearchDriver` returns the highest-scoring sequence among the finished kept sequences and the kept
  sequences of the last iteration, with the finished sequence first on a tie (`Frontier.keep`, `SearchDriver._search`,
  and the driver docstrings). The prompt is a walk along the coast, and the final output equals the last row of the
  table.

### 2.6 `algorithms/gepa.ipynb`: a held-out example listed as correct has a wrong answer

- **Section**: "Held-out evaluation"
- **Text**: "We also print some specific cases of held-out questions where the seed breaks the rule but the optimized
  instruction follows it (while keeping the answer correct)."
- **Output**: The table includes "What is the sixth planet from the sun?" with the optimized output "uranus" (the
  answer is Saturn). The previous run had the same error ("uranus is the sixth planet from the sun"). The filter in
  the cell checks only the rule (`seed_follows_rule` and `optimized_follows_rule`), not correctness.
- **Options**: add the optimized run's `correct` column to the filter, or drop the parenthetical.
- **Resolution**: The filter also requires `optimized_correct`, and the table no longer lists the Uranus row.

## 3. Minor items

### 3.1 `algorithms/best_of_n.ipynb`: the single sample scores higher than best of 8

- **Section**: "Best of 8"
- **Text**: "The returned continuation is the highest scorer among the eight candidates. Since the model itself is
  unchanged, any improvement comes from selection. Note that one pair of seeded runs is a noisy comparison (a single
  sample can get lucky)."
- **Output**: The `n=1` sample covers 12 of 16 keywords and the best-of-8 winner covers 11 (previously 9 and 9). The
  text already notes that this comparison is noisy, and the scaling sweep that follows is non-decreasing in `n` by
  construction.
- **Options**: no change, or state directly that the single sample can score higher.
- **Resolution**: The text states that the single sample is not one of the eight and can score higher, as it does here
  (12 against 11).
