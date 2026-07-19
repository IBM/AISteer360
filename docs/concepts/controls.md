# Steering Controls

!!! note
    This document provides a conceptual overview of model steering. To add your own steering control/method, please refer to
    the [tutorial](../tutorials/add_new_steering_method.md). For a better understanding of how steering
    methods can be composed, please see high-level outline on [steering pipelines](steering_pipelines.md).


There are various ways to steer a model. We structure steering methods across four categories of control, loosely
defined as:

- [**input**](#input-control): edits the prompt
- [**structural**](#structural-control): edits the weights/architecture
- [**state**](#state-control): edits the (hidden) states
- [**output**](#output-control): edits the decoding/sampling process

The category of a given steering method is dictated by what aspect of the model the method influences. We define each
category of control below.


## Input control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_\theta(\sigma(x))$

Input control methods describe algorithms that manipulate the input/prompt to guide model behavior. They do not change
the model itself. This is enabled in the toolkit through a prompt adapter $\sigma(x)$ applied to the original prompt
$x$.

For a control method to be deemed an input control method, it must satisfy the following requirements:

- *Control*: Method only influences the prompt supplied to the model; does not change model's internals (parameters/states/logits)

- *Persistence*: All changes are temporary; removing the prompt adapter $\sigma()$ yields the base model.

- *Access*: Implemented without requiring access to model's internals, e.g., hidden states.

Some examples of input control methods include: few-shot prompting, reasoning guidance (like CoT, ToT, GoT,
self-consistency), automatic prompting methods, and prompt routing. The toolkit implements:

- [`FewShot`](../reference/algorithms/input_control/few_shot.md) — pool- or runtime-supplied few-shot examples; pluggable selector. See the notebook: [FewShot](../examples/notebooks/few_shot.ipynb).
- [`PRewrite`](../reference/algorithms/input_control/prewrite.md) — RL-trained instruction rewriter ([Kong et al. 2024](https://arxiv.org/abs/2401.08189)); supports a greedy "inference" strategy and a best-of-K "search" strategy. The rewriter can optionally be trained with GRPO using a metric-in-the-loop reward (apply the rewrite with the frozen task model over a dev set and score with a `Metric`, the paper's reward).
- [`CPO`](../reference/algorithms/input_control/cpo.md) — causal prompt optimization ([Chen et al. 2026](https://arxiv.org/abs/2602.01711)); offline causal reward training (Double ML over PCA-reduced embeddings) plus per-query tree search.
- [`GEPA`](../reference/algorithms/input_control/gepa.md) — reflective genetic prompt evolution ([Agrawal et al. 2025](https://arxiv.org/abs/2507.19457)); single-module variant.

The few-shot retriever from [Rubin et al. 2021](https://arxiv.org/abs/2112.08633) (EPR) is shipped as a `BaseSelector`
that slots into `FewShot` rather than as a separate control; see
[`few_shot.selectors.epr`](../reference/algorithms/input_control/few_shot.md).

Reusable building blocks shared across these methods (memory containers, formatters, scorers, proposers, selectors,
Pareto / rollout-budget utilities) live in
[`input_control._common`](../reference/algorithms/input_control/_common.md).



## Structural control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_{\theta'}(x)$

Structural control methods alter the model’s parameters or architecture to steer its behaviour. These methods usually
allow for more aggressive changes to the model (compared to input control methods). Structural controls are implemented
via fine-tuning, adapter layers, or architectural modifications (e.g., merging) to yield an updated set of weights
$\theta'$.

Structural control methods satisfy the following requirements:

- *Control*: Produces a new or modified set of weights $\theta'$ or extends the network with additional modules/layers.

- *Persistence*: Changes are persistent and live inside the checkpoint; reverting requires reloading or undoing the weight edit.

- *Access*: Implementation requires access to parameters and (typically) gradient flows.

Examples of structural control methods include: fine-tuning methods (full, parameter efficient), soft prompting (prefix
tuning, p-tuning), and model merging. Many of the structural control methods in the toolkit are implemented as wrappers
around existing libraries. The toolkit implements:

- [`MergeKit`](../reference/algorithms/structural_control/mergekit_wrapper.md) — model merging via MergeKit[@goddard-etal-2024-arcees]; combines multiple checkpoints with strategies such as linear interpolation, SLERP, and TIES from a YAML/dict config. See the notebook: [MergeKit](../examples/notebooks/mergekit_wrapper.ipynb).
- [`TRL`](../reference/algorithms/structural_control/trl_wrapper.md) — weight-level training via Hugging Face TRL[@vonwerra2022trl]; exposes SFT, DPO, APO, PPO, and GRPO trainers, with optional LoRA/PEFT and a post-training merge. See the notebook: [TRL](../examples/notebooks/trl_wrapper.ipynb).


## State control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim p_{\theta}^a(x)$

State control methods modify the model's internal/hidden states (e.g., activations, attentions, etc.) at inference time.
These methods are implemented by defining hooks that are inserted/registered into the model to manipulate internal
variables during the forward pass.

State control methods satisfy requirements:

- *Control*: Writes to (augments) model's internal/hidden states; model weights remain fixed.

- *Persistence*: Changes are temporary; behavior reverts to baseline once hooks are removed.

- *Access*: Requires access to internal states (to define hooks).

Some examples of state control methods include: activation addition/steering, attention steering, and representation
patching. The toolkit implements:

- [`ActAdd`](../reference/algorithms/state_control/act_add.md) — activation addition[@turner2023activation]; adds a positional steering vector from a single contrast pair to the residual stream at one layer. See the notebook: [ActAdd](../examples/notebooks/act_add.ipynb).
- [`ActivationAdapter`](../reference/algorithms/state_control/activation_adapter.md) — the composable activation-steering atom; wires together the shared `_common` components (a transform that carries its own artifact, selector, gate, condition path, token scope) so a recipe is assembled without writing a new control class.
- [`AngularSteering`](../reference/algorithms/state_control/angular_steering.md) — angular steering[@vu2025angular]; rotates the hidden state within a per-layer 2D plane (feature axis + companion axis) to a target angle, leaving the orthogonal complement untouched. Norm-preserving by construction; vector addition and directional ablation are special cases.
- [`CAA`](../reference/algorithms/state_control/caa.md) — contrastive activation addition[@panickssery2023steering]; adds a learned mean-difference direction to the residual stream at a single layer. See the notebook: [CAA](../examples/notebooks/caa.ipynb).
- [`CAST`](../reference/algorithms/state_control/cast.md) — conditional activation steering[@lee2025programming]; applies behavior steering only when a learned condition direction crosses a threshold. The applied behavior transform is pluggable (additive by default; any `BaseTransform` via `behavior_transform`, e.g. directional ablation for conditional abliteration). See the notebook: [CAST](../examples/notebooks/cast.ipynb).
- [`DirectionalAblation`](../reference/algorithms/state_control/directional_ablation.md) — directional ablation / abliteration[@arditi2024refusal]; projects a learned feature direction (or subspace) out of the residual stream at masked positions, with a graded ablation strength.
- [`ITI`](../reference/algorithms/state_control/iti.md) — inference-time intervention[@li2023inference]; shifts activations at a sparse set of probe-selected attention heads during generation. See the notebook: [ITI](../examples/notebooks/iti.ipynb).
- [`PASTA`](../reference/algorithms/state_control/pasta.md) — post-hoc attention steering[@zhang2024tell]; rescales attention to targeted prompt substrings at selected layers and heads. See the notebook: [PASTA](../examples/notebooks/pasta.ipynb).

Reusable building blocks shared across the residual-stream methods (estimators, gates, selectors, transforms, steering
vectors, hook utilities) live in
[`state_control._common`](../reference/algorithms/state_control/_common.md).

`ActivationAdapter` is the **composition surface** for these building blocks: each adapter is a single-behavior atom
(one transform chain — which carries its own artifact — one gate, one token scope), and steering with several behaviors
is simply several adapters listed together in a pipeline's `controls`. Because a pipeline accepts
[multiple state controls](steering_pipelines.md) applied in list order, composition across behaviors is owned by that
ordered list — no separate composite abstraction is needed. Joint conditioning across adapters uses one shared gate
instance: a driver declares the condition path and updates the gate; followers pass the same instance with
`gate_driven_externally=True` and read its decision.



## Output control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim d(p_{\theta})(x)$

Output control methods modify model outputs or constrain/transform what leaves the decoder. The base distribution
$p_\theta$ is left intact; only the path through the distribution changes.

Output control methods satisfy:

- *Control*: Replaces or constrains the decoding operator; no prompts, hidden states, or weights are altered.

- *Persistence*: Changes are temporary; behavior is restored once decoding control is removed.

- *Access*: Requires access to logits, token-probabilities, and possibly hidden states (depending on the method).

Examples of output control methods include: sampling/search strategies, weighted decoding, and reward-augmented
decoding. The toolkit implements:

- [`DeAL`](../reference/algorithms/output_control/deal.md) — decoding-time alignment[@huang2024deal]; iterative lookahead beam search with reward-guided beam selection. See the notebook: [DeAL](../examples/notebooks/deal.ipynb).
- [`RAD`](../reference/algorithms/output_control/rad.md) — reward-augmented decoding[@deng-raffel-2023-reward]; shifts candidate-token logits by a reward from a unidirectional reward model. See the notebook: [RAD](../examples/notebooks/rad.ipynb).
- [`SASA`](../reference/algorithms/output_control/sasa.md) — self-disciplined autoregressive sampling[@ko2025large]; shifts logits toward a learned non-toxic subspace. See the notebook: [SASA](../examples/notebooks/sasa.ipynb).
- [`ThinkingIntervention`](../reference/algorithms/output_control/thinking_intervention.md) — thinking intervention[@wu2025effectively]; injects structured reasoning instructions into the chain of thought, then extracts the post-thinking output. See the notebook: [ThinkingIntervention](../examples/notebooks/thinking_intervention.ipynb).
