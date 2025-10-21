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
self-consistency), automatic prompting methods, and prompt routing. Few-shot prompting is implemented in our toolkit
under the control name `FewShot` (source code: `algorithms/input_control/few_shot/control.py`). See the notebook
here: [FewShot](../notebooks/controls/few_shot.ipynb).



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
tuning, p-tuning), and model merging. Many of the structural control methods in the toolkit are implemented using
wrappers around existing libraries, e.g., Hugging Face's PEFT library. Some implementations of structural control
methods can be found in the notebooks: [MergeKit](../notebooks/controls/mergekit_wrapper.ipynb)[@goddard-etal-2024-arcees],
[TRL](../notebooks/controls/trl_wrapper.ipynb)[@vonwerra2022trl].


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

Some examples of output control methods include: activation addition/steering, attention steering, and representation
patching. Example implementations of state control methods can be found in the following notebooks:
[CAST](../notebooks/controls/cast.ipynb)[@lee2025programming], [PASTA](../notebooks/controls/pasta.ipynb)[@zhang2024tell].



## Output control

**Baseline model**: $y \sim p_\theta(x)$

**Steered model**: $y \sim^d p_{\theta}(x)$

Output control methods modify model outputs or constrain/transform what leaves the decoder. The base distribution
$p_\theta$ is left intact; only the path through the distribution changes.

Output control methods satisfy:

- *Control*: Replaces or constrains the decoding operator; no prompts, hidden states, or weights are altered.

- *Persistence*: Changes are temporary; behavior is restored once decoding control is removed.

- *Access*: Requires access to logits, token-probabilities, and possibly hidden states (depending on the method).

Examples of output control methods include: sampling/search strategies, weighted decoding, and reward-augmented
decoding. Some example methods can be found in the following notebooks: [DeAL](../notebooks/controls/deal.ipynb)[@huang2024deal],
[RAD](../notebooks/controls/rad.ipynb)[@deng-raffel-2023-reward], [SASA](../notebooks/controls/sasa.ipynb)[@ko2025large],
[ThinkingIntervention](../notebooks/controls/thinking_intervention.ipynb)[@wu2025effectively].
