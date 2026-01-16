# Notebooks

Notebooks cover basic implementations of each control in our toolkit (including examples of how to implement methods
from wrappers), as well as implementations of benchmarks.

## Controls

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called. Current notebooks cover:

    :octicons-arrow-right-24: [FewShot](./controls/few_shot.ipynb)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights/architecture. Current notebooks cover:

    :octicons-arrow-right-24: [MergeKit wrapper](./controls/mergekit_wrapper.ipynb)

    :octicons-arrow-right-24: [TRL wrapper](./controls/trl_wrapper.ipynb)

-   __State control__

    ---

    State control methods influence the model's internal states (activation, attentions, etc.) at inference time. Current notebooks cover:

    :octicons-arrow-right-24: [CAST](./controls/cast.ipynb)

    :octicons-arrow-right-24: [PASTA](./controls/pasta.ipynb)

-   __Output control__

    ---

    Output control methods influence the model's behavior via the `generate()` method. Current notebooks cover:

    :octicons-arrow-right-24: [DeAL](./controls/deal.ipynb)

    :octicons-arrow-right-24: [RAD](./controls/rad.ipynb)

    :octicons-arrow-right-24: [SASA](./controls/sasa.ipynb)

    :octicons-arrow-right-24: [ThinkingIntervention](./controls/thinking_intervention.ipynb)


</div>


## Benchmarks

<div class="grid cards" markdown>

-   :material-comment-question-outline:  __Commonsense MCQA__

    ---

    This benchmark evaluates how well a steered model (under `FewShot` and `LoRA`) performs compared to a base model on
    answering commonsense multiple-choice questions.

    [:octicons-arrow-right-24: See the benchmark](./benchmarks/commonsense_mcqa.ipynb)

-   :material-list-box-outline:  __Instruction following__

    ---

    This benchmark evaluates a steered model's ability to follow instructions. We compare the performance of the
    baseline model to the steered model under `PASTA`, `DeAL`, and `ThinkingIntervention`.

    [:octicons-arrow-right-24: See the benchmark](./benchmarks/instruction_following.ipynb)

</div>
