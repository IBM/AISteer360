# Examples

We have prepared a collection of example notebooks for each of the implemented controls in our toolkit (including examples of how to implement methods
from wrappers), as well as demonstrations of more extensive benchmarks.

## Controls

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called. Current notebooks cover:

    :octicons-arrow-right-24: [FewShot](./notebooks/control_few_shot/few_shot.ipynb)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights/architecture. Current notebooks cover:

    :octicons-arrow-right-24: [MergeKit wrapper](./notebooks/wrapper_mergekit/mergekit_wrapper.ipynb)

    :octicons-arrow-right-24: [TRL wrapper](./notebooks/wrapper_trl/trl_wrapper.ipynb)

-   __State control__

    ---

    State control methods influence the model's internal states (activation, attentions, etc.) at inference time. Current notebooks cover:

    :octicons-arrow-right-24: [ActAdd](./notebooks/control_act_add/act_add.ipynb)

    :octicons-arrow-right-24: [CAA](./notebooks/control_caa/caa.ipynb)

    :octicons-arrow-right-24: [CAST](./notebooks/control_cast/cast.ipynb)

    :octicons-arrow-right-24: [ITI](./notebooks/control_iti/iti.ipynb)

    :octicons-arrow-right-24: [PASTA](./notebooks/control_pasta/pasta.ipynb)

-   __Output control__

    ---

    Output control methods influence the model's behavior via the `generate()` method. Current notebooks cover:

    :octicons-arrow-right-24: [DeAL](./notebooks/control_deal/deal.ipynb)

    :octicons-arrow-right-24: [RAD](./notebooks/control_rad/rad.ipynb)

    :octicons-arrow-right-24: [SASA](./notebooks/control_sasa/sasa.ipynb)

    :octicons-arrow-right-24: [ThinkingIntervention](./notebooks/control_thinking_intervention/thinking_intervention.ipynb)


</div>


## Benchmarks

<div class="grid cards" markdown>

-   :material-list-box-outline:  __Instruction following__

    ---

    This notebook studies the effect of post-hoc attention steering ([PASTA](https://arxiv.org/abs/2311.02262)) on a model's ability to follow instructions. We sweep over the steering strength and investigate the trade-off between a model's instruction following ability and general response quality.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmark_instruction_following/instruction_following.ipynb)

-   :material-comment-question-outline:  __Commonsense MCQA__

    ---

    This notebook benchmarks steering methods on the [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) dataset, comparing few-shot prompting against a LoRA adapter trained with DPO. We sweep over the number of few-shot examples and study how accuracy scales relative to the fine-tuned baseline across two models.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmark_commonsense_mcqa/commonsense_mcqa.ipynb)

</div>
