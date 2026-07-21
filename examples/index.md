# Examples

We have prepared a collection of example notebooks for each of the implemented controls in our toolkit (including examples of how to implement methods
from wrappers), as well as demonstrations of more extensive benchmarks.

## Controls

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called. Current notebooks cover:

    :octicons-arrow-right-24: [CPO](./notebooks/cpo.ipynb)

    :octicons-arrow-right-24: [FewShot](./notebooks/few_shot.ipynb)

    :octicons-arrow-right-24: [GEPA](./notebooks/gepa.ipynb)

    :octicons-arrow-right-24: [PRewrite](./notebooks/prewrite.ipynb)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights/architecture. Current notebooks cover:

    :octicons-arrow-right-24: [MergeKit wrapper](./notebooks/mergekit_wrapper.ipynb)

    :octicons-arrow-right-24: [TRL wrapper](./notebooks/trl_wrapper.ipynb)

-   __State control__

    ---

    State control methods influence the model's internal states (activation, attentions, etc.) at inference time. Current notebooks cover:

    :octicons-arrow-right-24: [ActAdd](./notebooks/act_add.ipynb)

    :octicons-arrow-right-24: [CAA](./notebooks/caa.ipynb)

    :octicons-arrow-right-24: [CAST](./notebooks/cast.ipynb)

    :octicons-arrow-right-24: [ITI](./notebooks/iti.ipynb)

    :octicons-arrow-right-24: [PASTA](./notebooks/pasta.ipynb)

-   __Output control__

    ---

    Output control methods influence the model's behavior via the `generate()` method. Current notebooks cover:

    :octicons-arrow-right-24: [DeAL](./notebooks/deal.ipynb)

    :octicons-arrow-right-24: [RAD](./notebooks/rad.ipynb)

    :octicons-arrow-right-24: [SASA](./notebooks/sasa.ipynb)

    :octicons-arrow-right-24: [ThinkingIntervention](./notebooks/thinking_intervention.ipynb)


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

-   :material-layers-triple-outline:  __Composite steering for truthfulness__

    ---

    One of the primary features of the toolkit is the ability to compose multiple steering methods into one model operation. This notebook composes a state control ([PASTA](https://arxiv.org/abs/2311.02262)) with an output control ([DeAL](https://arxiv.org/abs/2402.06147)) with the goal of improving the model's truthfulness (as measured on [TruthfulQA](https://huggingface.co/datasets/domenicrosati/TruthfulQA)) without significantly degrading informativeness. We sweep over the joint parameter space of the controls and study each control's performance (via the tradeoff between truthfulness and informativeness) to that of the composition.

    [:octicons-arrow-right-24: See the benchmark](./notebooks/benchmark_truthful_qa_composite_steering/truthful_qa_composite_steering.ipynb)

</div>
