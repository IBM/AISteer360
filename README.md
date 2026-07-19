![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_darkmode.png#gh-dark-mode-only)
![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_darkmode.png#gh-light-mode-only)

[//]: # (to add: arxiv; pypi; ci)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/AISteer360/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
[![GitHub License](https://img.shields.io/github/license/generative-computing/mellea)](https://img.shields.io/github/license/generative-computing/mellea)

---

The AI Steerability 360 toolkit is an extensible library for general purpose steering of LLMs. The toolkit allows for
the implementation of steering methods across a range of model control surfaces (input, structural, state, and output),
functionality to compose steering methods (into a `SteeringPipeline`), and the ability to compare steering methods
(and pipelines) on custom tasks/metrics.

To get started, please see the [documentation](https://ibm.github.io/AISteer360/).

## Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.11+). After installing `uv`, install
the toolkit by running:

```commandline
uv venv --python 3.11 && uv pip install .
```
Activate by running `source .venv/bin/activate`. Note that on Windows, you may need to split the above script into two separate commands (instead of chained via `&&`).

Inference is facilitated by Hugging Face. Before steering, create a `.env` file in the root directory for your Hugging
Face API key in the following format:
```
HUGGINGFACE_TOKEN=hf_***
```

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. Check that you have access via the model's Hub page with the same account whose token you pass to the toolkit.

> [!NOTE]
> AISteer360 runs the model inside your process. For efficient inference, please run the toolkit from a machine that
> has enough GPU memory for both the base checkpoint and the extra overhead your steering method/pipeline adds. 


## Featured applications

The ability to benchmark and compare steering methods on realistic use cases is one of the main features of the toolkit. The featured examples below showcase steering methods in action along with this benchmarking functionality.

| <div style="font-weight: bold; text-align: left;">Steering for selective refusal</div> |
|:------------------------------------------------------------------------------------------------------------------------|
| Selective refusal lets a model decline requests on a chosen topic while answering everything else as usual. This notebook applies conditional activation steering ([CAST](https://arxiv.org/abs/2409.05907)) to program refusal of legal questions, pairing a refusal behavior direction with a condition direction that recognizes requests for legal advice. We let the toolkit search for the condition layer and threshold, then verify that the steered model declines legal questions while answering general ones untouched.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/cast.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-weight: bold; text-align: left;">Composable activation steering</div> |
|:------------------------------------------------------------------------------------------------------------------------|
| Many activation steering methods are specific assignments of a transform, a layer choice, a gate, and a token scope. This notebook demonstrates the activation adapter, a control that exposes these components as arguments, and rebuilds contrastive activation addition ([CAA](https://arxiv.org/abs/2312.06681)) and directional ablation ([Arditi et al.](https://arxiv.org/abs/2406.11717)) from one fitted refusal direction. We swap transforms, select layers by fractional depth, and add a threshold gate so the ablation fires only on prompts that express the refusal feature.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/activation_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-weight: bold; text-align: left;">Steering for instruction following</div>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A model's instruction following ability is an important measure of its general usability. This notebook studies the effect of post-hoc attention steering ([PASTA](https://arxiv.org/abs/2311.02262)) on a model's ability to follow instructions. We sweep over the steering strength and investigate the trade-off between a model's instruction following ability and general response quality.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/benchmark_instruction_following/instruction_following.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Control library

Demonstrations of each of the implemented steering methods in the toolkit are provided as notebooks in the `examples/notebooks` directory; links to Colab notebooks are provided below.

| Method | Authors | Notebook |
|:-------|:----------|:---------|
| [Activation Addition (ActAdd)](https://arxiv.org/abs/2308.10248) | Turner et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/act_add.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Activation Adapter | - | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/activation_adapter.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Angular Steering](https://arxiv.org/abs/2510.26243) | Vu & Nguyen, 2025 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/angular_steering.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Contrastive Activation Addition (CAA)](https://arxiv.org/abs/2312.06681) | Panickssery et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/caa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Conditional Activation Steering (CAST)](https://arxiv.org/abs/2409.05907) | Lee et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/cast.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Causal Prompt Optimization (CPO)](https://arxiv.org/abs/2602.01711) | Chen et al., 2026 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/cpo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Decoding-time Alignment (DeAL)](https://arxiv.org/abs/2402.06147) | Huang et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/deal.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Directional Ablation](https://arxiv.org/abs/2406.11717) | Arditi et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/directional_ablation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Few-shot Learning](https://arxiv.org/abs/2005.14165) | Brown et al., 2020 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [GEPA](https://arxiv.org/abs/2507.19457) | Agrawal et al., 2025 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/gepa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Inference-Time Intervention (ITI)](https://arxiv.org/abs/2306.03341) | Li et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/iti.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Post-hoc Attention Steering (PASTA)](https://arxiv.org/abs/2311.02262) | Zhang et al., 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/pasta.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Prompt Rewriting with Reinforcement Learning (PRewrite)](https://arxiv.org/abs/2401.08189) | Kong et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/prewrite.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Reward-Augmented Decoding (RAD)](https://arxiv.org/abs/2310.09520) | Deng & Raffel, 2023 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/rad.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Self-Disciplined Autoregressive Sampling (SASA)](https://arxiv.org/abs/2410.03818) | Ko et al., 2025 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/sasa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [Thinking Intervention](https://arxiv.org/abs/2503.24370) | Wu et al., 2025 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/thinking_intervention.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

The toolkit also provides wrappers for the following external libraries.

| Wrapper | Authors | Notebook |
|:--------|:----------|:---------|
| [MergeKit](https://github.com/arcee-ai/mergekit) | Goddard et al., 2024 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/mergekit_wrapper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| [TRL](https://github.com/huggingface/trl) | von Werra et al., 2020 | <a href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/examples/notebooks/trl_wrapper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


## Contributing

We invite community contributions primarily on broadening the set of steering methods (via new controls) and evaluations
(via use cases and metrics). We additionally welcome reporting of any bugs/issues, improvements to the documentation,
and new features). Specifics on how to contribute can be found in our [contribution guidelines](CONTRIBUTING.md).
To make contributing easier, we have prepared the following tutorials.


### Adding a new steering method

If there is an existing steering method that is not yet in the toolkit but you feel should be, or you have developed a 
new steering method of your own, the toolkit has been designed to enable relatively easy contribution of new steering methods. 
Please see the tutorial on [adding your own steering method](./docs/tutorials/add_new_steering_method.md) for a detailed guide


### Adding a new use case / benchmark

Use cases enable comparison of different steering methods on a common task. The base `UseCase`
(`aisteer360/evaluation/use_cases/`) and the `Benchmark` class (`aisteer360/evaluation/benchmark.py`) enable this
comparison. If you'd like to compare various steering methods/pipelines on a novel use case, please see the tutorial 
on [adding your own use case](./docs/tutorials/add_new_use_case.md).


### Adding a new metric

Metrics are used by a given benchmark to quantify model performance across steering pipelines in a comparable way. We've
included a selection of generic metrics (see `aisteer360/evaluation/metrics/`). If you'd like to add new generic metrics
or custom metrics (for a new use case), please see the tutorial on
[adding your own metric](./docs/tutorials/add_new_metric.md).

## Reference

If you find the toolkit useful in your work, please cite the following:
```bibtex
@article{miehling2026aisteerability360,
  title = {AI Steerability 360: A Toolkit for Steering Large Language Models},
  author = {Miehling, Erik and Ramamurthy, Karthikeyan Natesan and Venkateswaran, Praveen and Ko, Irene and Dognin, Pierre and Singh, Moninder and Pedapati, Tejaswini and Balakrishnan, Avinash and Riemer, Matthew and Wei, Dennis and Vejsbjerg, Inge and Daly, Elizabeth M. and Varshney, Kush R.},
  journal = {arXiv preprint arXiv:2603.07837},
  year = {2026}
}
```

## IBM ❤️ Open Source AI

The AI Steerability 360 toolkit has been brought to you by IBM.
