![AISteer360](docs/assets/logo_wide_darkmode.png#gh-dark-mode-only)
![AISteer360](docs/assets/logo_wide_lightmode.png#gh-light-mode-only)

[//]: # (to add: arxiv; pypi; ci)
[![PyPI version](https://img.shields.io/pypi/v/aisteer360)](https://pypi.org/project/aisteer360/)
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
uv pip install aisteer360
```

Inference is facilitated by Hugging Face. Before steering, create a `.env` file in the root directory for your Hugging
Face API key in the following format:
```
HUGGINGFACE_TOKEN=hf_***
```

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. To gain access,
request access on the model’s Hub page (with the same account whose token you’ll pass to the toolkit) and wait for
the approval email. You may need to reauthenticate locally by running `huggingface-cli login`.


## Example library

> [!NOTE]
> AISteer360 runs the model inside your process. For efficient inference, please run the toolkit from a machine that
> has enough GPU memory for both the base checkpoint and the extra overhead your steering method/pipeline adds.

Notebook examples for each of the supported steering methods (and wrappers) can be found in the `notebooks/controls/`
directory. For some detailed notebooks on realistic use cases, please explore the following examples.

| <div style="font-size: 18px; font-weight: bold; text-align: left;">Steering for instruction following</div>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A model's instruction following ability is an important measure of its general usability. This notebook compares two methods: simple few-shot prompting and a more invasive attention-based method called post-hoc attention steering ([PASTA](https://arxiv.org/abs/2311.02262)). Using [MExGen](https://arxiv.org/abs/2403.14459) from the [ICX360](https://github.com/IBM/ICX360) toolkit, we study how steering impacts post-hoc attributions. We further investigate the trade-off between a model's instruction following ability and general response quality.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-size: 18px; font-weight: bold; text-align: left;">Mitigating catastrophic forgetting</div>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| When fine-tuning model behavior for a particular skill (e.g., via LoRA), other skills often degrade. This notebook investigates the degree to which a model catastrophically forgets under basic fine-tuning, and explores how we can mitigate this effect using meta-experience replay (see papers [here](https://arxiv.org/abs/1810.11910) and [here](https://arxiv.org/pdf/2508.01908)). <br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-size: 18px; font-weight: bold; text-align: left;">Many-shot learning</div>                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Current models are able to learn effectively from a small number of in-context examples. This notebook looks at a commonsense multiple-choice question-answering task and explores how effectively the model learns as we scale to hundreds of in-context examples. <br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-size: 18px; font-weight: bold; text-align: left;">Comparing activation steering methods</div>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Activation steering methods are a powerful method for influencing model behavior. However, this increased control can also inadvertently modify unintended behaviors. This notebook compares three activation steering methods ([CAA](https://aclanthology.org/2024.acl-long.828.pdf), [angular steering](https://openreview.net/pdf?id=scqQxchEyM), and [one-shot steering](https://arxiv.org/pdf/2502.18862)) on a range of metrics to better understand how each impacts the model.<br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

| <div style="font-size: 18px; font-weight: bold; text-align: left;">Composite steering pipelines</div>                                                                                                                                                                                                                                                                                                                                                                     |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A primary feature of the toolkit is the ability to compose steering methods from different control categories into a single steering pipeline. This notebook studies the effectiveness of combining ... <br /><br /><a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |




## Contributing

We invite community contributions on broadening the set of steering methods (via new controls) and evaluations (via novel
use cases and metrics). We additionally welcome reporting of any bugs/issues, improvements to the documentation,
and new features. Specifics on how to contribute can be found in our [contribution guidelines](CONTRIBUTING.md).
To make contributing easier, we have prepared the following tutorials.


### Adding a new steering method

If there is an existing steering method that is not yet in the toolkit, or you have developed a new steering method of
your own, the toolkit has been designed to enable relatively easy contribution of new steering methods. Please see the
tutorial on [adding your own steering method](./docs/tutorials/add_new_steering_method.md) for a detailed guide


### Adding a new use case / benchmark

Use cases enable comparison of different steering methods on a common task. The `UseCase`
(`aisteer360/evaluation/use_cases/`) and `Benchmark` classes (`aisteer360/evaluation/benchmark.py`) enable this
comparison. If you'd like to compare various steering methods/pipelines on a novel use case, please see the tutorial on
[adding your own use case](./docs/tutorials/add_new_use_case.md).


### Adding a new metric

Metrics are used by a given benchmark to quantify model performance across steering pipelines in a comparable way. We've
included a selection of generic metrics (see `aisteer360/evaluation/metrics/`). If you'd like to add new generic metrics
or custom metrics (for a new use case), please see the tutorial on
[adding your own metric](./docs/tutorials/add_new_metric.md).


## IBM ❤️ Open Source AI

The AI Steerability 360 toolkit has been brought to you by IBM.
