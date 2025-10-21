![AISteer360](docs/assets/logo_wide_darkmode.png#gh-dark-mode-only)
![AISteer360](docs/assets/logo_wide_lightmode.png#gh-light-mode-only)

[//]: # (to add: arxiv; pypi; ci)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/AISteer360/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
[![GitHub License](https://img.shields.io/github/license/generative-computing/mellea)](https://img.shields.io/github/license/generative-computing/mellea)

---

The AI Steerability 360 toolkit (AISteer360) is an extensible library for general purpose steering of LLMs. Primary
features of the toolkit include:

- Implementations of steering methods across a range of model control surfaces (input, structural, state, and output).

- Functionality to construct composite steering methods via a `SteeringPipeline`.

- Ability to compare steering pipelines on a given task (e.g., instruction following) via the `UseCase` and `Benchmark` classes.


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

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. To gain access:

1. Request access on the model’s Hub page with the same account whose token you’ll pass to the toolkit.
2. Wait for approval (you’ll receive an email).
3. (Re-)authenticate locally by running `huggingface-cli login` if your token has expired or was never saved.


> [!NOTE]
> AISteer360 runs the model inside your process. For efficient inference, please run the toolkit from a machine that
> has enough GPU memory for both the base checkpoint and the extra overhead your steering method/pipeline adds.


## Examples

We've constructed a range of example notebooks to illustrate the functionality of our toolkit. These notebooks cover
running individual steering methods/controls and running benchmarks, i.e., comparison of steering methods/pipelines on
a given task. Notebooks include:

| Description                              | Notebook                                                                                                                                                                                                                                                                                   |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Few-shot learning                        | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/few_shot.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                      |
| Post-hoc attention steering              | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/pasta.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                         |
| Conditional activation steering          | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/cast.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          |
| Self-disciplined autoregressive sampling | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/sasa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          |
| Thinking intervention                    | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/thinking_intervention.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                         |
| Reward-augmented decoding                | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/rad.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                           |
| Decoding-time alignment                  | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/deal.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                          |
| Model merging with MergeKit              | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/mergekit_wrapper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                              |
| Parameter-efficient fine-tuning with TRL | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/controls/trl_wrapper.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>                                   |
| Benchmark: Commonsense MCQA              | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/benchmarks/commonsense_mcqa/commonsense_mcqa.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>           |
| Benchmark: Instruction following         | <a target="_blank" rel="noopener noreferrer" href="https://colab.research.google.com/github/IBM/AISteer360/blob/main/notebooks/benchmarks/instruction_following/instruction_following.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |



## Contributing

We invite community contributions primarily on broadening the set of steering methods (via new controls) and evaluations
(via use cases and metrics). We additionally welcome reporting of any bugs/issues, improvements to the documentation,
and new features). Specifics on how to contribute can be found in our [contribution guidelines](CONTRIBUTING.md).
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
