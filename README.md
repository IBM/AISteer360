![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_darkmode.png#gh-dark-mode-only)
![AISteer360](https://github.com/IBM/AISteer360/raw/main/docs/assets/logo_wide_lightmode.png#gh-light-mode-only)

[//]: # (to add: arxiv; pypi; ci)
[![Docs](https://img.shields.io/badge/docs-live-brightgreen)](https://ibm.github.io/AISteer360/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
[![GitHub License](https://img.shields.io/github/license/IBM/AISteer360)](https://github.com/IBM/AISteer360/blob/main/LICENSE)

---

Welcome to AI Steerability 360 (AISteer360), a toolkit for steering large language models.

AISteer360 provides an expressive library of reusable components (termed generics) across four model control surfaces 
(input, structural, state, and output). This allows for the modular construction of novel steering methods, composition 
of steering methods into steering pipelines, and benchmarking of pipelines on custom use cases and metrics (including 
measurement of steering side effects). 

To get started, please see the documentation at <https://ibm.github.io/AISteer360/> and the [example notebooks](examples/index.md).

## Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.11+). After installing `uv`, install
the toolkit by running:

```commandline
uv venv --python 3.11 && uv pip install .
```
Activate by running `source .venv/bin/activate`. Note that on Windows, you may need to split the above script into two
separate commands (instead of chained via `&&`).

Optional features are available via extra. Install everything with `uv pip install ".[all]"`.

Inference is facilitated by Hugging Face. Before steering, create a `.env` file in the root directory for your Hugging
Face API key in the following format:
```
HUGGINGFACE_TOKEN=hf_***
```

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. Check that you have
access via the model's Hub page with the same account whose token you pass to the toolkit.

> [!NOTE]
> AISteer360 runs the model inside your process. For efficient inference, please run the toolkit from a machine that
> has enough GPU memory for both the base checkpoint and the extra overhead your steering method/pipeline adds.

## Contributing

We welcome contributions, particularly new steering methods (controls), use cases, and metrics, along with bug reports,
documentation improvements, and new features. See the [contribution guidelines](CONTRIBUTING.md) and the tutorials on
[adding a steering method](./docs/tutorials/add_new_steering_method.md),
[adding a use case](./docs/tutorials/add_new_use_case.md), and
[adding a metric](./docs/tutorials/add_new_metric.md).

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