# Installation

The toolkit uses [uv](https://docs.astral.sh/uv/) as the package manager (Python 3.11+). Make sure that `uv` is installed via:

=== "standalone installer"
    ```bash
    curl -Ls https://astral.sh/uv/install.sh | sh
    ```

=== "Homebrew"
    ```bash
    brew install astral-sh/uv/uv
    ```

## Installing the toolkit

Once `uv` is installed, install the `aisteer360` package via:

```commandline
uv venv --python 3.11 && uv pip install .
```

The above creates a `.venv` (if missing), installs `aisteer360` (in non-editable mode), and installs all dependencies
listed under `[project.dependencies]` in the `pyproject.toml` file. Activate by running `source .venv/bin/activate`.
To install an optional dependency group from `[project.optional-dependencies]`, e.g., `docs`, append it in quotes and
square brackets to the `install` command as follows:

```commandline
uv venv --python 3.11 && uv pip install '.[docs]'
```

## Accessing Hugging Face models

Inference is facilitated by Hugging Face. Before steering, create a `.env` file in the root directory for your Hugging
Face API key in the following format:
```
HUGGINGFACE_TOKEN=hf_***
```

Some Hugging Face models (e.g. `meta-llama/Meta-Llama-3.1-8B-Instruct`) are behind an access gate. To gain access:

1. Request access on the model’s Hub page with the same account whose token you use in your `.env` file.
2. Wait for approval (you’ll receive an email).
3. (Re-)authenticate locally by running `huggingface-cli login` if your token has expired or was never saved.

Once you have completed the above steps, please see our [quickstart](quickstart.md) guide to get up and running!
