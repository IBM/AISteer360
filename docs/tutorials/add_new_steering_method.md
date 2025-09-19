# Adding your own steering method

Steering methods span four categories of controls: *input*, *structural*, *state*, and *output*. The specific category of a
steering method is dictated by what aspects of the model the method influences. Please refer to the conceptual guide on
[model steering](../concepts/steering.md) for information on choosing the appropriate category for your method.

## Required files

Once you have determined the steering category, create the following files in `aisteer360/algorithms`:

```
aisteer360/
└── algorithms/
        └── <category>/
            └── <custom_control>/
                ├── utils/ (optional)
                ├── __init__.py
                ├── args.py
                └── control.py
```

where `<category>` must be one of the existing directories (`input_control`, `structural_control`, `state_control`, `output_control`) and
`<custom_control>` is the directory name for your method. We encourage you to keep your implementations as
self-contained as possible (within the control class), but any additional files/utils beyond the core implementation
can be placed in a `utils/` directory within `<custom_control>/`. The following outlines how each file (`__init__.py`,
`args.py`, `control.py`) are constructed.



### 1. Registry: `__init__.py`:

The `__init__.py` file exposes the method to the toolkit's registry.

```python
from .control import CustomControl
from .args import CustomControlArgs

REGISTRY_ENTRY = {
    "category": "<category>",
    "name": "CustomControl",
    "control": CustomControl,
    "args": CustomControlArgs,
}
```

### 2. Arguments dataclass: `args.py`:

The args file holds a dataclass that specifies the method's required arguments along with any associated validation
logic.

```python
from dataclasses import dataclass, field
from aisteer360.algorithms.core.base_args import BaseArgs

@dataclass
class CustomControlArgs(BaseArgs):

    prefix: str = field(
        default="You are an expert assistant.",
        metadata={"help": "Hard-coded text prepended to every user prompt."},
    )
    strip_newlines: bool = field(
        default=True,
        metadata={"help": "Remove trailing newlines from the original prompt before concatenation."},
    )

    # validate
    def __post_init__(self):

        if not self.prefix:
            raise ValueError("`prefix` must be non-empty.")
```

List all parameters that your method takes as input. Each parameter is written as a `field` with args: `default`
(included only if the parameter is optional; omit it if the parameter is required) and `metadata` (a dictionary
containing the description of the argument under key `help`). Include all validation logic for your method's parameters
in the `__post_init__` method to ensure that validation is run automatically (upon class initialization).

!!! warning
    Immutable defaults are safe with `default=`, i.e., `int`, `float`, `str`, and `bool` can be given directly (`default=5`, `default=True`, ...), but mutable defaults need `default_factory`. For example, for a `list`, `dict`, `set`, or any custom object you expect to mutate, you must write:
    ```python
    my_list: list[str] = field(default_factory=list, metadata={...})
    ```
    See the [example output control](./add_method_by_category/add_new_output_control.md) implementation for details.


### 3. Control implementation: `control.py`:

The control file holds the method's main implementation. The control class **does not** contain an `__init__` method.
Instead, the method's parameters are handled by the args class via the line `Args = CustomControlArgs`.[^1] The
`__init__` method of the control's base class automatically validates these fields (via `Args.validate`) and converts
them into class attributes.

[^1]: This is intended to minimize boilerplate code (parameter/argument parsing and validation) that would otherwise need to live in each control's `__init__` method.

Any one-time preparation of the steering method is done in the `.steer()` method of the control. This is optional for all
control categories *except* structural control methods; the `.steer()` method in a structural control method contains
the necessary logic for modifying the model's weights/architecture. Note that while including a steer method is optional
in every control type other than structural, it is often useful to include one for attaching necessary objects to the
control for later use (e.g., the tokenizer). This is illustrated in the tutorials below.

The implementation of a control method depends on its steering category. Specific instructions for how to add a method
under each of the four categories, via a simple example implementation, is detailed below:

<div class="grid cards" markdown>

-   __Input control__

    ---

    Input control methods adapt the input (prompt) before the model is called.

    *Required override*: `get_prompt_adapter`

    [:octicons-arrow-right-24: Add your own input control method](./add_method_by_category/add_new_input_control.md)

-   __Structural control__

    ---

    Structural control methods adapt the model's weights/architecture.

    *Required override*: `steer`

    [:octicons-arrow-right-24: Add your own structural control method](./add_method_by_category/add_new_structural_control.md)

-   __State control__

    ---

    State control methods influence the model's internal states (activation, attentions, etc.) at inference time.

    *Required override*: `get_hooks`

    [:octicons-arrow-right-24: Add your own state control method](./add_method_by_category/add_new_state_control.md)

-   __Output control__

    ---

    Output control methods influence the model's generations via the decoding process.

    *Required override*: `generate`

    [:octicons-arrow-right-24: Add your own output control method](./add_method_by_category/add_new_output_control.md)

</div>

!!! note
    If your steering method requires two distinct control knobs, e.g., both tweaks the prompt *and* constrains
    decoding, split it into two small controls and chain them together in `controls=[...]`. See the documentation on
    [composite controls](./add_composite_control.md) for additional details.


## Testing your method

To ensure your method is operating as intended, we ask that you write a small unit test in `./tests/controls/`. We
advise that these tests are written using a lightweight models (e.g., via
[Hugging Face internal testing](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM)). This allows
for the tests to be run locally (on your CPU) before submitting your PR. See the `tests/` directory for examples of
well-written tests.


## Document it and write a notebook


Ensure you have written a meaningful docstring for your method in the main control class. Docstrings should contain a
brief description of the method, a reference to the method's paper/documentation, and a list of the method's args
(please use the Google docstring format). An example class docstring (for the `DeAL` method) is given below:

```python
"""
Implementation of DeAL (Decoding-time Alignment) from Deng et al., 2024.

DeAL performs controlled text generation through iterative lookahead search and reward-guided beam selection. Unlike
training-time alignment methods, DeAL operates purely at inference time to steer language model outputs toward
desired behaviors.

The algorithm works in three phases:

1. **Lookahead Generation**: Generate multiple candidate continuations using beam search from the current context.

2. **Reward-based Scoring**: Evaluate each candidate continuation using a provided reward function that measures
alignment with the desired objective (e.g., helpfulness, safety).

3. **Iterative Refinement**: Select the top-k highest-scoring beams and repeat the process until termination
conditions are met (EOS token, max length, or max iterations reached).

This approach allows for flexible alignment with various objectives without requiring model retraining or
fine-tuning.

Args:
    reward_func (Callable): Function that scores generated continuations. Should accept
        (prompt: str, continuations: list[str], reward_params: dict) and return list[float].
    lookahead (int): Number of tokens to generate in each lookahead step. Defaults to 4.
    init_beams (int): Number of initial beams to generate at each iteration. Defaults to 8.
    topk (int): Number of top-scoring beams to retain for the next iteration. Defaults to 4.
    max_iterations (int): Maximum number of search iterations before termination. Defaults to 10.

Reference:

- "DeAL: Decoding-time Alignment for Large Language Models"
James Y. Huang, Sailik Sengupta, Daniele Bonadiman, Yi-an Lai, Arshit Gupta, Nikolaos Pappas, Saab Mansour,
Katrin Kirchhoff, Dan Roth
https://arxiv.org/abs/2402.06147
"""
```


Show off how cool your method is by writing a notebook (in `../notebooks/controls/<custom_control>/`). A good notebook
should contain the following:

- A description of what the method does and how it works
- How to initialize the control using the toolkit
- A simple example of it working; it's helpful to illustrate how the steered behavior compares with the baseline
(non-steered) behavior

See the [DeAL notebook](`../notebooks/controls/deal.ipynb`) for an example.
