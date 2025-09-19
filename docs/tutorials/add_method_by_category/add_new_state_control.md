# Adding a state control method

**Required override**: `get_hooks`

State control methods work by defining hooks that are then registered into the base model before inference. As part of
this tutorial, we’ll implement an `ActivationBias` method that adds a fixed bias (alpha) to the hidden state
output at a specified transformer layer.

First, create the registry file:

```python
from .control import ActivationBias
from .args import ActivationBiasArgs

REGISTRY_ENTRY = {
    "category": "state_control",
    "name": "activation_bias",
    "control": ActivationBias,
    "args": ActivationBiasArgs,
}
```

Next, define the arguments class. This is where we define the required arguments; the transformer layer (via
`layer_idx`) and the bias (via `alpha`):

```python
from dataclasses import dataclass, field
from aisteer360.algorithms.core.base_args import BaseArgs


@dataclass
class ActivationBiasArgs(BaseArgs):
    layer_idx: int = field(
        default=0,
        metadata={"help": "Transformer block to patch."}
    )
    alpha: float = field(
        default=0.02,
        metadata={"help": "Bias magnitude."}
    )

    def __post_init__(self):
        if self.layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
```

Lastly, the control is implemented as follows:

```python
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from aisteer360.algorithms.state_control.base import StateControl, HookSpec
from aisteer360.algorithms.state_control.activation_bias.args import ActivationBiasArgs


class ActivationBias(StateControl):
    """Adds alpha to hidden states at the selected layer."""

    Args = ActivationBiasArgs

    # class attributes (filled by steer)
    model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizer | None = None
    device: torch.device | str | None = None

    def steer(
            self,
            model: PreTrainedModel = None,
            tokenizer: PreTrainedTokenizer = None,
            **kwargs) -> None:
        self.model = model
        self.device = next(model.parameters()).device

    def get_hooks(
            self,
            input_ids: torch.Tensor,
            runtime_kwargs,
            **__
    ) -> dict[str, list[HookSpec]]:
        """Returns a forward hook that adds alpha to a specific layer's output.

        Args:
            input_ids (torch.Tensor): Input tensor (unused).
            runtime_kwargs: Optional runtime parameters (unused).

        Returns:
            dict[str, list[HookSpec]]: A dictionary mapping hook phases ("pre", "forward", "backward") to lists of hook
            specifications. Each HookSpec contains:
              - "module": The name of the module to hook
              - "hook_func": The hook function to apply (pre, forward, or backward)
        """

        def fwd_hook(module, args, kwargs, output):

            # handle different output formats
            if isinstance(output, tuple):
                return (output[0] + self.alpha,) + output[1:]
            elif isinstance(output, dict):
                output = output.copy()
                output['hidden_states'] += self.alpha
                return output
            else:  # direct tensor
                return output + self.alpha

        return {
            "pre": [],
            "forward": [{
                "module": f"model.layers.{self.layer_idx}",
                "hook_func": fwd_hook,
            }],
            "backward": [],
        }
```

The hooks are then registered into the model via the `register_hooks` method in the state control base class
(`aisteer360/algorithms/state_control/base.py`) such that they can be run on every `generate` call. The control can
then be called via:

```python
from aisteer360.algorithms.state_control.activation_bias.control import ActivationBias
from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

activation_bias_control = ActivationBias(layer_idx=2, alpha=0.03)

activation_bias_pipeline = SteeringPipeline(
    model_name_or_path=MODEL_NAME,
    controls=[activation_bias_control],
)
activation_bias_pipeline.steer()

prompt = "What should I do in Prague?"
chat = activation_bias_pipeline.tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True
)
inputs = activation_bias_pipeline.tokenizer(chat, return_tensors="pt")

print(activation_bias_pipeline.generate_text(inputs.input_ids, max_new_tokens=50))
```
