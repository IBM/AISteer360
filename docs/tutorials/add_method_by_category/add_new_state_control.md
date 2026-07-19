# Adding a state control method

**Required override**: `get_hooks`

State control methods work by defining hooks that are then registered into the base model before inference. As part of
this tutorial, we’ll implement an `ActivationBias` method that adds a fixed bias (alpha) to the hidden state
output at a specified transformer layer.

First, create the registry file:

```python
from .control import ActivationBias
from .args import ActivationBiasArgs

STEERING_METHOD = {
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

## Position tracking in hooks

Scoped controls (those honoring `token_scope="after_prompt"` or `"from_position"`) need to know each hook
invocation's absolute position in the full sequence. During prefill the hook sees the whole prompt
(`seq_len == prompt_len`); during KV-cached decode it sees only the newly generated token(s)
(`seq_len == 1`). Do **not** infer the phase by comparing `seq_len` to the prompt length — a length-1 prompt
makes prefill and decode indistinguishable, so steering silently never fires. Instead, track the phase
explicitly with a first-call flag, resetting it in both `reset()` and `get_hooks()`:

```python
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._position_offset: int = 0
    self._prefill_seen: bool = False

def reset(self):
    self._position_offset = 0
    self._prefill_seen = False

# inside the hook function:
seq_len = hidden.size(1)
if self._prefill_seen:            # decode step (or a later chunk)
    position_offset = self._position_offset
    self._position_offset += seq_len
else:                             # first pass since reset() == prefill
    position_offset = 0
    self._position_offset = seq_len
    self._prefill_seen = True

mask = make_token_mask(self.token_scope, seq_len=seq_len, prompt_lens=prompt_lens,
                       position_offset=position_offset)
```

If a control registers several hooks per pass (e.g. one per layer), designate a single hook to advance the
shared counter and gate both the advance and the flag flip on it, so earlier hooks in the same prefill pass
still read `position_offset = 0`. See `angular_steering` and `directional_ablation` for that variant.

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
print(activation_bias_pipeline.generate(prompt, max_new_tokens=50))
```
