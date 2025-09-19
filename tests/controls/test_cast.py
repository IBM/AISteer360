import numpy as np
import pytest
import torch

from aisteer360.algorithms.core.steering_pipeline import SteeringPipeline
from aisteer360.algorithms.state_control.cast.control import CAST
from aisteer360.algorithms.state_control.cast.utils.steering_vector import (
    SteeringVector,
)
from tests.utils.sweep import build_param_grid

PROMPT_TEXT = (
    "Answer truthfully. Therefore, when you respond: "
    "First, present your main point. "
    "Second, support it with evidence. "
    "Finally, conclude succinctly."
)

CAST_GRID = {
    'behavior_vector_strength': [1.0],
    'condition_vector_threshold': [0.043],
}


def create_dummy_steering_vector(model_type, hidden_size, num_layer):
    '''Creates a dummy steering vector for a given model_type, hidden_size and num_layer.'''

    directions = {k: np.zeros((hidden_size,), dtype=float) for k in range(num_layer)}
    explained_variances = {k: 0.5 for k in range(num_layer)}
    vec = SteeringVector(model_type=model_type, directions=directions, explained_variances=explained_variances)
    return vec


@pytest.mark.parametrize("conf", build_param_grid(CAST_GRID))
def test_cast(model_and_tokenizer, device: torch.device, conf: dict):
    """
    Verify that CAST steers and generates on every model/device/param combo.
    """

    # move model to target device
    base_model, tokenizer = model_and_tokenizer
    model = base_model.to(device)

    # get model_type, hidden_size and num_layer for current model
    model_type = model.config.model_type
    hidden_size = getattr(model.config, 'hidden_size') if model_type != 'gpt2' else getattr(model.config, 'n_embd')
    num_layer = getattr(model.config, 'num_hidden_layers') if model_type != 'gpt2' else getattr(model.config, 'n_layer')

    # create (dummy) behavior and condition vectors
    behavior_vector = create_dummy_steering_vector(model.config.model_type, model.config.hidden_size, num_layer)
    condition_vector = create_dummy_steering_vector(model_type, hidden_size, num_layer)

    # build pipeline with CAST control
    cast = CAST(
        behavior_vector=behavior_vector,
        behavior_layer_ids=[0, 1],
        behavior_vector_strength=conf['behavior_vector_strength'],
        condition_vector=condition_vector,
        condition_layer_ids=[1],
        condition_vector_threshold=conf['condition_vector_threshold'],
        condition_comparator_threshold_is='larger'
    )
    pipeline = SteeringPipeline(
        controls=[cast],
        lazy_init=True,
        device_map=device,
    )
    pipeline.model = model
    pipeline.tokenizer = tokenizer
    pipeline.steer()

    # prepare prompt & runtime kwargs
    prompt_ids = tokenizer(PROMPT_TEXT, return_tensors="pt").input_ids.to(device)

    # generate
    out_ids = pipeline.generate(
        input_ids=prompt_ids,
        max_new_tokens=8,
    )

    # assertions
    assert isinstance(out_ids, torch.Tensor), "Output is not torch.Tensor"
    assert out_ids.ndim == 2, "Expected (batch, seq_len) tensor"
    assert out_ids.size(1) >= 1, "No new tokens generated"
