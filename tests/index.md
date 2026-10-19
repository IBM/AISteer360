# Tests

Tests evaluate control implementations across a set of control arguments, models (listed in `tests/utils/ci_models.yaml`),
and the available devices on the machine. Each entry in `tests/utils/ci_models.yaml` is either a Hub id or a
`local:<builder>` entry for a model that a function in `tests/utils/tiny_models.py` builds in process.

The test tree is organized by area:

- `tests/controls/` covers individual steering controls
- `tests/core/` covers the pipeline, registry, and other core functionality
- `tests/internals/` covers the `core/internals` substrate and probes
- `tests/evaluation/` covers the Inspect AI evaluation stack (provider, collator, solver, scorer adapter, suite, runner)
- `tests/utils/` holds shared test utilities

## Executing tests

Running tests requires that the toolkit is installed with the `all` extra and the `dev` dependency group. First, run:
```commandline
uv sync --extra all
```
To execute tests for all controls, run:
```commandline
pytest tests/controls/
```
Tests for single controls are executed by specifying the file name, e.g.,:
```commandline
pytest tests/controls/test_pasta.py
```
Tests that download Hub models outside `tests/utils/ci_models.yaml` are marked `network`. On a machine with only the CI
models cached, deselect them with:
```commandline
pytest -m 'not network' tests/
```

## Gated and seed-dependent tests

Some tests run only when an environment variable is set, and some skip when the randomly initialized tiny model does
not separate the classes they need for the fixed seed:

| Test | Behavior |
| --- | --- |
| `tests/core/test_staged_steer.py` (TRL staged-steer smoke test) | Runs with `RUN_TRL_SMOKE=1`; trains a tiny SFT LoRA. |
| `tests/controls/test_grpo_wrapper.py` (GRPO smoke test) | Runs with `RUN_GRPO_SMOKE=1`; runs one GRPO step on a tiny model. |
| `tests/core/test_environment.py` | Fails when `STEERABILITY_REQUIRE_PLUGINS=1` (or `STEERABILITY_REQUIRE_MERGEKIT=1`) is set and `vllm_hook_plugins` (or `mergekit`) is not importable; skips otherwise. |
| `tests/controls/test_position_tracking_goldens.py` | `STEERABILITY_CAPTURE_GOLDENS=1` (with `-s`) prints the generated sequences to paste into `GOLDENS`. |
| `tests/controls/test_cast_conditional.py` (`test_rows_gated_independently`, `test_row_zero_closed_row_one_open`, `test_gate_open_ablates_gate_closed_untouched`) | Skips when the condition scores are not separable for the seed. |
| `tests/controls/test_routed_decoding.py` (`test_three_routes_in_one_call`) | Skips when the probe scores are not separable for the seed. |
| `tests/controls/test_probe_condition.py` (`test_adapter_decisions_match_probe_set_read`) | Skips when the probe scores are not separable for the seed. |

## Adding your own control test

To add your own test for a steering method, please follow the pattern in the existing test files. Specifically:

1. Create a test file named `test_your_control.py` in the `tests/controls/` directory

2. Define test parameters for your control using a grid dictionary and `build_param_grid()`:
```python
YOUR_CONTROL_GRID = {
    "param_1": [value_1_1, value_1_2],
    "param_2": [value_2_1, value_2_2, value_2_3],
    # ... other parameters your control accepts
}

@pytest.mark.parametrize("conf", build_param_grid(YOUR_CONTROL_GRID))
def test_your_control(model_and_tokenizer, device: torch.device, conf: dict):
```

3. Follow the standard test pattern:
   - Move model to target device
   - Initialize your control with parameters from `conf`
   - Create a `SteeringPipeline` with your control
   - Call `pipeline.steer()` to set up the model
   - Prepare prompt and any runtime kwargs (if necessary)
   - Call `pipeline.generate()`
   - Specify necessary assertions

Executing the test will automatically run your test across all available models and devices with every combination of parameters in your grid.
