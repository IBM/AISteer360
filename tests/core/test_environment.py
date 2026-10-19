"""Checks that a CI environment installed with an optional extra can import it.

Several suites skip when an optional dependency is missing (e.g., the spec/hook parity suite skips
without `vllm_hook_plugins`). A CI job installed with the extra sets the matching environment
variable, and the test below then fails instead of letting those suites skip silently.
"""
import importlib
import os

import pytest

REQUIRED_MODULES = {
    "STEERABILITY_REQUIRE_PLUGINS": "vllm_hook_plugins",
    "STEERABILITY_REQUIRE_MERGEKIT": "mergekit",
}


@pytest.mark.parametrize("variable, module", sorted(REQUIRED_MODULES.items()))
def test_required_optional_module_is_importable(variable, module):
    if os.environ.get(variable) != "1":
        pytest.skip(f"set {variable}=1 to require {module}")
    try:
        importlib.import_module(module)
    except ImportError as error:
        pytest.fail(f"{variable}=1 but {module} is not importable: {error}")
