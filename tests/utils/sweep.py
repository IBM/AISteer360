import itertools
from typing import Any, Callable

import pytest


def build_param_grid(
    grid: dict[str, list[Any]],
    fixed: dict[str, Any] | None = None,
    id_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[pytest.param]:
    """
    Generate a list of pytest.param objects from a parameter grid.
    """
    fixed = fixed or {}
    if not grid:  # no sweep dimensions
        conf = dict(fixed)
        return [pytest.param(conf, id=id_fn(conf) if id_fn else "default")]

    keys = list(grid)
    params: list[pytest.param] = []

    for combo in itertools.product(*(grid[k] for k in keys)):
        conf = {k: v for k, v in zip(keys, combo)}
        conf.update(fixed)
        params.append(pytest.param(conf, id=id_fn(conf) if id_fn else _default_id(conf, keys)))

    return params


def _default_id(conf: dict[str, Any], keys: list[str]) -> str:
    return "-".join(f"{key}={conf[key]}" for key in keys)
