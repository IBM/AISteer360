from typing import Any

import numpy as np


def to_1d_array(result: Any, n_examples: int) -> np.ndarray:
    """
    Normalize a metric's result into a 1d numpy array of length n_examples.
    """

    if isinstance(result, dict):
        if len(result) != 1:
            raise ValueError(f"Metric returned multiple values {list(result.keys())}; UseCase.evaluate expects exactly one.")
        result = next(iter(result.values()))

    array = np.asarray(result, dtype=float)
    if array.ndim == 0:
        array = np.full(n_examples, array.item(), dtype=float)
    elif array.ndim == 1:
        if array.size != n_examples:
            raise ValueError(f"Metric produced {array.size} values, but {n_examples} examples were expected.")
    else:
        raise ValueError(f"Metric returned an array with shape {array.shape}; only scalars or 1‑D arrays are supported.")

    return array
