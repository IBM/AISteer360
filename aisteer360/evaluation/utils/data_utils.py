from typing import Any, Mapping

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Conversion to json-safe format.

    - primitives: pass through
    - Path: str(path)
    - mappings: recurse, stringify keys
    - sequences: recurse on elements
    - numpy scalars/arrays: convert to Python / list
    - everything else: repr(obj)
    """
    from pathlib import Path as _Path

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    if isinstance(obj, _Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]

    return repr(obj)
