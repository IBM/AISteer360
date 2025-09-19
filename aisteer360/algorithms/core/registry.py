"""
Discovers steering methods at import‑time for cli reference.
"""
from importlib import import_module
from pathlib import Path
from typing import Dict, Type

ROOT = Path(__file__).resolve().parents[1] / "algorithms"

REGISTRY: Dict[str, Dict[str, "SteeringMethod"]] = {}


class SteeringMethod:
    """Container for a discovered steering method's metadata.

    Attributes:
       category: Category name (e.g., "state", "input")
       name: Method name (e.g., "pasta", "few_shot")
       control_cls: The control class implementation
       args_cls: The args dataclass for configuration
    """
    def __init__(self, category: str, name: str, control_cls: Type, args_cls: Type):
        self.category = category
        self.name = name
        self.control_cls = control_cls
        self.args_cls = args_cls


def _crawl_methods() -> None:
    """Auto-discover all steering methods by crawling the algorithms directory.

    Looks for STEERING_METHOD export in each method's __init__.py and populates the global REGISTRY for CLI and
    dynamic instantiation.
    """
    for category_dir in ROOT.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for method_dir in category_dir.iterdir():
            if not (method_dir / "__init__.py").exists():
                continue
            module_path = f"aisteer360.algorithms.{category}.{method_dir.name}"
            module = import_module(module_path)

            method = getattr(module, "STEERING_METHOD", None)
            if method is None:
                continue

            REGISTRY.setdefault(category + "_control", {})[method["name"]] = \
                SteeringMethod(category, method["name"], method["control"], method["args"])


_crawl_methods()
