"""
Discovers steering methods at import‑time for cli reference.
"""
import logging
from importlib import import_module
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

REGISTRY: dict[str, dict[str, "SteeringMethod"]] = {}


class SteeringMethod:
    """Container for a discovered steering method's metadata.

    Attributes:
       category: Category name (e.g., "state", "input")
       name: Method name (e.g., "pasta", "few_shot")
       control_cls: The control class implementation
       args_cls: The args dataclass for configuration
    """
    def __init__(self, category: str, name: str, control_cls: type, args_cls: type):
        self.category = category
        self.name = name
        self.control_cls = control_cls
        self.args_cls = args_cls


def _crawl_methods() -> None:
    """Auto-discover all steering methods by recursively crawling the algorithms directory.

    For each top-level category directory (input_control, structural_control, state_control,
    output_control), walks all nested __init__.py files and imports any module that exports a
    `STEERING_METHOD` dict. The exported dict is registered under the category's bucket keyed by
    the method name.
    """
    for category_dir in ROOT.iterdir():
        if not category_dir.is_dir() or category_dir.name == "core":
            continue
        category = category_dir.name.removesuffix("_control")

        for init_file in category_dir.rglob("__init__.py"):
            rel_parts = init_file.relative_to(ROOT).parent.parts
            if not rel_parts:
                continue
            module_path = "aisteer360.algorithms." + ".".join(rel_parts)
            try:
                module = import_module(module_path)
            except ImportError as exc:
                logger.warning("Skipping %s: missing optional dependency (%s).", module_path, exc)
                continue
            method = getattr(module, "STEERING_METHOD", None)
            if method is None:
                continue

            REGISTRY.setdefault(category + "_control", {})[method["name"]] = \
                SteeringMethod(category, method["name"], method["control"], method["args"])


_crawl_methods()
