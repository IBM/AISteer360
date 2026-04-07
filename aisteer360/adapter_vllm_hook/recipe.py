"""Serialize and reconstruct StateControl objects across process boundaries.

When using the vLLM backend, the StateControl must cross from the main process into vLLM-Hook's worker subprocess.  
We serialize a lightweight recipe (the class path, constructor arguments, and pre-computed steering vectors)
then reconstruct and re-steer inside the worker where the real model lives.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import fields as dc_fields

from aisteer360.algorithms.state_control.common.steering_vector import SteeringVector

logger = logging.getLogger(__name__)

def serialize_state_control(state_control, work_dir: str) -> dict:
    """Serialize a post-steer StateControl into a JSON-safe recipe.

    Args:
        state_control: A steered StateControl instance.
        work_dir: Directory for temporary ``.svec`` files.

    Returns:
        Recipe dict suitable for ``json.dump()``.
    """
    recipe: dict = {
        "class": f"{type(state_control).__module__}.{type(state_control).__qualname__}",
        "constructor_args": {},
        "steering_vectors": {},
        "post_steer_state": {},
    }

    # serialize constructor args from the Args dataclass
    if hasattr(state_control, "args") and state_control.args is not None:
        for f in dc_fields(state_control.args):
            value = getattr(state_control.args, f.name)
            if isinstance(value, SteeringVector):
                svec_path = os.path.join(work_dir, f"arg_{f.name}.svec")
                value.save(svec_path)
                recipe["steering_vectors"][f.name] = svec_path
            elif _is_json_safe(value):
                recipe["constructor_args"][f.name] = value
            else:
                recipe["constructor_args"][f.name] = None

    # save post-steer computed steering vector
    sv = getattr(state_control, "_steering_vector", None)
    if isinstance(sv, SteeringVector):
        svec_path = os.path.join(work_dir, "post_steer_vector.svec")
        sv.save(svec_path)
        recipe["post_steer_state"]["steering_vector_path"] = svec_path

    # save resolved layer info
    for attr in ("_layer_id", "_layer_names"):
        val = getattr(state_control, attr, None)
        if val is not None:
            recipe["post_steer_state"][attr] = val

    return recipe


def reconstruct_state_control(recipe: dict):
    """Reconstruct a StateControl from a recipe dict.

    The returned control has its steering vector pre-loaded so that
    ``steer()`` in the worker only needs to resolve layer names and
    build the transform (no prompt-pair extraction needed).

    Args:
        recipe: Recipe dict produced by :func:`serialize_state_control`.

    Returns:
        A StateControl instance ready for ``steer(model)``.
    """
    # import the class
    class_path = recipe["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    # rebuild constructor kwargs 
    kwargs = dict(recipe.get("constructor_args", {}))

    # Load steering vectors saved from constructor args
    for field_name, svec_path in recipe.get("steering_vectors", {}).items():
        kwargs[field_name] = SteeringVector.load(svec_path)

    # If prompts were used originally but we have a post-steer vector,
    # swap to vector mode (skip extraction in the worker)
    post_steer = recipe.get("post_steer_state", {})
    sv_path = post_steer.get("steering_vector_path")
    if sv_path and kwargs.get("steering_vector") is None:
        kwargs["steering_vector"] = SteeringVector.load(sv_path)
        # Remove prompt args to satisfy the "exactly one source" validation
        kwargs.pop("positive_prompt", None)
        kwargs.pop("negative_prompt", None)

    # Filter out None values for optional args that weren't serializable
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    control = cls(**kwargs)

    # Restore post-steer state so the worker's steer() is lightweight
    if "_layer_id" in post_steer:
        control._layer_id = post_steer["_layer_id"]
    if "_layer_names" in post_steer:
        control._layer_names = post_steer["_layer_names"]

    return control


def _is_json_safe(value) -> bool:
    """Check if a value can be JSON-serialized."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_safe(v) for k, v in value.items())
    return False
