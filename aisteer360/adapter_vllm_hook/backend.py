"""Boot a vLLM engine with AISteer360 steering applied via a custom worker."""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import tempfile

logger = logging.getLogger(__name__)


def boot_vllm_engine(
    model_name_or_path: str,
    state_control,
    vllm_kwargs: dict | None = None,
) -> object:
    """Create a HookLLM engine with the AISteer360 generic worker.

    Serializes the post-steer ``state_control`` into a recipe, 
    registers the :class:`AISteer360Worker` in vLLM-Hook's plugin registry, 
    and boots a ``HookLLM`` instance.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        state_control: A post-steer StateControl instance.
        vllm_kwargs: Extra kwargs forwarded to ``HookLLM``.

    Returns:
        A ``HookLLM`` instance ready for generation.
    """
    from .recipe import serialize_state_control

    vllm_kwargs = dict(vllm_kwargs or {})
    work_dir = tempfile.mkdtemp(prefix="aisteer360_vllm_")

    # serialize state control
    recipe = serialize_state_control(state_control, work_dir)
    recipe["tokenizer_name_or_path"] = str(model_name_or_path)

    config_path = os.path.join(work_dir, "aisteer360_recipe.json")
    with open(config_path, "w") as f:
        json.dump(recipe, f, indent=2)

    os.environ["VLLM_AISTEER360_CONFIG"] = config_path

    # vLLM env 
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # register worker in vLLM-Hook's plugin registry 
    import vllm.plugins
    from vllm_hook_plugins import PluginRegistry
    vllm.plugins.load_general_plugins()

    from .worker import AISteer360Worker
    PluginRegistry.register_worker("aisteer360", AISteer360Worker)

    # boot HookLLM 
    from vllm_hook_plugins import HookLLM

    engine_kwargs = {
        "gpu_memory_utilization": 0.9,
        "enforce_eager": True,
        "enable_prefix_caching": True,
    }
    engine_kwargs.update(vllm_kwargs)

    logger.info("Booting vLLM engine with AISteer360Worker for %s", model_name_or_path)
    return HookLLM(
        model=str(model_name_or_path),
        worker_name="aisteer360",
        config_file=config_path,
        enable_hook=True,
        **engine_kwargs,
    )
