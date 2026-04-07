"""AISteer360 generic worker for vLLM-Hook.

Reconstructs any AISteer360 :class:`StateControl` inside the vLLM worker
subprocess and registers hooks on the PyTorch model. 
"""
from __future__ import annotations

import json
import logging
import os

import torch
from vllm.v1.worker.gpu_worker import Worker as V1Worker

logger = logging.getLogger(__name__)


class AISteer360Worker(V1Worker):
    """vLLM-Hook worker that applies any AISteer360 StateControl."""

    def load_model(self, *args, **kwargs):
        """Load the model, then reconstruct and apply the StateControl."""
        result = super().load_model(*args, **kwargs)

        config_path = os.environ.get("VLLM_AISTEER360_CONFIG")
        if not config_path:
            logger.warning("VLLM_AISTEER360_CONFIG not set; no steering applied.")
            self._state_control = None
            return result

        try:
            self._install_aisteer360_hooks(config_path)
            logger.info("AISteer360 hooks installed successfully")
        except Exception:
            logger.exception("AISteer360 hook installation failed")
            self._state_control = None

        return result

    def _install_aisteer360_hooks(self, config_path: str) -> None:
        """Read recipe, reconstruct control, steer, and register hooks."""
        from aisteer360.adapter_vllm_hook.recipe import reconstruct_state_control

        with open(config_path) as f:
            recipe = json.load(f)

        # Reconstruct the StateControl (with pre-computed vectors)
        state_control = reconstruct_state_control(recipe)

        # Access vLLM's internal PyTorch model
        model = getattr(self.model_runner, "model", None)
        if model is None:
            raise RuntimeError("Could not access model_runner.model")

        # Load tokenizer if needed for steer()
        tokenizer = None
        tokenizer_path = recipe.get("tokenizer_name_or_path")
        if tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        # Run steer() on the real model — reuses AISteer360's full logic
        state_control.steer(model, tokenizer=tokenizer)

        # Get hooks from the state control
        dummy_ids = torch.zeros(1, 1, dtype=torch.long, device="cuda")
        hooks = state_control.get_hooks(dummy_ids, runtime_kwargs={})

        # Wrap each hook function with the flag file gate so that
        # HookLLM's use_hook=False (which removes the flag file) disables steering.
        hook_flag = os.environ.get("VLLM_HOOK_FLAG")
        for phase in ("pre", "forward", "backward"):
            for spec in hooks.get(phase, []):
                spec["hook_func"] = _gated_hook(spec["hook_func"], hook_flag, phase)

        state_control.set_hooks(hooks)
        state_control._model_ref = model
        state_control.register_hooks(model)

        # Store reference
        self._state_control = state_control
        self.hook_flag = hook_flag

        logger.info(
            "Registered %d hooks from %s",
            len(state_control.registered),
            type(state_control).__name__,
        )

    def execute_model(self, *args, **kwargs):
        return super().execute_model(*args, **kwargs)


def _is_gated_out(flag_path: str | None) -> bool:
    """Return True if steering is disabled by a missing flag file."""
    return bool(flag_path) and not os.path.exists(flag_path)


def _reshape_hidden_args(args, kwargs, *, squeeze: bool):
    """Reshape hidden_states between 2-D ``[N, H]`` and 3-D ``[B, T, H]``.

    When *squeeze* is False, unsqueezes 2-D → 3-D.
    When *squeeze* is True, squeezes 3-D (batch=1) → 2-D.
    """
    from aisteer360.algorithms.state_control.common.hook_utils import (
        _hidden_states_index,
        extract_hidden_states,
    )
    hidden = extract_hidden_states(args, kwargs)
    if hidden is None:
        return args, kwargs

    if squeeze:
        needs_reshape = hidden.ndim == 3 and hidden.size(0) == 1
        op = torch.Tensor.squeeze
    else:
        needs_reshape = hidden.ndim == 2
        op = torch.Tensor.unsqueeze

    if not needs_reshape:
        return args, kwargs

    new_hidden = op(hidden, 0)
    if args:
        idx = _hidden_states_index(args)
        if idx is not None:
            args = (*args[:idx], new_hidden, *args[idx + 1:])
    elif "hidden_states" in kwargs:
        kwargs = {**kwargs, "hidden_states": new_hidden}
    return args, kwargs


def _reshape_output(output, *, squeeze: bool):
    """Reshape tensors in forward-hook output between 2-D and 3-D."""
    if output is None:
        return None

    if squeeze:
        check = lambda t: t.ndim == 3 and t.size(0) == 1
        op = lambda t: t.squeeze(0)
    else:
        check = lambda t: t.ndim == 2
        op = lambda t: t.unsqueeze(0)

    if isinstance(output, torch.Tensor):
        return op(output) if check(output) else output
    if isinstance(output, tuple):
        return tuple(op(t) if isinstance(t, torch.Tensor) and check(t) else t for t in output)
    return output


def _gated_hook(original_hook, flag_path: str | None, phase: str):
    """Wrap a hook with flag-file gating and vLLM tensor shape normalization.

    AISteer360 hooks expect 3-D hidden states ``[B, T, H]`` but vLLM
    passes 2-D ``[N, H]``.  This wrapper unsqueezes inputs before the
    original hook and squeezes outputs back, keeping the original hooks
    untouched.
    """
    if flag_path is None and phase not in ("pre", "forward"):
        return original_hook

    if phase == "pre":
        def wrapper(module, args, kwargs):
            if _is_gated_out(flag_path):
                return None
            args, kwargs = _reshape_hidden_args(args, kwargs, squeeze=False)
            result = original_hook(module, args, kwargs)
            if result is None:
                return None
            return _reshape_hidden_args(*result, squeeze=True)
        return wrapper

    if phase == "forward":
        def wrapper(module, args, kwargs, output):
            if _is_gated_out(flag_path):
                return None
            output = _reshape_output(output, squeeze=False)
            result = original_hook(module, args, kwargs, output)
            return _reshape_output(result, squeeze=True)
        return wrapper

    def wrapper(*args, **kwargs):
        if _is_gated_out(flag_path):
            return None
        return original_hook(*args, **kwargs)
    return wrapper
