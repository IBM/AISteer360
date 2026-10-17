"""Digests identifying the model a fitted artifact was estimated on."""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from steerability.algorithms.core.execution.backend import SteeringSession


def config_digest_dict(config: PretrainedConfig) -> dict:
    """The config fields that identify a model, i.e., the config's diff dict without `transformers_version`.

    The diff dict keeps the fields that differ from the defaults (nested sub-configs included)
    and omits `_name_or_path`. Removing `transformers_version` keeps a digest over these fields
    stable across `transformers` upgrades.

    Args:
        config: The model config.

    Returns:
        The config fields as a JSON-serializable mapping.
    """
    config_dict = config.to_diff_dict()
    config_dict.pop("transformers_version", None)
    return config_dict


def _parameter_kind(name: str) -> str:
    """The parameter name with its numeric path components replaced by `*`, e.g.,
    `model.layers.3.self_attn.q_proj.weight` becomes `model.layers.*.self_attn.q_proj.weight`."""
    return ".".join("*" if part.isdigit() else part for part in name.split("."))


def model_fingerprint(model: PreTrainedModel) -> str:
    """Compute a deterministic digest identifying a model, from its configuration, dtype, and a sample of its weights.

    The digest is a sha256 over (a) the JSON of `config_digest_dict(model.config)` with sorted
    keys, (b) `str(model.dtype)`, and (c) for the first parameter of each parameter kind in
    `named_parameters()` order, the parameter name plus its first 64 elements flattened and cast
    to float32 bytes, truncated to 16 hex characters. A parameter's kind is its name with the
    numeric path components replaced by `*`, so every projection of a decoder layer
    (`model.layers.*.self_attn.q_proj.weight`, `model.layers.*.mlp.up_proj.weight`, ...) is sampled
    once, in the first layer that contains it.

    The digest has the following properties:

    - The same checkpoint at the same dtype yields equal digests in any process or device
        placement, since parameter values are deterministic under `from_pretrained`. The
        config's `transformers_version` is excluded, so the version that wrote or loaded the
        config does not enter the digest.
    - A weight edit that touches a sampled slice (SFT, DPO, merging, a LoRA merge) changes the
        digest; so does attaching unmerged adapters, since parameter names change. An edit that
        leaves the sampled slices unchanged is not detected, e.g., an adapter merged into a subset
        of layers that excludes the sampled layer of every kind it targets.
    - dtype is part of the identity, since a calibrated operating point is a numeric artifact and
        the same checkpoint loaded at a different dtype is a different runtime model for detection
        purposes.

    Args:
        model: The model to fingerprint.

    Returns:
        A 16-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(config_digest_dict(model.config), sort_keys=True, default=str).encode("utf-8"))
    digest.update(str(model.dtype).encode("utf-8"))

    sampled_kinds: set[str] = set()
    for name, param in model.named_parameters():
        kind = _parameter_kind(name)
        if kind in sampled_kinds:
            continue
        sampled_kinds.add(kind)
        digest.update(name.encode("utf-8"))
        sample = param.detach().reshape(-1)[:64].to(device="cpu", dtype=torch.float32).contiguous()
        digest.update(sample.numpy().tobytes())

    return digest.hexdigest()[:16]


def session_artifact_identity(session: SteeringSession | None) -> tuple[str, dict]:
    """`(model_type, meta)` recorded for a session-fitted artifact, from the session layout.

    The meta carries the layout's `model_fingerprint` and `model_ref` when present, so
    identity checks against the venue that will read the artifact stay possible without a
    live model. Returns `("unknown", {})` when no session layout is available.

    Args:
        session: The `SteeringSession` the artifact was fitted through, or None.

    Returns:
        The model type and the provenance mapping.
    """
    layout = None
    if session is not None:
        try:
            layout = session.layout
        except Exception:
            layout = None
    if layout is None:
        return "unknown", {}
    meta: dict = {}
    if layout.model_fingerprint:
        meta["model_fingerprint"] = layout.model_fingerprint
    if layout.model_ref:
        meta["model_ref"] = layout.model_ref
    return layout.model_type or "unknown", meta


def artifact_provenance_meta(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase | None = None) -> dict:
    """Provenance fingerprints for a fitted steering artifact.

    Always records the toolkit model fingerprint; when `vllm_hook_plugins` is installed, adds
    the plugin's config and chat-template fingerprint recipes, which cross-check against a
    serving engine's discovery payload.

    Args:
        model: The model the artifact was fitted on.
        tokenizer: The tokenizer used during fitting, for the chat-template fingerprint.

    Returns:
        The provenance mapping. Keys: `"model_fingerprint"`, and when the plugin is installed,
        `"config_fingerprint"` and (with a tokenizer) `"chat_template_fingerprint"`.
    """
    meta = {"model_fingerprint": model_fingerprint(model)}
    try:
        from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint, config_fingerprint
    except ImportError:
        return meta
    try:
        meta["config_fingerprint"] = config_fingerprint(model.config.to_dict())
    except (TypeError, ValueError):
        pass
    if tokenizer is not None:
        meta["chat_template_fingerprint"] = chat_template_fingerprint(
            getattr(tokenizer, "chat_template", None)
        )
    return meta


_ABSENT_TEMPLATE_DIGEST = hashlib.sha256(b"").hexdigest()


def is_absent_chat_template_fingerprint(fingerprint: str | None) -> bool:
    """Whether `fingerprint` denotes an absent chat template under the plugin recipe.

    A serving engine that exposes no chat template reports the fingerprint of an empty
    template, so a mismatch against such a value reflects exposure rather than divergence
    and comparisons should skip it. Uses the plugin recipe when `vllm_hook_plugins` is
    installed and falls back to the recipe's stable digest of an empty template otherwise.

    Args:
        fingerprint: The reported fingerprint, or None when none was reported.

    Returns:
        True when the fingerprint denotes an absent chat template.
    """
    if not fingerprint:
        return True
    try:
        from vllm_hook_plugins.core.fingerprints import chat_template_fingerprint
    except ImportError:
        return fingerprint in (f"sha256:{_ABSENT_TEMPLATE_DIGEST}", _ABSENT_TEMPLATE_DIGEST)
    return fingerprint in (chat_template_fingerprint(None), chat_template_fingerprint(""))
