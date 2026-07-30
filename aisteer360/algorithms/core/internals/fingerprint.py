"""Digests identifying the model a fitted artifact was estimated on."""
import hashlib

import torch
from transformers import PreTrainedModel


def model_fingerprint(model: PreTrainedModel) -> str:
    """Deterministic identity digest of a model's configuration, dtype, and sampled weights.

    The digest is a sha256 over (a) `model.config.to_json_string()`, (b) `str(model.dtype)`, and
    (c) for up to 8 evenly spaced named parameters, the parameter name plus its first 64 elements
    flattened and cast to float32 bytes, truncated to 16 hex characters.

    Properties:

    - The same checkpoint at the same dtype yields equal digests in any process or device
        placement, since parameter values are deterministic under `from_pretrained`.
    - Any weight edit that touches the sampled slices (SFT, DPO, merging, LoRA merge) changes the
        digest; so does attaching unmerged adapters, since parameter names change.
    - dtype is part of the identity, since a calibrated operating point is a numeric artifact and
        the same checkpoint loaded at a different dtype is a different runtime model for detection
        purposes.

    Args:
        model: The model to fingerprint.

    Returns:
        A 16-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    digest.update(model.config.to_json_string().encode("utf-8"))
    digest.update(str(model.dtype).encode("utf-8"))

    named_parameters = list(model.named_parameters())
    n = len(named_parameters)
    if n > 0:
        k = min(8, n)
        indices = [0] if k == 1 else [round(i * (n - 1) / (k - 1)) for i in range(k)]
        for idx in dict.fromkeys(indices):
            name, param = named_parameters[idx]
            digest.update(name.encode("utf-8"))
            sample = param.detach().reshape(-1)[:64].to(device="cpu", dtype=torch.float32).contiguous()
            digest.update(sample.numpy().tobytes())

    return digest.hexdigest()[:16]
