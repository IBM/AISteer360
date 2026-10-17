"""The `.spipe` value codec: round-tripping between constructor values and manifest JSON.

Plain JSON values pass through; everything else encodes as a single-key tagged object:

- `$artifact`: tensor-bearing or on-disk state, stored content-addressed (`store.py`).
- `$dc`: pure-data dataclasses and enums, decoded by importing the class and calling it.
- `$component`: state-control components (transforms, gates, selectors) and routing objects,
  decoded through `COMPONENT_KINDS`.
- `$ref`: module-level callables, decoded by import only under `allow_code=True`.
- `$data`: dataset references (`DataRef`), materialized at pipeline construction.
- `$map`: mappings with non-string scalar keys, kept as `[key, value]` pairs sorted by key so
  key types are retained through the JSON round trip and the encoding is independent of
  insertion order.

The codec round-trips; `identity.canonical_value` remains the separate, hash-only
canonicalizer. Digest computation for fit identities goes through `encode` in digest mode
(`EncodeContext(store=None)`), where tensors reduce to content ids without being written and
unhandled objects reduce to their type name.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from steerability.spipe.errors import SpipeCodeRefError, SpipeFormatError, SpipeSaveError
from steerability.spipe.store import (
    ArtifactRecord,
    ArtifactStore,
    check_artifact_id,
    check_sidecar,
    pickle_bearing_files,
    tensors_payload,
)

logger = logging.getLogger(__name__)

TAGS = ("$artifact", "$dc", "$component", "$ref", "$data", "$map")

TRUSTED_DC_PREFIX = "steerability."

# toolkit dataclasses that control args contain and whose construction only validates and stores
# its fields; `$dc` constructs any other dataclass only under `allow_code`
DECODABLE_DATACLASSES = frozenset({
    "steerability.algorithms.core.execution.payloads.ConstraintSource",
    "steerability.algorithms.core.internals.data.ContrastivePairs",
    "steerability.algorithms.core.internals.data.LabeledExamples",
    "steerability.algorithms.core.internals.probes.fitting.ProbeFitSpec",
    "steerability.algorithms.core.internals.probes.probe_set.ProbeSetFit",
    "steerability.algorithms.core.internals.stats.ActivationStats",
    "steerability.algorithms.core.internals.stats.StatsSpec",
    "steerability.algorithms.output_control.common.drivers.phased.Fixed",
    "steerability.algorithms.output_control.common.drivers.phased.Generated",
    "steerability.algorithms.output_control.routed_decoding.actions.Generate",
    "steerability.algorithms.output_control.routed_decoding.actions.Prefix",
    "steerability.algorithms.output_control.routed_decoding.actions.Respond",
    "steerability.algorithms.state_control.common.fit_specs.ConditionSearchSpec",
    "steerability.algorithms.state_control.common.fit_specs.VectorTrainSpec",
    "steerability.algorithms.state_control.common.selectors.condition_point.ConditionPoint",
    "steerability.algorithms.state_control.common.sources.ConditionPointSearch",
    "steerability.algorithms.state_control.common.sources.ContrastiveFit",
    "steerability.algorithms.state_control.common.sources.LayerFilteredFit",
    "steerability.algorithms.state_control.common.sources.SinglePairFit",
    "steerability.algorithms.state_control.pasta.profiling.HeadProfile",
})

# trainer state written next to trained weights by `Trainer.save_model` and trainer checkpoints
TRAINER_STATE_PATTERNS = (
    "training_args.bin", "optimizer.pt", "optimizer.bin", "scheduler.pt", "scaler.pt", "rng_state*.pth",
)


def _trusted_dc_qualnames() -> frozenset[str]:
    """Return the qualnames of the enums and dtypes outside `steerability.` that decode without `allow_code`."""
    from peft import PeftType, TaskType

    return frozenset({
        f"{PeftType.__module__}.{PeftType.__qualname__}",
        f"{TaskType.__module__}.{TaskType.__qualname__}",
        "torch.dtype",
    })


@dataclass(frozen=True)
class DataRef:
    """A reference to a dataset, resolved at pipeline construction.

    Attributes:
        kind: `"hf"` (a Hugging Face Hub dataset), `"path"` (a digest-pinned local file), or
            `"opaque"` (an in-memory dataset recorded by fingerprint only, not reloadable).
        repo_id: Hub dataset id (`"hf"` only).
        revision: Hub revision (`"hf"` only).
        split: Dataset split (`"hf"` only).
        subset: Dataset configuration name (`"hf"` only).
        path: Local file path (`"path"` only).
        sha256: Hex digest the file must match (`"path"` only).
        type: Type qualname of the recorded object (`"opaque"` only).
        fingerprint: Content fingerprint of the recorded object (`"opaque"` only), or None.
    """

    kind: str
    repo_id: str | None = None
    revision: str | None = None
    split: str | None = None
    subset: str | None = None
    path: str | None = None
    sha256: str | None = None
    type: str | None = None
    fingerprint: str | None = None

    def load(self) -> Any:
        """Materialize the reference.

        Returns:
            The loaded dataset for `"hf"`, the verified path string for `"path"`.

        Raises:
            SpipeFormatError: If the kind is `"opaque"` (not reloadable) or unknown.
            SpipeIntegrityError: If a `"path"` file does not match its recorded digest.
        """
        if self.kind == "hf":
            from datasets import load_dataset

            kwargs: dict[str, Any] = {}
            if self.revision is not None:
                kwargs["revision"] = self.revision
            if self.split is not None:
                kwargs["split"] = self.split
            if self.subset is not None:
                return load_dataset(self.repo_id, self.subset, **kwargs)
            return load_dataset(self.repo_id, **kwargs)
        if self.kind == "path":
            import hashlib

            from steerability.spipe.errors import SpipeIntegrityError

            digest = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
            if self.sha256 is not None and digest != self.sha256:
                raise SpipeIntegrityError(
                    f"Data file {self.path} hashes to {digest}, expected {self.sha256}."
                )
            return self.path
        raise SpipeFormatError(
            f"DataRef kind {self.kind!r} cannot be materialized; an opaque dataset reference "
            "records identity only. Re-supply the dataset (or a 'hf'/'path' DataRef) to re-steer."
        )


@dataclass(frozen=True)
class CodeRef:
    """Placeholder for a `$ref` callable that was decoded without importing it.

    Decoding with `code_mode="sentinel"` returns a `CodeRef` for each `$ref` callable. The
    staleness check, the `recipe_id` and `config_id` digests of a spipe without a lock section,
    and `SPipe.instantiate_entry(lenient=True)` decode this way. A `CodeRef` encodes back to the
    same `$ref`. Calling it raises `SpipeCodeRefError`.

    Attributes:
        target: The `"<module>:<qualname>"` reference.
    """

    target: str

    def __call__(self, *args, **kwargs):
        raise SpipeCodeRefError(
            f"Callable reference {self.target!r} was decoded without importing it (lenient or "
            "digest decoding); to import it, load the spipe with allow_code=True and instantiate "
            "without lenient=True."
        )


class AsPath:
    """Marker wrapping an artifact value whose decoded form is the payload path.

    Used by `frozen_form` implementations whose constructor field takes a path rather than the
    reconstructed object (e.g. `sasa`'s `wv_path`, `load_lora`'s `path`).
    """

    def __init__(self, value: Any):
        self.value = value


@dataclass
class EncodeContext:
    """State threaded through one encoding pass.

    Attributes:
        store: Destination artifact store, or None for digest mode (tensor ids are computed
            without writing, unhandled objects reduce to their type name, and callables never
            raise).
        records: Artifact records produced during the pass, keyed by artifact id.
        code_refs: `$ref` targets produced during the pass (drives `code_dependent`).
        provenance: Provenance mapping stamped onto artifact records.
        default_model_type: `model_type` stamped onto exported vectors recorded as
            `"unknown"`, when the producing model is known.
    """

    store: ArtifactStore | None = None
    records: dict[str, ArtifactRecord] = field(default_factory=dict)
    code_refs: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    default_model_type: str | None = None

    # per-artifact fields installed by the freeze layer before encoding one state value
    artifact_fields: dict = field(default_factory=dict)

    @property
    def digest_mode(self) -> bool:
        return self.store is None


@dataclass
class DecodeContext:
    """The artifact store and the policies that apply during one decoding pass.

    Attributes:
        store: The artifact store that `$artifact` references are read from, or None when no
            store is available. Decoding an `$artifact` reference without a store raises
            `SpipeIntegrityError`.
        allow_code: Whether decoding may import `$ref` callables, decode `$dc` values whose
            class is outside the data-only set, and load artifact payloads that contain pickled
            data. The data-only set consists of the enums in the `steerability.` namespace, the
            enums in `_trusted_dc_qualnames()`, and the dataclasses in `DECODABLE_DATACLASSES`.
        code_mode: How `$ref` callables decode. With `"strict"`, decoding imports the `$ref`
            target when `allow_code` is True and raises `SpipeCodeRefError` otherwise. With
            `"sentinel"`, a `$ref` decodes to a `CodeRef` without importing, whatever the value
            of `allow_code`.
        verify: The verification policy for frozen steering artifacts (`"strict"`, `"warn"`,
            or `"off"`).
        data_mode: How `$data` references decode. With `"load"`, a reference of kind `"hf"`
            or `"path"` is loaded, and an `"opaque"` reference decodes to a `DataRef`. With
            `"keep"`, every reference decodes to a `DataRef`.
        manifest_records: The artifact records of the manifest's resolved entries, keyed by
            artifact id. When an `$artifact` reference has a manifest record, the store
            sidecar must match the record's `encoding` and `type`. The record's
            `artifact_class` and `provenance` then determine whether and how a decoded
            `SteeringVector` is verified at `steer()`. The sidecar supplies the reconstruction
            metadata.
    """

    store: ArtifactStore | None = None
    allow_code: bool = False
    code_mode: str = "strict"
    verify: str = "strict"
    data_mode: str = "load"
    manifest_records: Mapping[str, ArtifactRecord] = field(default_factory=dict)


def _qualname(obj_type: type) -> str:
    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def _callable_target(value: Any, path: str) -> str:
    """The `"<module>:<qualname>"` target of a module-level callable.

    Raises:
        SpipeSaveError: If the callable has no stable module-level name (lambda, closure,
            `functools.partial`, bound method).
    """
    import functools

    if isinstance(value, functools.partial):
        raise SpipeSaveError(
            f"{path}: functools.partial does not serialize; give the function a module-level name."
        )
    if getattr(value, "__self__", None) is not None:
        raise SpipeSaveError(
            f"{path}: bound methods do not serialize; give the function a module-level name."
        )
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname or "<lambda>" in qualname or "<locals>" in qualname:
        raise SpipeSaveError(
            f"{path}: {qualname or type(value).__name__} does not serialize; give the function "
            "a module-level name."
        )
    return f"{module}:{qualname}"


def _steering_vector_type_meta(vector, default_model_type: str | None) -> dict:
    """Sidecar `type_meta` for a `SteeringVector`."""
    model_type = vector.model_type
    if model_type == "unknown" and default_model_type:
        model_type = default_model_type
    meta: dict[str, Any] = {"model_type": model_type}
    if vector.num_heads is not None:
        meta["num_heads"] = int(vector.num_heads)
    if vector.head_dim is not None:
        meta["head_dim"] = int(vector.head_dim)
    if vector.explained_variances:
        meta["explained_variances"] = {str(k): float(v) for k, v in vector.explained_variances.items()}
    if vector.probe_accuracies:
        meta["probe_accuracies"] = {f"{l}:{h}": float(a) for (l, h), a in vector.probe_accuracies.items()}
    if vector.meta:
        meta["meta"] = dict(vector.meta)
    return meta


def _artifact_ref(record: ArtifactRecord, ctx: EncodeContext, as_path: bool = False) -> dict:
    # first write wins: a metadata-rich record from the freeze walk is kept when the same
    # content is re-encoded later (e.g. inside recipe args)
    ctx.records.setdefault(record.id, record)
    ref: dict[str, Any] = {"$artifact": record.id}
    if as_path:
        ref["as"] = "path"
    return ref


def _record_fields(ctx: EncodeContext, type_name: str, type_meta: dict | None = None) -> dict:
    fields_ = {
        "type": type_name,
        "artifact_class": "opaque",
        "source": None,
        "fit_digest": None,
        "provenance": dict(ctx.provenance),
        "type_meta": type_meta or {},
    }
    fields_.update(ctx.artifact_fields)
    return fields_


def _encode_steering_vector(vector, ctx: EncodeContext, path: str) -> dict:
    tensors = {str(layer_id): direction for layer_id, direction in vector.directions.items()}
    type_meta = _steering_vector_type_meta(vector, ctx.default_model_type)
    record_fields = _record_fields(ctx, "SteeringVector", type_meta)
    if ctx.digest_mode:
        artifact_id, _ = tensors_payload(tensors)
        record = ArtifactRecord(id=artifact_id, encoding="tensors", **record_fields)
    else:
        record = ctx.store.put_tensors(tensors, record_fields)
    return _artifact_ref(record, ctx)


def _encode_tree_object(value, save_fn, type_name: str, ctx: EncodeContext, path: str,
                        type_meta: dict | None = None, as_path: bool = False) -> dict:
    if ctx.digest_mode:
        # digest mode never writes; identify the object by a stable, value-blind form
        return {"$data": {"kind": "opaque", "type": type_name}}
    from steerability.spipe.store import save_object_tree

    record = save_object_tree(save_fn, ctx.store, _record_fields(ctx, type_name, type_meta))
    return _artifact_ref(record, ctx, as_path=as_path)


def _encode_artifact_object(value: Any, ctx: EncodeContext, path: str, as_path: bool = False) -> dict | None:
    """Encode a typed artifact value to an `$artifact` ref, or return None when `value` is not
    an artifact type."""
    from steerability.algorithms.core.execution.payloads import CheckpointArtifact, LoRAArtifact
    from steerability.algorithms.core.internals.probes.probe import Probe
    from steerability.algorithms.core.internals.probes.probe_set import ProbeSet
    from steerability.algorithms.input_control.common.memory.pool import PoolMemory
    from steerability.algorithms.input_control.common.memory.text import TextMemory
    from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed, _Precomputed
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    if isinstance(value, torch.Tensor):
        record_fields = _record_fields(ctx, "Tensor")
        if ctx.digest_mode:
            artifact_id, _ = tensors_payload({"value": value})
            record = ArtifactRecord(id=artifact_id, encoding="tensors", **record_fields)
        else:
            record = ctx.store.put_tensors({"value": value}, record_fields)
        return _artifact_ref(record, ctx, as_path=as_path)
    if isinstance(value, SteeringVector):
        return _encode_steering_vector(value, ctx, path)
    if isinstance(value, VerifiedPrecomputed):
        return _encode_steering_vector(value.steering_vector, ctx, path)
    if isinstance(value, _Precomputed):
        return _encode_steering_vector(value._steering_vector, ctx, path)
    if isinstance(value, Probe):
        return _encode_tree_object(value, lambda d: value.save(d), "Probe", ctx, path, as_path=as_path)
    if isinstance(value, ProbeSet):
        def _save_probe_set(directory: Path) -> None:
            for name, probe in value.probes.items():
                probe.save(directory / name)

        return _encode_tree_object(
            value, _save_probe_set, "ProbeSet", ctx, path,
            type_meta={"names": list(value.names)}, as_path=as_path,
        )
    if isinstance(value, TextMemory):
        return _encode_tree_object(
            value, lambda d: value.save(d / "memory.json"), "TextMemory", ctx, path, as_path=as_path,
        )
    if isinstance(value, PoolMemory):
        return _encode_tree_object(
            value, lambda d: value.save(d / "memory.pkl"), "PoolMemory", ctx, path, as_path=as_path,
        )
    if type(value).__name__ == "CPOMemory" and hasattr(value, "causal_scorer"):
        return _encode_tree_object(
            value, lambda d: value.save(d), "CPOMemory", ctx, path, as_path=True,
        )
    if isinstance(value, LoRAArtifact):
        return _encode_tree_object(
            None, lambda d: _copy_tree_contents(value.path, d), "LoRAArtifact", ctx, path,
            type_meta={"base_model": value.base_model}, as_path=True,
        )
    if isinstance(value, CheckpointArtifact):
        return _encode_tree_object(
            None, lambda d: _copy_tree_contents(value.path, d), "CheckpointArtifact", ctx, path,
            as_path=True,
        )
    return None


def _copy_tree_contents(source: str | Path, dest: Path) -> None:
    """Copy the contents of an adapter or checkpoint directory into `dest`, leaving out trainer state.

    Files that match `TRAINER_STATE_PATTERNS` are not copied. They contain pickled data, and the
    `load_lora` and `load_checkpoint` controls do not read them. A warning is logged when the
    copied files still contain pickled data (e.g., `pytorch_model.bin` weights), since decoding
    the resulting artifact requires `allow_code=True`.

    Args:
        source: The adapter or checkpoint directory.
        dest: The directory to copy into, which may already exist.

    Raises:
        SpipeSaveError: If `source` is not a directory.
    """
    import shutil

    source = Path(source)
    if not source.is_dir():
        raise SpipeSaveError(f"Artifact directory {source} does not exist.")
    shutil.copytree(
        source, dest, symlinks=False, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*TRAINER_STATE_PATTERNS),
    )
    pickled = pickle_bearing_files(dest)
    if pickled:
        logger.warning(
            "Artifact directory %s contains pickled files (%s); loading a spipe that embeds it "
            "requires allow_code=True.", source, ", ".join(pickled),
        )


def _encode_dataclass(value: Any, ctx: EncodeContext, path: str) -> dict:
    encoded_fields = {}
    for f in fields(value):
        if not f.init:
            continue
        encoded_fields[f.name] = encode(getattr(value, f.name), ctx, f"{path}.{f.name}")
    return {"$dc": _qualname(type(value)), "fields": encoded_fields}


def _looks_like_hf_dataset(value: Any) -> bool:
    module = getattr(type(value), "__module__", "") or ""
    return module.split(".")[0] == "datasets" and hasattr(value, "_fingerprint")


def _is_live_model_or_tokenizer(value: Any) -> bool:
    if isinstance(value, torch.nn.Module):
        return True
    for cls in type(value).__mro__:
        if cls.__name__ in ("PreTrainedTokenizerBase", "PreTrainedModel"):
            return True
    return False


def encode(value: Any, ctx: EncodeContext, path: str = "$") -> Any:
    """Encode one constructor value to its manifest JSON form.

    Args:
        value: The value to encode.
        ctx: The encoding context; artifacts land in `ctx.store` (or reduce to ids in digest
            mode) and `$ref` targets accumulate on `ctx.code_refs`.
        path: Breadcrumb naming the position of `value`, used in error messages.

    Returns:
        A JSON-serializable form of `value`.

    Raises:
        SpipeSaveError: If `value` (or a nested value) has no serialized form; the message
            names the path and, where one exists, the alternative.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, AsPath):
        encoded = _encode_artifact_object(value.value, ctx, path, as_path=True)
        if encoded is None:
            raise SpipeSaveError(f"{path}: AsPath wraps a non-artifact value ({type(value.value).__name__}).")
        return encoded
    if isinstance(value, DataRef):
        payload = {k: v for k, v in (
            ("kind", value.kind), ("repo_id", value.repo_id), ("revision", value.revision),
            ("split", value.split), ("subset", value.subset), ("path", value.path),
            ("sha256", value.sha256), ("type", value.type), ("fingerprint", value.fingerprint),
        ) if v is not None}
        return {"$data": payload}
    if isinstance(value, CodeRef):
        ctx.code_refs.append(value.target)
        return {"$ref": value.target}
    if isinstance(value, torch.dtype):
        return {"$dc": "torch.dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, Enum):
        return {"$dc": _qualname(type(value)), "value": value.name}

    artifact = _encode_artifact_object(value, ctx, path)
    if artifact is not None:
        return artifact

    component = _encode_component(value, ctx, path)
    if component is not None:
        return component

    if _looks_like_hf_dataset(value):
        return {"$data": {
            "kind": "opaque",
            "type": _qualname(type(value)),
            "fingerprint": getattr(value, "_fingerprint", None),
        }}

    if is_dataclass(value) and not isinstance(value, type):
        return _encode_dataclass(value, ctx, path)

    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            encoded = {}
            for key, item in value.items():
                if key.startswith("$"):
                    raise SpipeSaveError(f"{path}: mapping key {key!r} uses the reserved '$' prefix.")
                encoded[key] = encode(item, ctx, f"{path}[{key!r}]")
            return encoded
        # non-string scalar keys keep their exact type through the $map form
        entries = []
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise SpipeSaveError(
                    f"{path}: mapping key {key!r} of type {type(key).__name__} has no "
                    "serialized form; mapping keys must be strings, ints, floats, or bools."
                )
            entries.append([key, encode(item, ctx, f"{path}[{key!r}]")])
        entries.sort(key=lambda entry: (type(entry[0]).__name__, repr(entry[0])))
        return {"$map": entries}
    if isinstance(value, (list, tuple)):
        return [encode(item, ctx, f"{path}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        encoded_items = [encode(item, ctx, path) for item in value]
        return sorted(encoded_items, key=lambda item: json.dumps(item, sort_keys=True))

    if _is_live_model_or_tokenizer(value):
        raise SpipeSaveError(
            f"{path}: live model/tokenizer objects do not serialize; pass the corresponding "
            "*_name_or_path reference instead."
        )

    if callable(value):
        if ctx.digest_mode:
            try:
                target = _callable_target(value, path)
            except SpipeSaveError:
                target = f"callable:{getattr(value, '__qualname__', type(value).__name__)}"
            return {"$ref": target}
        target = _callable_target(value, path)
        ctx.code_refs.append(target)
        return {"$ref": target}

    if ctx.digest_mode:
        return {"$type": _qualname(type(value))}
    raise SpipeSaveError(
        f"{path}: values of type {_qualname(type(value))} have no serialized form."
    )


def encoded_size(encoded: Any) -> int:
    """Byte length of the canonical JSON form of an encoded value."""
    return len(json.dumps(encoded, sort_keys=True).encode("utf-8"))


def digest_of(value: Any) -> str:
    """12-hex-character digest of a value's encoded form, in digest mode.

    Tensors reduce to content-addressed ids (float32 form), callables to their qualified
    names, and unhandled objects to their type names, which keeps the digest stable across a
    save and load round trip.

    Args:
        value: The value to digest.

    Returns:
        The digest.
    """
    import hashlib

    encoded = encode(value, EncodeContext(store=None))
    serialized = json.dumps(encoded, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def _resolve_target(module_name: str, qual: str, target: str, path: str) -> Any:
    """Import a module and return the object at a dotted attribute path in it.

    Args:
        module_name: The module to import.
        qual: The dotted attribute path in the module.
        target: The full reference, used in the error message.
        path: The position of the value in the manifest, used in the error message.

    Returns:
        The object at `qual` in the module.

    Raises:
        SpipeFormatError: If the module cannot be imported in this process, or if `qual` does not
            resolve in it.
    """
    try:
        obj: Any = import_module(module_name)
        for part in qual.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise SpipeFormatError(
            f"{path}: {target!r} does not resolve in this process ({type(exc).__name__}: {exc})."
        ) from exc
    return obj


def _import_qualname(target: str, ctx: DecodeContext, path: str) -> type:
    trusted = target.startswith(TRUSTED_DC_PREFIX) or target in _trusted_dc_qualnames()
    if not trusted and not ctx.allow_code:
        raise SpipeCodeRefError(
            f"{path}: decoding {target!r} requires importing code outside the steerability "
            "namespace; pass allow_code=True to load() to permit it."
        )
    module_name, _, qual = target.rpartition(".")
    return _resolve_target(module_name, qual, target, path)


def _decodable_without_code(cls: type) -> bool:
    """Return whether `$dc` may decode `cls` without `allow_code`.

    The check uses the qualname of the resolved class, which includes the module that defines the
    class. A class that a toolkit module imports from another package therefore does not qualify
    under the toolkit module's name. An enum qualifies when it is defined in the `steerability.`
    namespace or is in `_trusted_dc_qualnames()`, since decoding an enum only looks up a member by
    name. A dataclass qualifies only when it is in `DECODABLE_DATACLASSES`, since decoding a
    dataclass calls its constructor with the manifest's field values.

    Args:
        cls: The resolved `$dc` target, a dataclass or an enum.

    Returns:
        True when `cls` qualifies.
    """
    qualname = _qualname(cls)
    if issubclass(cls, Enum):
        return qualname.startswith(TRUSTED_DC_PREFIX) or qualname in _trusted_dc_qualnames()
    return qualname in DECODABLE_DATACLASSES


def _available_artifact_id(ref: Any, ctx: DecodeContext, path: str) -> str:
    """Return the artifact id of an `$artifact` reference, after checking that `ctx` has a store.

    The check does not look up the artifact in the store.

    Args:
        ref: The encoded `$artifact` reference.
        ctx: The decoding context.
        path: The position of `ref` in the manifest, used in error messages.

    Returns:
        The artifact id.

    Raises:
        SpipeFormatError: If `ref` is not an object with an `$artifact` key, or if the id is
            malformed.
        SpipeIntegrityError: If `ctx` has no artifact store.
    """
    from steerability.spipe.errors import SpipeIntegrityError

    if not isinstance(ref, Mapping) or "$artifact" not in ref:
        raise SpipeFormatError(f"{path}: expected an '$artifact' reference.")
    artifact_id = check_artifact_id(ref["$artifact"], path)
    if ctx.store is None:
        raise SpipeIntegrityError(
            f"{path}: artifact {artifact_id} is unavailable; this is a thin bundle, pass "
            "artifact_store= to load()."
        )
    return artifact_id


def _decode_artifact(ref: Mapping, ctx: DecodeContext, path: str) -> Any:
    artifact_id = _available_artifact_id(ref, ctx, path)
    record = ctx.store.record_for(artifact_id)
    # the manifest record, when present, decides the verification wrap and must agree with the
    # sidecar on the encoding and type; the sidecar keeps the reconstruction metadata
    manifest_record = ctx.manifest_records.get(artifact_id)
    if manifest_record is None:
        manifest_record = record
    else:
        check_sidecar(record, manifest_record)

    if record.type in ("CPOMemory", "PoolMemory") and not ctx.allow_code:
        raise SpipeCodeRefError(
            f"{path}: {record.type} payloads contain pickled data, which executes code "
            "when loaded; pass allow_code=True to load() to permit it."
        )

    if ref.get("as") == "path" and record.encoding != "tree":
        raise SpipeFormatError(f"{path}: 'as: path' applies to tree artifacts only.")

    if record.encoding == "tensors":
        tensors = ctx.store.load_tensors(artifact_id)
        if record.type == "Tensor":
            return tensors["value"]
        if record.type == "SteeringVector":
            vector = _rebuild_steering_vector(tensors, record, ctx)
            if manifest_record.artifact_class in ("direction", "calibrated"):
                from steerability.algorithms.state_control.common.sources import VerifiedPrecomputed

                return VerifiedPrecomputed(
                    vector,
                    provenance=manifest_record.provenance,
                    artifact_class=manifest_record.artifact_class,
                    policy=ctx.verify,
                )
            return vector
        raise SpipeFormatError(f"{path}: unknown tensors artifact type {record.type!r}.")

    payload = ctx.store.payload_path(artifact_id)
    if not ctx.allow_code:
        pickled = pickle_bearing_files(payload)
        if pickled:
            listed = ", ".join(pickled[:3]) + (f", and {len(pickled) - 3} more" if len(pickled) > 3 else "")
            raise SpipeCodeRefError(
                f"{path}: artifact {artifact_id} contains pickled data ({listed}), which executes "
                "code when loaded; pass allow_code=True to load() to permit it."
            )
    if ref.get("as") == "path":
        return str(payload)
    if record.type == "Probe":
        from steerability.algorithms.core.internals.probes.probe import Probe

        return Probe.load(payload)
    if record.type == "ProbeSet":
        from steerability.algorithms.core.internals.probes.probe import Probe
        from steerability.algorithms.core.internals.probes.probe_set import ProbeSet

        names = record.type_meta.get("names") or sorted(p.name for p in payload.iterdir() if p.is_dir())
        return ProbeSet({name: Probe.load(payload / name) for name in names})
    if record.type == "TextMemory":
        from steerability.algorithms.input_control.common.memory.text import TextMemory

        return TextMemory.load(payload / "memory.json")
    if record.type == "PoolMemory":
        from steerability.algorithms.input_control.common.memory.pool import PoolMemory

        return PoolMemory.load(payload / "memory.pkl")
    if record.type in ("LoRAArtifact", "CheckpointArtifact", "CPOMemory"):
        return str(payload)
    raise SpipeFormatError(f"{path}: unknown tree artifact type {record.type!r}.")


def _rebuild_steering_vector(tensors: Mapping[str, torch.Tensor], record: ArtifactRecord, ctx: DecodeContext):
    from steerability.algorithms.state_control.common.steering_vector import SteeringVector

    meta = record.type_meta
    explained = meta.get("explained_variances")
    accuracies = meta.get("probe_accuracies")
    probe_accuracies = None
    if accuracies:
        probe_accuracies = {}
        for key, acc in accuracies.items():
            layer_str, head_str = key.split(":")
            probe_accuracies[(int(layer_str), int(head_str))] = float(acc)
    return SteeringVector(
        model_type=meta.get("model_type", "unknown"),
        directions={int(name): tensor for name, tensor in tensors.items()},
        num_heads=meta.get("num_heads"),
        head_dim=meta.get("head_dim"),
        explained_variances={int(k): float(v) for k, v in explained.items()} if explained else None,
        probe_accuracies=probe_accuracies,
        meta=dict(meta.get("meta") or {}),
    )


def decode(value: Any, ctx: DecodeContext, path: str = "$") -> Any:
    """Decode one manifest JSON value to its constructor form.

    JSON scalars are returned unchanged. Lists and objects without a tag are decoded item by
    item.

    Args:
        value: The encoded value.
        ctx: The decoding context, which gives the artifact store and the decoding policies.
        path: The position of `value` in the manifest, used in error messages.

    Returns:
        The decoded value.

    Raises:
        SpipeFormatError: If a value is malformed or specifies an unknown component kind or
            artifact type, if a `$dc` or `$ref` target cannot be imported in this process, if a
            `$dc` target is neither a dataclass nor an enum, if the decoded values do not
            construct the target class or component, or if an artifact entry in the store is a
            symlink.
        SpipeCodeRefError: If decoding requires code and `ctx.allow_code` is False, e.g., for a
            `$ref` callable under `code_mode="strict"`, a `$dc` dataclass outside
            `DECODABLE_DATACLASSES`, or an artifact payload that contains pickled data.
        SpipeIntegrityError: If a referenced artifact is unavailable or fails verification, if
            its store sidecar differs from its manifest record on the encoding or type, or if a
            loaded `"path"` data file does not match its recorded digest.

    Warns:
        UserWarning: If `ctx.verify` is `"warn"` and a gate readout artifact records the
            fingerprint of the model that produced it. The rebuilt gate does not check that
            fingerprint under any policy other than `"strict"`.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [decode(item, ctx, f"{path}[{i}]") for i, item in enumerate(value)]
    if not isinstance(value, Mapping):
        raise SpipeFormatError(f"{path}: unexpected value of type {type(value).__name__}.")

    if "$artifact" in value:
        return _decode_artifact(value, ctx, path)
    if "$map" in value:
        entries = value["$map"]
        if not isinstance(entries, list) or not all(
            isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], (str, int, float, bool))
            for entry in entries
        ):
            raise SpipeFormatError(f"{path}: '$map' must be a list of [key, value] pairs with scalar keys.")
        return {key: decode(item, ctx, f"{path}[{key!r}]") for key, item in entries}
    if "$dc" in value:
        target = value["$dc"]
        if not isinstance(target, str):
            raise SpipeFormatError(f"{path}: '$dc' target must be a string; got {type(target).__name__}.")
        if target == "torch.dtype":
            name = value.get("value")
            dtype = getattr(torch, name, None) if isinstance(name, str) else None
            if not isinstance(dtype, torch.dtype):
                raise SpipeFormatError(f"{path}: {name!r} does not name a torch dtype.")
            return dtype
        cls = _import_qualname(target, ctx, path)
        if not isinstance(cls, type) or not (is_dataclass(cls) or issubclass(cls, Enum)):
            raise SpipeFormatError(
                f"{path}: '$dc' target {target!r} is neither a dataclass nor an enum; '$dc' "
                "decodes only those."
            )
        if not ctx.allow_code and not _decodable_without_code(cls):
            raise SpipeCodeRefError(
                f"{path}: '$dc' target {target!r} resolves to {_qualname(cls)!r}, which is not a "
                "data-only toolkit class; pass allow_code=True to load() to construct it."
            )
        if issubclass(cls, Enum):
            member = value.get("value")
            if not isinstance(member, str) or member not in cls.__members__:
                raise SpipeFormatError(f"{path}: {member!r} is not a member of {target!r}.")
            return cls[member]
        encoded_fields = value.get("fields", {})
        if not isinstance(encoded_fields, Mapping):
            raise SpipeFormatError(f"{path}: '$dc' fields must be an object.")
        decoded_fields = {
            name: decode(item, ctx, f"{path}.{name}")
            for name, item in encoded_fields.items()
        }
        try:
            return cls(**decoded_fields)
        except (AttributeError, LookupError, TypeError, ValueError) as exc:
            raise SpipeFormatError(
                f"{path}: the fields do not construct {target!r} ({type(exc).__name__}: {exc})."
            ) from exc
    if "$component" in value:
        return _decode_component(value, ctx, path)
    if "$ref" in value:
        target = value["$ref"]
        if not isinstance(target, str):
            raise SpipeFormatError(f"{path}: '$ref' target must be a string; got {type(target).__name__}.")
        if ctx.code_mode == "sentinel":
            return CodeRef(target)
        if ctx.allow_code:
            module_name, _, qual = target.partition(":")
            return _resolve_target(module_name, qual, target, path)
        raise SpipeCodeRefError(
            f"{path}: this spipe references code ({target!r}); pass allow_code=True to load() "
            "to import it."
        )
    if "$data" in value:
        payload = value["$data"]
        if not isinstance(payload, Mapping) or not all(
            isinstance(payload.get(data_field.name), (str, type(None))) for data_field in fields(DataRef)
        ):
            raise SpipeFormatError(f"{path}: '$data' must be an object whose values are strings.")
        ref = DataRef(
            kind=payload.get("kind", "opaque"),
            repo_id=payload.get("repo_id"),
            revision=payload.get("revision"),
            split=payload.get("split"),
            subset=payload.get("subset"),
            path=payload.get("path"),
            sha256=payload.get("sha256"),
            type=payload.get("type"),
            fingerprint=payload.get("fingerprint"),
        )
        if ctx.data_mode == "load" and ref.kind in ("hf", "path"):
            return ref.load()
        return ref

    return {key: decode(item, ctx, f"{path}[{key!r}]") for key, item in value.items()}


# component encoding


def _encode_component(value: Any, ctx: EncodeContext, path: str) -> dict | None:
    """Encode a state-control component or routing object, or return None when `value` is not
    a component."""
    from steerability.algorithms.state_control.common.gating import Gate
    from steerability.algorithms.state_control.common.selectors.base import BaseSelector
    from steerability.algorithms.state_control.common.transforms.base import BaseTransform

    if isinstance(value, BaseTransform):
        return _encode_transform(value, ctx, path)
    if isinstance(value, Gate):
        return _encode_gate(value, ctx, path)
    if isinstance(value, BaseSelector):
        return _encode_selector(value, ctx, path)

    from steerability.algorithms.output_control.routed_decoding.routing import Predicate, Route, Router

    if isinstance(value, Router):
        return {
            "$component": "router",
            "params": {
                "routes": [encode(route, ctx, f"{path}.routes[{i}]") for i, route in enumerate(value.routes)],
                "default_action": encode(value.default_action, ctx, f"{path}.default_action"),
            },
        }
    if isinstance(value, Route):
        return {
            "$component": "route",
            "params": {
                "name": value.name,
                "when": encode(value.when, ctx, f"{path}.when"),
                "action": encode(value.action, ctx, f"{path}.action"),
            },
        }
    if isinstance(value, Predicate):
        return _encode_predicate(value, ctx, path)
    return None


def _encode_transform(transform, ctx: EncodeContext, path: str) -> dict:
    to_config = getattr(type(transform), "to_config", None)
    kind = type(transform).wire_kind
    if to_config is None or kind is None:
        raise SpipeSaveError(
            f"{path}: transform {type(transform).__name__} declares no serialized form."
        )
    params, artifact, inner = transform.to_config()
    encoded: dict[str, Any] = {"$component": kind, "params": encode(params, ctx, f"{path}.params")}
    if artifact is not None:
        encoded["artifact"] = encode(artifact, ctx, f"{path}.artifact")
    if inner is not None:
        encoded["inner"] = _encode_transform(inner, ctx, f"{path}.inner")
    return encoded


def _encode_gate(gate, ctx: EncodeContext, path: str) -> dict:
    kind = "gate"
    params, readout_tensors = gate.to_config()
    encoded: dict[str, Any] = {"$component": kind, "params": encode(params, ctx, f"{path}.params")}
    if readout_tensors is not None:
        record_fields = _record_fields(ctx, "GateReadout")
        if not ctx.artifact_fields:
            record_fields["artifact_class"] = "calibrated"
        if ctx.digest_mode:
            artifact_id, _ = tensors_payload(readout_tensors)
            record = ArtifactRecord(id=artifact_id, encoding="tensors", **record_fields)
        else:
            record = ctx.store.put_tensors(readout_tensors, record_fields)
        encoded["artifact"] = _artifact_ref(record, ctx)
    return encoded


def _encode_selector(selector, ctx: EncodeContext, path: str) -> dict:
    kind = getattr(type(selector), "component_kind", None)
    to_config = getattr(type(selector), "to_config", None)
    if to_config is None or kind is None:
        raise SpipeSaveError(
            f"{path}: selector {type(selector).__name__} declares no serialized form."
        )
    return {"$component": kind, "params": encode(selector.to_config(), ctx, f"{path}.params")}


def _encode_predicate(predicate, ctx: EncodeContext, path: str) -> dict:
    from steerability.algorithms.output_control.routed_decoding.routing import _And, _Decision, _Not, _Or

    if isinstance(predicate, _Decision):
        return {"$component": "decision", "params": {"name": predicate.name}}
    if isinstance(predicate, _And):
        return {"$component": "and", "params": {
            "left": _encode_predicate(predicate.left, ctx, f"{path}.left"),
            "right": _encode_predicate(predicate.right, ctx, f"{path}.right"),
        }}
    if isinstance(predicate, _Or):
        return {"$component": "or", "params": {
            "left": _encode_predicate(predicate.left, ctx, f"{path}.left"),
            "right": _encode_predicate(predicate.right, ctx, f"{path}.right"),
        }}
    if isinstance(predicate, _Not):
        return {"$component": "not", "params": {
            "operand": _encode_predicate(predicate.operand, ctx, f"{path}.operand"),
        }}
    raise SpipeSaveError(f"{path}: predicate {type(predicate).__name__} cannot be serialized.")


def _component_kinds() -> dict[str, type]:
    """The `$component` kind table, built lazily from the participating classes."""
    from steerability.algorithms.state_control.common.gating import Gate
    from steerability.algorithms.state_control.common.selectors import (
        FixedLayerSelector,
        FractionalDepthSelector,
        LateThirdSelector,
        TopKHeadSelector,
    )
    from steerability.algorithms.state_control.common.transforms import (
        AdditiveTransform,
        AlignmentAdaptiveTransform,
        HeadAdditiveTransform,
        NormPreservingTransform,
        ProjectionTransform,
        RotationTransform,
    )

    return {
        "additive": AdditiveTransform,
        "projection": ProjectionTransform,
        "rotation": RotationTransform,
        "head_additive": HeadAdditiveTransform,
        "norm_preserving": NormPreservingTransform,
        "alignment_adaptive": AlignmentAdaptiveTransform,
        "gate": Gate,
        "fixed_layer": FixedLayerSelector,
        "fractional_depth": FractionalDepthSelector,
        "late_third": LateThirdSelector,
        "top_k_head": TopKHeadSelector,
    }


COMPONENT_KINDS = _component_kinds


def _decode_component(value: Mapping, ctx: DecodeContext, path: str) -> Any:
    """Decode a `$component` value to a state-control component or a routing object.

    Args:
        value: The encoded `$component` object.
        ctx: The decoding context.
        path: The position of `value` in the manifest, used in error messages.

    Returns:
        The rebuilt component.

    Raises:
        SpipeFormatError: If the kind is not a string, if `params` is not an object, if the kind
            is unknown, or if the params do not rebuild the component.
        SpipeCodeRefError: If a nested value requires code and `ctx.allow_code` is False.
        SpipeIntegrityError: If an artifact the component references is unavailable or fails
            verification.

    Warns:
        UserWarning: If `ctx.verify` is `"warn"` and a gate readout artifact records the
            fingerprint of the model that produced it.
    """
    kind = value["$component"]
    params_raw = value.get("params", {})
    if not isinstance(kind, str) or not isinstance(params_raw, Mapping):
        raise SpipeFormatError(f"{path}: '$component' must be a kind string with an object of params.")
    try:
        return _rebuild_component(kind, params_raw, value, ctx, path)
    except (AttributeError, LookupError, TypeError, ValueError) as exc:
        raise SpipeFormatError(
            f"{path}: the params do not rebuild a {kind!r} component ({type(exc).__name__}: {exc})."
        ) from exc


def _rebuild_component(kind: str, params_raw: Mapping, value: Mapping, ctx: DecodeContext, path: str) -> Any:
    if kind == "router":
        from steerability.algorithms.output_control.routed_decoding.routing import Router

        routes = [decode(route, ctx, f"{path}.routes[{i}]") for i, route in enumerate(params_raw.get("routes", []))]
        return Router(routes, default_action=decode(params_raw.get("default_action"), ctx, f"{path}.default_action"))
    if kind == "route":
        from steerability.algorithms.output_control.routed_decoding.routing import Route

        return Route(
            name=params_raw["name"],
            when=decode(params_raw["when"], ctx, f"{path}.when"),
            action=decode(params_raw.get("action"), ctx, f"{path}.action"),
        )
    if kind == "decision":
        from steerability.algorithms.output_control.routed_decoding.routing import P

        return P(params_raw["name"])
    if kind in ("and", "or", "not"):
        from steerability.algorithms.output_control.routed_decoding.routing import Predicate

        names = ("operand",) if kind == "not" else ("left", "right")
        operands = [decode(params_raw[name], ctx, f"{path}.{name}") for name in names]
        if not all(isinstance(operand, Predicate) for operand in operands):
            raise SpipeFormatError(f"{path}: the operands of a {kind!r} component must be predicates.")
        if kind == "not":
            return ~operands[0]
        return (operands[0] & operands[1]) if kind == "and" else (operands[0] | operands[1])

    table = _component_kinds()
    cls = table.get(kind)
    if cls is None:
        raise SpipeFormatError(
            f"{path}: unknown component kind {kind!r}; known kinds are {sorted(table)}."
        )

    if kind == "gate":
        params = decode(params_raw, ctx, f"{path}.params")
        readout_tensors = None
        if "artifact" in value:
            artifact_id = _available_artifact_id(value["artifact"], ctx, f"{path}.artifact")
            readout_tensors = ctx.store.load_tensors(artifact_id)
            record = ctx.store.record_for(artifact_id)
            params = dict(params)
            readout_params = dict(params.get("readout", {}))
            if ctx.verify == "strict":
                readout_params["model_fingerprint"] = (
                    readout_params.get("model_fingerprint") or record.provenance.get("model_fingerprint")
                )
            else:
                readout_params["model_fingerprint"] = None
                recorded = record.provenance.get("model_fingerprint")
                if ctx.verify == "warn" and recorded:
                    import warnings

                    warnings.warn(
                        "Gate readout fingerprint checks are disarmed under verify='warn'; "
                        f"the recorded producing fingerprint is {recorded!r}.",
                        UserWarning,
                    )
            params["readout"] = readout_params
        return cls.from_config(params, readout_tensors=readout_tensors)

    from steerability.algorithms.state_control.common.selectors.base import BaseSelector

    if isinstance(cls, type) and issubclass(cls, BaseSelector):
        params = decode(params_raw, ctx, f"{path}.params")
        return cls.from_config(params)

    # transform kinds; a direction/calibrated artifact decodes to a VerifiedPrecomputed
    # source, making the rebuilt transform bind (and verify) at steer
    params = decode(params_raw, ctx, f"{path}.params")
    artifact = None
    if "artifact" in value:
        artifact = decode(value["artifact"], ctx, f"{path}.artifact")
    inner = None
    if "inner" in value:
        inner = _decode_component(value["inner"], ctx, f"{path}.inner")
    return cls.from_config(params, artifact=artifact, inner=inner)
