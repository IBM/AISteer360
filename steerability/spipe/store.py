"""Content-addressed artifact store backing frozen `.spipe` bundles.

The store is a directory of artifact directories, each named by its content id. Two encodings
exist: `"tensors"` (a single `artifact.safetensors`, id and bytes produced by `artifact_id_for`
from `state_control/common/lowering.py`, byte-compatible with the vLLM-Hook plugin registry)
and `"tree"` (a directory copied verbatim under `payload/`, id a SHA-256 over the sorted
relative paths and per-file digests). Every artifact directory carries an `artifact.json`
sidecar duplicating its manifest record plus type-specific reconstruction metadata, which
keeps a detached artifact directory self-describing. The sidecar is the commit marker, written
last and atomically (to a temporary file that then replaces `artifact.json`). An artifact
directory without a sidecar is an interrupted write, and the next write of the same artifact
replaces it. Writes are idempotent and reads verify the content hash.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pickle
import pickletools
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from steerability.spipe.errors import SpipeFormatError, SpipeIntegrityError, SpipeSaveError

logger = logging.getLogger(__name__)

TENSOR_FILE = "artifact.safetensors"
SIDECAR_FILE = "artifact.json"
PAYLOAD_DIR = "payload"

PICKLE_SUFFIXES = frozenset({".pkl", ".pk", ".pickle", ".dill", ".joblib", ".pt", ".pth", ".ckpt"})

ARTIFACT_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

# leading bytes of a `.bin` file parsed as a pickle stream, and the opcode count that marks one
_PICKLE_SCAN_BYTES = 64 * 1024
_PICKLE_MIN_OPCODES = 8

# opcodes through which unpickling imports a callable
_PICKLE_IMPORT_OPCODES = frozenset({"GLOBAL", "INST", "STACK_GLOBAL", "EXT1", "EXT2", "EXT4"})


def is_artifact_id(value: Any) -> bool:
    """Return whether `value` is a well-formed artifact id (`sha256:` followed by 64 lowercase hex digits)."""
    return isinstance(value, str) and ARTIFACT_ID_PATTERN.fullmatch(value) is not None


def check_artifact_id(value: Any, where: str = "artifact id") -> str:
    """Return `value` after checking that it is a well-formed artifact id.

    Args:
        value: The candidate id.
        where: The position of the id, used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        SpipeFormatError: If `value` is not a string of `sha256:` followed by 64 lowercase hex
            digits.
    """
    if not is_artifact_id(value):
        raise SpipeFormatError(f"{where}: {value!r} is not an artifact id ('sha256:' and 64 lowercase hex digits).")
    return value


def _dir_name(artifact_id: str) -> str:
    """Directory name for an artifact id (`sha256:<hex>` becomes `sha256-<hex>`)."""
    return artifact_id.replace(":", "-", 1)


def _artifact_id_from_dir_name(name: str) -> str:
    """Artifact id for a store directory name."""
    return name.replace("-", ":", 1)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_id_for(root: Path) -> str:
    """The content id of a directory tree.

    The id is `"sha256:" + sha256` over the concatenation of
    `f"{posix_relpath}\\n{sha256_of_file_hex}\\n"` for every file, sorted by posix relative
    path. Symlinks are rejected.

    Args:
        root: Directory to hash.

    Returns:
        The `sha256:<hex>` id.

    Raises:
        SpipeSaveError: If `root` is not a directory, is empty, or contains a symlink.
    """
    root = Path(root)
    if not root.is_dir():
        raise SpipeSaveError(f"Tree artifact source {root} is not a directory.")
    entries: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SpipeSaveError(f"Tree artifact source contains a symlink: {path}.")
        if path.is_file():
            entries.append((path.relative_to(root).as_posix(), _file_sha256(path)))
    if not entries:
        raise SpipeSaveError(f"Tree artifact source {root} contains no files.")
    digest = hashlib.sha256()
    for relpath, file_hash in entries:
        digest.update(f"{relpath}\n{file_hash}\n".encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _is_pickle_or_zip(path: Path) -> bool:
    """Return whether a file is a zip archive or starts with a pickle stream.

    A file that starts with the zip signature (the format `torch.save` writes) is a zip archive.
    For any other file, the first 64 KB are parsed with `pickletools.genops`, and the file counts
    as a pickle stream when one of the following is true:

    - the first opcode is `PROTO` with a supported protocol number
    - the parse reaches `STOP` after at least one other opcode
    - the parse reaches an opcode that imports a callable (`GLOBAL`, `INST`, `STACK_GLOBAL`, or an
      `EXT` opcode)
    - the parse yields 8 opcodes
    - the file is longer than 64 KB, and the parse fails at a known opcode or at the end of the
      scanned bytes (the rest of the stream is not inspected)

    Data that is not a pickle rarely meets these conditions, since the parse stops at the first
    byte that is not an opcode.

    Args:
        path: The file to check.

    Returns:
        True when the file is a zip archive or starts with a pickle stream.
    """
    size = path.stat().st_size
    with open(path, "rb") as handle:
        prefix = handle.read(_PICKLE_SCAN_BYTES)
    if prefix.startswith(b"PK\x03\x04"):
        return True
    stream = io.BytesIO(prefix)
    count = 0
    end = 0  # offset just past the last opcode that parsed
    try:
        for opcode, argument, _ in pickletools.genops(stream):
            name = opcode.name
            if count == 0 and name == "PROTO" and argument <= pickle.HIGHEST_PROTOCOL:
                return True
            if name == "STOP":
                return count > 0
            if name in _PICKLE_IMPORT_OPCODES:
                return True
            count += 1
            if count >= _PICKLE_MIN_OPCODES:
                return True
            end = stream.tell()
    except ValueError:
        if size > len(prefix):
            next_code = prefix[end:end + 1].decode("latin-1")
            return next_code == "" or next_code in pickletools.code2op
    return False


def pickle_bearing_files(root: str | Path) -> list[str]:
    """Return the files under a directory that contain pickled data.

    A file counts when any of its suffixes is in `PICKLE_SUFFIXES`. A `.bin` file counts when it
    is a zip archive (the format `torch.save` writes) or when its first 64 KB parse as the start
    of a pickle stream of any protocol. Files with other suffixes, e.g., `.safetensors` or
    `.json`, are not inspected and do not count.

    Args:
        root: The directory to scan, searched recursively.

    Returns:
        The posix paths of the matching files relative to `root`, sorted.
    """
    root = Path(root)
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if suffixes & PICKLE_SUFFIXES or (path.suffix.lower() == ".bin" and _is_pickle_or_zip(path)):
            found.append(path.relative_to(root).as_posix())
    return found


def tensors_payload(tensors: Mapping[str, torch.Tensor]) -> tuple[str, bytes]:
    """The content id and serialized safetensors bytes of a tensor payload.

    Uses the id algorithm of `artifact_id_for` (float32, contiguous, CPU, sorted names,
    SHA-256 over the safetensors serialization), which makes tensor artifact directories
    byte-compatible with the vLLM-Hook plugin registry.

    Args:
        tensors: Mapping from tensor name to tensor.

    Returns:
        The `sha256:<hex>` id and the serialized bytes.
    """
    import safetensors.torch

    from steerability.algorithms.state_control.common.lowering import artifact_id_for

    artifact_id, prepared = artifact_id_for(tensors)
    data = safetensors.torch.save({name: prepared[name] for name in sorted(prepared)})
    return artifact_id, data


@dataclass
class ArtifactRecord:
    """One artifact's manifest record.

    Attributes:
        id: Content-addressed artifact id (`sha256:<hex>`).
        encoding: `"tensors"` or `"tree"`.
        type: Name of the producing class (e.g. `"SteeringVector"`, `"Probe"`).
        artifact_class: `"direction"`, `"calibrated"`, or `"opaque"`.
        source: Fit-source class name, or None for recipe-supplied artifacts.
        fit_digest: 12-hex-character digest of the producing fit identity, or None.
        provenance: Producing-side identity (`backend_spec_hash`, `model_fingerprint`,
            `model_ref`, `tokenizer_fingerprint`), each possibly None.
        type_meta: Type-specific reconstruction metadata (sidecar only, excluded from the
            manifest record).
    """

    id: str
    encoding: str
    type: str
    artifact_class: str = "opaque"
    source: str | None = None
    fit_digest: str | None = None
    provenance: dict = field(default_factory=dict)
    type_meta: dict = field(default_factory=dict)

    def manifest_entry(self) -> dict:
        """The record as it appears in `spipe.json` (without `type_meta`)."""
        return {
            "id": self.id,
            "encoding": self.encoding,
            "type": self.type,
            "artifact_class": self.artifact_class,
            "source": self.source,
            "fit_digest": self.fit_digest,
            "provenance": {
                "backend_spec_hash": self.provenance.get("backend_spec_hash"),
                "model_fingerprint": self.provenance.get("model_fingerprint"),
                "model_ref": self.provenance.get("model_ref"),
                "tokenizer_fingerprint": self.provenance.get("tokenizer_fingerprint"),
            },
        }

    def sidecar_entry(self) -> dict:
        """The record as written to the artifact's `artifact.json` sidecar."""
        return {**self.manifest_entry(), "type_meta": self.type_meta}

    @classmethod
    def from_mapping(cls, data: Mapping) -> "ArtifactRecord":
        return cls(
            id=data["id"],
            encoding=data["encoding"],
            type=data["type"],
            artifact_class=data.get("artifact_class", "opaque"),
            source=data.get("source"),
            fit_digest=data.get("fit_digest"),
            provenance=dict(data.get("provenance") or {}),
            type_meta=dict(data.get("type_meta") or {}),
        )


def check_sidecar(record: ArtifactRecord, expected: ArtifactRecord) -> None:
    """Check that an artifact's store sidecar matches its manifest record on the encoding and type.

    The sidecar determines how the payload is reconstructed, and `ArtifactStore.verify` checks
    only the payload bytes.

    Args:
        record: The record read from the store sidecar.
        expected: The artifact's record in the manifest.

    Raises:
        SpipeIntegrityError: If the sidecar's encoding or type differs from the manifest record.
    """
    if (record.encoding, record.type) != (expected.encoding, expected.type):
        raise SpipeIntegrityError(
            f"Artifact {record.id} has a store sidecar recording encoding {record.encoding!r} and "
            f"type {record.type!r}, but the manifest records encoding {expected.encoding!r} and "
            f"type {expected.type!r}."
        )


class ArtifactStore:
    """A directory of content-addressed artifacts, each in a subdirectory named after its id.

    Artifacts are written to `root`, and a write does nothing when `root` already contains the
    artifact. Reads look in `root` first. An artifact that is not in `root` is read from the
    directory that `resolver` returns for it. Loading a payload verifies its content hash first.

    Every method that takes an artifact id raises `SpipeFormatError` for a malformed id (see
    `check_artifact_id`) before it builds a path or calls the resolver. Reading an artifact from
    `root` raises `SpipeFormatError` when its directory, its sidecar, its tensor file, or its
    `payload/` directory is a symlink. Verification raises `SpipeFormatError` for a symlink
    anywhere inside a tree payload, including the payload of an artifact found through the
    resolver.

    Args:
        root: The store directory (`artifacts/` inside a spipe, or any external directory for
            thin exports). It is created on the first write.
        resolver: An optional callable that maps an artifact id to the directory containing
            that artifact, for artifacts stored outside `root` (e.g., the external store of a
            thin export). The directories it returns may be symlinks, but their tree payloads
            may not contain symlinks.

    Attributes:
        root: The store directory.
    """

    def __init__(self, root: str | Path, resolver: Callable[[str], Path] | None = None):
        self.root = Path(root)
        self._resolver = resolver

    def _dir_for(self, artifact_id: str) -> Path:
        local = self.root / _dir_name(check_artifact_id(artifact_id))
        if local.exists() or self._resolver is None:
            return local
        resolved = Path(self._resolver(artifact_id))
        return resolved if resolved.exists() else local

    def _read_dir(self, artifact_id: str) -> Path:
        """Return the directory of an artifact, after checking it for symlinks when it is in `root`.

        Args:
            artifact_id: The artifact id.

        Returns:
            The artifact's directory in `root`, or the directory the resolver returns when the
            artifact is not in `root` and that directory exists. The returned path may not
            exist.

        Raises:
            SpipeFormatError: If the id is malformed, or if the directory is in `root` and it,
                its sidecar, its tensor file, or its payload directory is a symlink.
        """
        directory = self._dir_for(artifact_id)
        if directory == self.root / _dir_name(artifact_id):
            for path in (directory, directory / SIDECAR_FILE, directory / TENSOR_FILE, directory / PAYLOAD_DIR):
                if path.is_symlink():
                    raise SpipeFormatError(f"Artifact {artifact_id}: {path} is a symlink; symlinks are rejected.")
        return directory

    def has(self, artifact_id: str) -> bool:
        """Whether the store holds `artifact_id`."""
        return (self._dir_for(artifact_id) / SIDECAR_FILE).exists()

    def ids(self) -> list[str]:
        """Sorted ids of every artifact under `root`."""
        if not self.root.is_dir():
            return []
        return sorted(
            _artifact_id_from_dir_name(child.name)
            for child in self.root.iterdir()
            if child.is_dir() and (child / SIDECAR_FILE).exists()
        )

    def _write_sidecar(self, directory: Path, record: ArtifactRecord) -> None:
        # write the commit marker to a temporary file and move it into place in one step
        temporary = directory / f"{SIDECAR_FILE}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record.sidecar_entry(), handle, sort_keys=True, indent=2)
        os.replace(temporary, directory / SIDECAR_FILE)

    def put_tensors(self, tensors: Mapping[str, torch.Tensor], record_fields: dict) -> ArtifactRecord:
        """Write a tensor payload as a `"tensors"` artifact, idempotently.

        Args:
            tensors: Mapping from tensor name to tensor.
            record_fields: Record fields other than `id` and `encoding` (`type`,
                `artifact_class`, `source`, `fit_digest`, `provenance`, `type_meta`).

        Returns:
            The artifact record.
        """
        artifact_id, data = tensors_payload(tensors)
        record = ArtifactRecord(id=artifact_id, encoding="tensors", **record_fields)
        directory = self.root / _dir_name(artifact_id)
        if not (directory / SIDECAR_FILE).exists():
            # a directory without a sidecar is an interrupted write
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / TENSOR_FILE).write_bytes(data)
            self._write_sidecar(directory, record)
        return record

    def put_tree(self, source: str | Path, record_fields: dict) -> ArtifactRecord:
        """Copy a directory as a `"tree"` artifact, idempotently.

        Args:
            source: Directory whose contents become `payload/`. Symlinks are rejected.
            record_fields: Record fields other than `id` and `encoding`.

        Returns:
            The artifact record.

        Raises:
            SpipeSaveError: If `source` is missing, empty, or contains a symlink.
        """
        source = Path(source)
        artifact_id = tree_id_for(source)
        record = ArtifactRecord(id=artifact_id, encoding="tree", **record_fields)
        directory = self.root / _dir_name(artifact_id)
        if not (directory / SIDECAR_FILE).exists():
            # a directory without a sidecar is an interrupted write
            shutil.rmtree(directory, ignore_errors=True)
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, directory / PAYLOAD_DIR, symlinks=False)
            self._write_sidecar(directory, record)
        return record

    def record_for(self, artifact_id: str) -> ArtifactRecord:
        """Return the record in a stored artifact's sidecar.

        Args:
            artifact_id: The artifact id.

        Returns:
            The `ArtifactRecord` read from the artifact's `artifact.json`, including
            `type_meta`.

        Raises:
            SpipeFormatError: If the id is malformed, or if the artifact is in `root` and its
                directory, sidecar, tensor file, or payload directory is a symlink.
            SpipeIntegrityError: If the store does not contain the artifact, if the sidecar is
                malformed, or if the sidecar's `id` differs from `artifact_id`.
        """
        directory = self._read_dir(artifact_id)
        sidecar = directory / SIDECAR_FILE
        if not sidecar.exists():
            raise SpipeIntegrityError(
                f"Artifact {artifact_id} is not in the store at {self.root}"
                + (" (no external resolver matched)" if self._resolver else
                   "; for a thin bundle, pass artifact_store= to load()")
                + "."
            )
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            record = ArtifactRecord.from_mapping(data)
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise SpipeIntegrityError(
                f"Artifact sidecar at {sidecar} is malformed ({type(exc).__name__}: {exc})."
            ) from exc
        if record.id != artifact_id:
            raise SpipeIntegrityError(
                f"Artifact sidecar at {directory} records id {record.id} but the directory "
                f"is named for {artifact_id}."
            )
        return record

    def verify(self, artifact_id: str) -> None:
        """Verify a stored artifact's bytes against its content id.

        Args:
            artifact_id: The artifact id.

        Raises:
            SpipeFormatError: If the id is malformed, if the artifact is in `root` and its
                directory, sidecar, tensor file, or payload directory is a symlink, or if a tree
                payload contains a symlink.
            SpipeIntegrityError: If the store does not contain the artifact, if the sidecar is
                malformed or records a different id, if a tensor file or tree payload is missing, if
                a tree payload is empty, or if the recomputed content hash differs from `artifact_id`.
        """
        record = self.record_for(artifact_id)
        directory = self._read_dir(artifact_id)
        if record.encoding == "tensors":
            data = self._tensor_file(artifact_id, directory).read_bytes()
            actual = "sha256:" + hashlib.sha256(data).hexdigest()
        else:
            payload = directory / PAYLOAD_DIR
            for path in payload.rglob("*"):
                if path.is_symlink():
                    raise SpipeFormatError(f"Artifact {artifact_id}: {path} is a symlink; symlinks are rejected.")
            try:
                actual = tree_id_for(payload)
            except SpipeSaveError as exc:
                raise SpipeIntegrityError(f"Artifact {artifact_id} failed integrity verification ({exc})") from exc
        if actual != artifact_id:
            raise SpipeIntegrityError(
                f"Artifact {artifact_id} failed integrity verification (content hashes to "
                f"{actual})."
            )

    @staticmethod
    def _tensor_file(artifact_id: str, directory: Path) -> Path:
        """Return the tensor file of a `"tensors"` artifact.

        Args:
            artifact_id: The artifact id, used in the error message.
            directory: The artifact's directory.

        Returns:
            The path of the artifact's `artifact.safetensors` file.

        Raises:
            SpipeIntegrityError: If the file does not exist.
        """
        path = directory / TENSOR_FILE
        if not path.is_file():
            raise SpipeIntegrityError(f"Artifact {artifact_id} has no tensor file at {path}.")
        return path

    def load_tensors(self, artifact_id: str) -> dict[str, torch.Tensor]:
        """Load and verify a `"tensors"` artifact.

        Returns:
            Mapping from tensor name to float32 CPU tensor.
        """
        import safetensors.torch

        self.verify(artifact_id)
        directory = self._read_dir(artifact_id)
        return safetensors.torch.load_file(str(directory / TENSOR_FILE))

    def payload_path(self, artifact_id: str) -> Path:
        """The verified `payload/` path of a `"tree"` artifact."""
        self.verify(artifact_id)
        return self._read_dir(artifact_id) / PAYLOAD_DIR

    def size_of(self, artifact_id: str) -> int:
        """Total on-disk bytes of an artifact's content (sidecar excluded)."""
        directory = self._read_dir(artifact_id)
        record = self.record_for(artifact_id)
        if record.encoding == "tensors":
            return self._tensor_file(artifact_id, directory).stat().st_size
        return sum(p.stat().st_size for p in (directory / PAYLOAD_DIR).rglob("*") if p.is_file())

    def copy_into(self, dest_root: str | Path, artifact_ids: list[str]) -> None:
        """Copy the named artifacts into another store directory (fat export)."""
        dest_root = Path(dest_root)
        for artifact_id in artifact_ids:
            source = self._read_dir(artifact_id)
            dest = dest_root / _dir_name(artifact_id)
            if not dest.exists():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, dest, symlinks=False, dirs_exist_ok=True)


def save_object_tree(save_fn: Callable[[Path], Any], store: ArtifactStore, record_fields: dict) -> ArtifactRecord:
    """Save an object into the store as a `"tree"` artifact via its own `save(path)` method.

    Args:
        save_fn: Callable writing the object into a directory (or file inside it).
        store: The destination store.
        record_fields: Record fields other than `id` and `encoding`.

    Returns:
        The artifact record.
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="spipe-artifact-") as tmp:
        save_fn(Path(tmp))
        return store.put_tree(tmp, record_fields)
