"""`SPipe`: a portable serialization of a `SteeringPipeline`.

One format holds both the recipe and the frozen resolution. A recipe-only spipe carries the
model reference and the controls as constructed; loading it and calling `steer()` re-runs
fits. A frozen spipe additionally pins what the resolution produced (fingerprints, resolved
bindings, per-fit digests) in a lock section and stores the products content-addressed. The
loaded controls therefore steer cheaply and model-free. The frozen form is itself a valid
recipe, since every resolved entry is constructor-valid for its method and loading takes the
ordinary construction and `steer()` path.
"""
from __future__ import annotations

import copy
import logging
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from steerability.spipe.codec import DecodeContext, decode, digest_of
from steerability.spipe.errors import (
    SpipeCodeRefError,
    SpipeFormatError,
    SpipeIntegrityError,
    SpipeSaveError,
    SpipeStaleError,
)
from steerability.spipe.format import (
    ARTIFACTS_DIR,
    MANIFEST_NAME,
    pack_zip,
    read_manifest,
    unpack_zip,
    validate_manifest,
    write_manifest,
)
from steerability.spipe.store import ArtifactRecord, ArtifactStore, check_sidecar, is_artifact_id

if TYPE_CHECKING:
    from steerability.algorithms.core.base_control import BaseControl
    from steerability.algorithms.core.execution.spec import BackendSpec
    from steerability.algorithms.core.steering_pipeline import SteeringPipeline

logger = logging.getLogger(__name__)


@dataclass
class SpipeReport:
    """The result of a model-free `SPipe.verify()`.

    Attributes:
        ok: True when no errors were found.
        errors: Findings that make the bundle unusable as-is.
        warnings: Findings worth knowing that do not block loading.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> str:
        """A human-readable summary."""
        lines = [f"spipe verify: {'ok' if self.ok else 'FAILED'}"]
        lines.extend(f"  error: {message}" for message in self.errors)
        lines.extend(f"  warning: {message}" for message in self.warnings)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


@dataclass(frozen=True)
class SpipeEntry:
    """A summary of one control entry in a manifest, as returned by `SPipe.entries`.

    Attributes:
        index: The position of the entry in the manifest's `controls` list, which is the value
            `SPipe.instantiate_entry()` accepts as `index`.
        method: The recipe method key (`"<category>_control/<name>"`).
        enabled: Whether the entry is enabled.
        args: A deep copy of the entry's recipe args in their encoded (manifest JSON) form.
        resolved: Whether the entry has a resolved (frozen) section.
    """

    index: int
    method: str
    enabled: bool
    args: dict
    resolved: bool


def _check_policies(prefer: str, verify: str) -> None:
    if prefer not in ("frozen", "recipe"):
        raise ValueError(f"prefer must be 'frozen' or 'recipe'; got {prefer!r}.")
    if verify not in ("strict", "warn", "off"):
        raise ValueError(f"verify must be 'strict', 'warn', or 'off'; got {verify!r}.")


def _resolved_items(entry: Mapping) -> list[Mapping]:
    resolved = entry.get("resolved")
    if resolved is None:
        return []
    return list(resolved) if isinstance(resolved, list) else [resolved]


def _collect_artifact_ids(value: Any, found: set[str]) -> None:
    """Add the well-formed artifact ids referenced in an encoded value to `found`.

    The value is searched recursively for `$artifact` and `id` keys, which cover `$artifact`
    references and artifact records. A malformed id is not collected. Decoding the entry that
    contains a malformed `$artifact` id raises `SpipeFormatError`.

    Args:
        value: The encoded value to search.
        found: The set that receives the ids, modified in place.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in ("$artifact", "id") and is_artifact_id(item):
                found.add(item)
            else:
                _collect_artifact_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _collect_artifact_ids(item, found)


def _manifest_records(manifest: Mapping) -> dict[str, ArtifactRecord]:
    """The artifact records of every resolved entry, keyed by artifact id."""
    records: dict[str, ArtifactRecord] = {}
    for entry in manifest["controls"]:
        for item in _resolved_items(entry):
            for record in (item.get("artifacts") or {}).values():
                records[record["id"]] = ArtifactRecord.from_mapping(record)
    return records


class SPipe:
    """A serialized steering pipeline, consisting of a manifest and an artifact store.

    The manifest contains the recipe (the model reference and each control's constructor args)
    and, for a frozen spipe, the resolved entries and a lock section. An `SPipe` is created by
    `SteeringPipeline.to_spipe()`, which keeps its artifacts in a temporary directory, or by
    `SPipe.load()`, which reads a saved bundle. `save()` writes the spipe to a `.spipe` file
    or a directory.

    The `pipeline()` and `instantiate_entry()` methods return controls built from the manifest.
    The staleness check and the `recipe_id` and `config_id` properties of a spipe without a
    lock section also construct controls from the recipe args, for internal use. These
    constructions decode each `$ref` callable to a `CodeRef` and leave `$data` references
    unloaded.
    """

    def __init__(
        self,
        manifest: dict,
        *,
        store: ArtifactStore | None,
        base_dir: Path | None,
        allow_code: bool,
        allow_stale: bool = False,
        _temp: Any = None,
    ):
        validate_manifest(manifest)
        self._manifest = manifest
        self._manifest_records = _manifest_records(manifest)
        self._store = store
        self._base_dir = base_dir
        self._allow_code = allow_code
        self._allow_stale = allow_stale
        self._temp = _temp
        self._stale_checked: set[int] = set()

    # inspection

    @property
    def manifest(self) -> dict:
        """A deep copy of the manifest."""
        return copy.deepcopy(self._manifest)

    @property
    def entries(self) -> tuple[SpipeEntry, ...]:
        """The manifest's control entries as `SpipeEntry` records, in pipeline order.

        The args are returned in encoded form without decoding. Listing the entries does not
        import code or read artifacts, and it works when a method key is not registered in this
        process.
        """
        return tuple(
            SpipeEntry(
                index=i,
                method=entry["method"],
                enabled=entry["enabled"],
                args=copy.deepcopy(entry["args"]),
                resolved=entry.get("resolved") is not None,
            )
            for i, entry in enumerate(self._manifest["controls"])
        )

    @property
    def recipe_id(self) -> str:
        """The recipe identity digest (model reference plus controls), 12 hex characters."""
        lock = self._manifest.get("lock")
        if lock is not None:
            return lock["recipe_id"]
        from steerability.algorithms.core.identity import config_digest

        return config_digest({
            "model": self._manifest["model"]["ref"],
            "controls": self._descriptor()["controls"],
        })

    @property
    def config_id(self) -> str:
        """The configuration identity digest over the recipe entries, 12 hex characters."""
        lock = self._manifest.get("lock")
        if lock is not None:
            return lock["config_id"]
        from steerability.algorithms.core.identity import config_digest

        return config_digest(self._descriptor())

    def _descriptor(self) -> dict:
        """The configuration descriptor recomputed from freshly instantiated recipe controls."""
        from steerability.algorithms.core.identity import config_descriptor_from_controls

        controls = [self._instantiate(entry["method"], entry["args"], self._decode_ctx(lenient=True), index=index)
                    for index, entry in enumerate(self._manifest["controls"])]
        for control, entry in zip(controls, self._manifest["controls"]):
            control.enabled = entry["enabled"]
        return config_descriptor_from_controls(controls)

    @property
    def code_dependent(self) -> bool:
        """Whether loading this spipe needs matching code and `allow_code=True`."""
        return bool(self._manifest["code_dependent"])

    @property
    def is_frozen(self) -> bool:
        """True iff every enabled entry with steer-time products carries a resolution.

        Entries with `resolved: null` have nothing to pin (their recipe is their frozen
        form), which makes a manifest with a lock section frozen.
        """
        return self._manifest.get("lock") is not None

    def describe(self) -> str:
        """A human-readable table of entries, frozen state, and artifact sizes."""
        lines = [
            f"spipe {self._manifest['format']}  model={self._manifest['model']['ref']}",
            f"  recipe_id={self.recipe_id}  config_id={self.config_id}  "
            f"frozen={self.is_frozen}  code_dependent={self.code_dependent}",
        ]
        for i, entry in enumerate(self._manifest["controls"]):
            items = _resolved_items(entry)
            frozen = "recipe"
            if items:
                methods = ", ".join(item["method"] for item in items)
                frozen = f"frozen -> {methods}"
            elif self.is_frozen and entry["enabled"]:
                frozen = "frozen (recipe is the frozen form)"
            enabled = "" if entry["enabled"] else "  [disabled]"
            lines.append(f"  [{i}] {entry['method']}{enabled}  ({frozen})")
            for item in items:
                for name, record in (item.get("artifacts") or {}).items():
                    size = ""
                    if self._store is not None and self._store.has(record["id"]):
                        try:
                            size = f"  {self._store.size_of(record['id'])} bytes"
                        except (SpipeFormatError, SpipeIntegrityError):
                            size = "  (unreadable; see verify())"
                    lines.append(f"        {name}: {record['type']} {record['id'][:19]}…{size}")
        return "\n".join(lines)

    # freeze state

    def thaw(self) -> "SPipe":
        """A recipe-only copy: every `resolved` section and the lock are dropped."""
        manifest = self.manifest
        for entry in manifest["controls"]:
            entry["resolved"] = None
        manifest["lock"] = None
        return SPipe(
            manifest, store=self._store, base_dir=self._base_dir,
            allow_code=self._allow_code, allow_stale=self._allow_stale, _temp=self._temp,
        )

    # verification

    def _referenced_artifact_ids(self) -> set[str]:
        found: set[str] = set()
        _collect_artifact_ids(self._manifest["controls"], found)
        return found

    def verify(self) -> SpipeReport:
        """Check the spipe without loading a model and return the findings as a `SpipeReport`.

        The method does not load a model or create a backend, and it does not raise an error for
        a failed check. It runs the following checks:

        1. **Format**: the manifest is validated against the `spipe/1` schema. A violation is
           an error.
        2. **Artifacts**: the referenced artifacts that are not in the store are listed in one
           warning (a thin bundle). Each present artifact is verified against its content id,
           and its store sidecar against its manifest record. A failure is an error.
        3. **Staleness**: the fit digest of each frozen entry is recomputed from its recipe
           args. The check stops at the first stale entry and reports it as an error. An entry
           whose recipe does not decode in this process is reported as a warning. A check that
           cannot run for another reason (e.g., a missing artifact) is also reported as a
           warning.
        4. **Versions**: a major version difference between the toolkit that wrote the spipe
           and the installed toolkit is a warning. A version recorded as `"unknown"` on either
           side is not compared.
        5. **Code dependence**: a manifest that references code (`$ref`) is reported as a
           warning.

        Returns:
            The report, whose `ok` is True when no check found an error.
        """
        errors: list[str] = []
        report_warnings: list[str] = []

        try:
            validate_manifest(self._manifest)
        except SpipeFormatError as exc:
            errors.append(str(exc))

        referenced = sorted(self._referenced_artifact_ids())
        missing = [
            artifact_id for artifact_id in referenced
            if self._store is None or not self._store.has(artifact_id)
        ]
        if missing:
            report_warnings.append(
                f"{len(missing)} of {len(referenced)} referenced artifact(s) are not present (thin "
                f"bundle); pass artifact_store= at load to resolve them: {', '.join(missing)}."
            )
        for artifact_id in referenced:
            if artifact_id not in missing:
                try:
                    self._verify_artifact(artifact_id)
                except (SpipeFormatError, SpipeIntegrityError) as exc:
                    errors.append(str(exc))

        try:
            unchecked = self._check_staleness(raise_on_stale=True, force=True)
            report_warnings.extend(f"staleness check could not run for {message}" for message in unchecked)
        except SpipeStaleError as exc:
            errors.append(str(exc))
        except Exception as exc:
            report_warnings.append(f"staleness check could not run: {exc}")

        from steerability.spipe.freeze import _toolkit_version

        saved = self._manifest.get("toolkit_version", "unknown")
        current = _toolkit_version()
        comparable = "unknown" not in (saved, current)
        if comparable and saved.split(".")[0] != current.split(".")[0]:
            report_warnings.append(
                f"spipe was written by steerability {saved}; this is {current} (major mismatch)."
            )
        if self.code_dependent:
            report_warnings.append(
                "the manifest references code ($ref); loading a pipeline from it requires "
                "allow_code=True and the referenced modules on the import path."
            )

        return SpipeReport(ok=not errors, errors=errors, warnings=report_warnings)

    def _verify_artifact(self, artifact_id: str) -> None:
        """Verify a stored artifact against its content id and its manifest record.

        The store sidecar is compared with the manifest record only when the manifest has a
        record for the artifact.

        Args:
            artifact_id: The id of an artifact that the store contains.

        Raises:
            SpipeFormatError: If the artifact is in the bundle's `artifacts/` directory and its
                directory, sidecar, tensor file, or payload directory is a symlink, or if a tree
                payload contains a symlink.
            SpipeIntegrityError: If the sidecar is malformed or records a different id, if a
                tree payload is missing or empty, if the content hash differs from the id, or if
                the sidecar's encoding or type differs from the manifest record.
        """
        self._store.verify(artifact_id)
        expected = self._manifest_records.get(artifact_id)
        if expected is not None:
            check_sidecar(self._store.record_for(artifact_id), expected)

    def _decode_ctx(
        self, *, lenient: bool, verify: str = "off", data_mode: str = "keep", allow_code: bool | None = None,
    ) -> DecodeContext:
        return DecodeContext(
            store=self._store,
            allow_code=self._allow_code if allow_code is None else allow_code,
            code_mode="sentinel" if lenient else "strict",
            verify=verify,
            data_mode=data_mode,
            manifest_records=self._manifest_records,
        )

    @staticmethod
    def _instantiate(method_key: str, encoded_args: Mapping, ctx: DecodeContext, *, index: int):
        """Decode the args of an entry or resolved item and construct its control.

        Args:
            method_key: The method key of the entry or resolved item.
            encoded_args: The args in encoded form.
            ctx: The decoding context.
            index: The position of the entry in the manifest's `controls` list, used in error
                messages.

        Returns:
            The constructed control.

        Raises:
            SpipeFormatError: If `method_key` does not match a registered method, if an encoded
                value is malformed, or if the control constructor rejects the decoded args with
                an `AttributeError`, `LookupError`, `TypeError`, or `ValueError`.
            SpipeCodeRefError: If decoding requires code that `ctx` does not permit.
            SpipeIntegrityError: If a referenced artifact is unavailable or fails verification.
        """
        from steerability.algorithms.core.registry import RegistryError, resolve_method_key

        try:
            method = resolve_method_key(method_key)
        except RegistryError as exc:
            raise SpipeFormatError(str(exc)) from exc
        kwargs = {name: decode(value, ctx, f"args.{name}") for name, value in encoded_args.items()}
        try:
            return method.control_cls(**kwargs)
        except (AttributeError, LookupError, TypeError, ValueError) as exc:
            raise SpipeFormatError(
                f"controls[{index}] ({method_key}): the decoded args do not construct the control "
                f"({type(exc).__name__}: {exc})."
            ) from exc

    def _check_staleness(self, *, raise_on_stale: bool, force: bool = False) -> list[str]:
        """Recompute the fit digest of each frozen entry from its current recipe args.

        Entries that an earlier call checked are skipped unless `force` is True. An entry whose
        recipe does not decode in this process is left unchecked, e.g., because its method key
        is not registered or its args require code that `allow_code` does not permit.
        `instantiate_entry()` raises its decoding error when it runs the staleness check for
        that entry. Unless the spipe was loaded with `allow_stale=True`, `load()` runs this
        check when every referenced artifact is available, and `pipeline()` runs it under
        `prefer="frozen"`.

        Args:
            raise_on_stale: Whether a stale entry raises `SpipeStaleError`. When False, a stale
                entry emits a `UserWarning` instead.
            force: Whether to check the entries that an earlier call checked.

        Returns:
            One message per entry left unchecked, giving the entry and the decoding error.

        Raises:
            SpipeStaleError: If `raise_on_stale` is True and a recorded fit digest differs from
                the recomputed one, or if computing an entry's fit identity fails.
            SpipeIntegrityError: If an artifact the recipe references is unavailable or fails
                verification.

        Warns:
            UserWarning: If `raise_on_stale` is False and a recorded fit digest differs from the
                recomputed one.
        """
        unchecked = []
        for index, entry in enumerate(self._manifest["controls"]):
            if index in self._stale_checked and not force:
                continue
            try:
                self._check_entry_staleness(index, raise_on_stale=raise_on_stale)
            except (SpipeFormatError, SpipeCodeRefError) as exc:
                prefix = f"controls[{index}] ({entry['method']}): "
                message = str(exc)
                unchecked.append(message if message.startswith(prefix) else prefix + message)
        return unchecked

    def _check_entry_staleness(self, index: int, *, raise_on_stale: bool) -> None:
        """Recompute the fit digest of one frozen entry from its current recipe args.

        An entry with no recorded fit digest is marked as checked without decoding its recipe.
        Otherwise the recipe args are decoded with `$ref` callables as `CodeRef` placeholders,
        and the entry is marked as checked when the comparison does not raise an error.

        Args:
            index: The position of the entry in the manifest's `controls` list.
            raise_on_stale: Whether a stale entry raises `SpipeStaleError`. When False, a stale
                entry emits a `UserWarning` instead.

        Raises:
            SpipeFormatError: If the entry's method key does not match a registered method, if
                an encoded recipe value is malformed, or if the control constructor rejects the
                decoded args.
            SpipeCodeRefError: If the recipe args contain a `$dc` class or an artifact payload
                that requires code and the spipe was loaded without `allow_code=True`.
            SpipeIntegrityError: If a recipe value references an artifact that is unavailable or
                fails verification.
            SpipeStaleError: If `raise_on_stale` is True and a recorded fit digest differs from
                the recomputed one, or if computing the fit identity fails (including a
                constructor error other than an `AttributeError`, `LookupError`, `TypeError`, or
                `ValueError`).

        Warns:
            UserWarning: If `raise_on_stale` is False and a recorded fit digest differs from the
                recomputed one.
        """
        entry = self._manifest["controls"][index]
        recorded: dict[str, str] = {}
        for item in _resolved_items(entry):
            for name, record in (item.get("artifacts") or {}).items():
                if record.get("fit_digest"):
                    recorded[name] = record["fit_digest"]
        if not recorded:
            self._stale_checked.add(index)
            return
        try:
            control = self._instantiate(entry["method"], entry["args"], self._decode_ctx(lenient=True), index=index)
            fit_identity = control.fit_identity()
        except (SpipeFormatError, SpipeCodeRefError, SpipeIntegrityError):
            raise
        except Exception as exc:
            raise SpipeStaleError(
                f"controls[{index}] ({entry['method']}): the recipe args no longer reconstruct "
                f"for the staleness check ({exc}); thaw() and re-steer(), or pass "
                "allow_stale=True."
            ) from exc
        current = digest_of(fit_identity) if fit_identity is not None else None
        for name, digest in recorded.items():
            if digest != current:
                message = (
                    f"controls[{index}] ({entry['method']}): frozen artifact {name!r} was "
                    f"produced from fit digest {digest} but the recipe now digests to "
                    f"{current}; the fit-relevant recipe fields were edited after "
                    "freezing. thaw() and re-steer(), or pass allow_stale=True."
                )
                if raise_on_stale:
                    raise SpipeStaleError(message)
                warnings.warn(message, UserWarning)
        self._stale_checked.add(index)

    # load / save

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        allow_code: bool = False,
        allow_stale: bool = False,
        artifact_store: str | Path | Callable[[str], Path] | None = None,
    ) -> "SPipe":
        """Load a spipe from a `.spipe` zip file or a spipe directory.

        A zip file is extracted to a temporary directory. Each referenced artifact that is
        present is verified against its content id and its manifest record. The staleness check
        runs when `allow_stale` is False and every referenced artifact is present. When some
        referenced artifacts are missing, the staleness check is deferred to `pipeline()` and
        `instantiate_entry()`, and each missing artifact is verified when it is decoded.

        An entry whose recipe does not decode in this process does not stop the load. The
        decoding error is raised when that entry is checked for staleness or instantiated from
        its recipe.

        Args:
            path: A `.spipe` zip file or a spipe directory.
            allow_code: Whether decoding may import `$ref` callables, decode `$dc` values whose
                class is not a toolkit enum or a data-only toolkit dataclass, and load artifact
                payloads that contain pickled data.
            allow_stale: Skip the staleness check.
            artifact_store: An external artifact source for thin bundles, either a store
                directory or a callable that maps an artifact id to the directory containing
                it. Artifacts in the bundle's `artifacts/` directory take precedence. Symlinked
                artifact directories, sidecars, tensor files, and payload directories are
                rejected only inside the bundle's `artifacts/` directory. A symlink inside a
                tree payload is rejected wherever the artifact is stored.

        Returns:
            The loaded `SPipe`.

        Raises:
            SpipeFormatError: If the format version is unsupported, if the manifest is missing
                or violates the schema, if the archive is malformed, if the bundle's
                `artifacts/` directory or one of its artifact entries is a symlink, or if a tree
                payload contains a symlink.
            SpipeIntegrityError: If a present artifact fails content verification, or if its
                store sidecar is malformed or disagrees with its manifest record.
            SpipeStaleError: If a frozen entry is stale and `allow_stale` is False.
        """
        path = Path(path)
        temp = None
        if path.is_dir():
            base_dir = path
            if (base_dir / ARTIFACTS_DIR).is_symlink():
                raise SpipeFormatError(
                    f"{base_dir / ARTIFACTS_DIR} is a symlink; a spipe directory must contain its artifacts."
                )
        else:
            temp = tempfile.TemporaryDirectory(prefix="spipe-load-")
            unpack_zip(path, temp.name)
            base_dir = Path(temp.name)

        manifest = read_manifest(base_dir)

        resolver: Callable[[str], Path] | None = None
        if callable(artifact_store):
            resolver = artifact_store
        elif artifact_store is not None:
            external = Path(artifact_store)
            resolver = lambda artifact_id: external / artifact_id.replace(":", "-", 1)  # noqa: E731
        store = ArtifactStore(base_dir / ARTIFACTS_DIR, resolver=resolver)

        spipe = cls(
            manifest, store=store, base_dir=base_dir,
            allow_code=allow_code, allow_stale=allow_stale, _temp=temp,
        )

        referenced = spipe._referenced_artifact_ids()
        available = [artifact_id for artifact_id in referenced if store.has(artifact_id)]
        for artifact_id in sorted(available):
            spipe._verify_artifact(artifact_id)
        if len(available) < len(referenced):
            logger.info(
                "Thin spipe: %d of %d referenced artifacts unavailable; integrity and "
                "staleness checks defer to pipeline construction.",
                len(referenced) - len(available), len(referenced),
            )
        elif not allow_stale:
            spipe._check_staleness(raise_on_stale=True)

        return spipe

    def save(self, path: str | Path, *, artifacts: str = "fat") -> Path:
        """Write the spipe to `path`.

        A path ending in `.spipe` produces a zip file; any other path produces (or replaces)
        a directory. `artifacts="fat"` (default) embeds every referenced artifact;
        `artifacts="thin"` writes the manifest only, leaving artifact ids resolvable at load
        via `artifact_store=`. A recipe-only spipe with no artifact references writes no
        `artifacts/` directory either way.

        Saving onto the bundle's own backing directory rewrites the manifest in place. A fat
        save there also embeds any referenced artifact the store resolves externally, and a
        thin save there raises `SpipeSaveError` when the directory embeds artifacts, since a
        thin export would have to delete them.

        Args:
            path: Destination file or directory.
            artifacts: `"fat"` or `"thin"`.

        Returns:
            The written path.

        Raises:
            SpipeSaveError: If a directory target exists and is neither empty nor a spipe
                directory, a referenced artifact is unavailable for a fat export, or a thin
                export targets the bundle's own directory while it embeds artifacts.
        """
        if artifacts not in ("fat", "thin"):
            raise SpipeSaveError(f"artifacts must be 'fat' or 'thin'; got {artifacts!r}.")
        path = Path(path)
        referenced = sorted(self._referenced_artifact_ids())
        if artifacts == "fat" and referenced:
            if self._store is None:
                raise SpipeSaveError("This spipe has no artifact store; only artifacts='thin' is possible.")
            for artifact_id in referenced:
                self._store.verify(artifact_id)

        if path.suffix == ".spipe":
            with tempfile.TemporaryDirectory(prefix="spipe-save-") as staging:
                write_manifest(self._manifest, staging)
                if artifacts == "fat" and referenced:
                    self._store.copy_into(Path(staging) / ARTIFACTS_DIR, referenced)
                path.parent.mkdir(parents=True, exist_ok=True)
                pack_zip(staging, path)
            return path

        if self._base_dir is not None and path.exists() \
                and path.resolve() == Path(self._base_dir).resolve():
            # saving onto the backing directory rewrites the manifest in place; a fat save also
            # embeds artifacts the store resolves externally, and a thin save is refused when the
            # directory embeds artifacts since it would have to delete them
            if artifacts == "thin" and (path / ARTIFACTS_DIR).is_dir():
                raise SpipeSaveError(
                    f"A thin export onto the bundle's own directory {path} would delete its "
                    "embedded artifacts; save the thin export to a new path."
                )
            write_manifest(self._manifest, path)
            if artifacts == "fat" and referenced:
                self._store.copy_into(path / ARTIFACTS_DIR, referenced)
            return path

        if path.exists():
            if not path.is_dir():
                raise SpipeSaveError(f"{path} exists and is not a directory.")
            occupied = any(path.iterdir())
            if occupied and not (path / MANIFEST_NAME).exists():
                raise SpipeSaveError(
                    f"{path} exists and is not a spipe directory; refusing to replace it."
                )
            for member in (path / MANIFEST_NAME, path / ARTIFACTS_DIR):
                if member.is_dir():
                    shutil.rmtree(member)
                elif member.exists():
                    member.unlink()
        path.mkdir(parents=True, exist_ok=True)
        write_manifest(self._manifest, path)
        if artifacts == "fat" and referenced:
            self._store.copy_into(path / ARTIFACTS_DIR, referenced)
        return path

    # pipeline construction

    def instantiate_entry(
        self,
        index: int,
        *,
        prefer: str = "frozen",
        verify: str = "off",
        lenient: bool = False,
    ) -> list[BaseControl]:
        """Instantiate the controls of one manifest entry.

        Under `prefer="frozen"`, an entry with a resolved section is instantiated from it, with
        one control per resolved item. Any other entry is instantiated from its recipe args.
        Each control's `enabled` flag is copied from the entry. `pipeline()` calls this method
        for each entry. Errors are raised for the requested entry only, and an entry that cannot
        be instantiated does not affect the others. Under `prefer="frozen"`, the staleness check
        runs first for an entry that has not been checked, unless `lenient=True` or the spipe
        was loaded with `allow_stale=True`.

        With `lenient=True`, the entry is decoded for inspection (e.g., of `steer_access()` or
        `requirements()`), and no bundle code is imported or run. Each `$ref` callable decodes
        to a `CodeRef`, which raises `SpipeCodeRefError` when called. Each `$data` reference
        decodes to a `DataRef` without loading. Every other value decodes as under
        `allow_code=False`, even when the spipe was loaded with `allow_code=True`. An entry that
        requires code (e.g., a `$dc` class outside the data-only set or an artifact payload
        that contains pickled data) therefore raises `SpipeCodeRefError`. The `verify` argument
        is treated as `"off"`.

        Args:
            index: The position of the entry in the manifest's `controls` list (see `entries`).
            prefer: `"frozen"` (default) or `"recipe"`.
            verify: The verification policy for frozen steering artifacts (`"strict"`,
                `"warn"`, or `"off"`), applied when the artifacts are bound at `steer()`. Under
                `"warn"` or `"off"`, frozen `routed_decoding` and `load_lora` controls are also
                constructed with `allow_model_mismatch=True` and `allow_base_mismatch=True`,
                respectively.
            lenient: Decode without loading datasets or importing or running bundle code.

        Returns:
            The entry's controls, one per resolved item in order, or a single control built from
            the recipe args.

        Raises:
            IndexError: If `index` is outside the manifest's `controls` list.
            ValueError: If `prefer` or `verify` is not a recognized value.
            SpipeFormatError: If the entry's method key does not match a registered method, if
                an encoded value is malformed, or if the control constructor rejects the decoded
                args.
            SpipeCodeRefError: If decoding the entry requires code and the spipe was loaded
                without `allow_code=True`, or if decoding under `lenient=True` requires code.
            SpipeIntegrityError: If a referenced artifact is unavailable, fails verification, or
                has a store sidecar that disagrees with its manifest record.
            SpipeStaleError: If the entry's frozen artifacts are stale.
        """
        _check_policies(prefer, verify)
        entries = self._manifest["controls"]
        if not 0 <= index < len(entries):
            raise IndexError(f"entry index {index} is out of range for {len(entries)} control entries.")
        entry = entries[index]

        if lenient:
            verify = "off"
            ctx = self._decode_ctx(lenient=True, allow_code=False)
        else:
            if prefer == "frozen" and not self._allow_stale and index not in self._stale_checked:
                self._check_entry_staleness(index, raise_on_stale=True)
            ctx = self._decode_ctx(lenient=False, verify=verify, data_mode="load")

        controls = []
        items = _resolved_items(entry) if prefer == "frozen" else []
        try:
            if items:
                for item in items:
                    args = dict(item["args"])
                    if verify != "strict" and item["method"] == "output_control/routed_decoding":
                        args["allow_model_mismatch"] = True
                    if verify != "strict" and item["method"] == "structural_control/load_lora":
                        args["allow_base_mismatch"] = True
                    controls.append(self._instantiate(item["method"], args, ctx, index=index))
            else:
                controls.append(self._instantiate(entry["method"], entry["args"], ctx, index=index))
        except SpipeCodeRefError as exc:
            if not (lenient and self._allow_code):
                raise
            raise SpipeCodeRefError(
                f"controls[{index}] ({entry['method']}): {exc} Lenient instantiation never runs bundle "
                "code, even under allow_code=True; instantiate without lenient=True to permit it."
            ) from exc

        # controls may hold paths into this spipe's extraction directory; retaining the spipe
        # on each control keeps that directory alive for the controls' lifetime
        for control in controls:
            control.enabled = entry["enabled"]
            control._spipe_retainer = self
        return controls

    def pipeline(
        self,
        *,
        backend: BackendSpec | str | None = None,
        prefer: str = "frozen",
        verify: str = "strict",
        **pipeline_kwargs,
    ) -> SteeringPipeline:
        """Construct a `SteeringPipeline` from this spipe.

        Under `prefer="frozen"` (default), each frozen entry is instantiated from its resolved
        section. With `prefer="recipe"`, every entry is instantiated from its recipe args, and
        `steer()` runs the fits again. The spipe provides the model reference and the controls.
        The backend and the model loading options (e.g., `device_map` and `hf_model_kwargs`)
        come from the arguments of this call. A model revision recorded in the manifest is
        added to `hf_model_kwargs` as `revision` unless the call's `hf_model_kwargs` sets one.
        Under `prefer="frozen"`, the staleness check runs for the entries that have not been
        checked, unless the spipe was loaded with `allow_stale=True`.

        Args:
            backend: The backend, forwarded to the `SteeringPipeline` constructor.
            prefer: `"frozen"` (default) or `"recipe"`.
            verify: The verification policy for frozen steering artifacts (`"strict"`,
                `"warn"`, or `"off"`), applied when the artifacts are bound at `steer()`. Under
                `"warn"` or `"off"`, frozen `routed_decoding` and `load_lora` controls are also
                constructed with `allow_model_mismatch=True` and `allow_base_mismatch=True`,
                respectively.
            **pipeline_kwargs: Other keyword arguments forwarded to the `SteeringPipeline`
                constructor.

        Returns:
            The constructed `SteeringPipeline`, not yet steered.

        Raises:
            ValueError: If `prefer` or `verify` is not a recognized value.
            SpipeFormatError: If a method key does not match a registered method, if an encoded
                value is malformed, or if a control constructor rejects the decoded args.
            SpipeCodeRefError: If decoding requires code and the spipe was loaded without
                `allow_code=True`.
            SpipeIntegrityError: If a referenced artifact is unavailable, fails verification, or
                has a store sidecar that disagrees with its manifest record.
            SpipeStaleError: If a frozen entry is stale, e.g., in a thin bundle whose staleness
                check `load()` deferred.
        """
        _check_policies(prefer, verify)
        if prefer == "frozen" and not self._allow_stale:
            self._check_staleness(raise_on_stale=True)

        controls = [
            control
            for index in range(len(self._manifest["controls"]))
            for control in self.instantiate_entry(index, prefer=prefer, verify=verify)
        ]

        from steerability.algorithms.core.steering_pipeline import SteeringPipeline

        revision = self._manifest["model"].get("revision")
        if revision is not None:
            hf_model_kwargs = dict(pipeline_kwargs.get("hf_model_kwargs") or {})
            hf_model_kwargs.setdefault("revision", revision)
            pipeline_kwargs["hf_model_kwargs"] = hf_model_kwargs

        return SteeringPipeline(
            model_name_or_path=self._manifest["model"]["ref"],
            controls=controls,
            backend=backend,
            **pipeline_kwargs,
        )
