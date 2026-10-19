"""Artifact store encodings, ids, idempotence, and integrity."""
import io
import json
import pickle
import zipfile

import pytest
import torch

from steerability.algorithms.state_control.common.lowering import artifact_id_for
from steerability.spipe.errors import SpipeFormatError, SpipeIntegrityError, SpipeSaveError
from steerability.spipe.store import ArtifactStore, pickle_bearing_files, tensors_payload, tree_id_for


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


RECORD_FIELDS = {"type": "Tensor", "artifact_class": "opaque", "source": None,
                 "fit_digest": None, "provenance": {}, "type_meta": {}}


def test_tensor_id_matches_artifact_id_for(store):
    tensors = {"1": torch.randn(1, 8, dtype=torch.bfloat16), "3": torch.randn(1, 8)}
    record = store.put_tensors(tensors, dict(RECORD_FIELDS))
    expected_id, _ = artifact_id_for(tensors)
    assert record.id == expected_id

    # stored bytes hash back to the id (byte-compatibility with the plugin registry)
    store.verify(record.id)
    loaded = store.load_tensors(record.id)
    assert loaded["1"].dtype == torch.float32
    assert torch.allclose(loaded["3"], tensors["3"])


def test_tensor_write_idempotent(store):
    tensors = {"value": torch.ones(4)}
    first = store.put_tensors(tensors, dict(RECORD_FIELDS))
    second = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    assert first.id == second.id
    assert store.ids() == [first.id]


def test_payload_without_a_sidecar_is_rewritten(store):
    tensors = {"value": torch.arange(4.0)}
    expected_id, data = tensors_payload(tensors)
    directory = store.root / expected_id.replace(":", "-", 1)
    directory.mkdir(parents=True)
    # an interrupted write: a truncated payload and a partially written marker
    (directory / "artifact.safetensors").write_bytes(data[: len(data) // 2])
    (directory / "artifact.json.tmp").write_text('{"id": ')
    assert not store.has(expected_id)

    record = store.put_tensors(tensors, dict(RECORD_FIELDS))
    assert record.id == expected_id
    assert store.has(expected_id)
    store.verify(expected_id)
    assert torch.equal(store.load_tensors(expected_id)["value"], tensors["value"])
    assert sorted(path.name for path in directory.iterdir()) == ["artifact.json", "artifact.safetensors"]


def test_tree_without_a_sidecar_is_rewritten(tmp_path, store):
    source = tmp_path / "src"
    source.mkdir()
    (source / "a.txt").write_text("alpha")
    artifact_id = tree_id_for(source)
    payload = store.root / artifact_id.replace(":", "-", 1) / "payload"
    payload.mkdir(parents=True)
    (payload / "partial.txt").write_text("partial")

    record = store.put_tree(source, {**RECORD_FIELDS, "type": "CheckpointArtifact"})
    assert store.has(record.id)
    assert sorted(path.name for path in store.payload_path(record.id).iterdir()) == ["a.txt"]


def test_tree_id_stability_and_order_independence(tmp_path, store):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("alpha")
    (src / "sub" / "b.txt").write_text("beta")
    first = tree_id_for(src)

    other = tmp_path / "other"
    (other / "sub").mkdir(parents=True)
    (other / "sub" / "b.txt").write_text("beta")
    (other / "a.txt").write_text("alpha")
    assert tree_id_for(other) == first

    record = store.put_tree(src, {**RECORD_FIELDS, "type": "CheckpointArtifact"})
    assert record.id == first
    assert (store.payload_path(record.id) / "sub" / "b.txt").read_text() == "beta"


def test_tree_symlink_rejected(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("alpha")
    (src / "link").symlink_to(src / "a.txt")
    with pytest.raises(SpipeSaveError, match="symlink"):
        tree_id_for(src)


def test_corruption_raises_integrity_error(store):
    record = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    tensor_file = store.root / record.id.replace(":", "-", 1) / "artifact.safetensors"
    tensor_file.write_bytes(tensor_file.read_bytes()[:-1] + b"\x00")
    with pytest.raises(SpipeIntegrityError, match="integrity"):
        store.verify(record.id)


def test_sidecar_dirname_mismatch_raises(store):
    record = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    sidecar = store.root / record.id.replace(":", "-", 1) / "artifact.json"
    data = json.loads(sidecar.read_text())
    data["id"] = "sha256:" + "0" * 64
    sidecar.write_text(json.dumps(data))
    with pytest.raises(SpipeIntegrityError, match="named for"):
        store.record_for(record.id)


@pytest.mark.parametrize("content", ["{not json", "[]", "{}"])
def test_malformed_sidecar_raises_integrity_error(store, content):
    record = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    sidecar = store.root / record.id.replace(":", "-", 1) / "artifact.json"
    sidecar.write_text(content)
    with pytest.raises(SpipeIntegrityError, match="malformed"):
        store.record_for(record.id)


def test_missing_tensor_file_raises_integrity_error(store):
    record = store.put_tensors({"value": torch.ones(4)}, dict(RECORD_FIELDS))
    (store.root / record.id.replace(":", "-", 1) / "artifact.safetensors").unlink()
    for read in (store.verify, store.size_of, store.load_tensors):
        with pytest.raises(SpipeIntegrityError, match="no tensor file"):
            read(record.id)


def test_missing_artifact_names_thin_hint(store):
    with pytest.raises(SpipeIntegrityError, match="artifact_store"):
        store.record_for("sha256:" + "a" * 64)


def test_pickle_bearing_files_by_suffix_and_header(tmp_path):
    import random

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("archive/data.pkl", pickle.dumps({}))
    files = {
        "config.json": b"{}",
        "weights.bin": b"not a pickle",
        "short.bin": b"abc",
        "text.bin": b"hello world",
        "floats.bin": torch.randn(1024, generator=torch.Generator().manual_seed(0)).numpy().tobytes(),
        **{f"random{seed}.bin": random.Random(seed).randbytes(4096) for seed in range(8)},
        "header_only.safetensors": b"\x80\x02" + b"\x00" * 6,
        "legacy.bin": pickle.dumps({}, protocol=2),
        "protocol0.bin": pickle.dumps({"w": [1.0]}, protocol=0),
        "protocol1.bin": pickle.dumps({"w": [1.0]}, protocol=1),
        "protocol5.bin": pickle.dumps({"w": [1.0]}, protocol=5),
        "small.bin": pickle.dumps(None, protocol=0),
        # a pickle cut off after it imports and calls a callable in three opcodes
        "truncated.bin": b"(S'x'\nibuiltins\nprint\n",
        # a memo store between the string operands and STACK_GLOBAL, with no PROTO and no STOP
        "memoized.bin": b"\x8c\x08builtins\x8c\x05printq\x00\x93\x8c\x05pwned\x85R",
        # a string operand longer than the scanned prefix, ahead of the import
        "long_operand.bin": b"\x8d" + (70_000).to_bytes(8, "little") + b"x" * 70_000 + b"\x8c\x05print\x93",
        "zipped.bin": archive.getvalue(),
        "nested/scorer.PKL": b"x",
        "state.pth": b"x",
        "model.ckpt": b"x",
        "cache.pk": b"x",
        "session.dill": b"x",
    }
    for name, data in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
    assert pickle_bearing_files(tmp_path) == [
        "cache.pk", "legacy.bin", "long_operand.bin", "memoized.bin", "model.ckpt", "nested/scorer.PKL",
        "protocol0.bin", "protocol1.bin", "protocol5.bin", "session.dill", "small.bin", "state.pth",
        "truncated.bin", "zipped.bin",
    ]


@pytest.mark.parametrize("artifact_id", [
    "sha256:x/../../outside",
    "sha256:" + "A" * 64,
    "sha256:" + "a" * 63,
    "md5:" + "a" * 32,
    5,
    None,
])
def test_malformed_artifact_ids_are_rejected(tmp_path, artifact_id):
    resolved = []
    store = ArtifactStore(tmp_path / "artifacts", resolver=lambda value: resolved.append(value) or tmp_path)
    for read in (store.has, store.record_for, store.verify, store.payload_path, store.load_tensors):
        with pytest.raises(SpipeFormatError, match="not an artifact id"):
            read(artifact_id)
    assert resolved == []


def test_symlinked_artifact_entries_are_rejected_under_the_store_root(tmp_path):
    import shutil

    outside = ArtifactStore(tmp_path / "outside")
    source = tmp_path / "tree"
    source.mkdir()
    (source / "memory.json").write_text("{}")
    tree = outside.put_tree(source, {**RECORD_FIELDS, "type": "TextMemory"})
    tensors = outside.put_tensors({"value": torch.ones(2)}, dict(RECORD_FIELDS))

    def artifact_dir(root, artifact_id):
        return root / artifact_id.replace(":", "-", 1)

    linked_dir = ArtifactStore(tmp_path / "linked_dir")
    linked_dir.root.mkdir()
    artifact_dir(linked_dir.root, tree.id).symlink_to(artifact_dir(outside.root, tree.id))

    def partly_linked(name, artifact_id, linked):
        store = ArtifactStore(tmp_path / name)
        directory = artifact_dir(store.root, artifact_id)
        shutil.copytree(artifact_dir(outside.root, artifact_id), directory)
        target = directory / linked
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        target.symlink_to(artifact_dir(outside.root, artifact_id) / linked)
        return store

    cases = [
        (linked_dir, tree.id),
        (partly_linked("linked_sidecar", tree.id, "artifact.json"), tree.id),
        (partly_linked("linked_payload", tree.id, "payload"), tree.id),
        (partly_linked("linked_tensors", tensors.id, "artifact.safetensors"), tensors.id),
    ]
    for store, artifact_id in cases:
        with pytest.raises(SpipeFormatError, match="symlink"):
            store.verify(artifact_id)
    with pytest.raises(SpipeFormatError, match="symlink"):
        cases[2][0].payload_path(tree.id)

    # a directory that the resolver provides is read as given
    resolved = ArtifactStore(tmp_path / "empty", resolver=lambda value: artifact_dir(linked_dir.root, value))
    resolved.verify(tree.id)
    assert (resolved.payload_path(tree.id) / "memory.json").read_text() == "{}"


def test_symlink_inside_a_tree_payload_is_rejected_in_any_store(tmp_path):
    import shutil

    source = tmp_path / "tree"
    source.mkdir()
    (source / "memory.json").write_text("{}")
    local = ArtifactStore(tmp_path / "local")
    tree = local.put_tree(source, {**RECORD_FIELDS, "type": "TextMemory"})
    external = tmp_path / "external" / tree.id.replace(":", "-", 1)
    shutil.copytree(local.root / tree.id.replace(":", "-", 1), external)
    resolved = ArtifactStore(tmp_path / "empty", resolver=lambda artifact_id: external)
    for store, directory in ((local, local.root / tree.id.replace(":", "-", 1)), (resolved, external)):
        payload_file = directory / "payload" / "memory.json"
        payload_file.unlink()
        payload_file.symlink_to(source / "memory.json")
        with pytest.raises(SpipeFormatError, match="symlink"):
            store.verify(tree.id)
