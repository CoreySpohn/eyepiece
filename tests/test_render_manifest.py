"""Portable manifests track output identities without rewriting unchanged files."""

import hashlib
import json
import os
import sys
from pathlib import Path

import hwostyle
import matplotlib
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.render as render
from eyepiece.render import Manifest


@pytest.fixture(autouse=True)
def manifest_versions(monkeypatch):
    """Stub serialization metadata and expose the collector for probe tests."""
    collect = render._manifest_versions
    monkeypatch.setattr(
        render,
        "_manifest_versions",
        lambda: {
            "eyepiece": "test-eyepiece",
            "matplotlib": "test-matplotlib",
            "numpy": "test-numpy",
            "hwostyle": "test-hwostyle",
            "ffmpeg": "ffmpeg version test",
        },
    )
    return collect


def test_public_export():
    assert ep.Manifest is Manifest
    assert "Manifest" in ep.__all__


def test_round_trip_and_relative_output_metadata(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    output = root / "clip.mp4"
    output.write_bytes(b"video")
    path = tmp_path / "metadata" / "manifest.json"
    manifest = Manifest(path, root=root)
    manifest.record(
        "clip", "doc-light", key="key", outputs=[output], frames=24, fps=12, dpi=100
    )
    expected = {
        "key": "key",
        "outputs": [
            {
                "path": "clip.mp4",
                "sha256": hashlib.sha256(b"video").hexdigest(),
                "bytes": 5,
            }
        ],
        "frames": 24,
        "fps": 12,
        "dpi": 100,
    }
    assert manifest.entry("clip", "doc-light") == expected
    assert manifest.save() is True
    before = path.read_bytes()
    loaded = Manifest.load(path, root=root)
    assert loaded.entry("clip", "doc-light") == expected
    assert loaded.current("clip", "doc-light", "key") is True
    assert loaded.verify("clip", "doc-light") == []
    assert loaded.save() is False
    assert path.read_bytes() == before


def test_missing_manifest_and_unknown_entries(tmp_path):
    path = tmp_path / "missing.json"
    manifest = Manifest.load(path)
    assert not path.exists()
    assert manifest.entry("missing", "doc-light") is None
    assert manifest.current("missing", "doc-light", "key") is False
    assert manifest.verify("missing", "doc-light") == []


def test_unchanged_save_preserves_mtime_even_after_record(tmp_path):
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"video")
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="key", outputs=["clip.mp4"])
    assert manifest.save() is True
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    before = path.stat().st_mtime_ns
    assert manifest.save() is False
    manifest.record("clip", "doc-light", key="key", outputs=[output])
    assert manifest.save() is False
    assert path.stat().st_mtime_ns == before


def test_sorted_portable_json_and_insertion_order(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    entries = [("zeta", "talk-dark"), ("alpha", "doc-light"), ("alpha", "doc-dark")]
    for path, order in zip(paths, (entries, entries[::-1]), strict=True):
        manifest = Manifest(path)
        for slug, variant in order:
            manifest.record(
                slug,
                variant,
                key="key",
                outputs=["clip.mp4"],
                extra=dict(reversed([("z", 1), ("a", {"b": 2, "a": 1})]))
                if path == paths[0]
                else {"z": 1, "a": {"a": 1, "b": 2}},
            )
        manifest.save()
    text = paths[0].read_text()
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert text == json.dumps(json.loads(text), sort_keys=True, indent=2) + "\n"
    document = json.loads(text)
    assert set(document) == {"schema", "versions", "clips"}
    assert set(document["versions"]) == {
        "eyepiece",
        "matplotlib",
        "numpy",
        "hwostyle",
        "ffmpeg",
    }
    assert set(document["clips"]) == {slug for slug, _ in entries}
    for slug, variants in document["clips"].items():
        assert set(variants) == {variant for name, variant in entries if name == slug}
        for entry in variants.values():
            assert set(entry) == {"key", "outputs", "frames", "fps", "dpi", "extra"}
            assert entry["extra"] == {"z": 1, "a": {"a": 1, "b": 2}}
            for output in entry["outputs"]:
                assert set(output) == {"path", "sha256", "bytes"}
                relative = Path(output["path"])
                assert not relative.is_absolute()
                assert ".." not in relative.parts
                assert (tmp_path / relative).resolve().is_relative_to(tmp_path)


def test_current_checks_key_existence_and_size_without_hashing(tmp_path):
    output = tmp_path / "clip.mp4"
    output.write_bytes(b"video")
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.record("clip", "doc-light", key="key", outputs=[output])
    assert manifest.current("clip", "doc-light", "key") is True
    assert manifest.current("clip", "doc-light", "different") is False
    output.write_bytes(b"other")
    assert manifest.current("clip", "doc-light", "key") is True
    output.write_bytes(b"longer video")
    assert manifest.current("clip", "doc-light", "key") is False
    output.unlink()
    assert manifest.current("clip", "doc-light", "key") is False


def test_verify_reports_same_size_tampering_and_missing_outputs(tmp_path):
    folder = tmp_path / "nested"
    folder.mkdir()
    for name in ("kept", "tampered", "missing"):
        (folder / name).write_bytes(b"video")
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.record(
        "clip",
        "doc-light",
        key="key",
        outputs=[f"nested/{name}" for name in ("kept", "tampered", "missing")],
    )
    (folder / "tampered").write_bytes(b"other")
    (folder / "missing").unlink()
    assert manifest.verify("clip", "doc-light") == ["nested/tampered", "nested/missing"]


@pytest.mark.parametrize("form", ["absolute", "relative", "symlink"])
def test_record_rejects_outputs_outside_root(tmp_path, form):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    path = outside if form == "absolute" else "../outside.mp4"
    if form == "symlink":
        path = root / "linked.mp4"
        path.symlink_to(outside)
    manifest = Manifest(root / "manifest.json")
    with pytest.raises(ValueError):
        manifest.record("clip", "doc-light", key="key", outputs=[path])
    assert manifest.entry("clip", "doc-light") is None


def test_missing_output_does_not_replace_an_entry(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.record("clip", "doc-light", key="old", outputs=["clip.mp4"])
    before = manifest.entry("clip", "doc-light")
    with pytest.raises(FileNotFoundError):
        manifest.record("clip", "doc-light", key="new", outputs=["missing.mp4"])
    assert manifest.entry("clip", "doc-light") == before


def test_entry_and_extra_are_independent_copies(tmp_path):
    extra = {"labels": ["original"]}
    manifest = Manifest(tmp_path / "manifest.json")
    manifest.record("clip", "doc-light", key="key", outputs=[], extra=extra)
    extra["labels"].append("changed")
    entry = manifest.entry("clip", "doc-light")
    entry["extra"]["labels"].append("also changed")
    entry["outputs"].append({"path": "invented"})
    assert manifest.entry("clip", "doc-light")["extra"] == {"labels": ["original"]}
    assert manifest.entry("clip", "doc-light")["outputs"] == []


def test_numpy_extra_round_trips_as_json(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record(
        "clip",
        "doc-light",
        key="key",
        outputs=[],
        extra={
            "array": np.array([[1, 2], [3, 4]]),
            "count": np.int64(4),
            "scale": np.float32(0.5),
            "flag": np.bool_(True),
            "nested": [{"value": np.array(2)}],
        },
    )
    manifest.save()
    assert Manifest.load(path).entry("clip", "doc-light")["extra"] == {
        "array": [[1, 2], [3, 4]],
        "count": 4,
        "scale": 0.5,
        "flag": True,
        "nested": [{"value": 2}],
    }


@pytest.mark.parametrize("extra", [{"bad": object()}, {"bad": {1, 2}}, [1], {1: "a"}])
def test_non_json_extra_is_rejected(tmp_path, extra):
    manifest = Manifest(tmp_path / "manifest.json")
    with pytest.raises(TypeError):
        manifest.record("clip", "doc-light", key="key", outputs=[], extra=extra)


@pytest.mark.parametrize("contents", ["{invalid", "null", "[]", "{}", "\ufeff{}"])
def test_invalid_manifest_names_the_path(tmp_path, contents):
    path = tmp_path / "invalid.json"
    path.write_text(contents)
    with pytest.raises(ValueError, match=r"invalid\.json"):
        Manifest.load(path)


def test_unreadable_manifest_names_the_path(tmp_path):
    with pytest.raises(ValueError, match=tmp_path.name):
        Manifest.load(tmp_path)


def test_wrong_schema_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.save()
    document = json.loads(path.read_text())
    document["schema"] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=r"manifest\.json"):
        Manifest.load(path)


def test_empty_save_collects_versions_and_preserves_them(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    expected = render._manifest_versions()
    manifest = Manifest(path)
    assert manifest.save() is True
    assert json.loads(path.read_text())["versions"] == expected
    assert json.loads(path.read_text())["clips"] == {}
    before = path.read_bytes()
    monkeypatch.setattr(render, "_manifest_versions", lambda: pytest.fail("reprobe"))
    assert manifest.save() is False
    assert Manifest.load(path).save() is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["frames", "fps", "dpi"])
def test_missing_numeric_entry_fields_are_rejected(tmp_path, field):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="key", outputs=[])
    manifest.save()
    document = json.loads(path.read_text())
    del document["clips"]["clip"]["doc-light"][field]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=r"manifest\.json"):
        Manifest.load(path)


@pytest.mark.parametrize("field", ["frames", "fps", "dpi"])
@pytest.mark.parametrize("value", ["24", True, [], {}, float("nan"), float("inf")])
def test_invalid_numeric_entry_fields_are_rejected(tmp_path, field, value):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="key", outputs=[])
    manifest.save()
    document = json.loads(path.read_text())
    document["clips"]["clip"]["doc-light"][field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=r"manifest\.json"):
        Manifest.load(path)


@pytest.mark.parametrize(
    ("frames", "fps", "dpi"), [(None, None, None), (24, 12, 100), (24, 12.5, 100.5)]
)
def test_numeric_metadata_round_trips(tmp_path, frames, fps, dpi):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record(
        "clip", "doc-light", key="key", outputs=[], frames=frames, fps=fps, dpi=dpi
    )
    manifest.save()
    assert Manifest.load(path).entry("clip", "doc-light") == manifest.entry(
        "clip", "doc-light"
    )


def test_fractional_frame_count_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="key", outputs=[])
    manifest.save()
    document = json.loads(path.read_text())
    document["clips"]["clip"]["doc-light"]["frames"] = 1.5
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=r"manifest\.json"):
        Manifest.load(path)


def test_versions_refresh_only_after_recording(
    tmp_path, monkeypatch, manifest_versions
):
    monkeypatch.setattr(render, "_manifest_versions", manifest_versions)
    monkeypatch.setattr(render, "_ffmpeg_version", lambda: "ffmpeg version initial")
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="key", outputs=[])
    manifest.save()
    before = path.read_bytes()
    versions = json.loads(before)["versions"]
    assert versions == {
        "eyepiece": ep.__version__,
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "hwostyle": hwostyle.__version__,
        "ffmpeg": "ffmpeg version initial",
    }
    loaded = Manifest.load(path)
    for package in (ep, matplotlib, np, hwostyle):
        monkeypatch.setattr(package, "__version__", "changed-version")
    monkeypatch.setattr(render, "_ffmpeg_version", lambda: "ffmpeg version changed")
    assert loaded.save() is False
    assert path.read_bytes() == before
    loaded.record("clip", "doc-light", key="key", outputs=[])
    assert loaded.save() is True
    versions = json.loads(path.read_text())["versions"]
    assert versions["eyepiece"] == "changed-version"
    assert versions["hwostyle"] == "changed-version"
    assert versions["ffmpeg"] == "ffmpeg version changed"


def test_hwostyle_version_is_optional(tmp_path, monkeypatch, manifest_versions):
    monkeypatch.setattr(render, "_manifest_versions", manifest_versions)
    monkeypatch.setattr(render, "_ffmpeg_version", lambda: "ffmpeg version test")
    monkeypatch.setitem(sys.modules, "hwostyle", None)
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="key", outputs=[])
    manifest.save()
    assert json.loads(path.read_text())["versions"]["hwostyle"] is None


def test_failed_atomic_save_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="old", outputs=[])
    manifest.save()
    before = path.read_bytes()
    manifest.record("clip", "doc-light", key="new", outputs=[])

    def fail_replace(source, destination):
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        manifest.save()
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_partial_write_failure_keeps_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    manifest.record("clip", "doc-light", key="old", outputs=[])
    manifest.save()
    before = path.read_bytes()
    manifest.record("clip", "doc-light", key="new", outputs=[])
    original_open = Path.open
    written = []

    def failing_open(file, mode="r", *args, **kwargs):
        stream = original_open(file, mode, *args, **kwargs)
        if file.parent == tmp_path and file != path and "w" in mode:
            write = stream.write

            def partial_write(payload):
                written.append(write(payload[: len(payload) // 2]))
                stream.flush()
                raise OSError("disk full during write")

            monkeypatch.setattr(stream, "write", partial_write)
        return stream

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", failing_open)
        with pytest.raises(OSError, match="disk full during write"):
            manifest.save()
    assert written and written[0] > 0
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert manifest.save() is True
    assert Manifest.load(path).entry("clip", "doc-light")["key"] == "new"


@pytest.mark.parametrize("state", ["missing", "changed", "unchanged"])
def test_save_reclaims_stale_temporary_file(tmp_path, state):
    path = tmp_path / "manifest.json"
    manifest = Manifest(path)
    if state == "unchanged":
        manifest.record("clip", "doc-light", key="new", outputs=[])
    if state != "missing":
        manifest.save()
    temporary = tmp_path / f".{path.name}.tmp"
    temporary.write_bytes(b"interrupted write")
    manifest.record("clip", "doc-light", key="new", outputs=[])
    assert manifest.save() is (state != "unchanged")
    assert Manifest.load(path).entry("clip", "doc-light")["key"] == "new"
    assert list(tmp_path.iterdir()) == [path]
