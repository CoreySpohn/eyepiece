"""Content identities depend on explicit inputs and rendering tool versions."""

import importlib
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass

import hwostyle
import matplotlib
import numpy as np
import pytest

import eyepiece as ep
import eyepiece._ffmpeg as ffmpeg
import eyepiece.render as render
from eyepiece.render import content_key


@dataclass
class Clock:
    beats: list
    fps: float


class Parts:
    def __init__(self, value):
        self.value = value

    def content_key_parts(self):
        return {"value": self.value}

    def __repr__(self):
        raise AssertionError("content keys must not use repr")


class Unsupported:
    def __repr__(self):
        raise AssertionError("content keys must not use repr")


def _encoder(path, version, *, status=0, log=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    source = "#!/bin/sh\n"
    if log is not None:
        source += f"echo probe >> {shlex.quote(str(log))}\n"
    source += f"printf '%s\\n' {shlex.quote(version)}\nexit {status}\n"
    path.write_text(source)
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def encoder_identity(tmp_path, monkeypatch):
    executable = _encoder(tmp_path / "bin" / "ffmpeg", "ffmpeg version fixture")
    monkeypatch.setenv("PATH", str(executable.parent))
    monkeypatch.setitem(matplotlib.rcParams, "animation.ffmpeg_path", "ffmpeg")
    return executable


def test_public_export():
    assert ep.content_key is content_key
    assert "content_key" in ep.__all__
    assert re.fullmatch(r"[0-9a-f]{64}", content_key())


def test_same_key_in_two_processes(tmp_path):
    script = tmp_path / "clip.py"
    script.write_text("draw()\n")
    kwargs = dict(
        script=script,
        clock=Clock([0, 1, 3], 12.0),
        modules=(render,),
        data=(np.arange(6, dtype=np.float32).reshape(2, 3), Parts([1, "two"])),
        settings={"dpi": 100, "fps": 12, "encoder": {"crf": 18}},
        extra=("stamp text",),
    )
    code = """
import sys
from pathlib import Path
import numpy as np
import eyepiece.render as render
import matplotlib
matplotlib.rcParams["animation.ffmpeg_path"] = "ffmpeg"
from test_render_keys import Clock, Parts
print(render.content_key(
    script=Path(sys.argv[1]), clock=Clock([0, 1, 3], 12.0), modules=(render,),
    data=(np.arange(6, dtype=np.float32).reshape(2, 3), Parts([1, 'two'])),
    settings={'encoder': {'crf': 18}, 'fps': 12, 'dpi': 100},
    extra=('stamp text',)))
"""
    expected = content_key(**kwargs)
    for seed in ("1", "42"):
        result = subprocess.run(
            [sys.executable, "-c", code, str(script)],
            env=dict(
                os.environ, PYTHONPATH=os.pathsep.join(sys.path), PYTHONHASHSEED=seed
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert result.stdout.strip() == expected


@pytest.mark.parametrize("field", ["script", "clock", "settings", "extra"])
def test_changed_canonical_input_changes_key(field):
    original = {
        "script": "draw A",
        "clock": [0, 1],
        "settings": {"dpi": 100},
        "extra": (1,),
    }
    replacement = {
        "script": "draw B",
        "clock": [0, 2],
        "settings": {"dpi": 101},
        "extra": (2,),
    }
    changed = {**original, field: replacement[field]}
    assert content_key(**original) != content_key(**changed)


def test_dict_and_keyword_order_do_not_matter():
    assert content_key(
        script="clip",
        clock={"b": [1, {"z": 3, "a": 2}], "a": 0},
        settings={"dpi": 100, "fps": 12},
        extra=("stamp",),
    ) == content_key(
        extra=("stamp",),
        settings={"fps": 12, "dpi": 100},
        clock={"a": 0, "b": [1, {"a": 2, "z": 3}]},
        script="clip",
    )


@pytest.mark.parametrize(
    "left, right",
    [
        (("ab", "c"), ("a", "bc")),
        (("ab",), ("a", "b")),
        (([1, 2],), ([2, 1],)),
        (([1, 2],), ((1, 2),)),
        (("1",), (1,)),
        ((True,), (1,)),
        ((1,), (1.0,)),
        ((b"abc",), ("abc",)),
        (({"ab": "c"},), ({"a": "bc"},)),
    ],
)
def test_encoding_preserves_boundaries_order_and_types(left, right):
    assert content_key(data=left) != content_key(data=right)


def test_input_domains_are_distinct():
    assert content_key(script="clip") != content_key(clock="clip")
    assert content_key(data=("clip",)) != content_key(extra=("clip",))


@pytest.mark.parametrize("field", ["script", "data"])
def test_paths_read_bytes_but_strings_are_content(tmp_path, field):
    path = tmp_path / "input.txt"
    path.write_bytes(b"abc")
    moved = tmp_path / "elsewhere" / "renamed.txt"
    moved.parent.mkdir()
    moved.write_bytes(path.read_bytes())

    def key(value):
        return content_key(**{field: (value,) if field == "data" else value})

    original = key(path)
    literal = key(str(path))
    assert original == key(moved)
    assert original != literal
    path.write_bytes(b"abd")
    assert key(path) != original
    assert key(str(path)) == literal
    path.unlink()
    with pytest.raises(FileNotFoundError):
        key(path)
    assert key(str(path)) == literal


@pytest.mark.parametrize("form", ["object", "name", "path"])
def test_module_source_inputs(tmp_path, monkeypatch, form):
    name = "key_test_module"
    path = tmp_path / f"{name}.py"
    path.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    item = (
        importlib.import_module(name)
        if form == "object"
        else name
        if form == "name"
        else path
    )
    try:
        original = content_key(modules=(item,))
        assert original == content_key(modules=(path,))
        path.write_text("VALUE = 2\n")
        assert content_key(modules=(item,)) != original
    finally:
        sys.modules.pop(name, None)


def test_package_hashes_all_python_files_by_relative_path(tmp_path, monkeypatch):
    package = tmp_path / "key_test_package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n")
    (package / "unused.py").write_text("VALUE = 2\n")
    nested = package / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text("")
    leaf = nested / "unused.py"
    leaf.write_text("VALUE = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(package.name)
    try:
        original = content_key(modules=(module,))
        assert original == content_key(modules=(package.name,))
        assert original == content_key(modules=(package,))
        moved = tmp_path / "moved"
        moved.mkdir()
        for path in reversed(sorted(package.rglob("*.py"))):
            destination = moved / path.relative_to(package)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        assert original == content_key(modules=(moved,))
        (package / "ignored.txt").write_text("not Python source")
        assert original == content_key(modules=(package,))
        for path in sorted(package.rglob("*.py")):
            source = path.read_bytes()
            path.write_bytes(source + b"# changed\n")
            assert original != content_key(modules=(package,))
            path.write_bytes(source)
        leaf.rename(leaf.with_name("renamed.py"))
        assert original != content_key(modules=(package,))
        leaf.unlink(missing_ok=True)
        (nested / "renamed.py").unlink()
        assert original != content_key(modules=(package,))
    finally:
        sys.modules.pop(package.name, None)


def test_missing_module_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        content_key(modules=(tmp_path / "missing.py",))


def test_array_dtype_shape_and_bytes_are_inputs():
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    original = content_key(data=(array,))
    changed = array.copy()
    changed[0, 0] += 1
    assert content_key(data=(changed,)) != original
    assert content_key(data=(array.reshape(3, 2),)) != original
    assert content_key(data=(array.view(np.float32),)) != original
    assert content_key(data=(np.asfortranarray(array),)) == original
    assert content_key(data=(array[:, ::-1],)) == content_key(
        data=(array[:, ::-1].copy(),)
    )
    assert content_key(data=(np.empty((0, 2)),)) != content_key(
        data=(np.empty((0, 3)),)
    )


def test_structured_array_field_names_are_part_of_dtype():
    left = np.array([(1, 2)], dtype=[("x", "i4"), ("y", "i4")])
    right = left.view(dtype=[("a", "i4"), ("b", "i4")])
    assert content_key(data=(left,)) != content_key(data=(right,))


@pytest.mark.parametrize(
    "array",
    [
        np.array([object()], dtype=object),
        np.empty(0, dtype=object),
        np.array([(1,)], dtype=[("value", object)]),
    ],
)
def test_object_arrays_are_rejected(array):
    with pytest.raises(TypeError, match="object"):
        content_key(data=(array,))


def test_dataclass_fields_are_canonical():
    first = Clock([0, 1], 12.0)
    assert content_key(clock=first) == content_key(clock=Clock([0, 1], 12.0))
    assert content_key(clock=first) != content_key(clock=Clock([0, 2], 12.0))
    assert content_key(clock=first) != content_key(clock=Clock([0, 1], 24.0))


def test_content_key_parts_are_canonical():
    assert content_key(clock=Parts({"a": 1, "b": [2]})) == content_key(
        clock=Parts({"b": [2], "a": 1})
    )
    assert content_key(clock=Parts(1)) != content_key(clock=Parts(2))
    with pytest.raises(TypeError):
        content_key(clock=Parts(Unsupported()))


@pytest.mark.parametrize("field", ["clock", "data", "settings", "extra"])
def test_unsupported_objects_raise_without_repr(field):
    value = Unsupported()
    if field in ("data", "extra"):
        value = (value,)
    with pytest.raises(TypeError):
        content_key(**{field: value})


def test_no_implicit_output_path_or_stamp_input(tmp_path, monkeypatch):
    original = content_key(script="clip", settings={"dpi": 100})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ep, "stamp", Unsupported())
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "123")
    assert content_key(script="clip", settings={"dpi": 100}) == original
    assert (
        content_key(script="clip", settings={"dpi": 100}, extra=("stamp",)) != original
    )


@pytest.mark.parametrize("package", [ep, matplotlib, np, hwostyle])
def test_package_versions_are_inputs(monkeypatch, package):
    original = content_key()
    monkeypatch.setattr(package, "__version__", "different-version")
    assert content_key() != original


def test_missing_eyepiece_version_is_supported(monkeypatch):
    present = content_key()
    monkeypatch.setattr(ep, "__version__", None)
    absent = content_key()
    assert absent != present
    monkeypatch.delattr(ep, "__version__")
    assert content_key() == absent


def test_unimportable_hwostyle_version_is_none(monkeypatch):
    present = content_key()
    with monkeypatch.context() as patch:
        patch.setattr(hwostyle, "__version__", None)
        absent = content_key()
    assert absent != present
    monkeypatch.setitem(sys.modules, "hwostyle", None)
    assert content_key() == absent


@pytest.mark.parametrize("selection", ["rc", "PATH"])
def test_switching_ffmpeg_executables_changes_key(tmp_path, monkeypatch, selection):
    first_log, second_log = tmp_path / "first.log", tmp_path / "second.log"
    first = _encoder(tmp_path / "one" / "ffmpeg", "ffmpeg version one", log=first_log)
    second = _encoder(tmp_path / "two" / "ffmpeg", "ffmpeg version two", log=second_log)

    def select(executable):
        if selection == "rc":
            monkeypatch.setitem(
                matplotlib.rcParams, "animation.ffmpeg_path", str(executable)
            )
        else:
            monkeypatch.setenv("PATH", str(executable.parent))

    select(first)
    original = content_key()
    assert content_key() == original
    select(second)
    assert content_key() != original
    select(first)
    assert content_key() == original
    assert first_log.read_text().splitlines() == ["probe"]
    assert second_log.read_text().splitlines() == ["probe"]


@pytest.mark.parametrize("failure", ["resolve", "probe", "empty"])
def test_missing_ffmpeg_records_none_and_recovers(tmp_path, monkeypatch, failure):
    present = content_key()
    with monkeypatch.context() as patch:
        patch.setattr(render, "_ffmpeg_version", lambda: None)
        absent = content_key()
    assert present != absent
    if failure == "resolve":
        resolve = ffmpeg.resolve_ffmpeg

        def missing(*args):
            raise RuntimeError("ffmpeg unavailable")

        monkeypatch.setattr(ffmpeg, "resolve_ffmpeg", missing)
    else:
        executable = _encoder(
            tmp_path / "recover" / "ffmpeg",
            "ffmpeg version fixture",
            status=1 if failure == "probe" else 0,
        )
        if failure == "empty":
            executable.write_text("#!/bin/sh\nexit 0\n")
        monkeypatch.setitem(
            matplotlib.rcParams, "animation.ffmpeg_path", str(executable)
        )
    assert content_key() == absent
    if failure == "resolve":
        monkeypatch.setattr(ffmpeg, "resolve_ffmpeg", resolve)
    else:
        _encoder(executable, "ffmpeg version fixture")
    assert content_key() == present


def test_package_follows_directory_symlinks_and_stops_cycles(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "__init__.py").write_text("")
    external = tmp_path / "external"
    external.mkdir()
    (external / "__init__.py").write_text("")
    source = external / "leaf.py"
    source.write_text("VALUE = 1\n")
    (package / "linked").symlink_to(external, target_is_directory=True)
    original = content_key(modules=(package,))
    source.write_text("VALUE = 2\n")
    changed = content_key(modules=(package,))
    assert changed != original
    (external / "cycle").symlink_to(package, target_is_directory=True)
    assert content_key(modules=(package,)) == changed
    cache = external / "__pycache__"
    cache.mkdir()
    (cache / "ignored.py").write_text("ignored cache source")
    (cache / "leaf.pyc").write_bytes(b"bytecode")
    (external / "leaf.pyc").write_bytes(b"bytecode")
    assert content_key(modules=(package,)) == changed
    (external / "leaf.pyc").write_bytes(b"changed bytecode")
    assert content_key(modules=(package,)) == changed
    (package / "linked").rename(package / "renamed")
    assert content_key(modules=(package,)) != changed


@pytest.mark.parametrize(
    "name", ["side_effect", "side_effect.child", "side_effect.nested.leaf"]
)
def test_named_module_resolution_does_not_execute_sources(tmp_path, monkeypatch, name):
    marker = tmp_path / "imported"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    package = tmp_path / "side_effect"
    package.mkdir()
    (package / "__init__.py").write_text(source)
    (package / "child.py").write_text(source)
    nested = package / "nested"
    nested.mkdir()
    (nested / "__init__.py").write_text(source)
    (nested / "leaf.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    path = {
        "side_effect": package,
        "side_effect.child": package / "child.py",
        "side_effect.nested.leaf": nested / "leaf.py",
    }[name]
    try:
        assert content_key(modules=(name,)) == content_key(modules=(path,))
        assert not marker.exists()
        assert not any(key.startswith("side_effect") for key in sys.modules)
    finally:
        for key in list(sys.modules):
            if key.startswith("side_effect"):
                sys.modules.pop(key)


@pytest.mark.parametrize("field", ["modules", "data", "extra"])
@pytest.mark.parametrize("container", [set, frozenset, dict])
def test_outer_inputs_must_be_lists_or_tuples(field, container):
    value = container({"eyepiece.render": "discarded value"})
    with pytest.raises(TypeError):
        content_key(**{field: value})


def test_dict_data_is_order_independent_across_hash_seeds():
    code = """
import matplotlib
from eyepiece.render import content_key
matplotlib.rcParams["animation.ffmpeg_path"] = "ffmpeg"
value = {key: key.upper() for key in {"alpha", "beta", "gamma"}}
print(content_key(data=(value,)))
"""
    expected = content_key(data=({"alpha": "ALPHA", "beta": "BETA", "gamma": "GAMMA"},))
    for seed in ("1", "42"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=dict(
                os.environ, PYTHONPATH=os.pathsep.join(sys.path), PYTHONHASHSEED=seed
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    "scalar", [np.float32(1), np.float64(1), np.int64(1), np.bool_(True)]
)
def test_numpy_scalars_preserve_dtype_as_zero_dimensional_arrays(scalar):
    assert content_key(data=(scalar,)) == content_key(data=(np.asarray(scalar),))
    assert content_key(data=(scalar,)) != content_key(data=(scalar.item(),))


@pytest.mark.parametrize(
    "dtype, unsigned, bits",
    [(np.float32, np.uint32, 0x7FC00001), (np.float64, np.uint64, 0x7FF8000000000001)],
)
def test_float_signed_zero_and_nan_payloads_are_preserved(dtype, unsigned, bits):
    assert content_key(data=(dtype(0.0),)) != content_key(data=(dtype(-0.0),))
    nan = np.array(bits, dtype=unsigned).view(dtype)[()]
    other_nan = np.array(bits + 1, dtype=unsigned).view(dtype)[()]
    assert np.isnan(nan) and np.isnan(other_nan)
    assert content_key(data=(nan,)) != content_key(data=(other_nan,))
    if dtype is np.float64:
        assert content_key(data=(float(nan),)) != content_key(data=(float(other_nan),))
        assert content_key(data=(0.0,)) != content_key(data=(-0.0,))
