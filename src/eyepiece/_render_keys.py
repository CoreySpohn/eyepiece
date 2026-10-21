"""Canonical content identities and rendering tool versions."""

import hashlib
import importlib.machinery
import importlib.util
import os
import shutil
import struct
import subprocess
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np

import eyepiece
from eyepiece import _ffmpeg

_LENGTH = struct.Struct("!Q")


_FFMPEG_VERSIONS = {}


def _key_field(tag, payload):
    """Frame both a type tag and its payload with unsigned byte lengths."""
    return _LENGTH.pack(len(tag)) + tag + _LENGTH.pack(len(payload)) + payload


def _canonical(value):
    """Encode supported values without addresses or ambiguous concatenations."""
    if isinstance(value, (np.ndarray, np.generic)):
        if value.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot enter a content key")
        dtype = value.dtype.descr if value.dtype.fields is not None else value.dtype.str
        payload = _canonical((dtype, value.shape, value.tobytes(order="C")))
        return _key_field(b"array", payload)
    if value is None:
        return _key_field(b"none", b"")
    if isinstance(value, bool):
        return _key_field(b"bool", bytes([value]))
    if isinstance(value, int):
        return _key_field(b"int", str(value).encode("ascii"))
    if isinstance(value, float):
        return _key_field(b"float", struct.pack("!d", value))
    if isinstance(value, complex):
        return _key_field(b"complex", struct.pack("!dd", value.real, value.imag))
    if isinstance(value, str):
        return _key_field(b"str", value.encode("utf-8"))
    if isinstance(value, bytes):
        return _key_field(b"bytes", value)
    if isinstance(value, Path):
        return _key_field(b"file", value.read_bytes())
    if isinstance(value, Mapping):
        entries = sorted((_canonical(k), _canonical(v)) for k, v in value.items())
        return _key_field(b"dict", b"".join(k + v for k, v in entries))
    if isinstance(value, (list, tuple)):
        tag = b"list" if isinstance(value, list) else b"tuple"
        return _key_field(tag, b"".join(_canonical(item) for item in value))
    kind = type(value)
    name = f"{kind.__module__}.{kind.__qualname__}"
    parts = getattr(value, "content_key_parts", None)
    if callable(parts):
        return _key_field(b"parts", _canonical((name, parts())))
    if is_dataclass(value) and not isinstance(value, type):
        values = {field.name: getattr(value, field.name) for field in fields(value)}
        return _key_field(b"dataclass", _canonical((name, values)))
    raise TypeError(f"Unsupported content key type: {name}")


def _package_source(roots):
    """Read logical package paths, following links except ancestor cycles."""
    sources = []
    for root in map(Path, roots):
        ancestors = {root: frozenset({root.resolve()})}
        for directory, directories, filenames in os.walk(root, followlinks=True):
            directory = Path(directory)
            seen = ancestors.pop(directory)
            keep = []
            for name in directories:
                child = directory / name
                real = child.resolve()
                if name != "__pycache__" and real not in seen:
                    keep.append(name)
                    ancestors[child] = seen | {real}
            directories[:] = keep
            for name in filenames:
                if name.endswith(".py"):
                    path = directory / name
                    sources.append(
                        (path.relative_to(root).as_posix(), path.read_bytes())
                    )
    return _key_field(b"package", _canonical(sorted(sources)))


def _module_spec(name):
    """Find source locations without importing a dotted name's parents."""
    parts = name.split(".")
    spec = importlib.util.find_spec(parts[0])
    for index in range(1, len(parts)):
        if spec is None or spec.submodule_search_locations is None:
            raise ModuleNotFoundError(f"No Python source found for {name}")
        spec = importlib.machinery.PathFinder.find_spec(
            ".".join(parts[: index + 1]), spec.submodule_search_locations
        )
    if spec is None:
        raise ModuleNotFoundError(f"No Python source found for {name}")
    return spec


def _module_source(module):
    """Encode source bytes, with package filenames relative to their roots."""
    if isinstance(module, (str, ModuleType)):
        if isinstance(module, str):
            spec = _module_spec(module)
            roots, source = spec.submodule_search_locations, spec.origin
        else:
            roots = getattr(module, "__path__", None)
            source = getattr(module, "__file__", None)
        if roots is not None:
            return _package_source(roots)
        if source is None or source in ("built-in", "frozen"):
            raise TypeError("A module must have Python source for a content key")
        module = Path(source)
        if module.suffix == ".pyc":
            module = Path(importlib.util.source_from_cache(str(module)))
    if not isinstance(module, Path):
        raise TypeError("modules must contain imported modules, names, or Paths")
    if module.is_dir():
        return _package_source((module,))
    return _key_field(b"module", module.read_bytes())


def _ffmpeg_version():
    """Resolve ffmpeg each time, caching successful probes by executable path."""
    try:
        executable = _ffmpeg.resolve_ffmpeg(
            matplotlib.rcParams["animation.ffmpeg_path"]
        )
        executable = str(Path(shutil.which(executable) or executable).resolve())
        if executable in _FFMPEG_VERSIONS:
            return _FFMPEG_VERSIONS[executable]
        result = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (RuntimeError, OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if not lines:
        return None
    _FFMPEG_VERSIONS[executable] = lines[0]
    return lines[0]


def _hwostyle_version():
    """Return the style package version, or None when unavailable."""
    try:
        import hwostyle
    except ImportError:
        return None
    return getattr(hwostyle, "__version__", None)


def content_key(
    *, script=None, clock=None, modules=(), data=(), settings=None, extra=()
) -> str:
    """Hash explicit clip inputs and rendering versions with SHA-256.

    The canonical encoding uses versioned domain separation and length-prefixed
    type tags and payloads. Mapping order is ignored; sequence order and types
    are preserved. NumPy scalars encode as zero-dimensional arrays with their
    dtype, distinct from Python scalars. Floating-point bits, including signed
    zero and NaN payloads, are preserved. Paths contribute file bytes, never
    absolute filenames.
    The eyepiece package source is always included: every .py file is hashed
    by logical relative path and bytes, using the same rules as modules.
    Source edits in editable installs invalidate keys without a version bump.
    Output destinations and provenance stamps are not implicit inputs.

    Args:
        script: Script text, a Path whose bytes are read, or None.
        clock: Canonical value, dataclass instance, array, or an object exposing
            content_key_parts(). Dataclasses and these objects include their
            qualified type name and their fields or returned parts.
        modules: List or tuple of imported modules, importable names, or source
            Paths. Names resolve without executing their source. Packages
            include every .py file, following directory symlinks without cycles,
            sorted by logical path relative to the package, excluding __pycache__.
        data: List or tuple of Paths, arrays, or canonical values. Arrays include
            dtype, shape, and bytes in C order, independent of memory layout.
        settings: Canonical render settings supplied by the caller, including
            fps, dpi, encoder, ffmpeg arguments, rc, holds, and frames.
        extra: List or tuple of additional canonical inputs such as stamp text.

    Returns:
        A SHA-256 hex digest including eyepiece, matplotlib, numpy, hwostyle,
        and ffmpeg versions, and the eyepiece source hash. Unavailable versions
        are recorded as None.

    Raises:
        TypeError: If a value is unsupported or an array has an object dtype.
        FileNotFoundError: If an input Path does not exist.
        ImportError: If a named module cannot be located.
    """
    if script is not None and not isinstance(script, (str, Path)):
        raise TypeError("script must be text, a Path, or None")
    for name, value in (("modules", modules), ("data", data), ("extra", extra)):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{name} must be a list or tuple")
    values = {
        "eyepiece_source": hashlib.sha256(_module_source(eyepiece)).hexdigest(),
        "script": script,
        "clock": clock,
        "modules": tuple(_module_source(module) for module in modules),
        "data": tuple(data),
        "settings": settings,
        "extra": tuple(extra),
        "versions": {
            "eyepiece": getattr(eyepiece, "__version__", None),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "hwostyle": _hwostyle_version(),
            "ffmpeg": _ffmpeg_version(),
        },
    }
    encoded = _key_field(b"eyepiece.content-key.v1", _canonical(values))
    return hashlib.sha256(encoded).hexdigest()


# Keep public pickle references stable through the facade.
content_key.__module__ = "eyepiece.render"
