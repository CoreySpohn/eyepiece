"""Render independent frames in spawned workers into one atomic mp4 output.

Builders and frame sources must be picklable and deterministic. A draw must
replace every artist property it uses; ``frame_faults`` can diagnose some
violations of that contract. Only an explicit caller key enables chunk reuse.
"""

import hashlib
import importlib.machinery
import importlib.util
import json
import math
import operator
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zlib
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._pylab_helpers import Gcf
from PIL import Image

import eyepiece
from eyepiece import _ffmpeg
from eyepiece._motion import frame_count
from eyepiece.anim import RawSink, _agg_canvas, _rc_overrides, _Recorder

_SCHEMA = 1
_MAGIC = b"EPCHUNK\0"
_HEADER = struct.Struct("!8sI32sIII")
_LENGTH = struct.Struct("!Q")
_MARKER = ".eyepiece-render"
_MARKER_TEXT = "eyepiece render chunks\n"
_CHUNK_NAME = re.compile(r"chunk-[0-9]+\.(?:bin|part)\Z")
_LOCK_PART_NAME = re.compile(r"lock-([0-9]+)-[0-9a-f]+\.part\Z")
_LOCK_GRACE_SECONDS = 5
_FFMPEG_VERSIONS = {}
_PROTOCOL_NAME = re.compile(
    r"(?:claim-[0-9a-f]+-[0-9]+|worker-[0-9]+\.(?:ready|part|error(?:\.part)?|log)|"
    r"chunk-[0-9]+\.error(?:\.part)?|job\.pkl|tasks\.(?:pkl|part))\Z"
)


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
        and ffmpeg versions. Unavailable versions are recorded as None.

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


_MANIFEST_SCHEMA = 1


def _jsonable(value):
    """Copy JSON values, converting NumPy arrays and scalars recursively."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, np.generic):
            return _jsonable(item)
    elif value is None or isinstance(value, (str, bool, int)):
        return value
    elif isinstance(value, float) and math.isfinite(value):
        return value
    elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {key: _jsonable(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def _manifest_versions():
    """Collect portable version strings using the content key's ffmpeg probe."""
    return {
        "eyepiece": getattr(eyepiece, "__version__", None),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "hwostyle": _hwostyle_version(),
        "ffmpeg": _ffmpeg_version(),
    }


class Manifest:
    """Record portable output identities and save deterministic JSON atomically.

    A manifest file has a single writer.

    Args:
        path: JSON manifest path.
        root: Folder containing outputs. Defaults to the manifest's parent.
    """

    def __init__(self, path, *, root=None):
        """Create an empty manifest with output paths relative to root."""
        self.path = Path(path)
        self.root = Path(root if root is not None else self.path.parent).resolve()
        self._document = {
            "schema": _MANIFEST_SCHEMA,
            "versions": dict.fromkeys(
                ("eyepiece", "matplotlib", "numpy", "hwostyle", "ffmpeg")
            ),
            "clips": {},
        }
        self._recorded = False
        self._versions_collected = False

    @classmethod
    def load(cls, path, *, root=None) -> "Manifest":
        """Load a manifest, returning an empty one when the file is missing.

        Args:
            path: JSON manifest path.
            root: Folder containing outputs, or None for the manifest's parent.

        Returns:
            A manifest preserving its saved version metadata.

        Raises:
            ValueError: If the file is unreadable, invalid, or unsupported.
        """
        manifest = cls(path, root=root)
        try:
            text = manifest.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return manifest
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read manifest {path}: {exc}") from exc
        try:
            document = json.loads(text)
            manifest._validate(document)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid manifest {path}: {exc}") from exc
        manifest._document = document
        manifest._versions_collected = True
        return manifest

    def _output_path(self, path):
        """Resolve an output while enforcing containment under the root."""
        path = Path(path)
        absolute = (path if path.is_absolute() else self.root / path).resolve()
        if not absolute.is_relative_to(self.root):
            raise ValueError(f"Output path is outside manifest root: {path}")
        return absolute

    def _validate(self, document):
        """Reject malformed schemas and output paths that escape the root."""
        if not isinstance(document, dict) or (
            type(document.get("schema")) is not int
            or document["schema"] != _MANIFEST_SCHEMA
        ):
            raise ValueError("unsupported schema version")
        versions, clips = document["versions"], document["clips"]
        if not isinstance(versions, dict) or not isinstance(clips, dict):
            raise ValueError("versions and clips must be objects")
        if set(versions) != set(self._document["versions"]) or any(
            value is not None and not isinstance(value, str)
            for value in versions.values()
        ):
            raise ValueError("invalid version metadata")
        for variants in clips.values():
            if not isinstance(variants, dict):
                raise ValueError("clip variants must be objects")
            for entry in variants.values():
                if not isinstance(entry, dict) or not isinstance(entry["key"], str):
                    raise ValueError("entries must have string keys")
                if not isinstance(entry["outputs"], list):
                    raise ValueError("outputs must be a list")
                if entry["frames"] is not None and type(entry["frames"]) is not int:
                    raise ValueError("frames must be an integer or None")
                for field in ("fps", "dpi"):
                    value = entry[field]
                    if value is not None and (
                        type(value) not in (int, float)
                        or (isinstance(value, float) and not math.isfinite(value))
                    ):
                        raise ValueError(f"{field} must be a finite number or None")
                for output in entry["outputs"]:
                    if not isinstance(output, dict):
                        raise ValueError("outputs must be objects")
                    path = output["path"]
                    if not isinstance(path, str) or Path(path).is_absolute():
                        raise ValueError("output paths must be relative strings")
                    self._output_path(path)
                    if (
                        not isinstance(output["sha256"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", output["sha256"]) is None
                        or type(output["bytes"]) is not int
                        or output["bytes"] < 0
                    ):
                        raise ValueError("invalid output hash or size")
                if "extra" in entry and not isinstance(entry["extra"], dict):
                    raise ValueError("extra must be an object")
        _jsonable(document)

    def entry(self, slug, variant) -> dict | None:
        """Return an independent copy of an entry, or None when absent."""
        return deepcopy(self._document["clips"].get(slug, {}).get(variant))

    def discard(self, slug, variant) -> bool:
        """Remove an entry, returning whether it existed.

        Other variants are preserved. Empty clip groups are removed, and the
        next save persists the deletion.

        Args:
            slug: Clip identifier.
            variant: Variant identifier.

        Returns:
            True if an entry was removed, False if it was absent.
        """
        clips = self._document["clips"]
        entries = clips.get(slug)
        if entries is None or variant not in entries:
            return False
        del entries[variant]
        if not entries:
            del clips[slug]
        return True

    def record(
        self,
        slug,
        variant,
        *,
        key,
        outputs,
        frames=None,
        fps=None,
        dpi=None,
        extra=None,
    ):
        """Record output hashes and sizes, replacing an entry after validation.

        Args:
            slug: Clip identifier.
            variant: Variant identifier.
            key: Content key for this variant.
            outputs: Paths, absolute or relative to root, read from disk.
            frames: Integer encoded frame count, or None.
            fps: Finite numeric frame rate, or None.
            dpi: Finite numeric raster resolution, or None.
            extra: Optional JSON dictionary; NumPy values are converted.

        Raises:
            FileNotFoundError: If an output does not exist.
            ValueError: If an output resolves outside root.
            TypeError: If identifiers or metadata cannot be stored as JSON.
        """
        if not all(isinstance(value, str) for value in (slug, variant, key)):
            raise TypeError("slug, variant, and key must be strings")
        if extra is not None and not isinstance(extra, dict):
            raise TypeError("extra must be a JSON dictionary")
        entry = _jsonable({"key": key, "frames": frames, "fps": fps, "dpi": dpi})
        if extra is not None:
            entry["extra"] = _jsonable(extra)
        recorded = []
        for output in outputs:
            path = self._output_path(output)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                size = os.fstat(stream.fileno()).st_size
            recorded.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": digest,
                    "bytes": size,
                }
            )
        entry["outputs"] = recorded
        self._document["clips"].setdefault(slug, {})[variant] = entry
        self._recorded = True

    def current(self, slug, variant, key) -> bool:
        """Check the key and output sizes without recomputing output hashes."""
        entry = self.entry(slug, variant)
        if entry is None or entry["key"] != key:
            return False
        for output in entry["outputs"]:
            try:
                path = self._output_path(output["path"])
                if not path.is_file() or path.stat().st_size != output["bytes"]:
                    return False
            except (OSError, ValueError):
                return False
        return True

    def verify(self, slug, variant) -> list[str]:
        """List relative paths with missing, unreadable, or changed contents."""
        entry = self.entry(slug, variant)
        failures = []
        for output in entry["outputs"] if entry is not None else ():
            try:
                with self._output_path(output["path"]).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest == output["sha256"]:
                    continue
            except (OSError, ValueError):
                pass
            failures.append(output["path"])
        return failures

    def save(self) -> bool:
        """Atomically save sorted JSON, preserving the file when bytes match.

        Version metadata is collected on the first save and refreshed after
        recording an entry. Loaded versions are preserved otherwise. Parent
        directories are created as needed, and failed writes leave the previous
        file in place. Each save reclaims its deterministic temporary file.

        Returns:
            True if the file was replaced, False if its bytes were unchanged.
        """
        document = deepcopy(self._document)
        if self._recorded or not self._versions_collected:
            document["versions"] = _manifest_versions()
        payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        try:
            unchanged = self.path.read_bytes() == payload
        except FileNotFoundError:
            unchanged = False
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            if not unchanged:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("wb") as stream:
                    stream.write(payload)
                os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self._document = document
        self._recorded = False
        self._versions_collected = True
        return not unchanged


class Venue(NamedTuple):
    """Rendering settings for an output destination.

    The frame budget keeps silent clips short. Narrated clips are exempt
    because speech determines their length.

    Attributes:
        name: Destination name used in variant filenames.
        rc: Matplotlib overrides; None means an empty mapping.
        dpi: Raster resolution, defaulting to 100 dots per inch.
        fps: Video frame rate, defaulting to 30 frames per second.
        frame_budget: Maximum silent frame source length, or None.
    """

    name: str
    rc: dict | None = None
    dpi: int = 100
    fps: float = 30
    frame_budget: int | None = None


class Variant(NamedTuple):
    """A destination, style mode, and optional narration track.

    Attributes:
        venue: Destination name matching a Venue.
        mode: Style mode passed to hwostyle.use.
        narrated: Whether to mux the spec's audio track.
    """

    venue: str
    mode: str
    narrated: bool = False

    @property
    def name(self) -> str:
        """Return the filename and manifest identifier for this variant."""
        suffix = "-narrated" if self.narrated else ""
        return f"{self.venue}-{self.mode}{suffix}"


def variants(venues, modes=("light", "dark"), narrated=()) -> list[Variant]:
    """List silent variants in venue/mode order, then narrated variants.

    Args:
        venues: Ordered destination names.
        modes: Ordered style modes, defaulting to light and dark.
        narrated: Destination names also receiving narrated variants.

    Returns:
        The ordered list of variants.

    Raises:
        ValueError: If a narrated destination is absent from venues.
    """
    venues, modes, narrated = tuple(venues), tuple(modes), tuple(narrated)
    for venue in narrated:
        if venue not in venues:
            raise ValueError(f"Unknown narrated venue: {venue}")
    return [Variant(venue, mode) for venue in venues for mode in modes] + [
        Variant(venue, mode, True)
        for venue in venues
        if venue in narrated
        for mode in modes
    ]


@dataclass(frozen=True)
class ClipSpec:
    """Explicit inputs for key-driven rendering of a clip's variants.

    Venue budgets apply only to silent variants; speech determines narrated
    length. Each render starts from fresh default style state, then applies
    the variant mode and venue rc. Parent toggles such as ep.use_video are not
    inherited. For render_frames, pass style changes through setup and rc so
    both the in-process and spawned paths receive them explicitly.

    Attributes:
        slug: Clip identifier and output filename prefix.
        build: Picklable callable taking (venue, mode), returning (fig, draw).
        frames: Frame item sequence or integer count.
        venues: Tuple of destination settings.
        frame_at: Optional picklable index-to-item callable. Its behavior must
            be represented by clock or the named source inputs for keying.
        hold_s: Extra first/last frame holds in seconds.
        audio: Optional callable taking a Variant and returning an audio path.
        after: Optional callable taking (variant, out_dir), returning extra
            output paths. Runs for rendered and kept variants.
        script: Script text or a Path to source bytes for the key.
        modules: Module source inputs for the key.
        data: Tuple of data inputs, or a callable returning one per Variant.
        clock: Canonical clock inputs for the key.
        encoder: Optional encoder profile overrides.
        extra_ffmpeg_args: Optional additional encoding arguments.
        extra: Additional content key inputs, including stamp text if needed.
        poster: Optional frame index for each silent variant's PNG poster.
    """

    slug: str
    build: object
    frames: object
    venues: tuple[Venue, ...]
    frame_at: object = None
    hold_s: tuple[float, float] = (0.0, 0.0)
    audio: object = None
    after: object = None
    script: str | Path | None = None
    modules: tuple = ()
    data: object = ()
    clock: object = None
    encoder: dict | None = None
    extra_ffmpeg_args: object = None
    extra: tuple = ()
    poster: int | None = None


class VariantResult(NamedTuple):
    """One rendered or kept variant and its portable output identities.

    Attributes:
        variant: Destination, mode, and narration selection.
        outputs: POSIX output paths relative to out_dir.
        key: Content key computed before rendering.
        rendered: True when the video was rendered during this call.
        frames: Encoded frame count, including holds.
    """

    variant: Variant
    outputs: list[str]
    key: str
    rendered: bool
    frames: int


def _variant_video(spec, variant, venue, out_dir, key, hold, audio, workers):
    """Render once, atomically replacing a narrated target after mux succeeds."""
    import hwostyle

    target = out_dir / f"{spec.slug}-{variant.name}.mp4"
    silent = out_dir / f".{target.stem}.silent.mp4" if variant.narrated else target
    muxed = out_dir / f".{target.stem}.mux.mp4"
    try:
        result = render_frames(
            partial(spec.build, venue, variant.mode),
            spec.frames,
            silent,
            fps=venue.fps,
            dpi=venue.dpi,
            workers=workers,
            frame_at=spec.frame_at,
            hold=hold,
            encoder=spec.encoder,
            extra_ffmpeg_args=spec.extra_ffmpeg_args,
            setup=partial(hwostyle.use, variant.mode),
            rc=dict(venue.rc or {}),
            work_dir=out_dir / f".chunks-{spec.slug}-{variant.name}",
            key=key,
        )
        if variant.narrated:
            from eyepiece import narration

            narration.mux(silent, audio, muxed)
            os.replace(muxed, target)
        return target, result.frames
    finally:
        if variant.narrated:
            silent.unlink(missing_ok=True)
            muxed.unlink(missing_ok=True)


def render_variants(
    spec, variants, out_dir, *, manifest, workers=None, force=False
) -> list[VariantResult]:
    """Render each requested variant once, skipping current manifest entries.

    All destinations, silent budgets, poster indices, audio requirements, and
    keys are checked before any figure is built. Output paths and implicit
    stamps never enter keys. Duplicate requests are coalesced in their first
    occurrence's order. Silent variants also save the requested PNG poster.
    Completed entries are saved even if a later variant fails.
    A replaced variant's old entry is discarded before rendering, so failures
    cannot leave it marked current. Narrated variants are exempt from venue
    frame budgets because speech determines their length.

    Args:
        spec: Frozen ClipSpec describing explicit inputs and figure builders.
        variants: Iterable of Variant selections.
        out_dir: Folder for videos, posters, extra outputs, and per-variant chunks.
        manifest: Manifest whose root contains all outputs.
        workers: Worker count forwarded to render_frames.
        force: Render even when the manifest entry is current.

    Returns:
        Results in first-request order with paths relative to out_dir.

    Raises:
        ValueError: For duplicate or unknown venues, invalid modes or venue
            settings, exceeded silent frame budgets, missing narrated audio,
            invalid frames or poster indices, or outputs outside out_dir.
        TypeError: If an explicit key input is unsupported.
        RuntimeError: If rendering or audio muxing fails.
    """
    import hwostyle

    requested = list(dict.fromkeys(variants))
    venues = {venue.name: venue for venue in spec.venues}
    if len(venues) != len(spec.venues):
        raise ValueError("Venue names must be unique")
    count = _frame_source(spec.frames, spec.frame_at)
    if spec.poster is not None and (
        not isinstance(spec.poster, Integral) or not 0 <= spec.poster < count
    ):
        raise ValueError(f"poster must be a frame index in [0, {count})")
    for venue in venues.values():
        for name in ("dpi", "fps"):
            value = getattr(venue, name)
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    f"Venue {venue.name}: {name} must be positive and finite"
                )
        if venue.rc is not None and not isinstance(venue.rc, Mapping):
            raise ValueError(f"Venue {venue.name}: rc must be a mapping or None")
        try:
            matplotlib.RcParams(venue.rc or {})
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Venue {venue.name}: invalid rc settings: {exc}") from exc
    for variant in requested:
        if (
            not isinstance(variant.mode, str)
            or not variant.mode
            or variant.mode not in hwostyle.core.MODE_RC
        ):
            raise ValueError(f"Invalid style mode: {variant.mode}")
        if variant.venue not in venues:
            raise ValueError(f"Unknown venue: {variant.venue}")
        venue = venues[variant.venue]
        if (
            not variant.narrated
            and venue.frame_budget is not None
            and count > venue.frame_budget
        ):
            raise ValueError(
                f"{spec.slug} venue {venue.name}: {count} frames exceeds "
                f"budget {venue.frame_budget}"
            )
        if variant.narrated and spec.audio is None:
            raise ValueError(f"Narrated variant {variant.name} requires audio")
    jobs = []
    for variant in requested:
        venue = venues[variant.venue]
        hold = tuple(frame_count(seconds, venue.fps) for seconds in spec.hold_s)
        audio = Path(spec.audio(variant)) if variant.narrated else None
        extra = spec.extra
        if audio is not None:
            with audio.open("rb") as stream:
                extra = (*extra, hashlib.file_digest(stream, "sha256").hexdigest())
        key = content_key(
            script=spec.script,
            clock=spec.clock,
            modules=spec.modules,
            data=spec.data(variant) if callable(spec.data) else spec.data,
            settings={
                "variant": variant._asdict(),
                "fps": venue.fps,
                "dpi": venue.dpi,
                "rc": dict(venue.rc or {}),
                "hold": hold,
                "frames": spec.frames,
                "frame_at": spec.frame_at is not None,
                "encoder": spec.encoder,
                "extra_ffmpeg_args": spec.extra_ffmpeg_args,
                **({"poster": spec.poster} if spec.poster is not None else {}),
            },
            extra=extra,
        )
        jobs.append((variant, venue, hold, audio, key))
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        for variant, venue, hold, audio, key in jobs:
            entry = manifest.entry(spec.slug, variant.name)
            target = out_dir / f"{spec.slug}-{variant.name}.mp4"
            expected = [target]
            poster = out_dir / f"{spec.slug}-{variant.name}-poster.png"
            if spec.poster is not None and not variant.narrated:
                expected.append(poster)
            recorded = (
                []
                if entry is None
                else [manifest.root / output["path"] for output in entry["outputs"]]
            )
            rendered = (
                force
                or recorded[: len(expected)] != expected
                or any(not path.is_relative_to(out_dir) for path in recorded)
                or not manifest.current(spec.slug, variant.name, key)
            )
            if rendered:
                manifest.discard(spec.slug, variant.name)
                target, frames = _variant_video(
                    spec, variant, venue, out_dir, key, hold, audio, workers
                )
                outputs = [target]
                if spec.poster is not None and not variant.narrated:
                    outputs.append(
                        render_poster(
                            partial(spec.build, venue, variant.mode),
                            spec.frames,
                            spec.poster,
                            poster,
                            dpi=venue.dpi,
                            frame_at=spec.frame_at,
                            setup=partial(hwostyle.use, variant.mode),
                            rc=dict(venue.rc or {}),
                        )
                    )
            else:
                frames = entry["frames"]
                outputs = [
                    manifest.root / output["path"] for output in entry["outputs"]
                ]
            if spec.after is not None:
                outputs.extend(
                    path if path.is_absolute() else out_dir / path
                    for path in map(Path, spec.after(variant, out_dir))
                )
            outputs = list(dict.fromkeys(path.resolve() for path in outputs))
            relative = [path.relative_to(out_dir).as_posix() for path in outputs]
            manifest.record(
                spec.slug,
                variant.name,
                key=key,
                outputs=outputs,
                frames=frames,
                fps=venue.fps,
                dpi=venue.dpi,
            )
            results.append(VariantResult(variant, relative, key, rendered, frames))
    finally:
        manifest.save()
    return results


class RenderResult(NamedTuple):
    """An encoded clip and the work used to produce it.

    Attributes:
        path: Output path.
        frames: Encoded frame count, including holds.
        chunks: Number of chunks in this run.
        resumed: Number of validated chunks reused.
        workers: Worker count, capped by the number of chunks.
        seconds: Elapsed wall time in seconds.
    """

    path: Path
    frames: int
    chunks: int
    resumed: int
    workers: int
    seconds: float


def _frame_source(frames, frame_at):
    """Validate a sized, indexable source without evaluating frame_at."""
    if frame_at is not None and not isinstance(frames, int):
        raise ValueError("frame_at requires an int frames count")
    if isinstance(frames, int):
        count = frames
    else:
        count = len(frames)
        if not hasattr(frames, "__getitem__"):
            raise ValueError("frames must be a sequence or an int")
    if count < 1:
        raise ValueError("frames must not be empty")
    return count


def _item(state, index):
    if state["frame_at"] is not None:
        return state["frame_at"](index)
    frames = state["frames"]
    return index if isinstance(frames, int) else frames[index]


@contextmanager
def _default_style():
    """Reset style globals to fresh-interpreter defaults and restore on exit."""
    import hwostyle

    from eyepiece import _style

    # use() rebinds these globals. There is no public context manager that can
    # restore the unactivated state or a caller's exact palette objects.
    names = (
        "_current_mode",
        "_current_family",
        "_activated",
        "palette",
        "cmaps",
        "roles",
    )
    saved = {name: getattr(hwostyle.core, name) for name in names}
    video = _style.use_video(False)
    try:
        # hwostyle initializes dark objects without activating a mode. The
        # enclosing rc_context separately establishes default matplotlib rc.
        hwostyle.use("dark")
        hwostyle.core._activated = False
        yield
    finally:
        for name, value in saved.items():
            setattr(hwostyle.core, name, value)
        _style.use_video(video)


def _close_new_figures(existing):
    """Close pyplot figures created since the manager snapshot was taken."""
    for manager in Gcf.get_all_fig_managers():
        if manager not in existing:
            plt.close(manager.canvas.figure)


@contextmanager
def _figure(build, frames, dpi, frame_at, setup, rc):
    """Build and freeze a figure under the same state in every process."""
    with ExitStack() as stack:
        stack.enter_context(matplotlib.rc_context())
        stack.enter_context(_default_style())
        # Spawned interpreters start with defaults. Also reset the in-process
        # path so unpassed parent rc does not silently change its pixels.
        stack.enter_context(matplotlib.rc_context(matplotlib.rcParamsDefault))
        stack.callback(_close_new_figures, set(Gcf.get_all_fig_managers()))
        if setup is not None:
            setup()
        stack.enter_context(matplotlib.rc_context(rc))
        stack.enter_context(matplotlib.rc_context(_rc_overrides([], snap=False)))
        fig, draw = build()
        stack.callback(plt.close, fig)
        stack.enter_context(_agg_canvas(fig))
        state = dict(fig=fig, draw=draw, frames=frames, frame_at=frame_at)
        draw(fig, _item(state, 0))
        _Recorder(fig, [], {"frames": 0}, dpi)._freeze_layout()
        fig.dpi = dpi
        width, height = fig.canvas.get_width_height()
        state["size"] = (width // 2 * 2, height // 2 * 2)
        yield state


def _watch_parent(parent_pid):
    """Exit when the parent dies or closes its private lifetime pipe."""
    import select

    while os.getppid() == parent_pid:
        readable, _, _ = select.select([sys.stdin.buffer], [], [], 0.25)
        if readable and not os.read(sys.stdin.fileno(), 1):
            break
    os._exit(1)


def _check_parent(state):
    parent_pid = state.get("parent_pid")
    if parent_pid is not None and os.getppid() != parent_pid:
        os._exit(1)


def _raster(state, index):
    _check_parent(state)
    fig = state["fig"]
    state["draw"](fig, _item(state, index))
    fig.canvas.draw()
    width, height = state["size"]
    return np.asarray(fig.canvas.buffer_rgba())[:height, :width].copy()


def render_poster(
    build, frames, index, path, *, dpi, frame_at=None, setup=None, rc=None
) -> Path:
    """Atomically save a worker's even-cropped frame pixels as a PNG.

    Layout is frozen after drawing frame zero, exactly as for a video worker.
    The PNG contains RGBA pixels without metadata. Figure, style, and rc state
    are restored even when drawing or writing fails.

    Args:
        build: Callable returning (fig, draw), as for render_frames.
        frames: Frame item sequence or integer frame count.
        index: Zero-based frame index before any video holds.
        path: Destination PNG; parent directories are created as needed.
        dpi: Rasterization dots per inch.
        frame_at: Optional index-to-item callable for an integer frame count.
        setup: Callable run before applying rc and building the figure.
        rc: Matplotlib overrides held open during building and drawing.

    Returns:
        The destination Path.

    Raises:
        ValueError: For a non-PNG suffix, invalid frame source, or nonpositive dpi.
        IndexError: If index lies outside the frame source.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("render_poster requires a .png path")
    count = _frame_source(frames, frame_at)
    index = operator.index(index)
    if not 0 <= index < count:
        raise IndexError(f"poster frame {index} outside [0, {count})")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with _figure(build, frames, dpi, frame_at, setup, rc) as state:
        pixels = _raster(state, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def render_still(build, path, *, dpi=None, setup=None, rc=None, metadata=None) -> Path:
    """Build and save a deterministic still through eyepiece.save_fig.

    Default style state, setup, and rc are applied in the same order as for
    workers. The built figure is closed and the parent's state restored on exit.

    Args:
        build: Callable returning a matplotlib Figure.
        path: Destination PNG, PDF, or SVG; parent directories are created.
        dpi: PNG resolution override; ignored for vector formats.
        setup: Callable run before applying rc and building the figure.
        rc: Matplotlib overrides held open during building and saving.
        metadata: Optional format-specific metadata passed to save_fig.

    Returns:
        The saved Path returned by save_fig.
    """
    path = Path(path)
    with ExitStack() as stack:
        stack.enter_context(matplotlib.rc_context())
        stack.enter_context(_default_style())
        stack.enter_context(matplotlib.rc_context(matplotlib.rcParamsDefault))
        stack.callback(_close_new_figures, set(Gcf.get_all_fig_managers()))
        if setup is not None:
            setup()
        stack.enter_context(matplotlib.rc_context(rc))
        fig = build()
        stack.callback(plt.close, fig)
        overrides = {}
        if dpi is not None and path.suffix.lower() in ("", ".png"):
            overrides["dpi"] = dpi
        if metadata is not None:
            overrides["metadata"] = metadata
        return eyepiece.save_fig(fig, path.name, dir=path.parent, **overrides)


def same_picture(a, b, *, tol=2, ignore_bottom_px=0) -> bool:
    """Compare PNG RGBA levels, excluding an optional bottom band.

    Images without alpha are treated as fully opaque.

    Args:
        a: First PNG path.
        b: Second PNG path.
        tol: Maximum allowed absolute difference in any RGBA channel, inclusive.
        ignore_bottom_px: Number of bottom rows excluded from comparison.

    Returns:
        Whether shapes match and every compared RGBA difference is within tol.

    Raises:
        ValueError: If either argument is negative or the bottom band excludes
            the entire image.
    """
    if tol < 0 or ignore_bottom_px < 0:
        raise ValueError("tol and ignore_bottom_px must be nonnegative")
    with Image.open(a) as image:
        first = np.array(image.convert("RGBA"))
    with Image.open(b) as image:
        second = np.array(image.convert("RGBA"))
    if ignore_bottom_px >= min(first.shape[0], second.shape[0]):
        raise ValueError("ignore_bottom_px must be less than the image height")
    if first.shape != second.shape:
        return False
    rows = first.shape[0] - ignore_bottom_px
    difference = first[:rows].astype(np.int16) - second[:rows].astype(np.int16)
    return bool(np.max(np.abs(difference)) <= tol)


def _identity(key, count, hold, dpi, size, fps, start, stop):
    values = [key, _SCHEMA, count, list(hold), dpi, size, fps, start, stop]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).digest()


def _write_chunk(job, state):
    """Stream a chunk to a partial file and publish it only when complete."""
    path, identity, start, stop = job
    _check_parent(state)
    partial = path.with_suffix(".part")
    lengths = []
    crc = 0
    try:
        with partial.open("wb") as stream:
            stream.write(bytes(_HEADER.size + _LENGTH.size * (stop - start)))
            for index in range(start, stop):
                payload = zlib.compress(_raster(state, index).tobytes(), level=1)
                lengths.append(len(payload))
                crc = zlib.crc32(payload, crc)
                stream.write(payload)
            stream.seek(0)
            length_data = b"".join(_LENGTH.pack(length) for length in lengths)
            stream.write(
                _HEADER.pack(
                    _MAGIC,
                    _SCHEMA,
                    identity,
                    len(lengths),
                    crc,
                    zlib.crc32(length_data),
                )
            )
            stream.write(length_data)
        _check_parent(state)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path


def _chunk_lengths(path, identity, count):
    """Return lengths only if the identity, size, and payload CRC agree."""
    try:
        with path.open("rb") as stream:
            magic, schema, stored, nframes, crc, lengths_crc = _HEADER.unpack(
                stream.read(_HEADER.size)
            )
            if (magic, schema, stored, nframes) != (_MAGIC, _SCHEMA, identity, count):
                return None
            length_data = stream.read(_LENGTH.size * count)
            if len(length_data) != _LENGTH.size * count:
                return None
            if zlib.crc32(length_data) != lengths_crc:
                return None
            lengths = [entry[0] for entry in _LENGTH.iter_unpack(length_data)]
            if any(length == 0 for length in lengths):
                return None
            if path.stat().st_size != stream.tell() + sum(lengths):
                return None
            actual = 0
            while payload := stream.read(1024 * 1024):
                actual = zlib.crc32(payload, actual)
            return lengths if actual == crc else None
    except (FileNotFoundError, struct.error):
        return None


def _read_chunk(job, size):
    path, identity, start, stop = job
    lengths = _chunk_lengths(path, identity, stop - start)
    if lengths is None:
        raise RuntimeError(f"Invalid chunk for frames [{start}, {stop})")
    width, height = size
    with path.open("rb") as stream:
        stream.seek(_HEADER.size + _LENGTH.size * len(lengths))
        for length in lengths:
            raw = zlib.decompress(stream.read(length))
            if len(raw) != width * height * 4:
                raise RuntimeError(f"Invalid raster for frames [{start}, {stop})")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


def _pid_alive(pid):
    if pid < 1:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


@contextmanager
def _workspace(folder):
    """Claim an owned directory and release its exclusive pid lock on exit."""
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / _MARKER
    if not marker.exists():
        if any(folder.iterdir()):
            raise RuntimeError(
                f"Refusing non-empty folder without ownership marker: {folder}"
            )
        try:
            with marker.open("x") as stream:
                stream.write(_MARKER_TEXT)
        except FileExistsError:
            pass
    if marker.read_text() != _MARKER_TEXT:
        raise RuntimeError(f"Unrecognized ownership marker in {folder}")
    lock = folder / "render.lock"
    candidate = folder / f"lock-{os.getpid()}-{uuid.uuid4().hex}.part"
    try:
        with candidate.open("x") as stream:
            stream.write(str(os.getpid()))
        while True:
            try:
                # Exclusive linking publishes a fully written pid atomically.
                os.link(candidate, lock)
                break
            except FileExistsError:
                # Serialize stale-lock takeover, checking the inode before
                # deleting so a competing owner cannot lose its lock.
                import fcntl

                try:
                    with lock.open() as stale:
                        fcntl.flock(stale, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        info = os.fstat(stale.fileno())
                        if info.st_ino != lock.stat().st_ino:
                            continue
                        try:
                            pid = int(stale.read())
                        except ValueError:
                            # Older interrupted runs may leave empty locks.
                            # Give a recent writer time to finish first.
                            if time.time() - info.st_mtime < _LOCK_GRACE_SECONDS:
                                raise RuntimeError(
                                    f"Render folder has an unreadable lock: {folder}"
                                ) from None
                        else:
                            if _pid_alive(pid):
                                raise RuntimeError(
                                    f"Render folder is locked by pid {pid}: {folder}"
                                ) from None
                        lock.unlink()
                except FileNotFoundError:
                    continue
                except BlockingIOError:
                    raise RuntimeError(
                        f"Render folder lock is being taken over: {folder}"
                    ) from None
    finally:
        candidate.unlink(missing_ok=True)
    try:
        for path in folder.iterdir():
            lock_part = _LOCK_PART_NAME.fullmatch(path.name)
            if (
                _PROTOCOL_NAME.fullmatch(path.name)
                or (_CHUNK_NAME.fullmatch(path.name) and path.suffix == ".part")
                or (lock_part is not None and not _pid_alive(int(lock_part[1])))
            ):
                path.unlink()
        yield
    finally:
        lock.unlink(missing_ok=True)


def _worker_main(folder, worker_index):
    """Load one trusted local job, build once, and claim chunks until done."""
    active = None
    try:
        threading.Thread(
            target=_watch_parent, args=(os.getppid(),), daemon=True
        ).start()
        matplotlib.use("Agg")
        with (folder / "job.pkl").open("rb") as stream:
            # Bootstrap import paths before unpickling caller-defined objects.
            sys.path[:0] = pickle.load(stream)
            job = pickle.load(stream)
        parent_pid = job["parent_pid"]
        _check_parent({"parent_pid": parent_pid})
        with _figure(**job["figure"]) as state:
            state["parent_pid"] = parent_pid
            ready = folder / f"worker-{worker_index}.ready"
            partial = ready.with_suffix(".part")
            partial.write_text(json.dumps(state["size"]))
            partial.replace(ready)
            tasks = folder / "tasks.pkl"
            while not tasks.exists():
                _check_parent(state)
                time.sleep(0.02)
            with tasks.open("rb") as stream:
                jobs = pickle.load(stream)
            for index, task in jobs:
                _check_parent(state)
                claim = folder / f"claim-{job['run']}-{index}"
                try:
                    with claim.open("x") as stream:
                        stream.write(str(worker_index))
                except FileExistsError:
                    continue
                active = index
                _write_chunk(task, state)
                active = None
    except BaseException:
        message = traceback.format_exc()
        name = f"chunk-{active}" if active is not None else f"worker-{worker_index}"
        # If writing the error also fails, stderr still carries the traceback.
        print(message, file=sys.stderr, flush=True)
        error = folder / f"{name}.error"
        partial = error.with_suffix(".error.part")
        partial.write_text(message)
        partial.replace(error)
        return 1
    return 0


class _Workers:
    """Own subprocesses, their lifetime pipes, and their on-disk protocol."""

    def __init__(self, folder, workers, figure, first_stop):
        self.folder = folder
        self.run = uuid.uuid4().hex
        self.processes = []
        self.jobs = []
        self.completed = set()
        self.first_stop = first_stop
        self.figure = figure
        self.workers = workers

    def __enter__(self):
        try:
            with (self.folder / "job.pkl").open("wb") as stream:
                pickle.dump([os.path.abspath(p) for p in sys.path], stream)
                pickle.dump(
                    dict(figure=self.figure, parent_pid=os.getpid(), run=self.run),
                    stream,
                )
            for index in range(self.workers):
                with (self.folder / f"worker-{index}.log").open("wb") as log:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "eyepiece.render",
                            str(self.folder),
                            str(index),
                        ],
                        stdin=subprocess.PIPE,
                        stdout=log,
                        stderr=log,
                    )
                self.processes.append(process)
            while True:
                self.check()
                ready = [self.folder / f"worker-{i}.ready" for i in range(self.workers)]
                if all(path.exists() for path in ready):
                    sizes = [tuple(json.loads(path.read_text())) for path in ready]
                    if any(size != sizes[0] for size in sizes):
                        raise RuntimeError("Worker figure sizes disagree")
                    self.size = sizes[0]
                    return self
                time.sleep(0.02)
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
        for process in self.processes:
            process.wait()
            process.stdin.close()

    def start_jobs(self, jobs, reused):
        self.jobs = jobs
        partial = self.folder / "tasks.part"
        with partial.open("wb") as stream:
            pickle.dump(
                [(i, job) for i, job in enumerate(jobs) if job[0] not in reused], stream
            )
        partial.replace(self.folder / "tasks.pkl")

    def _failure(self, index, message):
        start, stop = self.jobs[index][2:] if self.jobs else (0, self.first_stop)
        raise RuntimeError(f"Rendering chunk frames [{start}, {stop}):\n{message}")

    def check(self):
        """Surface errors promptly, including crashes without an error file."""
        for error in sorted(self.folder.glob("*.error")):
            if not _PROTOCOL_NAME.fullmatch(error.name):
                continue
            if error.name.startswith("chunk-"):
                self._failure(int(error.stem.split("-")[1]), error.read_text())
            if error.name.startswith("worker-"):
                self._failure(0, error.read_text())
        for index, process in enumerate(self.processes):
            status = process.poll()
            if status is not None:
                active = None
                for claim in self.folder.glob(f"claim-{self.run}-*"):
                    if claim.read_text() == str(index):
                        k = int(claim.name.rsplit("-", 1)[1])
                        if k not in self.completed and not self.jobs[k][0].exists():
                            active = k
                            break
                if status != 0 or not self.jobs or active is not None:
                    log = self.folder / f"worker-{index}.log"
                    self._failure(
                        active if active is not None else 0,
                        f"Worker {index} exited ({status}):\n"
                        + log.read_text(errors="replace"),
                    )


def _ordered_chunks(pool, jobs, reused, state):
    """Yield chunks in order, observing failures from every subprocess."""
    if pool is None:
        for job in jobs:
            if job[0] not in reused:
                try:
                    _write_chunk(job, state)
                except BaseException as exc:
                    exc.add_note(f"Rendering chunk frames [{job[2]}, {job[3]})")
                    raise
            yield job
        return
    pool.start_jobs(jobs, reused)
    for index, job in enumerate(jobs):
        while True:
            pool.check()
            if job[0].exists():
                pool.completed.add(index)
                yield job
                break
            if all(p.poll() is not None for p in pool.processes):
                # A worker may publish and exit after the existence check.
                # Once every worker is reaped the folder can be scanned again
                # without racing any more chunk or error publications.
                pool.check()
                if job[0].exists():
                    pool.completed.add(index)
                    yield job
                    break
                pool._failure(index, "Workers exited without a complete chunk")
            time.sleep(0.02)
    # A late worker failure must be observed before RawSink commits the output.
    while any(p.poll() is None for p in pool.processes):
        pool.check()
        time.sleep(0.02)
    pool.check()


@contextmanager
def _encoding_sink(path, size, fps, encoder, extra_ffmpeg_args):
    """Abort and clean up even when the sink fails during finalization."""
    sink = RawSink(
        path, size, fps, encoder=encoder, extra_ffmpeg_args=extra_ffmpeg_args
    )
    try:
        yield sink
    except BaseException:
        sink.__exit__(*sys.exc_info())
        raise
    else:
        try:
            sink.close()
        except BaseException as exc:
            # close() may fail flushing stdin, before waiting or cleaning up.
            # Read stderr after ffmpeg stops, before the abort closes its log.
            sink._proc.kill()
            sink._proc.wait()
            message = sink._stderr()
            sink.__exit__(type(exc), exc, exc.__traceback__)
            if isinstance(exc, Exception):
                raise RuntimeError(
                    f"ffmpeg failed writing {path}: {message or exc}"
                ) from exc
            raise


def _encode(
    pool,
    state,
    size,
    frames,
    path,
    fps,
    dpi,
    hold,
    chunk_frames,
    folder,
    key,
    resume,
    keep_chunks,
    encoder,
    extra_ffmpeg_args,
):
    count = _frame_source(frames, None)
    jobs = []
    reused = set()
    for index, start in enumerate(range(0, count, chunk_frames)):
        stop = min(count, start + chunk_frames)
        chunk_path = folder / f"chunk-{index}.bin"
        identity = _identity(key, count, hold, dpi, size, fps, start, stop)
        job = (chunk_path, identity, start, stop)
        jobs.append(job)
        if resume and _chunk_lengths(chunk_path, identity, stop - start) is not None:
            reused.add(chunk_path)
    # Delete only reserved files, including chunks from old partitions.
    for chunk_path in folder.iterdir():
        if _CHUNK_NAME.fullmatch(chunk_path.name) and chunk_path not in reused:
            chunk_path.unlink()
    with _encoding_sink(path, size, fps, encoder, extra_ffmpeg_args) as sink:
        for job in _ordered_chunks(pool, jobs, reused, state):
            for index, pixels in enumerate(_read_chunk(job, size), start=job[2]):
                sink.write(pixels)
                repeats = hold[0] if index == 0 else 0
                repeats += hold[1] if index == count - 1 else 0
                for _ in range(repeats):
                    sink.write(pixels)
            if not resume:
                job[0].unlink()
    if not keep_chunks or not resume:
        for job in jobs:
            job[0].unlink(missing_ok=True)
    return len(jobs), len(reused)


def render_frames(
    build,
    frames,
    path,
    *,
    fps,
    dpi,
    workers=None,
    chunk_frames=None,
    frame_at=None,
    hold=(0, 0),
    encoder=None,
    extra_ffmpeg_args=None,
    setup=None,
    rc=None,
    work_dir=None,
    key=None,
    resume=True,
    keep_chunks=False,
) -> RenderResult:
    """Render independent frames in chunks and encode them in index order.

    ``build()`` returns ``(fig, draw)``. Draw overwrites the state needed for
    an item with ``draw(fig, item)``. Builders, setup, and frame_at must be
    picklable module-level callables (partials are supported). Each worker
    builds once, draws frame zero, and freezes the layout at the sink dpi.
    A fresh worker starts from default style state. Parent toggles such as
    ep.use_video are not inherited, including with workers=1. Pass style
    changes through setup and rc to apply them identically in both paths.
    The in-process path restores rcParams, hwostyle globals, and the video
    style switch when it exits, including after a setup or build failure.

    Args:
        build: Deterministic figure and draw builder.
        frames: A sequence of picklable items, or an integer frame count.
        path: Destination mp4, replaced only after successful encoding.
        fps: Encoded frames per second.
        dpi: Rasterization dots per inch.
        workers: Process count; None uses available CPUs up to chunk count.
            One runs the same worker functions in the parent process.
        chunk_frames: Frames per chunk; None targets four chunks per worker,
            with at least eight frames per chunk.
        frame_at: Optional index-to-item callable, evaluated in the worker.
            Requires an integer frames count.
        hold: Extra repetitions of the first and last raster, respectively.
        encoder: Overrides for the RawSink encoder profile.
        extra_ffmpeg_args: Additional output arguments for RawSink.
        setup: Callable run before rc and build in each worker.
        rc: Matplotlib rc mapping held open while building and drawing.
        work_dir: Owned chunk folder; defaults to .<stem>.chunks beside path.
            Reserves .eyepiece-render, render.lock, lock-<pid>-<run>.part,
            job.pkl, tasks.pkl,
            chunk files, run claims, and worker readiness and log files.
        key: Caller input identity string; None disables chunk reuse.
        resume: Reuse complete chunks with matching identity and CRC.
        keep_chunks: Retain chunks after success when reuse is enabled.

    Returns:
        Output path, frame and chunk counts, reused chunks, workers, and time.

    Raises:
        ValueError: For invalid frame sources, suffix, counts, or settings.
        TypeError: If key is neither a string nor None.
        RuntimeError: If the chunk folder is unowned or locked, or a subprocess
            fails. Worker failures include their chunk range and traceback or
            stderr. Completed chunks remain available for resume.
    """
    started = time.monotonic()
    path = Path(path)
    count = _frame_source(frames, frame_at)
    if path.suffix.lower() != ".mp4":
        raise ValueError("render_frames requires an .mp4 path")
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive int")
    requested = workers if workers is not None else (os.cpu_count() or 1)
    if chunk_frames is None:
        chunk_frames = max(8, math.ceil(count / (4 * requested)))
    if not isinstance(chunk_frames, int) or chunk_frames < 1:
        raise ValueError("chunk_frames must be a positive int")
    if len(hold) != 2 or any(not isinstance(n, int) or n < 0 for n in hold):
        raise ValueError("hold must contain two nonnegative integers")
    if dpi <= 0 or fps <= 0:
        raise ValueError("dpi and fps must be positive")
    if key is not None and not isinstance(key, str):
        raise TypeError("key must be a string or None")
    workers = min(requested, math.ceil(count / chunk_frames))
    folder = (
        Path(work_dir) if work_dir is not None else path.parent / f".{path.stem}.chunks"
    )
    folder = folder.resolve()
    resume = bool(resume and key is not None)
    with _workspace(folder), ExitStack() as stack:
        pool = None
        state = None
        try:
            if workers == 1:
                state = stack.enter_context(
                    _figure(build, frames, dpi, frame_at, setup, rc)
                )
                size = state["size"]
            else:
                pool = stack.enter_context(
                    _Workers(
                        folder,
                        workers,
                        dict(
                            build=build,
                            frames=frames,
                            dpi=dpi,
                            frame_at=frame_at,
                            setup=setup,
                            rc=rc,
                        ),
                        min(count, chunk_frames),
                    )
                )
                size = pool.size
            chunks, resumed = _encode(
                pool,
                state,
                size,
                frames,
                path,
                fps,
                dpi,
                hold,
                chunk_frames,
                folder,
                key,
                resume,
                keep_chunks,
                encoder,
                extra_ffmpeg_args,
            )
        except BaseException as exc:
            if "Rendering chunk" not in str(exc) and not any(
                "Rendering chunk" in note for note in getattr(exc, "__notes__", ())
            ):
                exc.add_note(f"Rendering chunk frames [0, {min(count, chunk_frames)})")
            raise
    return RenderResult(
        path, count + sum(hold), chunks, resumed, workers, time.monotonic() - started
    )


def frame_faults(
    build, frames, *, dpi, frame_at=None, setup=None, rc=None, sample=4
) -> list[int]:
    """Find sampled frame indices whose pixels depend on prior draws.

    This diagnostic is not a proof of purity. Sampled frames are drawn in
    order for a baseline. Each is then drawn in a fresh figure after frame
    zero and a different frame, and compared with its baseline pixels.

    Args:
        build: Deterministic callable returning a figure and draw function.
        frames: Sequence of items or integer count.
        dpi: Rasterization dots per inch.
        frame_at: Optional index-to-item callable for an integer count.
        setup: Callable run before rc and build.
        rc: Matplotlib rc mapping for building and drawing.
        sample: Maximum number of evenly spaced indices to check.

    Returns:
        Sampled indices whose pixels differ from the ordered baseline.

    Raises:
        ValueError: If frames is empty, frame_at is incompatible, or sample
            is not a positive integer.
    """
    count = _frame_source(frames, frame_at)
    if not isinstance(sample, int) or sample < 1:
        raise ValueError("sample must be a positive int")
    indices = np.linspace(0, count - 1, min(sample, count), dtype=int).tolist()
    with _figure(build, frames, dpi, frame_at, setup, rc) as state:
        baseline = [_raster(state, index) for index in indices]
    faults = []
    for index, expected in zip(indices, baseline, strict=True):
        with _figure(build, frames, dpi, frame_at, setup, rc) as state:
            if count > 1:
                _raster(state, (index + 1) % count)
            if not np.array_equal(_raster(state, index), expected):
                faults.append(index)
    return faults


def assert_frames_pure(
    build, frames, *, dpi, frame_at=None, setup=None, rc=None, sample=4
) -> None:
    """Raise if sampled frame pixels depend on their draw history.

    Args:
        build: Deterministic callable returning a figure and draw function.
        frames: Sequence of items or integer count.
        dpi: Rasterization dots per inch.
        frame_at: Optional index-to-item callable for an integer count.
        setup: Callable run before rc and build.
        rc: Matplotlib rc mapping for building and drawing.
        sample: Maximum number of evenly spaced indices to check.

    Raises:
        AssertionError: Naming sampled indices with differing pixels.
    """
    faults = frame_faults(
        build, frames, dpi=dpi, frame_at=frame_at, setup=setup, rc=rc, sample=sample
    )
    if faults:
        raise AssertionError(f"Draw depends on prior frames at indices {faults}")


if __name__ == "__main__":
    sys.exit(_worker_main(Path(sys.argv[1]), int(sys.argv[2])))
