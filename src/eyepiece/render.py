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
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zlib
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import eyepiece
from eyepiece import _ffmpeg
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
        A SHA-256 hex digest including eyepiece, matplotlib, numpy, and ffmpeg
        versions. An unavailable eyepiece or ffmpeg version is recorded as None.

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
            "ffmpeg": _ffmpeg_version(),
        },
    }
    encoded = _key_field(b"eyepiece.content-key.v1", _canonical(values))
    return hashlib.sha256(encoded).hexdigest()


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
def _figure(build, frames, dpi, frame_at, setup, rc):
    """Build and freeze a figure under the same state in every process."""
    with ExitStack() as stack:
        # Spawned interpreters start with defaults. Also reset the in-process
        # path so unpassed parent rc does not silently change its pixels.
        stack.enter_context(matplotlib.rc_context(matplotlib.rcParamsDefault))
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
