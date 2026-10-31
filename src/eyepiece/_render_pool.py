"""Spawned frame workers, rasterization, chunk protocol, and encoding."""

import io
import json
import math
import os
import pickle
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
import warnings
import zlib
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._pylab_helpers import Gcf
from matplotlib.animation import adjusted_figsize

from eyepiece._render_chunks import (
    _CHUNK_NAME,
    _HEADER,
    _LENGTH,
    _MAGIC,
    _PROTOCOL_NAME,
    _SCHEMA,
    _chunk_lengths,
    _delete_chunk,
    _identity,
    _read_chunk,
    _read_scale_diagnostic,
    _scale_diagnostic,
    _workspace,
)
from eyepiece.anim import (
    RawSink,
    ScaleDriftError,
    _agg_canvas,
    _drift_warnings,
    _rc_overrides,
    _Recorder,
    _scale_signature,
)


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
def _figure(
    build, frames, dpi, frame_at, setup, rc, on_rescale="ignore", frame_key=None
):
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
        fig, draw = build()
        stack.enter_context(matplotlib.rc_context(_rc_overrides([], snap=False)))
        stack.callback(plt.close, fig)
        stack.enter_context(_agg_canvas(fig))
        # MovieWriter adjusts the figure before the caller draws frame zero.
        # Use the same roundoff correction as record's H.264 writer.
        fig.set_size_inches(adjusted_figsize(*fig.get_size_inches(), dpi, 2))
        state = dict(
            fig=fig,
            draw=draw,
            frames=frames,
            frame_at=frame_at,
            dpi=dpi,
            frame_key=frame_key,
        )
        state["initial_item"] = _item(state, 0)
        draw(fig, state["initial_item"])
        state["initial_draw"] = True
        _Recorder(fig, [], {"frames": 0}, dpi)._freeze_layout()
        state["on_rescale"] = on_rescale
        if on_rescale != "ignore":
            # Norm autoscaling and aspect adjustments settle during drawing.
            # Take the baseline at the same point as every checked raster.
            fig.canvas.draw()
            state["first_scales"] = _scale_signature(fig)
        state["size"] = tuple(int(inches * dpi) for inches in fig.get_size_inches())
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
    initial_draw = state.pop("initial_draw", False)
    item = state["initial_item"] if index == 0 and initial_draw else _item(state, index)
    key = state["frame_key"](item) if state["frame_key"] is not None else None
    fig = state["fig"]
    if index != 0 or not initial_draw:
        state["draw"](fig, item)
    if key is not None:
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("frame_key must return a hashable value or None") from exc
        previous = state.get("previous_frame")
        if previous is not None and index == previous[0] + 1 and key == previous[1]:
            _check_scales(state, index)
            state["previous_frame"] = (index, key, previous[2])
            return previous[2]
    # Frame zero was already drawn before layout froze. A second draw could
    # measure the settled axes and move labels compared with record.
    # record invokes draw at the figure dpi, then savefig at the sink dpi.
    # Saving also applies facecolor and transparency just as MovieWriter does.
    with io.BytesIO() as buffer:
        fig.savefig(
            buffer, format="rgba", dpi=state["dpi"], facecolor=fig.get_facecolor()
        )
        width, height = state["size"]
        pixels = np.frombuffer(buffer.getvalue(), dtype=np.uint8).reshape(
            height, width, 4
        )
    _check_scales(state, index)
    if key is not None:
        state["previous_frame"] = (index, key, pixels)
    else:
        state.pop("previous_frame", None)
    return pixels


def _validate_on_rescale(on_rescale):
    """Reject unsupported scale policies before creating any render state."""
    if on_rescale not in ("warn", "raise", "ignore"):
        raise ValueError(
            f"on_rescale must be one of ('warn', 'raise', 'ignore'), got {on_rescale!r}"
        )


def _check_scales(state, index):
    """Compare each chunk's frames against the frozen frame-zero data scales."""
    if state["on_rescale"] == "ignore" or state.get("scale_warning"):
        return
    messages = _drift_warnings(state["first_scales"], _scale_signature(state["fig"]))
    if not messages:
        return
    text = (
        "data scales changed during recording, so a quantity is drawn at "
        "a different size in different frames and the change will read as "
        "a change in the data: "
        + "; ".join(messages)
        + ". Pin the limits from the union over all frames before the "
        'frame loop, or pass on_rescale="ignore" if the rescaling is the '
        f"point. (frame {index})"
    )
    if state["on_rescale"] == "raise":
        raise ScaleDriftError(text)
    state["scale_warning"] = text


def _write_chunk(job, state):
    """Stream a chunk to a partial file and publish it only when complete."""
    path, identity, start, stop = job
    _check_parent(state)
    partial = path.with_suffix(".part")
    lengths = []
    crc = 0
    state["scale_warning"] = ""
    warning = path.with_suffix(".warn")
    warning_partial = path.with_suffix(".warn.part")
    try:
        with partial.open("wb") as stream:
            stream.write(bytes(_HEADER.size + _LENGTH.size * (stop - start)))
            previous_pixels = None
            for index in range(start, stop):
                pixels = _raster(state, index)
                if pixels is not previous_pixels:
                    payload = zlib.compress(memoryview(pixels), level=1)
                    previous_pixels = pixels
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
        if state["on_rescale"] != "ignore":
            # Every checked chunk has a sidecar, with an empty message if its
            # scales stayed fixed. Publish it before publishing the raster.
            diagnostic = _scale_diagnostic(identity, crc, state["scale_warning"])
            warning_partial.write_text(
                json.dumps(diagnostic, sort_keys=True) + "\n", encoding="utf-8"
            )
            warning_partial.replace(warning)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        warning_partial.unlink(missing_ok=True)
        warning.unlink(missing_ok=True)
        raise
    return path


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
        error_type = (
            ScaleDriftError
            if re.search(r"(?m)^(?:eyepiece\.anim\.)?ScaleDriftError:", message)
            else RuntimeError
        )
        raise error_type(f"Rendering chunk frames [{start}, {stop}):\n{message}")

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
                except ScaleDriftError as exc:
                    raise ScaleDriftError(
                        f"Rendering chunk frames [{job[2]}, {job[3]}): {exc}"
                    ) from exc
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
    # Buffer small rasters together. Large frames pass straight through without
    # a staging copy, and close() flushes the final partial buffer before wait().
    sink._proc.stdin = io.BufferedWriter(sink._proc.stdin, buffer_size=1024 * 1024)
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


def _write_frame(sink, pixels):
    """Write a validated, contiguous worker raster without RawSink's conversion."""
    try:
        sink._proc.stdin.write(memoryview(pixels).cast("B"))
    except BrokenPipeError:
        raise RuntimeError(
            f"ffmpeg stopped accepting frames: {sink._stderr()}"
        ) from None
    sink.frames += 1


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
    on_rescale,
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
        if (
            resume
            and _chunk_lengths(chunk_path, identity, stop - start) is not None
            and (on_rescale == "ignore" or _read_scale_diagnostic(job) is not None)
        ):
            reused.add(chunk_path)
    # Delete only reserved files, including chunks from old partitions.
    for chunk_path in folder.iterdir():
        if (
            _CHUNK_NAME.fullmatch(chunk_path.name)
            and chunk_path.with_suffix(".bin") not in reused
        ):
            _delete_chunk(chunk_path)
    warned = False
    with _encoding_sink(path, size, fps, encoder, extra_ffmpeg_args) as sink:
        for job in _ordered_chunks(pool, jobs, reused, state):
            if on_rescale != "ignore":
                message = _read_scale_diagnostic(job)
                if message is None:
                    raise RuntimeError(
                        f"Invalid scale diagnostic for frames [{job[2]}, {job[3]})"
                    )
                if message:
                    if on_rescale == "raise":
                        raise ScaleDriftError(
                            f"Rendering chunk frames [{job[2]}, {job[3]}): {message}"
                        )
                    if not warned:
                        warnings.warn(message, RuntimeWarning, stacklevel=3)
                        warned = True
            for index, pixels in enumerate(_read_chunk(job, size), start=job[2]):
                _write_frame(sink, pixels)
                repeats = hold[0] if index == 0 else 0
                repeats += hold[1] if index == count - 1 else 0
                for _ in range(repeats):
                    _write_frame(sink, pixels)
            if not resume:
                _delete_chunk(job[0])
    if not keep_chunks or not resume:
        for job in jobs:
            _delete_chunk(job[0])
    return len(jobs), len(reused)


def _available_cpus():
    """Respect process CPU restrictions when the platform exposes them."""
    process_count = getattr(os, "process_cpu_count", None)
    if process_count is not None:
        return process_count() or 1
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _available_memory():
    """Return free physical bytes without an optional process dependency."""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return pages * page_size if pages > 0 and page_size > 0 else None


def _default_workers():
    """Reserve a CPU and 1 GiB for encoding, then budget 1 GiB per worker."""
    memory = _available_memory()
    memory_workers = max(1, memory // 1024**3 - 1) if memory is not None else 1
    return min(8, max(1, _available_cpus() - 1), memory_workers)


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
    on_rescale="warn",
    frame_key=None,
) -> RenderResult:
    """Render independent frames in chunks and encode them in index order.

    ``build()`` returns ``(fig, draw)``. Draw overwrites the state needed for
    an item with ``draw(fig, item)``. Builders, setup, and frame_at must be
    picklable module-level callables (partials are supported). Each worker
    builds once before recording rc overrides, resizes the figure to even
    sink pixels, draws frame zero at the figure dpi, and freezes layout at
    the sink dpi. Each later draw runs at the figure dpi; capture uses savefig
    at the sink dpi, matching record's default H.264 writer.
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
        workers: Process count; None uses available CPUs and free memory,
            reserving one CPU and 1 GiB for encoding and allowing 1 GiB per
            worker, up to eight workers and the chunk count. If memory cannot
            be probed, uses one worker. Explicit counts override this heuristic.
            One runs the same worker functions in the parent process.
        chunk_frames: Frames per chunk; None targets four chunks per worker,
            with at least eight frames per chunk.
        frame_at: Optional index-to-item callable, evaluated in the worker.
            Requires an integer frames count.
        frame_key: Optional picklable callable taking a frame item before draw.
            Equal consecutive non-None hashable keys promise identical pixels
            and allow reusing rasterization. The draw callback still runs on
            every frame, including scale diagnostics. None disables reuse for
            that item. Draw must set absolute state even after repeated
            items. Include the callback's source and inputs in the caller key.
            Defaults to None, drawing every frame; clocks are never inferred.
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
        on_rescale: "warn" emits one RuntimeWarning for data-scale drift,
            "raise" raises ScaleDriftError, and "ignore" disables checks.
            Every worker compares against frame zero after layout freeze.
            Checked chunks retain their diagnostic for resume; the policy
            does not affect raster identities or content keys.

    Returns:
        Output path, frame and chunk counts, reused chunks, workers, and time.

    Raises:
        ValueError: For invalid frame sources, suffix, counts, or settings.
        TypeError: If key is neither a string nor None.
        RuntimeError: If the chunk folder is unowned or locked, or a subprocess
            fails. Worker failures include their chunk range and traceback or
            stderr. Completed chunks remain available for resume.
        ScaleDriftError: If a data scale drifts with on_rescale="raise".
            The destination is preserved. New chunks that drift are not published.
    """
    _validate_on_rescale(on_rescale)
    started = time.monotonic()
    path = Path(path)
    count = _frame_source(frames, frame_at)
    if path.suffix.lower() != ".mp4":
        raise ValueError("render_frames requires an .mp4 path")
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive int")
    if frame_key is not None and not callable(frame_key):
        raise TypeError("frame_key must be callable or None")
    requested = workers if workers is not None else _default_workers()
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
                    _figure(
                        build, frames, dpi, frame_at, setup, rc, on_rescale, frame_key
                    )
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
                            on_rescale=on_rescale,
                            frame_key=frame_key,
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
                on_rescale,
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


# Keep public pickle references stable through the facade.
RenderResult.__module__ = "eyepiece.render"
render_frames.__module__ = "eyepiece.render"
frame_faults.__module__ = "eyepiece.render"
assert_frames_pure.__module__ = "eyepiece.render"
