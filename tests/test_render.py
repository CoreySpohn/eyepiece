"""Contracts for chunked frame rendering and resumable output."""

import os
import select
import subprocess
import sys
import time
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_render_helpers import (
    build_clip,
    build_crashing_clip,
    build_tiny_clip,
    build_waiting_clip,
    frame_at,
    setup_style,
)

import eyepiece as ep
import eyepiece._render_pool as render
from eyepiece._render_chunks import _workspace
from eyepiece.anim import _agg_canvas, _rc_overrides, _Recorder


def test_render_needs_no_process_pool(tmp_path):
    code = """
import sys
from pathlib import Path
sys.modules['multiprocessing'] = None
sys.modules['concurrent.futures.process'] = None
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'multiprocessing' or fullname.startswith('concurrent.futures'):
            raise ImportError('process pools are unavailable')
sys.meta_path.insert(0, Blocker())
import eyepiece.render
from test_render_helpers import build_clip
eyepiece.render.render_frames(build_clip, 2, Path(sys.argv[1]), fps=12,
    dpi=100, workers=2, chunk_frames=1)
assert sys.modules['multiprocessing'] is None
assert sys.modules['concurrent.futures.process'] is None
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "clip.mp4")],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("killed, exitcode", [(False, 7), (True, 7), (False, 0)])
def test_worker_crash_reports_stderr_and_preserves_output(tmp_path, killed, exitcode):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"previous output")
    with pytest.raises(RuntimeError, match=r"[Cc]hunk.*0.*2") as error:
        ep.render_frames(
            partial(build_crashing_clip, killed=killed, exitcode=exitcode),
            4,
            path,
            fps=12,
            dpi=100,
            workers=2,
            chunk_frames=2,
            work_dir=tmp_path / "chunks",
        )
    assert "deliberate worker crash" in str(error.value)
    assert path.read_bytes() == b"previous output"


def test_killed_parent_stops_workers_and_allows_resume(tmp_path):
    folder = tmp_path / "chunks"
    code = """
import sys
from functools import partial
from pathlib import Path
import eyepiece as ep
import eyepiece._render_pool as render
from test_render_helpers import build_waiting_clip, setup_style
folder = Path(sys.argv[1])
worker_fds = iter(map(int, sys.argv[2:]))
original_popen = render.subprocess.Popen
def start_process(args, **kwargs):
    if args[1:3] == ['-m', 'eyepiece.render']:
        kwargs['pass_fds'] = (next(worker_fds),)
    return original_popen(args, **kwargs)
render.subprocess.Popen = start_process
ep.render_frames(partial(build_waiting_clip, folder=folder), 8,
    folder.parent / 'clip.mp4', fps=12, dpi=100, workers=2,
    chunk_frames=2, key='killed', work_dir=folder, keep_chunks=True,
    setup=setup_style)
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    # Each worker alone holds one write end after the parent dies. EOF proves
    # exit even if the worker remains a zombie awaiting reaping by the OS.
    pipes = [os.pipe() for _ in range(2)]
    readers, writers = zip(*pipes, strict=True)
    parent = subprocess.Popen(
        [sys.executable, "-c", code, str(folder), *map(str, writers)],
        env=env,
        pass_fds=writers,
    )
    for fd in writers:
        os.close(fd)
    pids = []
    try:
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"previous output")
        deadline = time.monotonic() + 30
        barrier = [
            folder / name
            for name in ("chunk-0.bin", "chunk-1.bin", "blocked-5", "blocked-7")
        ]
        while not all(path.exists() for path in barrier):
            assert parent.poll() is None
            assert time.monotonic() < deadline, "workers did not reach blocked chunks"
            time.sleep(0.02)
        pids = [int(p.read_text()) for p in folder.glob("pid-*")]
        assert len(pids) == 2
        parent.kill()
        parent.wait(timeout=10)
        deadline = time.monotonic() + 10
        while _workers_running(readers):
            assert time.monotonic() < deadline, "orphan workers did not exit"
            time.sleep(0.02)
        assert path.read_bytes() == b"previous output"
        (folder / "continue").touch()
        result = ep.render_frames(
            partial(build_waiting_clip, folder=folder),
            8,
            tmp_path / "clip.mp4",
            fps=12,
            dpi=100,
            workers=2,
            chunk_frames=2,
            key="killed",
            work_dir=folder,
            keep_chunks=True,
            setup=setup_style,
        )
        assert result.frames == 8
        assert result.resumed == 2
        assert path.read_bytes() == serial_reference(
            build_clip, list(range(8)), tmp_path / "reference.mp4"
        )
        assert not list(folder.glob("*.part"))
        assert not list(tmp_path.glob(".*.partial.mp4"))
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)
        if _workers_running(readers):
            for pid in pids:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
        for fd in readers:
            os.close(fd)


def _workers_running(readers):
    closed, _, _ = select.select(readers, [], [], 0)
    assert all(os.read(fd, 1) == b"" for fd in closed)
    return len(closed) < len(readers)


@pytest.mark.parametrize("failed", [False, True])
def test_chunk_published_as_last_worker_exits(tmp_path, failed):
    job = (tmp_path / "chunk-0.bin", b"identity", 0, 2)

    class PublishingWorker:
        polls = 0

        def poll(self):
            self.polls += 1
            if self.polls == 1:
                return None
            if failed:
                (tmp_path / "chunk-0.error").write_text("deliberate late failure")
            else:
                job[0].touch()
            return 0

    pool = render._Workers(tmp_path, 1, {}, 2)
    pool.processes = [PublishingWorker()]
    if failed:
        with pytest.raises(RuntimeError, match="deliberate late failure"):
            list(render._ordered_chunks(pool, [job], set(), None))
    else:
        assert list(render._ordered_chunks(pool, [job], set(), None)) == [job]


def test_interrupted_lock_publication_allows_takeover(tmp_path):
    folder = tmp_path / "chunks"
    signal = tmp_path / "published"
    code = """
import os
import sys
import time
from pathlib import Path
from eyepiece._render_chunks import _workspace
folder, signal = map(Path, sys.argv[1:])
original_open, original_link = os.open, os.link
def published(path):
    if Path(path).name == 'render.lock':
        signal.touch()
        while True:
            time.sleep(0.02)
def interrupted_open(path, flags, *args, **kwargs):
    fd = original_open(path, flags, *args, **kwargs)
    if flags & os.O_EXCL:
        published(path)
    return fd
def interrupted_link(source, target, *args, **kwargs):
    original_link(source, target, *args, **kwargs)
    published(target)
os.open, os.link = interrupted_open, interrupted_link
with _workspace(folder):
    raise AssertionError('publication was not intercepted')
"""
    parent = subprocess.Popen([sys.executable, "-c", code, str(folder), str(signal)])
    try:
        deadline = time.monotonic() + 30
        while not signal.exists():
            assert parent.poll() is None
            assert time.monotonic() < deadline, "lock was not published"
            time.sleep(0.02)
        parent.kill()
        parent.wait(timeout=10)
        assert (folder / "render.lock").read_text() == str(parent.pid)
        with _workspace(folder):
            assert (folder / "render.lock").read_text() == str(os.getpid())
        assert not list(folder.glob("*.part"))
        assert not (folder / "render.lock").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)


def test_unreadable_lock_blocks_when_fresh_and_is_reclaimed_when_old(tmp_path):
    folder = tmp_path / "chunks"
    with _workspace(folder):
        pass
    lock = folder / "render.lock"
    lock.touch()
    with pytest.raises(RuntimeError, match="lock"):
        with _workspace(folder):
            pytest.fail("fresh unreadable lock was taken over")
    old = time.time() - 60
    os.utime(lock, (old, old))
    with _workspace(folder):
        assert lock.read_text() == str(os.getpid())
    assert not lock.exists()


def test_encoder_close_failure_cleans_up_and_preserves_output(tmp_path, monkeypatch):
    sinks = []

    class FailedEncoderSink(ep.RawSink):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            sinks.append(self)

        def close(self):
            # A tiny raster stays buffered until close. Wait for the bad option
            # to stop ffmpeg so flushing that buffer reliably hits a broken pipe.
            self._proc.wait(timeout=10)
            super().close()

    monkeypatch.setattr(render, "RawSink", FailedEncoderSink)
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"previous output")
    folder = tmp_path / "chunks"
    with pytest.raises(RuntimeError, match=r"ffmpeg.*no_such_render_option"):
        ep.render_frames(
            build_tiny_clip,
            1,
            path,
            fps=12,
            dpi=100,
            workers=1,
            work_dir=folder,
            key="encoder-failure",
            extra_ffmpeg_args=["-no_such_render_option"],
        )
    assert path.read_bytes() == b"previous output"
    assert sinks[0]._proc.returncode is not None
    assert sinks[0]._proc.stdin.closed
    assert sinks[0]._log.closed
    assert (folder / "chunk-0.bin").exists()
    assert not list(folder.glob("*.part"))
    assert not list(tmp_path.glob(".*.partial.mp4"))
    assert not (folder / "render.lock").exists()


def serial_reference(build, frames, path, *, dpi=100, rc=None, hold=(0, 0)):
    """Encode a serial raster loop independent of the render driver."""
    with matplotlib.rc_context(matplotlib.rcParamsDefault):
        setup_style()
        with (
            matplotlib.rc_context(rc),
            matplotlib.rc_context(_rc_overrides([path], snap=False)),
        ):
            fig, draw = build()
            try:
                with _agg_canvas(fig):
                    draw(fig, frames[0])
                    _Recorder(fig, [], {"frames": 0}, dpi)._freeze_layout()
                    fig.dpi = dpi
                    width, height = fig.canvas.get_width_height()
                    size = (width // 2 * 2, height // 2 * 2)
                    with ep.RawSink(path, size, fps=12) as sink:
                        for index, frame in enumerate(frames):
                            draw(fig, frame)
                            fig.canvas.draw()
                            pixels = np.asarray(fig.canvas.buffer_rgba())
                            pixels = pixels[: size[1], : size[0]]
                            sink.write(pixels)
                            repeats = hold[0] if index == 0 else 0
                            if index == len(frames) - 1:
                                repeats += hold[1]
                            for _ in range(repeats):
                                sink.write(pixels)
            finally:
                plt.close(fig)
    return path.read_bytes()


@pytest.mark.parametrize("hold", [(0, 0), (3, 2)])
def test_workers_chunks_and_reference_are_byte_identical(tmp_path, hold):
    rc = {"font.size": 8, "axes.facecolor": "#dedede"}
    with matplotlib.rc_context(rc):
        expected_rc = dict(matplotlib.rcParams)
        expected = serial_reference(
            build_clip, list(range(24)), tmp_path / "reference.mp4", rc=rc, hold=hold
        )
        for workers, chunk_frames in [(1, 12), (2, 6), (4, 7)]:
            path = tmp_path / f"pool-{workers}.mp4"
            result = ep.render_frames(
                build_clip,
                list(range(24)) if workers == 2 else 24,
                path,
                fps=12,
                dpi=100,
                workers=workers,
                chunk_frames=chunk_frames,
                frame_at=frame_at if workers == 4 else None,
                hold=hold,
                setup=setup_style,
                rc=rc,
            )
            assert isinstance(result, ep.RenderResult)
            assert result.path == path
            assert result.frames == 24 + sum(hold)
            assert result.chunks == (24 + chunk_frames - 1) // chunk_frames
            assert result.resumed == 0
            assert result.workers == min(workers, result.chunks)
            assert result.seconds > 0
            assert path.read_bytes() == expected
            assert not list((tmp_path / f".pool-{workers}.chunks").glob("chunk-*.bin"))
        assert dict(matplotlib.rcParams) == expected_rc


@pytest.mark.parametrize("existing", [False, True])
def test_resume_failure_identity_and_corruption(tmp_path, existing):
    folder = tmp_path / "chunks"
    build = partial(build_clip, failure_dir=folder)
    path = tmp_path / "clip.mp4"
    if existing:
        path.write_bytes(b"existing output")
    kwargs = dict(
        fps=12,
        dpi=100,
        workers=2,
        chunk_frames=6,
        setup=setup_style,
        work_dir=folder,
        key="first",
        keep_chunks=True,
    )
    with pytest.raises((RuntimeError, OSError), match=r"12.*18"):
        ep.render_frames(build, 24, path, **kwargs)
    if existing:
        assert path.read_bytes() == b"existing output"
    else:
        assert not path.exists()
    assert len(list(folder.glob("chunk-*.bin"))) == 3
    assert not list(tmp_path.glob(".*.partial.mp4"))
    result = ep.render_frames(build, 24, path, **kwargs)
    assert result.resumed == 3
    expected = serial_reference(build_clip, list(range(24)), tmp_path / "ref.mp4")
    assert path.read_bytes() == expected
    kwargs["workers"] = 1
    chunks = sorted(folder.glob("chunk-*.bin"))
    damaged = bytearray(chunks[1].read_bytes())
    damaged[-1] ^= 1
    chunks[1].write_bytes(damaged)
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 3
    assert path.read_bytes() == expected
    kwargs["key"] = "second"
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 0
    kwargs["dpi"] = 110
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 0
    kwargs["keep_chunks"] = False
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 4
    assert not list(folder.glob("chunk-*.bin"))
    assert not list(folder.glob("*.part"))
    assert not list(tmp_path.glob(".*.partial.mp4"))
    assert not list(folder.glob("*.lock"))


def test_frame_at_odd_size_and_untrusted_chunks(tmp_path):
    build = partial(build_clip, odd=True)
    path = tmp_path / "clip.mp4"
    folder = tmp_path / "chunks"
    expected = serial_reference(
        build, [frame_at(i) for i in range(24)], tmp_path / "ref.mp4"
    )
    kwargs = dict(
        fps=12,
        dpi=100,
        workers=1,
        chunk_frames=12,
        setup=setup_style,
        work_dir=folder,
        key="a",
        keep_chunks=True,
    )
    ep.render_frames(build, 24, path, frame_at=frame_at, **kwargs)
    assert path.read_bytes() == expected
    kwargs["key"] = None
    assert ep.render_frames(build, 24, path, frame_at=frame_at, **kwargs).resumed == 0
    assert not list(folder.glob("chunk-*.bin"))
    kwargs.update(key="a", resume=False)
    assert ep.render_frames(build, 24, path, frame_at=frame_at, **kwargs).resumed == 0
    assert not list(folder.glob("chunk-*.bin"))


def test_folder_ownership_and_locks(tmp_path):
    folder = tmp_path / "chunks"
    folder.mkdir()
    foreign = folder / "keep.txt"
    foreign.write_text("owned by caller")
    kwargs = dict(fps=12, dpi=100, workers=1, work_dir=folder)
    with pytest.raises(RuntimeError, match=r"marker|own|empty"):
        ep.render_frames(build_clip, 1, tmp_path / "clip.mp4", **kwargs)
    foreign.unlink()
    ep.render_frames(build_clip, 1, tmp_path / "clip.mp4", **kwargs)
    # A builder observes the public work-folder lock during an active run.
    ep.render_frames(
        partial(build_clip, observe_dir=folder), 1, tmp_path / "clip.mp4", **kwargs
    )
    lock = folder / (folder / "lock-name.txt").read_text()
    lock.write_text(str(os.getpid()))
    with pytest.raises(RuntimeError, match=r"lock|running|use"):
        ep.render_frames(build_clip, 1, tmp_path / "clip.mp4", **kwargs)
    lock.write_text("999999999")
    foreign.write_text("keep me")
    (folder / "unrelated.partial").write_text("keep me too")
    ep.render_frames(build_clip, 1, tmp_path / "clip.mp4", **kwargs)
    assert foreign.read_text() == "keep me"
    assert (folder / "unrelated.partial").read_text() == "keep me too"
    assert not lock.exists()


def test_purity_diagnostic():
    assert ep.frame_faults(build_clip, 24, dpi=100, setup=setup_style) == []
    ep.assert_frames_pure(build_clip, 24, dpi=100)
    build = partial(build_clip, accumulating=True)
    faults = ep.frame_faults(build, 24, dpi=100)
    assert faults == [0, 7, 15, 23]
    with pytest.raises(AssertionError, match="frame") as error:
        ep.assert_frames_pure(build, 24, dpi=100)
    assert all(str(index) in str(error.value) for index in faults)
    assert ep.frame_faults(build_clip, 24, dpi=100, frame_at=frame_at, sample=3) == []


@pytest.mark.parametrize(
    "changes",
    [
        {"workers": 0},
        {"frames": []},
        {"frames": 0},
        {"path": "bad.gif"},
        {"frames": [0, 1], "frame_at": frame_at},
    ],
)
def test_invalid_inputs(tmp_path, changes):
    kwargs = dict(
        build=build_clip, frames=2, path=tmp_path / "a.mp4", fps=12, dpi=100, workers=1
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        ep.render_frames(**kwargs)


def test_serial_resume_validation_and_failure(tmp_path):
    folder = tmp_path / "chunks"
    path = tmp_path / "clip.mp4"
    build = partial(build_clip, failure_dir=folder, wait_for_chunks=False)
    kwargs = dict(
        fps=12,
        dpi=100,
        workers=1,
        chunk_frames=6,
        key="a",
        work_dir=folder,
        keep_chunks=True,
        setup=setup_style,
    )
    with pytest.raises(OSError, match=r"12.*18"):
        ep.render_frames(build, 24, path, **kwargs)
    assert not path.exists()
    assert len(list(folder.glob("chunk-*.bin"))) == 2
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 2
    expected = serial_reference(build_clip, list(range(24)), tmp_path / "ref.mp4")
    assert path.read_bytes() == expected
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 4
    chunks = sorted(folder.glob("chunk-*.bin"))
    # Both incomplete headers and incomplete payloads are untrusted.
    chunks[0].write_bytes(b"incomplete")
    chunks[1].write_bytes(chunks[1].read_bytes()[:-1])
    damaged = bytearray(chunks[2].read_bytes())
    damaged[-1] ^= 1
    chunks[2].write_bytes(damaged)
    assert ep.render_frames(build, 24, path, **kwargs).resumed == 1
    assert path.read_bytes() == expected
    for changes in ({"key": "b"}, {"dpi": 110}, {"fps": 15}, {"hold": (1, 2)}):
        kwargs.update(changes)
        assert ep.render_frames(build, 24, path, **kwargs).resumed == 0
    assert ep.render_frames(build, 23, path, **kwargs).resumed == 0
    kwargs["keep_chunks"] = False
    assert ep.render_frames(build, 23, path, **kwargs).resumed == 4
    assert not list(folder.glob("chunk-*.bin"))


def test_unpassed_parent_rc_is_isolated_and_default_chunking(tmp_path):
    path = tmp_path / "clip.mp4"
    expected = serial_reference(build_clip, list(range(3)), tmp_path / "ref.mp4")
    before_figures = plt.get_fignums()
    with matplotlib.rc_context({"font.size": 23, "axes.facecolor": "red"}):
        before = dict(matplotlib.rcParams)
        result = ep.render_frames(
            build_clip, 3, path, fps=12, dpi=100, setup=setup_style
        )
        assert dict(matplotlib.rcParams) == before
    assert result.chunks == result.workers == 1
    assert path.read_bytes() == expected
    assert plt.get_fignums() == before_figures
