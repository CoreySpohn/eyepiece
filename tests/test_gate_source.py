"""Video samples keep seek selection while using one decoder."""

import os
import signal
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece.testing as ept
from eyepiece._ffmpeg import resolve_ffmpeg


@pytest.fixture(params=["ffv1", "libx264"])
def video(tmp_path, request):
    codec = request.param
    path = tmp_path / ("samples.mkv" if codec == "ffv1" else "samples.mp4")
    subprocess.run(
        [
            resolve_ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=s=160x90:r=30:d=1", "-c:v", codec, str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    return path


@pytest.fixture
def decoder_starts(monkeypatch):
    starts = []
    original = subprocess.Popen

    class CountedPopen(original):
        def __init__(self, command, *args, **kwargs):
            starts.append(command)
            super().__init__(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", CountedPopen)
    return starts


def test_source_samples_match_separate_seeks_including_tail_and_duplicates(
    video, decoder_starts
):
    times = [
        0.9,
        0.0,
        0.2,
        0.201,
        0.95,
        1.0,
        2.0,
        0.2,
        0.033333,
        0.033334,
        0.0333335,
        0.0334995,
        0.0335005,
    ]
    expected = [ept._source_frame(video, t) for t in times]
    decoder_starts.clear()
    actual = list(ept._source_samples(video, times))
    assert len(actual) == len(expected)
    assert len(decoder_starts) == 1
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(before, after)


def test_requests_wholly_past_video_tail_keep_last_frame(video):
    expected = ept._source_frame(video, 2.0)
    actual = list(ept._source_samples(video, [2.0, 3.0]))
    assert len(actual) == 2
    for frame in actual:
        np.testing.assert_array_equal(frame, expected)


def test_callable_path_uses_one_video_decoder_and_rejects_plan_keys(
    video, decoder_starts
):
    class CallablePath:
        def __fspath__(self):
            return str(video)

        def __call__(self, time):
            raise AssertionError("a video path is not a pixel callback")

    source = CallablePath()
    assert len(list(ept._source_samples(source, [0.1, 0.4, 0.8]))) == 3
    assert len(decoder_starts) == 1
    with pytest.raises(ValueError, match="callable source"):
        with ept.sample_source(source, [0], frame_key=lambda t: 0):
            pass


def test_variable_rate_samples_match_input_seeks(video, tmp_path):
    sparse = tmp_path / "sparse.mkv"
    subprocess.run(
        [
            resolve_ffmpeg(), "-v", "error", "-y", "-i", str(video),
            "-vf", "select='eq(n,0)+eq(n,2)+eq(n,10)+eq(n,29)'",
            "-vsync", "0", "-c:v", "ffv1", str(sparse),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    times = [0.01, 0.067, 0.07, 0.8, 1.0]
    for t, frame in zip(times, ept._source_samples(sparse, times), strict=True):
        np.testing.assert_array_equal(frame, ept._source_frame(sparse, t))


def test_sample_source_video_shared_consumers_decode_once(video, decoder_starts):
    with ept.sample_source(video, [0, 0.1, 0.2]) as source:
        first = [source(t).copy() for t in [0, 0.1, 0.2]]
        second = [source(t) for t in [0.2, 0.1, 0]]
        for a, b in zip(first, reversed(second), strict=True):
            np.testing.assert_array_equal(a, b)
    assert len(decoder_starts) == 1


def test_motion_source_callable_matches_drawn_pixels_and_geometry():
    fig, ax = plt.subplots(figsize=(2, 1), dpi=60)
    (dot,) = ax.plot([0.2], [0.5], "o", color="black")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    times = [0, 0.2, 0.4, 0.6, 0.8]

    def at(t):
        dot.set_xdata([0.2 + t * 0.6])

    try:
        frames = {}
        for t in times:
            at(t)
            fig.canvas.draw()
            frames[t] = np.asarray(fig.canvas.buffer_rgba()).copy()
        expected = ept.motion_faults(fig, at, times, rest=[dot])
        actual = ept.motion_faults(
            fig, at, times, rest=[dot], source=lambda t: frames[t]
        )

        def records(faults):
            return [
                tuple(getattr(f, field.name) for field in fields(f)) for f in faults
            ]

        assert records(actual) == records(expected)
        assert any(f.kind == "drift" for f in actual)
    finally:
        plt.close(fig)


def test_motion_source_refreshes_geometry_without_any_paint(monkeypatch):
    fig, ax = plt.subplots(figsize=(2, 1), dpi=60)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    (dot,) = ax.plot([0.2], [0.5], "o")
    times = [0, 0.1, 0.2]
    frames = {}
    paints = []

    def at(t):
        dot.set_xdata([0.2 + t])

    native = agg._RendererAgg

    class CountRenderer(native):
        pass

    for name in (
        "draw_path",
        "draw_markers",
        "draw_path_collection",
        "draw_image",
        "draw_gouraud_triangles",
    ):

        def paint(self, *args, _name=name, **kwargs):
            paints.append(_name)
            return getattr(native, _name)(self, *args, **kwargs)

        setattr(CountRenderer, name, paint)

    text_draw = agg.RendererAgg.draw_text

    def text_paint(self, *args, **kwargs):
        paints.append("draw_text")
        return text_draw(self, *args, **kwargs)

    try:
        for t in times:
            at(t)
            fig.canvas.draw()
            frames[t] = np.asarray(fig.canvas.buffer_rgba()).copy()
        # Replace the renderer factory at class level and obtain a new
        # renderer with the instrumented native paint methods.
        monkeypatch.setattr(agg, "_RendererAgg", CountRenderer)
        monkeypatch.setattr(agg.RendererAgg, "draw_text", text_paint)
        agg.FigureCanvasAgg(fig)
        faults = ept.motion_faults(
            fig, at, times, source=lambda t: frames[t], rest=[dot]
        )
        assert any(f.kind == "drift" for f in faults)
        assert paints == []
    finally:
        plt.close(fig)


def test_motion_source_size_mismatch_raises_and_restores_dpi():
    fig = plt.figure(figsize=(2, 1), dpi=60)
    try:
        with pytest.raises(ValueError, match="size"):
            ept.motion_faults(
                fig,
                lambda t: None,
                [0, 0.1, 0.2],
                dpi=80,
                source=lambda t: np.zeros((10, 10, 3), dtype=np.uint8),
            )
        assert fig.dpi == 60
    finally:
        plt.close(fig)


def test_motion_source_preserves_custom_canvas_nonpainting_updates():
    fig = plt.figure(figsize=(2, 1), dpi=60)
    label = fig.text(0.2, 0.5, "label")

    class Canvas(agg.FigureCanvasAgg):
        def __init__(self, figure):
            super().__init__(figure)
            self.time = 0.0
            self.updates = []

        def draw(self):
            label.set_x(0.2 + self.time)
            self.updates.append(label.get_position())
            super().draw()

    canvas = Canvas(fig)

    def at(t):
        canvas.time = t

    try:
        faults = ept.motion_faults(
            fig,
            at,
            [0, 0.1, 0.2],
            kinds=["drift"],
            rest=[label],
            source=lambda t: np.zeros((60, 120, 3), dtype=np.uint8),
        )
        assert canvas.updates == [(0.2, 0.5), (0.2 + 0.1, 0.5), (0.4, 0.5)]
        assert any(f.kind == "drift" for f in faults)
        assert label.get_position()[0] == 0.4
    finally:
        plt.close(fig)


def test_closing_video_iterator_reaps_decoder_without_pipe_deadlock(video):
    # An isolated process lets the regression fail within a fixed timeout,
    # and lets the test reap the whole child group on the old hanging path.
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from eyepiece.testing import _source_frames; "
        "frames = _source_frames(sys.argv[2], seconds=1, fps=30); "
        "next(frames); frames.close(); print('closed')"
    )
    command = [
        sys.executable,
        "-c",
        code,
        str(Path(ept.__file__).parents[1]),
        str(video),
    ]
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
    ) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            pytest.fail("closing the video iterator blocked on an unread pipe")
    assert proc.returncode == 0, stderr.decode()
    assert stdout.strip() == b"closed"
