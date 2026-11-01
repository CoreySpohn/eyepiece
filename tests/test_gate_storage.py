"""Lossless bounded pixel storage for long checks."""

import gc
import sys
import weakref

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece.testing as ept
from eyepiece import _check_samples as samples


def test_frame_store_releases_rasters_and_round_trips_exactly():
    rng = np.random.default_rng(48)
    expected = []
    references = []
    with samples._FrameStore(max_bytes=128) as store:
        for _ in range(20):
            frame = rng.integers(0, 256, (4, 8, 4), dtype=np.uint8)
            expected.append(frame.copy())
            references.append(weakref.ref(frame))
            store.append(frame)
        del frame
        gc.collect()
        assert sum(ref() is not None for ref in references) <= 1
        assert store.nbytes <= 128
        for k in [19, 0, 11, 1, 18]:
            np.testing.assert_array_equal(store[k], expected[k])
            assert not store[k].flags.writeable
            assert store.nbytes <= 128
        assert len(store) == 20
        with pytest.raises(IndexError):
            _ = store[20]
        np.testing.assert_array_equal(store[-1], expected[-1])
    assert store.closed


def test_motion_trace_uses_store_without_changing_any_observations(monkeypatch):
    fig, ax = plt.subplots(figsize=(2, 1), dpi=60)
    (line,) = ax.plot([0.2], [0.5], "o")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    times = [0, 0.2, 0.4, 0.6, 0.8]

    def at(t):
        line.set_xdata([0.2 + t * 0.6])

    try:
        trace = ept._sample_trace(fig, at, times, dpi=None, keep_frames=True)
        assert isinstance(trace.frames, samples._FrameStore)
        with monkeypatch.context() as patch:
            patch.setattr(ept, "_FrameStore", list)
            original = ept._sample_trace(fig, at, times, dpi=None, keep_frames=True)
        assert trace.boxes.keys() == original.boxes.keys()
        for artist, boxes in trace.boxes.items():
            assert boxes.keys() == original.boxes[artist].keys()
            for k, record in boxes.items():
                for actual, expected in zip(
                    record, original.boxes[artist][k], strict=True
                ):
                    np.testing.assert_equal(actual, expected)
        assert trace.tick_text == original.tick_text
        assert trace.shapes == original.shapes
        for k in range(len(times)):
            np.testing.assert_array_equal(trace.frames[k], original.frames[k])
    finally:
        plt.close(fig)


def test_frame_store_repeated_frames_share_one_record_and_cleanup():
    frame = np.zeros((4, 8, 4), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="stop"):
        with samples._FrameStore(max_bytes=256) as store:
            store.append(frame)
            store.append(frame.copy())
            assert store[0] is store[1]
            assert store.disk_bytes < frame.nbytes
            frame[:] = 255
            store.append(frame)
            assert np.all(store[0] == 0)
            assert np.all(store[2] == 255)
            raise RuntimeError("stop")
    assert store.closed


def test_motion_analysis_closes_pixels_when_analysis_raises(monkeypatch):
    fig = plt.figure(figsize=(2, 1), dpi=60)
    traces = []
    original = ept._sample_trace

    def capture(*args, **kwargs):
        trace = original(*args, **kwargs)
        traces.append(trace)
        return trace

    def fail(*args, **kwargs):
        raise RuntimeError("analysis")

    monkeypatch.setattr(ept, "_sample_trace", capture)
    monkeypatch.setattr(ept, "_analyze", fail)
    try:
        with pytest.raises(RuntimeError, match="analysis"):
            ept.motion_faults(fig, lambda t: None, [0, 0.1, 0.2])
        assert traces[0].frames.closed
    finally:
        plt.close(fig)


def test_store_creation_error_preserves_original_error_without_cleanup_failure(
    monkeypatch,
):
    unraisable = []

    def fail():
        raise OSError("no temporary storage")

    monkeypatch.setattr(samples.tempfile, "TemporaryFile", fail)
    monkeypatch.setattr(
        sys, "unraisablehook", lambda event: unraisable.append(event.exc_value)
    )
    with pytest.raises(OSError, match="no temporary storage"):
        samples._FrameStore()
    gc.collect()
    assert unraisable == []
