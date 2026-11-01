"""A decoded sample bank is reusable and has an explicit lifetime."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backends.backend_agg import RendererAgg

import eyepiece.testing as ept


def test_sample_source_reads_declared_times_once_and_reuses_pixels():
    calls = []
    frame = np.zeros((4, 8, 3), dtype=np.uint8)

    def source(t):
        calls.append(t)
        frame[:] = round(t * 100)
        return frame

    with ept.sample_source(source, [0.2, 0.0, 0.1]) as sample:
        assert calls == [0.2, 0.0, 0.1]
        for _ in range(3):
            assert np.all(sample(0.1) == 10)
            assert np.all(sample(0.2) == 20)
            assert np.all(sample(0.0) == 0)
        assert not sample(0.1).flags.writeable
        with pytest.raises(KeyError):
            sample(0.3)
    assert calls == [0.2, 0.0, 0.1]
    with pytest.raises(RuntimeError, match="closed"):
        sample(0.1)


def test_sample_source_closes_on_failure_and_rejects_nonfinite_times():
    with pytest.raises(RuntimeError, match="consumer"):
        with ept.sample_source(lambda t: np.zeros((4, 8, 3), np.uint8), [0]) as sample:
            raise RuntimeError("consumer")
    with pytest.raises(RuntimeError, match="closed"):
        sample(0)
    with pytest.raises(ValueError, match="finite"):
        with ept.sample_source(lambda t: None, [float("nan")]):
            pass


def test_sample_source_key_reuses_consecutive_pixels_without_aliasing():
    rendered = []
    buffer = np.zeros((4, 8, 3), dtype=np.uint8)
    states = {0: 0, 0.1: 0, 0.2: 1, 0.3: 1, 0.4: 0}

    def source(t):
        buffer[:] = states[t] * 100
        rendered.append(t)
        return buffer

    with ept.sample_source(source, states, frame_key=states.__getitem__) as sample:
        assert rendered == [0, 0.2, 0.4]
        for t, value in states.items():
            assert np.all(sample(t) == value * 100)


def test_sample_source_none_keys_capture_every_frame_and_bad_keys_raise():
    rendered = []

    def source(t):
        rendered.append(t)
        return np.zeros((4, 8, 3), dtype=np.uint8)

    with ept.sample_source(source, [0, 0.1], frame_key=lambda t: None):
        pass
    assert rendered == [0, 0.1]
    with pytest.raises(TypeError, match="hashable"):
        with ept.sample_source(source, [0], frame_key=lambda t: []):
            pass
    with pytest.raises(ValueError, match="callable source"):
        with ept.sample_source("movie.mp4", [0], frame_key=lambda t: 0):
            pass


def test_sample_source_key_paints_only_changed_states(monkeypatch):
    fig = plt.figure(figsize=(2, 1), dpi=60)
    text = fig.text(0.5, 0.5, "0")
    states = {0: 0, 0.1: 0, 0.2: 1, 0.3: 1, 0.4: 0}
    paints = []
    original = RendererAgg.draw_text

    def draw_text(self, *args, **kwargs):
        paints.append(1)
        return original(self, *args, **kwargs)

    def source(t):
        text.set_text(str(states[t]))
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3]

    monkeypatch.setattr(RendererAgg, "draw_text", draw_text)
    try:
        with ept.sample_source(source, states, frame_key=states.__getitem__) as sample:
            assert len(paints) == 3
            np.testing.assert_array_equal(sample(0), sample(0.4))
            assert not np.array_equal(sample(0), sample(0.2))
    finally:
        plt.close(fig)
