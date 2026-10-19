"""Emphasis without a hue: a comet passes round a target, a pulse holds its peak."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece.attention import _pulse


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, facecolor="white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    note = ax.text(0.5, 0.5, "this one", ha="center", va="center", fontsize=18)
    fig.canvas.draw()
    return fig, ax, note


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def test_flash_starts_and_ends_on_the_still_and_leaves_nothing_behind():
    fig, _, note = _board()
    still = _pixels(fig)
    flash = ep.circumscribe(note, style="flash")
    for u in (0.0, 1.0):
        flash.update(u)
        np.testing.assert_array_equal(_pixels(fig), still)
    for u in (0.3, 0.6, 0.9, 1.0):
        flash.update(u)
        _pixels(fig)
    np.testing.assert_array_equal(_pixels(fig), still)


def test_flash_window_is_time_width_long_and_runs_empty_to_empty():
    _, _, note = _board()
    result = ep.circumscribe(note, style="flash", time_width=0.25)
    line = result.artists["line"]
    heads = []
    for u in np.linspace(0.01, 0.99, 99):
        result.update(u)
        lo, hi = line._window
        assert hi - lo == pytest.approx(0.25)
        heads.append(hi)
    assert np.all(np.diff(heads) > 0.0)
    assert heads[0] < 0.01
    assert heads[-1] > 1.24


def test_flash_comet_tapers_to_its_tail():
    fig, _, note = _board()
    flash = ep.circumscribe(note, style="flash", lw=4.0, color="black")
    flash.update(0.5)
    line = flash.artists["line"]
    seen = []
    draw = type(line).__mro__[1].draw

    def spy(self, renderer):
        seen.append((self.get_linewidth(), self.get_alpha()))
        return draw(self, renderer)

    type(line).__mro__[1].draw = spy
    try:
        fig.canvas.draw()
    finally:
        type(line).__mro__[1].draw = draw
    widths = [w for w, _ in seen]
    alphas = [a for _, a in seen]
    assert len(seen) == 8
    assert widths == sorted(widths) and alphas == sorted(alphas)
    assert widths[-1] == pytest.approx(4.0) and alphas[-1] == pytest.approx(1.0)
    assert widths[0] == pytest.approx(4.0 * (0.4 + 0.6 / 8))
    assert line.get_linewidth() == pytest.approx(4.0) and line.get_alpha() is None


def test_flash_never_touches_the_target():
    fig, _, note = _board()
    before = (note.get_color(), note.get_fontsize(), note.get_position())
    flash = ep.circumscribe(note, style="flash")
    for u in np.linspace(0.0, 1.0, 21):
        flash.update(u)
        fig.canvas.draw()
        assert (note.get_color(), note.get_fontsize(), note.get_position()) == before


def test_flash_clip_has_no_motion_fault():
    fig, _, note = _board()
    flash = ep.circumscribe(note, style="flash", shape="ellipse")
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, flash.update, frames, moves=[flash]) == []


def test_flash_and_style_validation():
    _, _, note = _board()
    with pytest.raises(ValueError, match="style"):
        ep.circumscribe(note, style="comet")
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="time_width"):
            ep.circumscribe(note, style="flash", time_width=bad)


def test_trace_style_is_unchanged_by_default():
    fig, _, note = _board()
    a = ep.circumscribe(note)
    a.update(0.3)
    traced = _pixels(fig)
    a.update(1.0)
    b = ep.circumscribe(note, style="trace")
    b.update(0.3)
    np.testing.assert_array_equal(_pixels(fig), traced)


def test_plateau_holds_its_peak_through_the_middle():
    us = np.linspace(0.0, 1.0, 1001)
    k = np.array([_pulse(u, "plateau") for u in us])
    assert k[0] == 0.0 and k[-1] == 0.0
    middle = (us >= 0.35) & (us <= 0.65)
    assert np.all(k[middle] == 1.0)
    rise = us < 0.35
    assert np.all(np.diff(k[rise]) >= 0.0) and np.all(k[rise] < 1.0)
    np.testing.assert_allclose(k, k[::-1], atol=1e-12)
    sine = np.array([_pulse(u, "sine") for u in us])
    assert np.sum(sine >= 0.999) < np.sum(k >= 0.999) / 10


def test_indicate_plateau_at_scale_1_2_swells_and_returns_exactly():
    fig, ax, note = _board()
    (curve,) = ax.plot([0.1, 0.9], [0.2, 0.3], lw=2.0, color="C0")
    still = _pixels(fig)
    pulse = ep.indicate([curve, note], scale=1.2, shape="plateau")
    widths = []
    for u in np.linspace(0.0, 1.0, 41):
        pulse.update(u)
        widths.append(curve.get_linewidth())
    assert max(widths) == pytest.approx(2.4)
    assert sum(w == pytest.approx(2.4) for w in widths) >= 12
    np.testing.assert_array_equal(_pixels(fig), still)


def test_indicate_default_shape_is_the_sine_pulse_and_bad_shape_raises():
    _, ax, _ = _board()
    (curve,) = ax.plot([0.1, 0.9], [0.2, 0.3], lw=2.0)
    pulse = ep.indicate(curve)
    pulse.update(0.25)
    assert curve.get_linewidth() == pytest.approx(2.0 * (1.0 + 0.15 * 0.5))
    with pytest.raises(ValueError, match="shape"):
        ep.indicate(curve, shape="square")


def test_indicate_plateau_clip_has_no_motion_fault():
    # Marks only: a pulsed text steps in whole pixels (see the indicate docs).
    fig, ax, _ = _board()
    (curve,) = ax.plot([0.1, 0.9], [0.2, 0.3], lw=2.0)
    dots = ax.scatter([0.3, 0.7], [0.75, 0.8], s=60)
    pulse = ep.indicate([curve, dots], scale=1.2, shape="plateau")
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, pulse.update, frames, moves=[pulse]) == []
