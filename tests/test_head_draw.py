"""The head reveal draws at constant arc-length speed and rests on the plain still."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board(face="white"):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, facecolor=face)
    ax.set(xlim=(0, 1), ylim=(0, 1), facecolor=face)
    ax.set_axis_off()
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def _tip(fig, move, i=0):
    """The tip dot's display position, the last piece the copy drew."""
    fig.canvas.draw()
    return move.contract.moves[i].get_xydata()[-1].copy()


def _along(display, point):
    """Arc length from the start of a display polyline to its point nearest `point`."""
    seg = np.diff(display, axis=0)
    done = np.r_[0.0, np.cumsum(np.hypot(*seg.T))]
    best = (np.inf, 0.0)
    for k, (a, d) in enumerate(zip(display[:-1], seg, strict=True)):
        t = np.clip(np.dot(point - a, d) / np.dot(d, d), 0.0, 1.0)
        gap = np.hypot(*(a + t * d - point))
        best = min(best, (gap, done[k] + t * np.hypot(*d)))
    return best[1], done[-1]


@pytest.mark.parametrize("alpha", [None, 0.5])
def test_rest_is_the_plain_still_and_start_has_no_ink(alpha):
    fig, ax = _board()
    (line,) = ax.plot([0.1, 0.3, 0.7, 0.9], [0.2, 0.8, 0.4, 0.6], "o--", alpha=alpha)
    without = _pixels(fig)
    move = ep.head_draw(line)
    np.testing.assert_array_equal(_pixels(fig), without)
    for u in (0.3, 0.7, 0.95):
        move.update(u, level=0.4)
        _pixels(fig)
    move.update(0.0)
    assert np.all(_pixels(fig) == 255)
    move.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    assert line.get_alpha() == alpha


def test_rest_matches_the_still_within_half_a_pixel_inside_an_unsnapped_recording():
    fig, ax = _board()
    x = np.linspace(0.1, 0.9, 120)
    (line,) = ax.plot(x, 0.5 + 0.3 * np.sin(6 * x), color="C0", lw=2.5)
    with plt.rc_context({"path.snap": False}):
        without = _pixels(fig)
        move = ep.head_draw(line)
        move.update(0.5)
        _pixels(fig)
        move.update(1.0)
        np.testing.assert_array_equal(_pixels(fig), without)


@pytest.mark.parametrize("xscale", ["linear", "log"])
def test_tip_moves_at_constant_arc_length_speed_on_uneven_paths(xscale):
    fig, ax = _board()
    ax.set_xscale(xscale)
    ax.set_xlim(1, 100)
    # A tall thin rectangle with uneven vertex spacing: a short side, a long
    # side sampled densely and one sampled with two points.
    xs = np.r_[np.geomspace(2, 90, 40), 90, 2, 2]
    ys = np.r_[np.full(40, 0.2), 0.3, 0.3, 0.2]
    (line,) = ax.plot(xs, ys, color="black")
    move = ep.head_draw(line, ease="linear")
    fig.canvas.draw()
    display = line.get_transform().transform(np.c_[xs, ys])
    for u in np.linspace(0.05, 0.95, 13):
        move.update(u)
        reached, total = _along(display, _tip(fig, move))
        assert reached == pytest.approx(u * total, abs=0.75)


def test_default_ease_is_the_library_reveal_ease():
    fig, ax = _board()
    (line,) = ax.plot([0.1, 0.9], [0.5, 0.5], color="black")
    move = ep.head_draw(line)
    fig.canvas.draw()
    display = line.get_transform().transform([[0.1, 0.5], [0.9, 0.5]])
    move.update(0.3)
    reached, total = _along(display, _tip(fig, move))
    assert reached / total == pytest.approx(ep.ease(0.3), abs=1e-3)


def test_head_brightens_toward_ink_and_dissolves_into_the_body_at_rest():
    fig, ax = _board(face="black")
    with plt.rc_context({"text.color": "white"}):
        (line,) = ax.plot([0.1, 0.9], [0.5, 0.5], color="#1f77b4", lw=4)
        move = ep.head_draw(line, ease="linear")
    fig.canvas.draw()
    (x0, y), (x1, _) = line.get_transform().transform([[0.1, 0.5], [0.9, 0.5]])
    body, ink = np.array(to_rgb("#1f77b4")) * 255, np.full(3, 255.0)
    row = round(fig.bbox.height - y)

    def color_at(u, f):
        move.update(u)
        pixels = _pixels(fig)
        return pixels[row, round(x0 + f * (x1 - x0))]

    # Mid-move: well behind the tip is the body, just behind it is near ink.
    np.testing.assert_allclose(color_at(0.5, 0.2), body, atol=3)
    near = color_at(0.5, 0.48)
    assert np.linalg.norm(near - ink) < np.linalg.norm(near - body)
    # Almost done: the head has dissolved, so the end is the body color.
    np.testing.assert_allclose(color_at(0.999, 0.97), body, atol=6)


def test_last_frame_differs_from_rest_only_at_the_tip():
    fig, ax = _board()
    x = np.linspace(0.1, 0.9, 50)
    (line,) = ax.plot(x, 0.3 + 0.4 * x, color="C2", lw=3)
    rest = _pixels(fig)
    end = line.get_transform().transform([[0.9, 0.3 + 0.4 * 0.9]])[0]
    move = ep.head_draw(line)
    move.update(59 / 60)
    almost = _pixels(fig)
    # Away from the tip only antialiasing differs; the head color is gone.
    rows, cols = np.nonzero(np.abs(almost - rest).max(axis=-1) > 3)
    distance = np.hypot(cols - end[0], (fig.bbox.height - rows) - end[1])
    assert distance.max(initial=0.0) <= 4.0


def test_lag_staggers_lines_one_after_another():
    fig, ax = _board()
    a, b = ax.plot([0.1, 0.9], [[0.3, 0.7], [0.3, 0.7]], color="black")
    move = ep.head_draw([a, b], lag=1.0, ease="linear")
    move.update(0.5)
    _pixels(fig)
    assert a.get_visible() and not b.get_visible()
    assert not move.contract.moves[1].get_visible()
    move.update(0.75)
    fig.canvas.draw()
    assert move.contract.moves[1]._fraction == pytest.approx(0.5)
    together = ep.head_draw([a, b], lag=0.0, ease="linear")
    together.update(0.4)
    assert [c._fraction for c in together.contract.moves] == [0.4, 0.4]


def test_clip_has_no_motion_fault():
    fig, ax = _board()
    th = np.linspace(0, 2 * np.pi, 181)
    (ring,) = ax.plot(0.5 + 0.3 * np.cos(th), 0.5 + 0.3 * np.sin(th), color="C0", lw=2)
    (curve,) = ax.plot([0.1, 0.5, 0.9], [0.1, 0.2, 0.15], color="C1", lw=2)
    move = ep.head_draw([ring, curve], lag=0.4)
    frames = np.linspace(0.0, 1.0, 61)
    assert ept.motion_faults(fig, move.update, frames, moves=[move]) == []


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_lag_outside_unit_interval_raises(bad):
    _, ax = _board()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="lag"):
        ep.head_draw(line, lag=bad)


def test_rejects_non_lines_and_stepped_lines():
    _, ax = _board()
    with pytest.raises(TypeError):
        ep.head_draw([ax.text(0, 0, "x")])
    (step,) = ax.plot([0, 1], [0, 1], drawstyle="steps")
    with pytest.raises(ValueError, match="stepped"):
        ep.head_draw(step)
