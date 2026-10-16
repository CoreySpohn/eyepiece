"""Line reveals use current display geometry and restore the original still."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100, facecolor="white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _ink(fig, ax):
    pixels = _pixels(fig)
    x0, y0, x1, y1 = ax.bbox.extents.astype(int)
    area = pixels[pixels.shape[0] - y1 : pixels.shape[0] - y0, x0:x1, :3]
    return np.count_nonzero(np.any(area < 250, axis=-1))


@pytest.mark.parametrize("many", [False, True])
@pytest.mark.parametrize("alpha", [None, 0.45, 0.0])
def test_draw_on_rest_is_pixel_identical_with_markers_and_dashes(many, alpha):
    fig, ax = _board()
    (line,) = ax.plot([0.1, 0.3, 0.7, 0.9], [0.2, 0.8, 0.4, 0.6], "o--", alpha=alpha)
    (other,) = ax.plot([0.1, 0.5, 0.9], [0.1, 0.3, 0.2], "s-.")
    items = [line, other] if many else line
    original = line.get_data()
    without = _pixels(fig)
    move = ep.draw_on(items)
    assert isinstance(move, ep.PlotResult)
    np.testing.assert_array_equal(_pixels(fig), without)
    for u in (0.0, 0.4, 0.9, 0.2):
        move.update(u, level=0.35)
        _pixels(fig)
    move.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    assert line.get_alpha() == alpha
    for before, after in zip(original, line.get_data(), strict=True):
        np.testing.assert_array_equal(before, after)


def test_draw_on_zero_has_no_ink_in_axes_even_with_markers():
    fig, ax = _board()
    lines = ax.plot([0.1, 0.9], [[0.2, 0.7], [0.4, 0.6]], "o--")
    move = ep.draw_on(lines)
    assert _ink(fig, ax) > 0
    move.update(0.0)
    assert _ink(fig, ax) == 0


def test_draw_on_ink_grows_monotonically():
    fig, ax = _board()
    x = np.linspace(0.1, 0.9, 200)
    (line,) = ax.plot(x, 0.5 + 0.25 * np.sin(2 * np.pi * x), color="black", lw=2)
    move = ep.draw_on(line)
    counts = []
    for u in np.linspace(0, 1, 11):
        move.update(u)
        counts.append(_ink(fig, ax))
    assert counts[0] == 0
    assert np.all(np.diff(counts) > 0), counts


def test_draw_on_clip_has_no_jump_or_spike():
    fig, ax = _board()
    x = np.linspace(0.1, 0.9, 101)
    (line,) = ax.plot(x, 0.25 + 0.5 * x, lw=2)
    move = ep.draw_on(line)
    frames = np.linspace(0.0, 1.0, 31)
    assert (
        ept.motion_faults(
            fig,
            move.update,
            frames,
            moves=[move],
            kinds=("jump", "spike"),
        )
        == []
    )


def test_draw_on_reads_new_data_between_frames_and_at_draw_time():
    fig, ax = _board()
    (line,) = ax.plot([0.1, 0.9], [0.2, 0.2], color="black")
    move = ep.draw_on(line)
    move.update(0.5)
    _pixels(fig)
    old = _pixels(fig)
    line.set_data([0.2, 0.8], [0.7, 0.7])
    move.update(0.5)
    new = _pixels(fig)
    assert not np.array_equal(new, old)
    copy = move.contract.moves[0]
    expected = ax.transData.transform([[0.2, 0.7], [0.5, 0.7]])
    np.testing.assert_allclose(copy.get_xydata(), expected, atol=1e-8)
    # A caller can also replace the data after update and before drawing.
    line.set_data([0.1, 0.7], [0.4, 0.4])
    _pixels(fig)
    np.testing.assert_allclose(
        copy.get_xydata(), ax.transData.transform([[0.1, 0.4], [0.4, 0.4]])
    )
    move.update(1)
    _pixels(fig)
    np.testing.assert_array_equal(line.get_xdata(), [0.1, 0.7])


def test_draw_on_log_axis_cuts_equal_display_arc_length_after_layout_changes():
    fig, ax = _board()
    ax.set(xscale="log", xlim=(1, 1000))
    xy = np.array([[2, 0.2], [8, 0.75], [500, 0.4]])
    (line,) = ax.plot(*xy.T)
    move = ep.draw_on(line)
    # Resolve the transform at draw time, after both update and a resize.
    move.update(0.4)
    fig.set_size_inches(6, 2)
    ax.set_xlim(1, 2000)
    _pixels(fig)
    display = line.get_transform().transform(xy)
    lengths = np.linalg.norm(np.diff(display, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    ends = []
    for u in np.linspace(0.05, 0.95, 10):
        move.update(u)
        _pixels(fig)
        copy = move.contract.moves[0]
        cut = copy.get_transform().transform(copy.get_xydata())
        distance = u * cumulative[-1]
        segment = min(np.searchsorted(cumulative, distance, side="right") - 1, 1)
        fraction = (distance - cumulative[segment]) / lengths[segment]
        expected = display[segment] + fraction * (
            display[segment + 1] - display[segment]
        )
        np.testing.assert_allclose(cut[-1], expected, atol=1)
        assert np.linalg.norm(np.diff(cut, axis=0), axis=1).sum() == pytest.approx(
            distance, abs=1
        )
        ends.append(cut[-1, 0])
    assert np.all(np.diff(ends) > 0)


@pytest.mark.parametrize("easing", ["smoothstep", lambda u: u**2])
def test_draw_on_easing_and_level_are_independent(easing):
    fig, ax = _board()
    (line,) = ax.plot([0.1, 0.9], [0.4, 0.4], alpha=0.6)
    move = ep.draw_on(line, ease=easing)
    fraction = ep.ease(0.3, easing) if isinstance(easing, str) else easing(0.3)
    move.update(0.3)
    _pixels(fig)
    copy = move.contract.moves[0]
    full = copy.get_xydata().copy()
    move.update(0.3, level=0.2)
    _pixels(fig)
    np.testing.assert_array_equal(copy.get_xydata(), full)
    assert copy.get_alpha() == pytest.approx(0.12)
    expected = ax.transData.transform([0.1 + 0.8 * fraction, 0.4])
    np.testing.assert_allclose(full[-1], expected)
    move.update(1, level=0)
    assert _ink(fig, ax) == 0
    move.update(1)
    assert line.get_alpha() == 0.6


def test_draw_on_nan_gaps_do_not_add_length_or_connect_runs():
    fig, ax = _board()
    xy = np.array([[0.1, 0.2], [0.3, 0.2], [np.nan, np.nan], [0.6, 0.8], [0.8, 0.8]])
    (line,) = ax.plot(*xy.T)
    move = ep.draw_on(line)
    move.update(0.75)
    _pixels(fig)
    copy = move.contract.moves[0]
    cut = copy.get_xydata()
    np.testing.assert_allclose(cut[:2], ax.transData.transform(xy[:2]))
    assert np.isnan(cut[2]).all()
    np.testing.assert_allclose(cut[-1], ax.transData.transform([0.7, 0.8]))


@pytest.mark.parametrize("xy", [np.empty((0, 2)), np.array([[0.5, 0.5]])])
def test_draw_on_empty_and_single_point_paths(xy):
    fig, ax = _board()
    (line,) = ax.plot(*xy.T, marker="o")
    without = _pixels(fig)
    move = ep.draw_on(line)
    for u in (0.0, 0.5, 1.0):
        move.update(u)
        pixels = _pixels(fig)
        if u == 0:
            assert _ink(fig, ax) == 0
        elif u == 1:
            np.testing.assert_array_equal(pixels, without)


def test_draw_on_rejects_invalid_lines_without_changing_valid_ones():
    _, ax = _board()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="at least one"):
        ep.draw_on([])
    with pytest.raises(TypeError, match="Line2D"):
        ep.draw_on([line, ax.patch])
    assert len(ax.lines) == 1
    assert line.get_visible()


def test_draw_on_rejects_stepped_drawstyles():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1, 2], [0, 1, 0], drawstyle="steps-mid")
    with pytest.raises(ValueError, match="stepped"):
        ep.draw_on(line)
    plt.close(fig)
