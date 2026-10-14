"""Stage, axis builds, and collapses: what the viewer sees, not how.

A stage at rest is the figure as drawn, and a dimmed group keeps the
fraction of its contrast the stage names, on a light figure and a dark one;
an axis build ends on the finished axes and only ever adds ink; a collapse
draws nothing at rest and lands its bar where the caller's bar is drawn.
"""

from itertools import pairwise

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import eyepiece as ep

MODES = ["light", "dark"]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _scene():
    """One axes with a thick line, a labeled marker, and a colormapped mesh."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100, layout="constrained")
    x = np.linspace(0.0, 1.0, 50)
    (line,) = ax.plot(x, 0.5 + 0.0 * x, lw=12, color="tab:blue", solid_capstyle="butt")
    (dot,) = ax.plot([0.2], [0.8], "o", ms=10, color="tab:red")
    note = ax.text(0.6, 0.8, "the dot", bbox=ep.backing())
    mesh = ax.pcolormesh(
        np.linspace(0.0, 1.0, 6), [0.05, 0.25], np.arange(5.0)[None, :], cmap="viridis"
    )
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax, {"line": [line], "marks": [dot, note], "mesh": [mesh]}


def _ink(image, background):
    """Pixels that differ from the background color."""
    bg = np.round(255 * np.asarray(to_rgba(background))[:3])
    return int(np.any(np.abs(image[..., :3].astype(float) - bg) > 2, axis=-1).sum())


# Stage


@pytest.mark.parametrize("mode", MODES)
def test_stage_rest_is_the_figure_as_drawn_and_never_compounds(mode):
    hwostyle.use(mode)
    fig, _ax, groups = _scene()
    plain = _pixels(fig)
    stage = ep.Stage(groups)
    np.testing.assert_array_equal(_pixels(fig), plain)  # made at rest
    for _ in range(3):
        stage.set(shown=["line"], dimmed=["marks", "mesh"])
    once = _pixels(fig)
    stage.set(shown=["line"], dimmed=["marks", "mesh"])
    np.testing.assert_array_equal(_pixels(fig), once)  # idempotent
    stage.set(shown=["line"], dimmed=["marks"], start=stage.state(), u=0.4)
    stage.rest()
    np.testing.assert_array_equal(_pixels(fig), plain)
    stage.set()  # every group shown: the rest state again
    np.testing.assert_array_equal(_pixels(fig), plain)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("level", [0.35, 0.6])
def test_a_dimmed_group_keeps_its_level_of_contrast(mode, level):
    hwostyle.use(mode)
    fig, ax, groups = _scene()
    stage = ep.Stage(groups, dim=level)
    bg = np.asarray(to_rgba(ax.get_facecolor()))[:3]
    probe = ax.transData.transform((0.5, 0.5)).astype(int)
    row, col = fig.bbox.height - probe[1], probe[0]

    def contrast():
        pixel = _pixels(fig)[int(row), int(col), :3] / 255.0
        return np.abs(pixel - bg).max()

    full = contrast()
    assert full > 0.2
    stage.set(shown=["marks", "mesh"], dimmed=["line"])
    assert contrast() == pytest.approx(level * full, abs=2.0 / 255.0)


@pytest.mark.parametrize("mode", MODES)
def test_an_absent_group_draws_nothing(mode):
    hwostyle.use(mode)
    fig, _ax, groups = _scene()
    stage = ep.Stage(groups)
    stage.set(shown=["line"])
    staged = _pixels(fig)
    stage.rest()
    for artist in (*groups["marks"], *groups["mesh"]):
        artist.set_visible(False)
    np.testing.assert_array_equal(staged, _pixels(fig))


def test_stage_moves_between_states_and_checks_names():
    _fig, _ax, groups = _scene()
    stage = ep.Stage(groups)
    start = stage.state(shown=["line"])
    half = stage.set(shown=["line", "marks"], start=start, u=0.5)
    assert half["line"] == (1.0, 1.0)
    assert 0.0 < half["marks"][0] < 1.0
    assert half["mesh"] == (0.0, 1.0)
    with pytest.raises(KeyError):
        stage.state(shown=["nothing"])
    with pytest.raises(ValueError, match="dim"):
        ep.Stage(groups, dim=1.5)


def test_weigh_sets_only_the_groups_it_names():
    fig, _ax, groups = _scene()
    plain = _pixels(fig)
    stage = ep.Stage(groups)
    stage.weigh({"line": (1.0, 0.35)})
    line_only = _pixels(fig)
    stage.weigh({"marks": 0.0})
    # The line stays dimmed: weighing the marks did not return it to rest.
    assert not np.array_equal(_pixels(fig), line_only)
    stage.weigh({"marks": 1.0})
    np.testing.assert_array_equal(_pixels(fig), line_only)
    stage.rest()
    np.testing.assert_array_equal(_pixels(fig), plain)
    with pytest.raises(KeyError):
        stage.weigh({"nothing": 1.0})


# Axis build


def _axes(log=False):
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100, layout="constrained")
    ax.set(xlim=(0.0, 1.0), ylim=(0.5, 2.0), xlabel="albedo", ylabel="radius")
    if log:
        ax.set_yscale("log")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("log", [False, True])
def test_axis_build_ends_on_the_finished_axes_and_only_adds_ink(mode, log):
    hwostyle.use(mode)
    fig, ax = _axes(log)
    plain = _pixels(fig)
    limits = (ax.get_xlim(), ax.get_ylim())
    build = ep.build_axes(ax)
    np.testing.assert_array_equal(_pixels(fig), plain)  # made at rest
    face = fig.get_facecolor()
    inks = []
    for u in np.linspace(0.0, 1.0, 21):
        build.update(u)
        inks.append(_ink(_pixels(fig), face))
        assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert inks[0] < 0.2 * inks[-1]
    assert all(b >= a - 3 for a, b in pairwise(inks)), inks
    np.testing.assert_array_equal(_pixels(fig), plain)
    build.update(0.3)
    build.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), plain)


def test_axis_build_draws_one_axis_at_a_time():
    fig, ax = _axes()
    build = ep.build_axes(ax, axis="x")
    build.update(1.0)
    plain = _pixels(fig)
    build.update(0.0)
    ax.yaxis.set_visible(True)
    hidden = _pixels(fig)
    # The x axis is gone and the y axis still drawn: something remains.
    face = fig.get_facecolor()
    assert 0 < _ink(hidden, face) < _ink(plain, face)
    assert not ax.xaxis.get_major_ticks()[1].label1.get_alpha()
    with pytest.raises(ValueError, match="axis"):
        ep.build_axes(ax, axis="z")


def test_axis_label_writes_on_after_the_spine():
    fig, ax = _axes()
    build = ep.build_axes(ax, axis="x", spine_end=0.6)
    face = fig.get_facecolor()
    label = ax.xaxis.label
    box = label.get_window_extent()
    rows = slice(int(fig.bbox.height - box.y1), int(fig.bbox.height - box.y0) + 1)
    cols = slice(int(box.x0), int(box.x1) + 1)

    def label_ink(u):
        build.update(u)
        return _ink(_pixels(fig)[rows, cols], face)

    assert label_ink(0.6) == 0
    assert 0 < label_ink(0.8) < label_ink(1.0)


# Collapse


def _two_panels():
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(6.0, 3.0), dpi=100, sharey=True, layout="constrained"
    )
    for ax in (left, right):
        ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, left, right


def _cells(n=10, y=0.5):
    edges = np.linspace(0.1, 0.7, n + 1)
    return np.stack([np.c_[edges[:-1], [y] * n], np.c_[edges[1:], [y] * n]], axis=1)


@pytest.mark.parametrize("mode", MODES)
def test_collapse_is_absent_at_rest_and_lands_on_its_bar(mode):
    hwostyle.use(mode)
    fig, left, right = _two_panels()
    plain = _pixels(fig)
    weights = np.array([0, 0, 1, 3, 4, 2, 0, 0, 0, 0], dtype=float)
    start = ep.Anchor.from_data(left, (0.1, 0.5))
    end = ep.Anchor.from_data(left, (0.4, 0.5))
    land = (
        ep.Anchor.from_data(right, (0.0, 0.5)),
        ep.Anchor.from_data(right, (0.3, 0.5)),
    )
    move = ep.collapse(
        left,
        _cells(),
        weights,
        start,
        end,
        fig=fig,
        colors="tab:gray",
        color="tab:pink",
        to=land,
    )
    np.testing.assert_array_equal(_pixels(fig), plain)  # made at rest
    ghost = move.artists["line"]
    move.update(0.0)
    np.testing.assert_array_equal(_pixels(fig), plain)
    move.update(1.0)
    fig.canvas.draw()
    segs = np.asarray(ghost.get_segments())
    p0, p1 = start.display(), end.display()
    # Shares in proportion to the weights, end to end along the bar.
    assert np.hypot(*(segs[0][0] - p0)) < 0.5
    assert np.hypot(*(segs[-1][1] - p1)) < 0.5
    lengths = np.hypot(*(segs[:, 1] - segs[:, 0]).T)
    np.testing.assert_allclose(
        lengths / lengths.sum(), weights / weights.sum(), atol=1e-6
    )
    np.testing.assert_allclose(ghost.get_colors()[3], to_rgba("tab:pink"))
    move.update(1.0, carry=1.0 - 1e-9)
    fig.canvas.draw()
    segs = np.asarray(ghost.get_segments())
    assert np.hypot(*(segs[0][0] - land[0].display())) < 0.5
    assert np.hypot(*(segs[-1][1] - land[1].display())) < 0.5
    move.update(1.0, carry=1.0)
    np.testing.assert_array_equal(_pixels(fig), plain)
    assert (left.get_xlim(), right.get_xlim()) == ((0.0, 1.0), (0.0, 1.0))


def test_collapse_rejects_bad_weights():
    fig, left, _right = _two_panels()
    a = ep.Anchor.from_data(left, (0.1, 0.5))
    with pytest.raises(ValueError, match="sum to zero"):
        ep.collapse(left, _cells(), np.zeros(10), a, a, fig=fig, colors="k")
    with pytest.raises(ValueError, match="negative"):
        ep.collapse(left, _cells(), -np.ones(10), a, a, fig=fig, colors="k")
    with pytest.raises(ValueError, match="shape"):
        ep.collapse(left, _cells(), np.ones(3), a, a, fig=fig, colors="k")


# Callout pieces and rise


@pytest.mark.parametrize("mode", MODES)
def test_callout_pieces_draw_each_term_in_its_color_on_one_line(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    res = ep.callout(
        dot, ["add up the ", ("row", "tab:red"), " of cells"], side="right"
    )
    caption = res.artists["text"]
    fig.canvas.draw()
    assert caption.get_text() == "add up the row of cells"
    pieces = caption.get_children()
    assert [p.get_text() for p in pieces] == ["add up the ", "row", " of cells"]
    assert to_rgba(pieces[1].get_color()) == to_rgba("tab:red")
    assert to_rgba(pieces[0].get_color()) == to_rgba(caption.get_color())
    boxes = [p.get_window_extent() for p in pieces]
    for a, b in pairwise(boxes):
        assert abs(b.x0 - a.x1) <= 1.0
    whole = caption.get_window_extent()
    assert abs(boxes[0].x0 - whole.x0) <= 1.0
    assert abs(boxes[-1].x1 - whole.x1) <= 2.0
    with pytest.raises(ValueError, match="words"):
        ep.callout(dot, ["one two three ", ("four five six seven eight nine", "k")])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("side", ["right", "left"])
def test_a_heavier_key_term_grows_the_caption_away_from_its_subject(mode, side):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=100)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.5], [0.5], "o")
    text = "the posterior: planets the data allow"
    res = ep.callout(
        dot,
        [(text[:13], None, "bold"), text[13:]],
        side=side,
        leader=False,
        box={"boxstyle": "square,pad=0.2", "fc": "w", "ec": "k"},
    )
    caption = res.artists["text"]
    fig.canvas.draw()
    assert caption.get_text() == text
    key, rest = caption.get_children()
    assert key.get_fontweight() in ("bold", 700)
    assert rest.get_fontweight() == caption.get_fontweight()
    assert to_rgba(key.get_color()) == to_rgba(caption.get_color())
    plain = caption.get_window_extent()
    left, right = key.get_window_extent().x0, rest.get_window_extent().x1
    patch = caption.get_bbox_patch().get_window_extent()
    # The box holds the whole line.
    assert patch.x0 <= left and patch.x1 >= right
    # The extra width goes away from the subject; the near edge stays put.
    if side == "right":
        assert left == pytest.approx(plain.x0, abs=0.5) and right > plain.x1
    else:
        assert right == pytest.approx(plain.x1, abs=0.5) and left < plain.x0


def test_callout_rises_into_place_as_it_fades_in():
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    res = ep.callout(dot, "the dot", side="right", leader=False, rise_pt=6.0)
    caption = res.artists["text"]
    fig.canvas.draw()
    rest = caption.get_window_extent().y0
    res.update(0.3)
    fig.canvas.draw()
    assert caption.get_window_extent().y0 < rest - 1.0
    res.update(1.0)
    fig.canvas.draw()
    assert caption.get_window_extent().y0 == pytest.approx(rest)
