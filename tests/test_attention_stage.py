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


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "axis, before_label, during_label",
    [("x", 0.5, 0.8), ("y", 0.5, 0.8), ("both", 0.75, 0.9)],
)
@pytest.mark.parametrize("title", [False, True])
@pytest.mark.parametrize("label", [False, True])
def test_axis_build_titles_arrive_last_only_when_requested(
    mode, axis, before_label, during_label, title, label
):
    hwostyle.use(mode)
    fig, ax = _axes()
    center = ax.set_title("Center")
    left = ax.set_title("Left", loc="left", alpha=0.65)
    right = ax.set_title("Right", loc="right", alpha=0.4)
    right.set_visible(False)
    titles = (center, left, right)
    alphas = [text.get_alpha() for text in titles]
    plain = _pixels(fig)
    build = ep.build_axes(ax, axis=axis, label=label, spine_end=0.6, title=title)
    np.testing.assert_array_equal(_pixels(fig), plain)

    for u in (0.0, before_label):
        build.update(u)
        if title:
            assert all(text.get_alpha() == 0.0 for text in titles)
        else:
            assert [text.get_alpha() for text in titles] == alphas
        assert not right.get_visible()

    build.update(during_label)
    if title:
        assert 0.0 < center.get_alpha() < 1.0
        assert 0.0 < left.get_alpha() < 0.65
    else:
        assert [text.get_alpha() for text in titles] == alphas
    build.update(1.0)
    assert [text.get_alpha() for text in titles] == alphas
    np.testing.assert_array_equal(_pixels(fig), plain)
    build.update(0.0)
    build.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), plain)


@pytest.mark.parametrize("loc", ["center", "left", "right"])
def test_axis_build_includes_a_plain_title_set_after_the_call(loc):
    _fig, (ax, reference) = plt.subplots(1, 2)
    for axes in (ax, reference):
        axes.plot([0.0, 1.0], [0.0, 1.0])
    made = reference.set_title("late", loc=loc)
    reference_build = ep.build_axes(reference, title=True)
    build = ep.build_axes(ax, title=True)
    text = ax.set_title("late", loc=loc)
    position = text.get_position()
    for u in np.linspace(0.0, 1.0, 11):
        build.update(u)
        reference_build.update(u)
        assert text.get_position() == position
        if u == 0.5:
            assert (1.0 if text.get_alpha() is None else text.get_alpha()) < 1.0
    assert text.get_alpha() == made.get_alpha()
    assert text.get_color() == made.get_color()


@pytest.mark.parametrize("loc", ["center", "left", "right"])
def test_axis_build_includes_a_styled_title_set_after_the_call(loc):
    _fig, ax = _axes()
    ax.plot([0.0, 1.0], [0.5, 2.0])
    build = ep.build_axes(ax, title=True)
    text = ax.set_title("late", loc=loc, alpha=0.6)
    position = text.get_position()
    for u in np.linspace(0.0, 1.0, 11):
        build.update(u)
        assert text.get_position() == position
        if u == 0.5:
            assert text.get_alpha() < 0.6
    assert text.get_alpha() == 0.6


@pytest.mark.parametrize("loc", ["center", "left", "right"])
def test_axis_build_adopts_a_title_restyled_between_frames(loc):
    _fig, ax = _axes()
    text = ax.set_title("old", loc=loc)
    position = text.get_position()
    build = ep.build_axes(ax, title=True)
    build.update(0.5)
    assert text.get_position() == position
    ax.set_title("new", loc=loc, alpha=0.5)
    for u in np.linspace(0.5, 1.0, 6):
        build.update(u)
        assert text.get_position() == position
    assert text.get_alpha() == 0.5
    assert text.get_text() == "new"


@pytest.mark.parametrize("loc", ["center", "left", "right"])
@pytest.mark.parametrize("alpha", [None, 0.6])
def test_axis_build_preserves_a_title_set_before_the_call(loc, alpha):
    _fig, ax = _axes()
    text = ax.set_title("made", loc=loc, alpha=alpha)
    position = text.get_position()
    build = ep.build_axes(ax, title=True)
    for u in np.linspace(0.0, 1.0, 11):
        build.update(u)
        assert text.get_position() == position
        if u == 0.5:
            assert text.get_alpha() < (1.0 if alpha is None else alpha)
    assert text.get_alpha() == alpha
    # Replaying the build must not adopt its own faded alpha as the made state.
    build.update(0.9)
    build.update(0.9)
    build.update(1.0)
    assert text.get_alpha() == alpha
    assert text.get_position() == position


@pytest.mark.parametrize("loc", ["center", "left", "right"])
@pytest.mark.parametrize("alpha", [None, 0.6])
def test_axis_build_leaves_late_titles_untouched_when_not_requested(loc, alpha):
    _fig, ax = _axes()
    build = ep.build_axes(ax, title=False)
    text = ax.set_title("late", loc=loc, alpha=alpha)
    position = text.get_position()
    for u in np.linspace(0.0, 1.0, 11):
        build.update(u)
        assert text.get_alpha() == alpha
        assert text.get_position() == position


def test_axis_build_preserves_title_clipping_changed_between_frames():
    _fig, ax = _axes()
    text = ax.set_title("made")
    build = ep.build_axes(ax, title=True)
    build.update(0.5)
    text.set_clip_on(not text.get_clip_on())
    clip_on = text.get_clip_on()
    text.set_clip_box(ax.bbox)
    for u in np.linspace(0.5, 1.0, 6):
        build.update(u)
        assert text.get_clip_on() == clip_on
        assert text.get_clip_box() is ax.bbox


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


def _line_contrast(fig, ax, bg):
    probe = ax.transData.transform((0.5, 0.5)).astype(int)
    pixel = _pixels(fig)[int(fig.bbox.height - probe[1]), probe[0], :3] / 255.0
    return float(np.abs(pixel - bg).max())


@pytest.mark.parametrize("mode", MODES)
def test_an_eased_weigh_fades_gently_and_keeps_both_ends(mode):
    hwostyle.use(mode)
    fig, ax, groups = _scene()
    stage = ep.Stage(groups, dim=0.35)
    bg = np.asarray(to_rgba(ax.get_facecolor()))[:3]
    full = _line_contrast(fig, ax, bg)

    def presence(u, **kw):
        stage.weigh({"line": u}, **kw)
        return _line_contrast(fig, ax, bg) / full

    tol = 2.0 / 255.0 / full
    assert presence(0.0, ease="smoothstep") == pytest.approx(0.0, abs=tol)
    assert presence(1.0, ease="smoothstep") == pytest.approx(1.0, abs=tol)
    assert presence(0.5, ease="smoothstep") == pytest.approx(0.5, abs=tol)
    assert presence(0.2, ease="smoothstep") < presence(0.2) - 0.05
    assert presence(0.8, ease="smoothstep") > presence(0.8) + 0.05


@pytest.mark.parametrize("contrast", [0.35, 0.6, 1.0])
def test_an_eased_weigh_takes_a_contrast_as_given(contrast):
    hwostyle.use("light")
    fig, ax, groups = _scene()
    stage = ep.Stage(groups, dim=0.35)
    bg = np.asarray(to_rgba(ax.get_facecolor()))[:3]
    full = _line_contrast(fig, ax, bg)
    tol = 2.0 / 255.0 / full
    stage.weigh({"line": (1.0, contrast)}, ease="smoothstep")
    assert _line_contrast(fig, ax, bg) / full == pytest.approx(contrast, abs=tol)


def test_weigh_refuses_an_unknown_easing():
    _fig, _ax, groups = _scene()
    stage = ep.Stage(groups, dim=0.35)
    with pytest.raises(ValueError, match="unknown easing"):
        stage.weigh({"line": 0.5}, ease="bouncy")


@pytest.mark.parametrize("kind", ["back_out", "running_start"])
def test_stage_set_levels_never_pass_their_ends_under_an_overshoot_ease(kind):
    _fig, _ax, groups = _scene()
    stage = ep.Stage(groups, dim=0.35)
    for u in np.linspace(0.0, 1.0, 41):
        levels = stage.set(dimmed=["line"], u=u, ease=kind)
        for name, (presence, contrast) in levels.items():
            end = stage.state(dimmed=["line"])[name]
            lo, hi = sorted((1.0, end[1]))
            assert min(1.0, end[0]) - 1e-9 <= presence <= max(1.0, end[0]) + 1e-9
            assert lo - 1e-9 <= contrast <= hi + 1e-9
    stage.rest()


@pytest.mark.parametrize("kind", ["back_out", "running_start"])
def test_collapse_colors_stay_between_the_cells_and_the_target(kind):
    fig, left, _right = _two_panels()
    weights = np.array([0, 0, 1, 3, 4, 2, 0, 0, 0, 0], dtype=float)
    move = ep.collapse(
        left,
        _cells(),
        weights,
        ep.Anchor.from_data(left, (0.1, 0.5)),
        ep.Anchor.from_data(left, (0.4, 0.5)),
        fig=fig,
        colors="tab:gray",
        color="tab:pink",
        ease=kind,
    )
    ghost = move.artists["line"]
    begin = np.asarray(to_rgba("tab:gray"))[:3]
    end = np.asarray(to_rgba("tab:pink"))[:3]
    low, high = np.minimum(begin, end), np.maximum(begin, end)
    for u in np.linspace(0.0, 1.0, 41):
        move.update(u)
        fig.canvas.draw()
        if not ghost.get_visible():
            continue
        colors = np.asarray(ghost.get_colors())[:, :3]
        assert np.all(colors >= low - 1e-9) and np.all(colors <= high + 1e-9)


@pytest.mark.parametrize("level", [(0.5, 1.0), (1.0, 0.5)])
def test_a_staged_mesh_scales_each_colormap_alpha_and_restores(level):
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        [[0.0, 1.0]],
        cmap=ListedColormap([(1, 0, 0, 0.2), (0, 0, 1, 0.8)]),
        edgecolors=(0, 0, 0, 0.6),
        linewidth=1,
    )
    stage = ep.Stage({"mesh": mesh})
    for _ in range(2):
        stage.weigh({"mesh": level})
        fig.canvas.draw()
        np.testing.assert_allclose(mesh.get_facecolors()[:, 3], [0.1, 0.4])
        np.testing.assert_allclose(mesh.get_edgecolors()[:, 3], 0.3)
    mesh.set_array(np.array([[1.0, 0.0]]))
    stage.weigh({"mesh": level})
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_facecolors()[:, 3], [0.4, 0.1])
    mesh.set_array(np.array([[0.0, 1.0]]))
    stage.rest()
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_facecolors()[:, 3], [0.2, 0.8])
    np.testing.assert_allclose(mesh.get_edgecolors()[:, 3], 0.6)


def test_mesh_edges_that_follow_faces_keep_following_changed_data():
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        [[0.0, 1.0]],
        cmap=ListedColormap([(1, 0, 0, 0.2), (0, 0, 1, 0.8)]),
        edgecolors="face",
    )
    fig.canvas.draw()
    stage = ep.Stage({"mesh": mesh})
    stage.weigh({"mesh": (0.5, 1.0)})
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_edgecolors()[:, 3], [0.1, 0.4])
    mesh.set_array(np.array([[1.0, 0.0]]))
    stage.weigh({"mesh": (0.5, 1.0)})
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_edgecolors()[:, 3], [0.4, 0.1])
    stage.rest()
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_edgecolors()[:, 3], [0.8, 0.2])


@pytest.mark.parametrize("level", [(0.5, 1.0), (1.0, 0.5)])
def test_staging_only_a_mesh_leaves_its_colorbar_unchanged(level):
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(
        [[0.0, 1.0]],
        cmap=ListedColormap([(1, 0, 0, 0.2), (0, 0, 1, 0.8)]),
    )
    colorbar = fig.colorbar(mesh, ax=ax)
    fig.canvas.draw()
    bar_colors = colorbar.solids.get_facecolors().copy()
    stage = ep.Stage({"mesh": mesh})
    for _ in range(2):
        stage.weigh({"mesh": level})
        fig.canvas.draw()
        np.testing.assert_allclose(mesh.get_facecolors()[:, 3], [0.1, 0.4])
        np.testing.assert_allclose(colorbar.solids.get_facecolors(), bar_colors)
    stage.rest()
    fig.canvas.draw()
    np.testing.assert_allclose(mesh.get_facecolors()[:, 3], [0.2, 0.8])
    np.testing.assert_allclose(colorbar.solids.get_facecolors(), bar_colors)
