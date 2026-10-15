"""Attention moves: what the viewer sees, not how it is drawn.

A flight lands on its destination and leaves no trace at either end; a
leader keeps the connector rules; a callout sits beside its subject and is
absent before its beat; all of it in light and dark.
"""

import itertools
import math

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb, to_rgba

import eyepiece as ep
from eyepiece import _style

MODES = ["light", "dark"]
X = np.linspace(-0.2, 0.8, 201)


def _gauss(x, mu, s=0.1):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2.0 * np.pi))


def _two_panels():
    """A curve read at y = 0.35 on top, an empty likelihood axis below."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.0, 5.0), layout="constrained")
    top.plot(X, _gauss(X, 0.25))
    top.axvline(0.35, lw=0.8)
    (dot,) = top.plot([0.35], [_gauss(0.35, 0.25)], "o", ms=7)
    (landed,) = bottom.plot(
        [0.25], [_gauss(0.35, 0.25)], "o", ms=7, color=dot.get_color()
    )
    for ax in (top, bottom):
        ax.set(xlim=(-0.2, 0.8), ylim=(0.0, 6.0))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, top, bottom, dot, landed


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# Anchors


def test_data_anchor_follows_limits_and_resizes():
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    anchor = ep.Anchor.from_data(ax, (0.5, 0.5))
    fig.canvas.draw()
    np.testing.assert_allclose(anchor.display(), ax.transData.transform((0.5, 0.5)))
    ax.set_xlim(0.0, 2.0)
    fig.set_size_inches(6.0, 3.0)
    fig.canvas.draw()
    np.testing.assert_allclose(anchor.display(), ax.transData.transform((0.5, 0.5)))
    assert anchor.axes is ax and anchor.artist is None and anchor.figure is fig


def test_artist_anchor_resolves_a_marker_to_its_point_and_follows_new_data():
    _fig, ax = plt.subplots()
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.3], [0.6], "o", ms=9)
    anchor = ep.Anchor.from_artist(dot)
    np.testing.assert_allclose(anchor.display(), ax.transData.transform((0.3, 0.6)))
    dot.set_data([0.7], [0.2])
    np.testing.assert_allclose(anchor.display(), ax.transData.transform((0.7, 0.2)))
    top = ep.Anchor.from_artist(dot, loc="top").display()
    assert top[1] > anchor.display()[1]
    assert anchor.axes is ax and anchor.artist is dot


def test_anchor_rejects_bad_inputs():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError):
        ep.Anchor.from_data(fig, (0.0, 0.0))
    with pytest.raises(TypeError):
        ep.Anchor.from_artist("not an artist")
    with pytest.raises(ValueError, match="loc"):
        ep.Anchor.from_artist(ax.patch, loc="middle")


# Flight


@pytest.mark.parametrize("mode", MODES)
def test_flight_ghost_is_absent_at_both_ends_and_at_rest(mode):
    hwostyle.use(mode)
    fig, _top, bottom, dot, _landed = _two_panels()
    without = _pixels(fig)
    carry = ep.flight(
        dot, ep.Anchor.from_data(bottom, (0.25, _gauss(0.35, 0.25))), fig=fig
    )
    np.testing.assert_array_equal(_pixels(fig), without)  # the rest state
    for u in (0.0, 1.0):
        carry.update(u)
        np.testing.assert_array_equal(_pixels(fig), without)
    carry.update(0.5)
    assert not np.array_equal(_pixels(fig), without)


@pytest.mark.parametrize("mode", MODES)
def test_flight_lands_on_its_destination_mark(mode):
    hwostyle.use(mode)
    fig, _top, _bottom, dot, landed = _two_panels()
    carry = ep.flight(dot, ep.Anchor.from_artist(landed), fig=fig, arc=0.2)
    ghost = carry.artists["line"]
    for u, target in ((1e-6, dot), (1.0 - 1e-6, landed)):
        carry.update(u)
        fig.canvas.draw()
        where = ghost.get_xydata()[0]
        mark = target.get_transform().transform(target.get_xydata())[0]
        assert np.hypot(*(where - mark)) < 0.5
    # The ghost wears the mark it copies.
    assert to_rgba(ghost.get_color()) == to_rgba(dot.get_color())
    assert ghost.get_marker() == dot.get_marker()
    assert ghost.get_markersize() == dot.get_markersize()
    assert ghost.get_markerfacecolor() == dot.get_markerfacecolor()


def test_flight_lands_where_the_destination_is_drawn_now():
    fig, _top, bottom, dot, _landed = _two_panels()
    carry = ep.flight(dot, ep.Anchor.from_data(bottom, (0.25, 2.0)), fig=fig)
    bottom.set_xlim(0.0, 0.5)  # the destination axis changes after the flight is made
    carry.update(1.0 - 1e-6)
    fig.canvas.draw()
    where = carry.artists["line"].get_xydata()[0]
    assert np.hypot(*(where - bottom.transData.transform((0.25, 2.0)))) < 0.5


def test_flight_ghost_belongs_to_the_figure_and_moves_no_limits():
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    (dot,) = ax.plot([0.5], [0.5], "o")
    limits = (ax.get_xlim(), ax.get_ylim())
    carry = ep.flight(dot, ep.Anchor.from_data(ax, (5.0, 5.0)), fig=fig)
    ghost = carry.artists["line"]
    assert ghost in fig.artists and ghost not in ax.get_children()
    for u in np.linspace(0.0, 1.0, 7):
        carry.update(u)
        fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert set(carry.artists) <= ep.ARTIST_KEYS
    assert list(carry.axes) == [ax]


def test_flight_bows_left_of_its_direction_by_arc_of_the_distance():
    fig, ax = plt.subplots()
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    a, b = ep.Anchor.from_data(ax, (0.2, 0.5)), ep.Anchor.from_data(ax, (0.8, 0.5))
    carry = ep.flight(a, b, fig=fig, arc=0.25, mark=dot)
    carry.update(0.5)
    fig.canvas.draw()
    p0, p1 = a.display(), b.display()
    mid = carry.artists["line"].get_xydata()[0]
    distance = np.hypot(*(p1 - p0))
    # Rightward travel: left is up, and the widest bow is a quarter of the way.
    assert mid[0] == pytest.approx(0.5 * (p0[0] + p1[0]), abs=0.5)
    assert mid[1] - p0[1] == pytest.approx(0.25 * distance, abs=0.5)


@pytest.mark.parametrize("mode", MODES)
def test_flight_into_a_label_crossfades_and_rests_on_the_label(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    value = ax.text(0.2, 0.8, "2.41", bbox=ep.backing())
    label = ax.text(0.8, 0.2, "2.41", bbox=ep.backing())
    fig.canvas.draw()
    with_label = _pixels(fig)
    carry = ep.flight(value, label, fig=fig, crossfade=label, fade=0.4)
    assert "text" in carry.artists
    np.testing.assert_array_equal(_pixels(fig), with_label)  # rest: landed
    carry.update(0.0)
    assert label.get_alpha() == 0.0
    carry.update(0.5)
    assert label.get_alpha() == 0.0 and carry.artists["text"].get_visible()
    carry.update(0.8)
    assert 0.0 < label.get_alpha() < 1.0
    carry.update(1.0)
    assert not carry.artists["text"].get_visible()
    np.testing.assert_array_equal(_pixels(fig), with_label)


def test_flight_update_is_idempotent():
    fig, _top, _bottom, dot, landed = _two_panels()
    carry = ep.flight(dot, landed, fig=fig)
    carry.update(0.3)
    first = _pixels(fig)
    carry.update(0.9)
    carry.update(0.3)
    np.testing.assert_array_equal(_pixels(fig), first)


def test_flight_rejects_what_it_cannot_copy():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="mark"):
        ep.flight(
            ep.Anchor.from_data(ax, (0, 0)), ep.Anchor.from_data(ax, (1, 1)), fig=fig
        )
    with pytest.raises(TypeError):
        ep.flight(ax.imshow(np.zeros((2, 2))), ep.Anchor.from_data(ax, (1, 1)), fig=fig)
    (dot,) = ax.plot([0.0], [0.0], "o")
    with pytest.raises(ValueError, match="fade"):
        ep.flight(dot, dot, fig=fig, fade=0.0)


# Leader


def _leader_ends(line):
    xy = line.get_xydata()
    return xy[0], xy[-1]


@pytest.mark.parametrize("mode", MODES)
def test_leader_ends_on_its_mark_and_takes_a_neutral(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.5], [0.5], "o", ms=8)
    note = ax.text(0.62, 0.62, "the dot")
    res = ep.leader(note, dot)
    line = res.artists["line"]
    fig.canvas.draw()
    start, end = _leader_ends(line)
    mark = dot.get_window_extent().padded(0.5)
    assert mark.contains(*end)
    assert not note.get_window_extent().contains(*start)
    assert line.faults() == []
    color = to_rgb(line.get_color())
    assert color == pytest.approx(_style.neutral(0.6))
    assert color != pytest.approx(to_rgb(ax.get_facecolor()))
    assert color != pytest.approx(to_rgb(plt.rcParams["text.color"]))


def test_leader_grows_from_the_text():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.5], [0.5], "o")
    res = ep.leader(ax.text(0.62, 0.62, "the dot"), dot)
    line = res.artists["line"]
    fig.canvas.draw()
    start, end = _leader_ends(line)
    res.update(0.5)
    fig.canvas.draw()
    s, tip = _leader_ends(line)
    np.testing.assert_allclose(s, start)
    np.testing.assert_allclose(tip, start + 0.5 * (end - start))
    res.update(0.0)
    hidden = _pixels(fig)
    line.remove()
    np.testing.assert_array_equal(_pixels(fig), hidden)


def test_leader_too_long_or_across_data_is_a_fault():
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.plot([0.0, 1.0], [0.3, 0.3], label="baseline")
    (dot,) = ax.plot([0.5], [0.15], "o")
    far = ax.text(0.95, 0.95, "the dot")
    res = ep.leader(far, dot, strict=False)
    faults = res.artists["line"].faults()
    assert any("quarter" in f for f in faults)
    assert any("crosses data: baseline" in f for f in faults)
    near = ax.text(0.55, 0.22, "the dot")
    assert ep.leader(near, dot).artists["line"].faults() == []


def test_a_broken_leader_draws_and_the_audit_names_it():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.05], [0.05], "o")
    far = ep.leader(ax.text(0.95, 0.95, "far away"), dot)
    fig.canvas.draw()  # a still or a PDF save is never aborted mid-draw
    assert len(far.artists["line"].get_xydata()) == 2
    with pytest.raises(ep.AttentionFault, match="quarter") as info:
        ep.audit(fig)
    assert isinstance(info.value, ValueError)
    assert far.artists["line"] in info.value.faults
    found = ep.audit(fig, raise_on_fault=False)
    assert any("quarter" in m for m in found[far.artists["line"]])


def test_audit_skips_lenient_and_hidden_lines_and_passes_clean_ones():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.05], [0.05], "o")
    ep.leader(ax.text(0.95, 0.95, "far away"), dot, strict=False)
    assert ep.audit(fig)  # reported, not raised
    ep.leader(ax.text(0.1, 0.3, "near"), dot)
    near_fig, near_ax = plt.subplots(figsize=(6.0, 4.0))
    near_ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (mark,) = near_ax.plot([0.5], [0.5], "o", ms=8)
    ep.leader(near_ax.text(0.55, 0.22, "the dot"), mark)
    assert ep.audit(near_fig) == {}


def test_leader_does_not_run_through_its_mark():
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    circle = ep.overlay_circle(ax, (0.5, 0.5), 0.05).artists["ellipse"][-1]
    note = ax.text(0.5, 0.3, "ring", ha="center", va="top")
    # Anchored on the ring's top while the text sits below it: the leader
    # would cross the ring's box to get there.
    res = ep.leader(note, ep.Anchor.from_artist(circle, loc="top"), strict=False)
    assert any("runs through" in f for f in res.artists["line"].faults())


def test_leader_in_a_data_color_is_a_fault():
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (curve,) = ax.plot([0.0, 1.0], [0.9, 0.9], color="tab:orange")
    (dot,) = ax.plot([0.5], [0.5], "o")
    res = ep.leader(
        ax.text(0.6, 0.6, "dot"), dot, color=curve.get_color(), strict=False
    )
    assert any("data color" in f for f in res.artists["line"].faults())


# Callout


@pytest.mark.parametrize("mode", MODES)
def test_callout_sits_beside_its_subject_on_a_mode_box(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.3], [0.5], "o", ms=8)
    res = ep.callout(dot, "the measured height", side="right")
    caption = res.artists["text"]
    fig.canvas.draw()
    box, mark = caption.get_window_extent(), dot.get_window_extent()
    em = caption.get_fontsize() * fig.dpi / 72.0
    assert 0.0 < box.x0 - mark.x1 < em
    assert box.y0 < 0.5 * (mark.y0 + mark.y1) < box.y1
    face = caption.get_bbox_patch().get_facecolor()
    assert to_rgb(face) == pytest.approx(to_rgb(ax.get_facecolor()))
    assert to_rgb(caption.get_color()) == pytest.approx(
        to_rgb(plt.rcParams["text.color"])
    )
    assert caption in fig.artists and caption not in ax.texts
    # Adjacent to its subject, the caption needs no leader.
    line = res.artists["line"]
    assert line._ends() is None


def test_callout_auto_side_keeps_the_caption_in_its_panel():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.canvas.draw()
    (dot,) = ax.plot([0.97], [0.5], "o")
    caption = ep.callout(dot, "near the right edge").artists["text"]
    fig.canvas.draw()
    box = caption.get_window_extent()
    assert box.x1 <= ax.bbox.x1 and box.x0 >= ax.bbox.x0
    assert box.x1 < dot.get_window_extent().x0  # it went to the left


@pytest.mark.parametrize("mode", MODES)
def test_callout_is_absent_before_its_beat_and_whole_at_rest(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.3], [0.5], "o")
    without = _pixels(fig)
    res = ep.callout(dot, "the dot", at=ep.Anchor.from_data(ax, (0.45, 0.62)))
    rest = _pixels(fig)
    assert not np.array_equal(rest, without)
    res.update(0.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    res.update(0.25)  # the caption fading in, no leader yet
    assert not res.artists["line"].get_visible()
    res.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), rest)
    # Placed away from its subject, the caption has a leader that keeps the rules.
    line = res.artists["line"]
    assert line._ends() is not None and line.faults() == []


def test_callout_caps_its_words():
    _fig, ax = plt.subplots()
    (dot,) = ax.plot([0.3], [0.5], "o")
    with pytest.raises(ValueError, match="words"):
        ep.callout(dot, "one two three four five six seven eight nine")
    with pytest.raises(ValueError, match="side"):
        ep.callout(dot, "short", side="north")


def test_leader_to_a_plotted_data_point_ignores_its_marker_and_the_scenery():
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.plot([0.5], [0.5], "o", ms=10)
    ax.axvline(0.55)
    ax.axvspan(0.52, 0.58, alpha=0.2)
    note = ax.text(0.62, 0.62, "the dot")
    res = ep.leader(note, ep.Anchor.from_data(ax, (0.5, 0.5)))
    assert res.artists["line"].faults() == []


def test_leader_across_another_text_is_a_fault():
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.5], [0.5], "o")
    ax.text(0.56, 0.56, "other", ha="center", va="center")
    res = ep.leader(ax.text(0.62, 0.62, "the dot"), dot, strict=False)
    assert any("crosses text: 'other'" in f for f in res.artists["line"].faults())


@pytest.mark.parametrize("mode", MODES)
def test_callout_dims_and_restores(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.3], [0.5], "o")
    without = _pixels(fig)
    res = ep.callout(dot, "the dot", at=ep.Anchor.from_data(ax, (0.45, 0.62)))
    rest = _pixels(fig)
    res.update(1.0, level=0.35)
    dimmed = _pixels(fig)
    assert not np.array_equal(dimmed, rest)
    assert not np.array_equal(dimmed, without)
    res.update(1.0, level=0.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    res.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), rest)


def test_flight_refuses_a_crossfade_label_drawn_invisible():
    fig, ax = plt.subplots()
    (dot,) = ax.plot([0.2], [0.2], "o")
    label = ax.text(0.8, 0.8, "2.41", alpha=0.0)
    with pytest.raises(ValueError, match="alpha 0"):
        ep.flight(dot, label, fig=fig, crossfade=label)


def test_anchor_on_an_artist_outside_a_figure_says_so():
    from matplotlib.lines import Line2D

    with pytest.raises(ValueError, match="not in a figure"):
        _ = ep.Anchor.from_artist(Line2D([0.0], [0.0])).figure


# Indicate


def _marks(mode):
    """One of each artist kind a pulse works on, in a figure of `mode`."""
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (line,) = ax.plot([0.1, 0.5, 0.9], [0.2, 0.7, 0.3], "o-", lw=2.0, ms=7, color="C1")
    note = ax.text(0.15, 0.85, "a note", bbox=ep.backing())
    bar = ax.add_patch(plt.Rectangle((0.6, 0.1), 0.2, 0.15, fc="C2", ec="C3", lw=1.5))
    dots = ax.scatter([0.3, 0.5], [0.4, 0.5], s=40, color="C4")
    fig.canvas.draw()
    return fig, ax, line, note, bar, dots


@pytest.mark.parametrize("mode", MODES)
def test_indicate_leaves_both_ends_pixel_identical(mode):
    fig, _ax, line, note, bar, dots = _marks(mode)
    without = _pixels(fig)
    pulse = ep.indicate([line, note, bar, dots])
    np.testing.assert_array_equal(_pixels(fig), without)  # the rest state
    for u in (0.0, 1.0, -3.0, 7.0):
        pulse.update(u)
        np.testing.assert_array_equal(_pixels(fig), without)
    pulse.update(0.5)
    assert not np.array_equal(_pixels(fig), without)
    pulse.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), without)


@pytest.mark.parametrize("mode", MODES)
def test_indicate_peaks_mid_way(mode):
    fig, _ax, line, note, bar, dots = _marks(mode)
    without = _pixels(fig).astype(float)
    pulse = ep.indicate([line, note, bar, dots])
    change = []
    for u in (0.1, 0.25, 0.5, 0.75, 0.9):
        pulse.update(u)
        change.append(np.abs(_pixels(fig).astype(float) - without).sum())
    assert 0.0 < change[0] < change[1] < change[2]
    assert 0.0 < change[4] < change[3] < change[2]
    assert change[1] == pytest.approx(change[3], rel=0.15)


@pytest.mark.parametrize("mode", MODES)
def test_indicate_swells_by_the_scale_and_tints_toward_the_text_color(mode):
    _fig, _ax, line, note, bar, dots = _marks(mode)
    made = {
        "lw": line.get_linewidth(),
        "ms": line.get_markersize(),
        "font": note.get_fontsize(),
        "bar_lw": bar.get_linewidth(),
        "dot_lw": float(dots.get_linewidths()[0]),
        "dot_s": float(dots.get_sizes()[0]),
        "color": np.array(to_rgb(line.get_color())),
        "face": np.array(bar.get_facecolor()[:3]),
    }
    pulse = ep.indicate([line, note, bar, dots], scale=1.15, strength=0.5)
    pulse.update(0.5)
    assert line.get_linewidth() == pytest.approx(1.15 * made["lw"])
    assert line.get_markersize() == pytest.approx(1.15 * made["ms"])
    assert note.get_fontsize() == pytest.approx(1.15 * made["font"])
    assert bar.get_linewidth() == pytest.approx(1.15 * made["bar_lw"])
    assert float(dots.get_linewidths()[0]) == pytest.approx(1.15 * made["dot_lw"])
    assert float(dots.get_sizes()[0]) == pytest.approx(1.15**2 * made["dot_s"])
    text = np.array(to_rgb(plt.rcParams["text.color"]))
    for before, after in (
        (made["color"], np.array(to_rgb(line.get_color()))),
        (made["face"], np.array(bar.get_facecolor()[:3])),
    ):
        # Half way to the text color along the straight line: no new hue.
        np.testing.assert_allclose(after, before + 0.5 * (text - before), atol=1e-6)
    assert bar.get_facecolor()[3] == 1.0


def test_indicate_swells_a_callout_with_its_box_and_pieces():
    hwostyle.use("dark")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.5], [0.5], "o", ms=8)
    call = ep.callout(
        dot,
        ["the ", ("dot", "C3", "bold")],
        side="right",
        leader=False,
    )
    text = call.artists["text"]
    fig.canvas.draw()
    without = _pixels(fig)
    width = text.get_window_extent().width
    sizes = [p.get_fontsize() for p in text._pieces]
    pulse = ep.indicate(text)
    pulse.update(0.5)
    fig.canvas.draw()
    assert text.get_window_extent().width > width
    for piece, size in zip(text._pieces, sizes, strict=True):
        assert piece.get_fontsize() == pytest.approx(1.15 * size)
    pulse.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), without)


def test_indicate_is_idempotent_and_order_free():
    fig, _ax, line, note, bar, dots = _marks("light")
    pulse = ep.indicate([line, note, bar, dots])
    pulse.update(0.3)
    first = _pixels(fig)
    pulse.update(0.8)
    pulse.update(0.3)
    pulse.update(0.3)
    np.testing.assert_array_equal(_pixels(fig), first)


@pytest.mark.parametrize("mode", MODES)
def test_indicate_returns_to_the_stage_level_not_the_made_colors(mode):
    fig, _ax, line, note, bar, dots = _marks(mode)
    stage = ep.Stage({"group": [line, note, bar, dots]}, dim=0.4)
    made = _pixels(fig)
    pulse = ep.indicate([line, note, bar, dots])
    # The caller sets the stage, then the pulse, every frame.
    stage.set(dimmed=["group"])
    pulse.update(0.0)
    dimmed = _pixels(fig)
    assert not np.array_equal(dimmed, made)
    for u in (0.0, 0.3, 0.5, 0.8, 1.0):
        stage.set(dimmed=["group"])
        pulse.update(u)
        if u in (0.0, 1.0):
            np.testing.assert_array_equal(_pixels(fig), dimmed)
    # At the peak the line is between its dimmed color and the text color,
    # darker or lighter than the dimmed level, never the made color.
    stage.set(dimmed=["group"])
    dim_color = np.array(to_rgb(line.get_color()))
    pulse.update(0.5)
    peak = np.array(to_rgb(line.get_color()))
    text = np.array(to_rgb(plt.rcParams["text.color"]))
    t = np.dot(peak - dim_color, text - dim_color) / np.dot(
        text - dim_color, text - dim_color
    )
    assert 0.0 < t < 1.0
    np.testing.assert_allclose(peak, dim_color + t * (text - dim_color), atol=1e-6)
    # The stage going back to full strength while the pulse is over restores
    # the made colors exactly.
    pulse.update(1.0)
    stage.set(shown=["group"])
    np.testing.assert_array_equal(_pixels(fig), made)


def test_indicate_rejects_what_it_cannot_pulse():
    fig, ax = plt.subplots()
    image = ax.imshow(np.zeros((2, 2)))
    with pytest.raises(TypeError, match="indicate"):
        ep.indicate(image)
    with pytest.raises(ValueError, match="scale"):
        ep.indicate(ax.plot([0], [0])[0], scale=0.9)
    with pytest.raises(ValueError, match="strength"):
        ep.indicate(ax.plot([0], [0])[0], strength=1.5)
    with pytest.raises(ValueError, match="no artist"):
        ep.indicate([])
    assert fig


# Circumscribe


def _ink_box(fig, without):
    """Display-space box (x0, x1, y0, y1) of the pixels that differ, or None."""
    now = _pixels(fig)
    rows, cols = np.nonzero(np.any(now != without, axis=-1))
    if not len(rows):
        return None
    height = now.shape[0]
    return cols.min(), cols.max() + 1, height - rows.max() - 1, height - rows.min()


def _ink(fig, without):
    return int(np.any(_pixels(fig) != without, axis=-1).sum())


def _labelled_panel():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.plot([0.0, 1.0], [0.1, 0.2], color=ep.neutral(0.3))
    label = ax.text(0.5, 0.6, "this pixel", ha="center", va="center")
    fig.canvas.draw()
    return fig, ax, label


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("shape", ["rect", "ellipse"])
def test_circumscribe_is_absent_at_both_ends_and_at_rest(mode, shape):
    hwostyle.use(mode)
    fig, _ax, label = _labelled_panel()
    without = _pixels(fig)
    ring = ep.circumscribe(label, shape=shape)
    for u in (None, 0.0, 1.0):
        if u is not None:
            ring.update(u)
        np.testing.assert_array_equal(_pixels(fig), without)
    ring.update(0.4)
    assert _ink(fig, without) > 0


def test_circumscribe_rect_encloses_its_target_at_the_pad():
    fig, _ax, label = _labelled_panel()
    without = _pixels(fig)
    box = label.get_window_extent()
    ring = ep.circumscribe(label, pad_pt=6.0, lw=1.0)
    pad = 6.0 * fig.dpi / 72.0
    ring.update(0.6)  # fully drawn, not yet fading
    x0, x1, y0, y1 = _ink_box(fig, without)
    for got, want in (
        (x0, box.x0 - pad),
        (x1, box.x1 + pad),
        (y0, box.y0 - pad),
        (y1, box.y1 + pad),
    ):
        assert abs(got - want) <= 3.0
    # Nothing inside the target's box.
    now = _pixels(fig)
    h = now.shape[0]
    inside = (slice(h - int(box.y1), h - int(box.y0)), slice(int(box.x0), int(box.x1)))
    np.testing.assert_array_equal(now[inside], without[inside])


def test_circumscribe_ellipse_passes_outside_the_corners_of_the_box():
    fig, _ax, label = _labelled_panel()
    without = _pixels(fig)
    box = label.get_window_extent()
    ring = ep.circumscribe(label, shape="ellipse", pad_pt=2.0)
    ring.update(0.6)
    x0, x1, y0, y1 = _ink_box(fig, without)
    assert x0 < box.x0 and x1 > box.x1 and y0 < box.y0 and y1 > box.y1
    # Its horizontal extent is that of the ellipse through the corners.
    half = 0.5 * (x1 - x0)
    assert half == pytest.approx(
        math.sqrt(0.5) * box.width + 2.0 * fig.dpi / 72.0, abs=2.0
    )


def test_circumscribe_draws_on_then_fades():
    fig, _ax, label = _labelled_panel()
    without = _pixels(fig)
    ring = ep.circumscribe(label, draw=0.6)
    inked = []
    for u in (0.1, 0.3, 0.5, 0.6):
        ring.update(u)
        inked.append(_ink(fig, without))
    assert 0 < inked[0] < inked[1] < inked[2] < inked[3]
    full = _pixels(fig).astype(float)
    strength = []
    for u in (0.6, 0.75, 0.9):
        ring.update(u)
        strength.append(np.abs(_pixels(fig).astype(float) - without).sum())
    assert strength[0] > strength[1] > strength[2] > 0.0
    np.testing.assert_array_equal(full.shape, without.shape)


def test_circumscribe_follows_its_target_and_takes_a_given_color():
    fig, _ax, label = _labelled_panel()
    without = _pixels(fig)
    ring = ep.circumscribe(label, color="C3")
    ring.update(0.6)
    first = _ink_box(fig, without)
    ring.update(0.0)
    label.set_position((0.7, 0.6))
    fig.canvas.draw()
    moved_without = _pixels(fig)
    ring.update(0.0)
    np.testing.assert_array_equal(_pixels(fig), moved_without)
    ring.update(0.6)
    second = _ink_box(fig, moved_without)
    assert second[0] > first[0] + 20
    assert to_rgba(ring.artists["line"].get_color()) == to_rgba("C3")


def test_circumscribe_belongs_to_the_figure_and_moves_no_limits():
    fig, ax, _label = _labelled_panel()
    limits = (ax.get_xlim(), ax.get_ylim())
    ring = ep.circumscribe(ep.Anchor.from_data(ax, (0.5, 0.5)), shape="ellipse")
    line = ring.artists["line"]
    assert line in fig.artists and line not in ax.get_children()
    for u in np.linspace(0.0, 1.0, 9):
        ring.update(u)
        fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert set(ring.artists) <= ep.ARTIST_KEYS
    # A data point has no extent: a ring of the pad around it.
    ring.update(0.6)
    ends = line.outline()
    center = ax.transData.transform((0.5, 0.5))
    np.testing.assert_allclose(np.hypot(*(ends - center).T), 4.0 * fig.dpi / 72.0)


def test_circumscribe_rejects_bad_inputs():
    _fig, ax, label = _labelled_panel()
    with pytest.raises(ValueError, match="shape"):
        ep.circumscribe(label, shape="star")
    with pytest.raises(ValueError, match="draw"):
        ep.circumscribe(label, draw=1.0)
    with pytest.raises(TypeError):
        ep.circumscribe("label")
    assert ax


# Term row


def _row_panel():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    ax.plot([0.0, 1.0], [0.05, 0.1], color=ep.neutral(0.3))
    return fig, ax


EQUATION = [r"$\chi^2$", "=", r"$\sum_i (y_i - m_i)^2$", "=", "0.00"]


def _boxes(row, fig):
    fig.canvas.draw()
    return [t.get_window_extent() for t in row.artists["text"]]


@pytest.mark.parametrize("mode", MODES)
def test_term_row_terms_read_left_to_right_without_overlap_on_one_baseline(mode):
    hwostyle.use(mode)
    fig, ax = _row_panel()
    start = ep.Anchor.from_data(ax, (0.1, 0.6))
    row = ep.term_row(start, EQUATION, gap_em=0.5)
    boxes = _boxes(row, fig)
    em = plt.rcParams["font.size"] * fig.dpi / 72.0
    for left, right in itertools.pairwise(boxes):
        assert right.x0 - left.x1 == pytest.approx(0.5 * em, abs=1.5)
    assert boxes[0].x0 == pytest.approx(start.display()[0], abs=1.5)
    ys = {round(t.get_position()[1], 3) for t in row.artists["text"]}
    assert ys == {round(float(start.display()[1]), 3)}


def test_term_row_shares_a_baseline_in_the_pixels():
    fig, ax = _row_panel()
    row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.6)), ["x", "$x$", "x"])
    texts = row.artists["text"]
    fig.canvas.draw()
    blank = {}
    for i in range(3):
        for j, t in enumerate(texts):
            t.set_alpha(1.0 if i == j else 0.0)
        blank[i] = _pixels(fig)
    for t in texts:
        t.set_alpha(0.0)
    empty = _pixels(fig)
    bottoms = []
    for i in range(3):
        rows, _cols = np.nonzero(np.any(blank[i] != empty, axis=-1))
        bottoms.append(rows.max())
    assert max(bottoms) - min(bottoms) <= 1


def test_term_row_relays_out_when_a_term_changes():
    fig, ax = _row_panel()
    row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.6)), EQUATION)
    first = _boxes(row, fig)
    texts = row.artists["text"]
    texts[2].set_text("$\\sum_i^{N} (y_i - m_i)^2 / \\sigma_i^2$")
    after = _boxes(row, fig)
    assert after[2].width > first[2].width
    grown = after[2].width - first[2].width
    # The neighbors made room, by exactly what the term grew.
    assert after[3].x0 - first[3].x0 == pytest.approx(grown, abs=1.0)
    assert after[0].x0 == pytest.approx(first[0].x0)
    for left, right in itertools.pairwise(after):
        assert right.x0 >= left.x1
    texts[2].set_text("")  # a blank readout has no slot and no gap
    blank = _boxes(row, fig)
    assert blank[3].x0 < first[3].x0


def test_term_row_terms_are_anchors_before_any_draw_and_follow_the_readout():
    fig, ax = _row_panel()
    row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.6)), EQUATION)
    anchor = ep.Anchor.from_artist(row.artists["text"][4], loc="left")
    first = anchor.display()  # no draw yet
    fig.canvas.draw()
    np.testing.assert_allclose(anchor.display(), first, atol=0.01)
    row.artists["text"][2].set_text(r"$\sum$")
    assert anchor.display()[0] < first[0] - 20


def test_term_row_ha_places_the_row_by_its_end():
    fig, ax = _row_panel()
    at = ep.Anchor.from_data(ax, (0.5, 0.6))
    spans = {}
    for ha in ("left", "center", "right"):
        row = ep.term_row(at, EQUATION, ha=ha)
        boxes = _boxes(row, fig)
        spans[ha] = (boxes[0].x0, boxes[-1].x1)
        for t in row.artists["text"]:
            t.remove()
    x = at.display()[0]
    assert spans["left"][0] == pytest.approx(x, abs=1.5)
    assert 0.5 * sum(spans["center"]) == pytest.approx(x, abs=1.5)
    assert spans["right"][1] == pytest.approx(x, abs=1.5)


def test_term_row_moves_no_data_limit_and_belongs_to_the_figure():
    fig, ax = _row_panel()
    limits = (ax.get_xlim(), ax.get_ylim())
    row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.6)), EQUATION)
    for t in row.artists["text"]:
        assert t in fig.artists and t not in ax.get_children()
    row.artists["text"][4].set_text("12345.67")
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert set(row.artists) <= ep.ARTIST_KEYS


@pytest.mark.parametrize("mode", MODES)
def test_term_row_terms_stage_flight_and_indicate_alone(mode):
    hwostyle.use(mode)
    fig, ax = _row_panel()
    row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.6)), EQUATION)
    terms = row.artists["text"]
    fig.canvas.draw()
    made = _pixels(fig)
    boxes = [t.get_window_extent() for t in terms]
    stage = ep.Stage({"lhs": terms[:1], "rest": terms[1:]})
    stage.set(shown=["lhs"])
    fig.canvas.draw()
    assert to_rgba(terms[0].get_color())[3] == 1.0
    assert to_rgba(terms[3].get_color())[3] == 0.0
    # Hiding terms by the stage leaves the layout where it was.
    for t, b in zip(terms, boxes, strict=True):
        assert t.get_window_extent().x0 == pytest.approx(b.x0)
    stage.rest()
    np.testing.assert_array_equal(_pixels(fig), made)
    pulse = ep.indicate(terms[2])
    pulse.update(0.5)
    fig.canvas.draw()
    swollen = terms[2].get_window_extent()
    assert swollen.width > boxes[2].width
    for k in (0, 1, 3, 4):  # no neighbor moved
        assert terms[k].get_window_extent().x0 == pytest.approx(boxes[k].x0)
    pulse.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), made)
    carry = ep.flight(
        terms[4], ep.Anchor.from_data(ax, (0.8, 0.3)), fig=fig, mark=terms[4]
    )
    carry.update(1.0 - 1e-6)
    fig.canvas.draw()
    where = np.array(carry.artists["text"].get_position())
    assert np.hypot(*(where - ax.transData.transform((0.8, 0.3)))) < 0.5


def test_term_row_writes_on_term_by_term_and_rests_whole():
    fig, ax = _row_panel()
    row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.6)), EQUATION)
    texts = row.artists["text"]
    fig.canvas.draw()
    whole = _pixels(fig)
    row.update(0.0)
    assert not any(t.get_visible() for t in texts)
    row.update(0.5)  # 5 terms: two done, the third arriving
    assert [t.get_visible() for t in texts] == [True, True, True, False, False]
    assert texts[0].get_alpha() in (None, 1.0) and 0.0 < texts[2].get_alpha() < 1.0
    row.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), whole)


def test_term_row_rejects_bad_inputs():
    _fig, ax = _row_panel()
    at = ep.Anchor.from_data(ax, (0.1, 0.6))
    with pytest.raises(ValueError, match="no terms"):
        ep.term_row(at, [])
    with pytest.raises(ValueError, match="ha"):
        ep.term_row(at, ["x"], ha="top")
    with pytest.raises(ValueError, match="gap"):
        ep.term_row(at, ["x"], gap_em=-1.0)
    with pytest.raises(TypeError):
        ep.term_row("here", ["x"])


# Turning flight


def _stub_panels():
    """A short slanted segment and a bar in one panel, a velocity axis beside it."""
    fig, (star, speed) = plt.subplots(1, 2, figsize=(7.0, 3.0), layout="constrained")
    star.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    speed.set(xlim=(0.0, 1.0), ylim=(-1.0, 1.0))
    (stub,) = star.plot(
        [0.1, 0.4], [0.1, 0.3], lw=3.0, color="C2", solid_capstyle="butt"
    )
    bar = star.add_patch(
        plt.Rectangle((-0.6, -0.5), 0.3, 0.1, fc="C3", ec="C4", lw=2.0)
    )
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, star, speed, stub, bar


def _ghost_geometry(carry, fig, u):
    carry.update(u)
    fig.canvas.draw()
    xy = carry.artists["line"].get_xydata()
    return xy[0], xy[-1], 0.5 * (xy.min(axis=0) + xy.max(axis=0))


@pytest.mark.parametrize("mode", MODES)
def test_turning_flight_is_absent_at_both_ends_and_at_rest(mode):
    hwostyle.use(mode)
    fig, _star, speed, stub, bar = _stub_panels()
    without = _pixels(fig)
    for mark in (stub, bar):
        carry = ep.flight(
            mark, ep.Anchor.from_data(speed, (0.5, 0.0)), fig=fig, turn_deg=90.0
        )
        np.testing.assert_array_equal(_pixels(fig), without)  # the rest state
        for u in (0.0, 1.0):
            carry.update(u)
            np.testing.assert_array_equal(_pixels(fig), without)
        carry.update(0.5)
        assert not np.array_equal(_pixels(fig), without)
        carry.update(1.0)


def test_turning_flight_carries_a_segment_and_turns_it_a_quarter_turn():
    fig, _star, speed, stub, _bar = _stub_panels()
    dest = ep.Anchor.from_data(speed, (0.5, 0.2))
    carry = ep.flight(stub, dest, fig=fig, turn_deg=90.0)
    ghost = carry.artists["line"]
    src = stub.get_transform().transform(stub.get_xydata())
    vec = src[1] - src[0]
    length = np.hypot(*vec)
    # Just after take-off it is the source segment, on the source segment.
    a, b, center = _ghost_geometry(carry, fig, 1e-6)
    np.testing.assert_allclose(b - a, vec, atol=0.5)
    np.testing.assert_allclose(center, src.mean(axis=0), atol=0.5)
    # Half way (smoothstep is 0.5 there) it has turned an eighth of a turn.
    a, b, _ = _ghost_geometry(carry, fig, 0.5)
    got = b - a
    assert np.hypot(*got) == pytest.approx(length, abs=0.5)
    turned = math.degrees(math.atan2(got[1], got[0]) - math.atan2(vec[1], vec[0]))
    assert turned == pytest.approx(45.0, abs=0.5)
    # Just before landing it is the segment turned 90 degrees, on the destination.
    a, b, center = _ghost_geometry(carry, fig, 1.0 - 1e-6)
    np.testing.assert_allclose(b - a, [-vec[1], vec[0]], atol=0.5)
    assert np.hypot(*(center - dest.display())) < 0.5
    # It wears the segment it copies.
    assert ghost.get_linewidth() == stub.get_linewidth()
    assert to_rgba(ghost.get_color()) == to_rgba(stub.get_color())
    assert ghost.get_solid_capstyle() == "butt"


def test_turning_flight_turns_clockwise_for_a_negative_angle():
    fig, _star, speed, stub, _bar = _stub_panels()
    carry = ep.flight(
        stub, ep.Anchor.from_data(speed, (0.5, 0.2)), fig=fig, turn_deg=-90.0
    )
    src = stub.get_transform().transform(stub.get_xydata())
    vec = src[1] - src[0]
    a, b, _ = _ghost_geometry(carry, fig, 1.0 - 1e-6)
    np.testing.assert_allclose(b - a, [vec[1], -vec[0]], atol=0.5)


def test_flight_without_a_turn_keeps_the_segment_orientation():
    fig, _star, speed, stub, _bar = _stub_panels()
    carry = ep.flight(stub, ep.Anchor.from_data(speed, (0.5, 0.2)), fig=fig)
    src = stub.get_transform().transform(stub.get_xydata())
    a, b, _ = _ghost_geometry(carry, fig, 0.7)
    np.testing.assert_allclose(b - a, src[1] - src[0], atol=0.5)


def test_flight_carries_a_bar_patch_and_turns_it():
    fig, _star, speed, _stub, bar = _stub_panels()
    dest = ep.Anchor.from_data(speed, (0.5, -0.4))
    carry = ep.flight(bar, dest, fig=fig, turn_deg=90.0)
    ghost = carry.artists["fill"]
    assert ghost in fig.artists and set(carry.artists) <= ep.ARTIST_KEYS
    box = bar.get_window_extent()

    def ghost_box(u):
        carry.update(u)
        fig.canvas.draw()
        return ghost.get_path().get_extents()

    start = ghost_box(1e-6)
    assert start.width == pytest.approx(box.width, abs=1.5)  # strokes aside
    np.testing.assert_allclose(
        [0.5 * (start.x0 + start.x1), 0.5 * (start.y0 + start.y1)],
        [0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)],
        atol=0.5,
    )
    end = ghost_box(1.0 - 1e-6)
    assert end.width == pytest.approx(start.height, abs=0.5)
    assert end.height == pytest.approx(start.width, abs=0.5)
    center = [0.5 * (end.x0 + end.x1), 0.5 * (end.y0 + end.y1)]
    assert np.hypot(*(np.array(center) - dest.display())) < 0.5
    assert to_rgba(ghost.get_facecolor()) == to_rgba(bar.get_facecolor())
    assert to_rgba(ghost.get_edgecolor()) == to_rgba(bar.get_edgecolor())
    assert ghost.get_linewidth() == bar.get_linewidth()


def test_turning_flight_ghost_moves_no_limits_and_is_idempotent():
    fig, star, speed, stub, bar = _stub_panels()
    limits = [(a.get_xlim(), a.get_ylim()) for a in (star, speed)]
    carries = [
        ep.flight(m, ep.Anchor.from_data(speed, (0.5, 0.0)), fig=fig, turn_deg=90.0)
        for m in (stub, bar)
    ]
    for u in np.linspace(0.0, 1.0, 7):
        for carry in carries:
            carry.update(u)
        fig.canvas.draw()
    assert [(a.get_xlim(), a.get_ylim()) for a in (star, speed)] == limits
    for carry in carries:
        carry.update(0.4)
    first = _pixels(fig)
    for carry in carries:
        carry.update(0.9)
        carry.update(0.4)
    np.testing.assert_array_equal(_pixels(fig), first)


def test_flight_turns_a_text_and_refuses_to_turn_a_marker():
    fig, ax = plt.subplots()
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    note = ax.text(0.2, 0.8, "v")
    carry = ep.flight(note, ep.Anchor.from_data(ax, (0.8, 0.2)), fig=fig, turn_deg=90.0)
    carry.update(1.0 - 1e-6)
    fig.canvas.draw()
    assert carry.artists["text"].get_rotation() == pytest.approx(90.0, abs=0.01)
    (dot,) = ax.plot([0.5], [0.5], "o")
    with pytest.raises(ValueError, match="marker"):
        ep.flight(dot, ep.Anchor.from_data(ax, (0.8, 0.2)), fig=fig, turn_deg=45.0)
