"""Attention moves: what the viewer sees, not how it is drawn.

A flight lands on its destination and leaves no trace at either end; a
leader keeps the connector rules; a callout sits beside its subject and is
absent before its beat; all of it in light and dark.
"""

import io
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


@pytest.mark.parametrize("mode", MODES)
def test_a_fading_callout_box_is_always_fainter_than_its_words(mode):
    hwostyle.use(mode)
    _fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.imshow(np.linspace(0.0, 1.0, 64).reshape(8, 8), extent=(0, 1, 0, 1))
    (dot,) = ax.plot([0.3], [0.5], "o")
    res = ep.callout(dot, "the dot", at=ep.Anchor.from_data(ax, (0.45, 0.62)))
    caption = res.artists["text"]
    box = caption.get_bbox_patch()
    full = to_rgba(box.get_facecolor())[3]
    for level in (0.1, 0.4, 0.7, 0.95):
        res.update(1.0, level=level)
        words = caption.get_alpha()
        assert to_rgba(box.get_facecolor())[3] / full < words, level
    res.update(1.0)
    assert to_rgba(box.get_facecolor())[3] == pytest.approx(full)


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


# Link


def _image_and_phasor():
    """An image panel on the left, a phasor panel on the right, a gap between."""
    fig, (image, phasor) = plt.subplots(
        1, 2, figsize=(8.0, 3.5), gridspec_kw={"wspace": 0.5}
    )
    image.imshow(np.random.default_rng(0).random((16, 16)), cmap="gray")
    phasor.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    phasor.axhline(0.0, lw=0.6)
    phasor.axvline(0.0, lw=0.6)
    phasor.plot([-0.6, -0.2], [0.5, 0.9], color="C0")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, image, phasor


@pytest.mark.parametrize("mode", MODES)
def test_link_is_whole_at_rest_absent_at_zero_and_grows_from_the_source(mode):
    hwostyle.use(mode)
    fig, image, phasor = _image_and_phasor()
    without = _pixels(fig)
    src = ep.Anchor.from_data(image, (12, 4))
    dst = ep.Anchor.from_data(phasor, (0.0, 0.0))
    res = ep.link(src, dst)
    line = res.artists["line"]
    whole = _pixels(fig)
    assert not np.array_equal(whole, without)  # the rest state is drawn
    fig.canvas.draw()
    xy = line.get_xydata()
    np.testing.assert_allclose(xy[0], src.display(), atol=0.5)
    np.testing.assert_allclose(xy[-1], dst.display(), atol=0.5)
    res.update(0.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    res.update(0.5)  # smoothstep is 0.5 half way
    fig.canvas.draw()
    tip = line.get_xydata()[-1]
    np.testing.assert_allclose(tip, 0.5 * (src.display() + dst.display()), atol=0.5)
    np.testing.assert_allclose(line.get_xydata()[0], src.display(), atol=0.5)
    res.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), whole)


@pytest.mark.parametrize("mode", MODES)
def test_link_fades_with_its_level_and_returns_whole(mode):
    hwostyle.use(mode)
    fig, image, phasor = _image_and_phasor()
    without = _pixels(fig)
    res = ep.link(
        ep.Anchor.from_data(image, (12, 4)), ep.Anchor.from_data(phasor, (0.0, 0.0))
    )
    whole = _pixels(fig)
    res.update(1.0, level=0.35)
    dimmed = _pixels(fig)
    assert not np.array_equal(dimmed, whole)
    assert not np.array_equal(dimmed, without)
    res.update(1.0, level=0.0)
    np.testing.assert_array_equal(_pixels(fig), without)
    res.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), whole)


def test_link_audit_reads_titles_and_axis_labels_made_after_it():
    fig, image, phasor = _image_and_phasor()
    # From the image up past the phasor panel's top edge, where its title sits.
    res = ep.link(
        ep.Anchor.from_data(image, (8, 8)),
        ep.Anchor.from_data(phasor, (0.5, 1.6)),
        strict=False,
    )
    assert not any("crosses text" in f for f in res.artists["line"].faults())
    phasor.set_title("a title in the way")
    phasor.set_xlabel("an axis label")
    fig.canvas.draw()
    faults = res.artists["line"].faults()
    assert any("a title in the way" in f for f in faults), faults


def test_link_takes_a_neutral_and_keeps_the_cross_panel_length():
    fig, image, phasor = _image_and_phasor()
    res = ep.link(
        ep.Anchor.from_data(image, (12, 4)), ep.Anchor.from_data(phasor, (0.0, 0.0))
    )
    line = res.artists["line"]
    fig.canvas.draw()
    assert to_rgba(line.get_color())[:3] == pytest.approx(_style.neutral(0.6))
    assert line.faults() == []  # longer than a quarter panel, and no fault
    xy = line.get_xydata()
    assert np.hypot(*(xy[-1] - xy[0])) > 0.25 * max(image.bbox.width, image.bbox.height)
    assert ep.audit(fig) == {}


def test_link_crossing_data_near_either_end_is_a_fault_and_audit_names_it():
    fig, image, phasor = _image_and_phasor()
    src = ep.Anchor.from_data(image, (12, 4))
    # A curve across the phasor panel on the way to its origin.
    phasor.plot([-0.3, -0.3], [-1.0, 1.0], color="C1", label="arm")
    res = ep.link(src, ep.Anchor.from_data(phasor, (0.5, 0.5)))
    fig.canvas.draw()
    assert any("crosses data: arm" in f for f in res.artists["line"].faults())
    # A curve in the source panel's overlay, near the starting end.
    image.autoscale(False)
    image.plot([13.0, 13.0], [0.0, 8.0], color="C2", label="edge")
    res2 = ep.link(src, ep.Anchor.from_data(phasor, (0.9, -0.9)), strict=False)
    assert any("crosses data: edge" in f for f in res2.artists["line"].faults())
    with pytest.raises(ep.AttentionFault, match="arm"):
        ep.audit(fig)
    found = ep.audit(fig, raise_on_fault=False)
    assert res.artists["line"] in found and res2.artists["line"] in found


def test_link_drawing_never_raises_and_scenery_is_not_data():
    fig, image, phasor = _image_and_phasor()
    # The reference lines through the origin and the image are not data.
    clean = ep.link(
        ep.Anchor.from_data(image, (12, 4)), ep.Anchor.from_data(phasor, (0.0, 0.0))
    )
    assert clean.artists["line"].faults() == []
    phasor.plot([-0.3, -0.3], [-1.0, 1.0], color="C1")
    assert clean.artists["line"].faults()
    fig.canvas.draw()  # a broken link still draws
    fig.savefig(io.BytesIO(), format="pdf")


def test_link_ending_on_a_marker_or_in_a_data_color_is_judged_as_a_leader_is():
    _fig, image, phasor = _image_and_phasor()
    (dot,) = phasor.plot([0.0], [0.0], "o", color="C4", ms=8)
    res = ep.link(ep.Anchor.from_data(image, (12, 4)), dot)
    assert res.artists["line"].faults() == []  # ends on the mark it names
    other = ep.link(
        ep.Anchor.from_data(image, (12, 4)),
        ep.Anchor.from_data(phasor, (0.0, 0.0)),
        color="C0",
    )
    assert any("data color" in f for f in other.artists["line"].faults())


def test_link_leaves_an_anchored_artist_at_its_edge():
    fig, image, phasor = _image_and_phasor()
    ring = ep.overlay_circle(image, (12, 4), 1.5).artists["ellipse"][-1]
    res = ep.link(ring, ep.Anchor.from_data(phasor, (0.0, 0.0)))
    fig.canvas.draw()
    box = ring.get_window_extent()
    start = res.artists["line"].get_xydata()[0]
    assert box.x1 - 1.0 <= start[0] <= box.x1 + 1.0  # the right edge, toward the target
    assert res.artists["line"].faults() == []


def test_link_follows_the_layout_and_moves_no_limits():
    fig, image, phasor = _image_and_phasor()
    limits = [(a.get_xlim(), a.get_ylim()) for a in (image, phasor)]
    dst = ep.Anchor.from_data(phasor, (0.0, 0.0))
    res = ep.link(ep.Anchor.from_data(image, (12, 4)), dst)
    line = res.artists["line"]
    assert line in fig.artists and line not in image.get_children()
    phasor.set_xlim(-2.0, 2.0)
    fig.set_size_inches(10.0, 4.0)
    fig.canvas.draw()
    np.testing.assert_allclose(line.get_xydata()[-1], dst.display(), atol=0.5)
    phasor.set_xlim(-1.0, 1.0)
    for u in np.linspace(0.0, 1.0, 6):
        res.update(u)
        fig.canvas.draw()
    assert [(a.get_xlim(), a.get_ylim()) for a in (image, phasor)] == limits
    assert set(res.artists) <= ep.ARTIST_KEYS
    assert list(res.axes) == [image, phasor]


def test_link_update_is_idempotent_and_rejects_bad_inputs():
    fig, image, phasor = _image_and_phasor()
    res = ep.link(
        ep.Anchor.from_data(image, (12, 4)), ep.Anchor.from_data(phasor, (0.0, 0.0))
    )
    res.update(0.3)
    first = _pixels(fig)
    res.update(0.9)
    res.update(0.3)
    np.testing.assert_array_equal(_pixels(fig), first)
    with pytest.raises(TypeError):
        ep.link("here", ep.Anchor.from_data(phasor, (0.0, 0.0)))


# Morph

MORPH_TOL = 1e-9


def _segments_cross(p, q, r, s):
    """Whether open segments pq and rs cross at a point interior to both."""

    def side(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = side(p, q, r), side(p, q, s)
    d3, d4 = side(r, s, p), side(r, s, q)
    return d1 * d2 < 0.0 and d3 * d4 < 0.0


def _is_simple(ring):
    """Whether a closed polygon (rows, not repeated) has no self-crossing."""
    n = len(ring)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            if _segments_cross(ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n]):
                return False
    return True


def _same_ring(p, q, tol=MORPH_TOL):
    """Whether two rings are the same points up to a cyclic shift."""
    return any(np.allclose(p, np.roll(q, k, axis=0), atol=tol) for k in range(len(q)))


def _morph_axes():
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set(xlim=(-2.0, 2.0), ylim=(-2.0, 2.0))
    ax.set_aspect("equal")
    fig.canvas.draw()
    return fig, ax


def _circle(r=1.0, k=64, clockwise=False):
    t = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    if clockwise:
        t = -t
    return np.c_[r * np.cos(t), r * np.sin(t)]


def _square(h=1.0):
    return np.array([[h, h], [-h, h], [-h, -h], [h, -h]])


def _data(ax, vertices):
    return ax.transData.inverted().transform(vertices)


def test_morph_is_exported_and_not_shadowed_by_a_module():
    assert callable(ep.morph)
    import eyepiece.attention

    assert ep.morph is eyepiece.attention.morph


def test_morph_endpoints_are_the_resampled_paths():
    from eyepiece.attention import _resample

    _fig, ax = _morph_axes()
    a, b = _circle(1.0), _square(1.2)
    move = ep.morph(a, b, ax=ax, closed=True, n=120)
    ghost = move.artists["fill"]
    px = lambda v: ax.transData.transform(v)  # noqa: E731
    ((start, closed0),) = ghost.shape(0.0)
    ((stop, closed1),) = ghost.shape(1.0)
    assert closed0 and closed1
    # Exact to float precision: a is resampled by arc length, b is the same
    # ring as resampled, rolled and possibly reversed to sit on a.
    assert np.allclose(start, px(_resample(a, True, 120)), atol=MORPH_TOL)
    assert _same_ring(stop, px(_resample(b, True, 120)), 1e-6) or _same_ring(
        stop, px(_resample(b, True, 120))[::-1], 1e-6
    )


def test_morph_ghost_is_visible_only_between_the_ends():
    fig, ax = _morph_axes()
    move = ep.morph(_circle(), _square(), ax=ax)
    ghost = move.artists["fill"]
    assert not ghost.get_visible()  # rest state
    for u, shown in ((0.0, False), (0.3, True), (0.999, True), (1.0, False)):
        move.update(u)
        assert ghost.get_visible() is shown
    # Drawn at both ends the frame is the frame without the move.
    before = _pixels(fig)
    move.update(0.0)
    assert np.array_equal(before, _pixels(fig))
    move.update(1.0)
    assert np.array_equal(before, _pixels(fig))


def test_morph_circle_to_square_stays_simple():
    _fig, ax = _morph_axes()
    move = ep.morph(_circle(1.0, 90), _square(0.9), ax=ax, closed=True, n=90)
    ghost = move.artists["fill"]
    for u in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        ((ring, closed),) = ghost.shape(u)
        assert closed
        assert _is_simple(ring), f"self-intersection at u={u}"


def test_morph_simple_test_catches_a_crossing():
    bow = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert not _is_simple(bow)


def test_morph_eased_progress_is_monotone():
    fig, ax = _morph_axes()
    move = ep.morph(_circle(), _square(), ax=ax, ease="smoothstep")
    ghost = move.artists["fill"]
    us = np.linspace(0.0, 1.0, 41)
    eased = []
    for u in us:
        move.update(u)
        fig.canvas.draw()
        eased.append(ghost.eased if 0.0 < u < 1.0 else u)
    assert np.all(np.diff(eased) >= 0.0)
    assert np.allclose(eased, ep.ease(us, "smoothstep"))
    # A tracked vertex moves farther from its start as the progress grows.
    first = [ghost.shape(u)[0][0][0] for u in us]
    travel = np.hypot(*(np.array(first) - first[0]).T)
    assert np.all(np.diff(travel) >= -1e-9)


def test_morph_interpolates_stroke_fill_and_width():
    fig, ax = _morph_axes()
    a, b = _circle(), _square()
    move = ep.morph(
        a,
        b,
        ax=ax,
        a_kw={"edgecolor": "#ff0000", "facecolor": "none", "linewidth": 1.0},
        b_kw={"edgecolor": "#0000ff", "facecolor": "#00ff00", "linewidth": 5.0},
    )
    ghost = move.artists["fill"]
    move.update(0.5)
    fig.canvas.draw()
    assert np.allclose(ghost.get_edgecolor(), (0.5, 0.0, 0.5, 1.0))
    # No fill at the start blends in from clear, in the other end's hue.
    assert np.allclose(ghost.get_facecolor(), (0.0, 1.0, 0.0, 0.5))
    assert ghost.get_linewidth() == pytest.approx(3.0)
    move.update(0.25)
    fig.canvas.draw()
    e = ep.ease(0.25)
    assert np.allclose(ghost.get_edgecolor(), (1.0 - e, 0.0, e, 1.0))
    move.update(0.5, level=0.5)
    fig.canvas.draw()
    assert ghost.get_edgecolor()[3] == pytest.approx(0.5)


def test_morph_takes_line_and_pathpatch_artists():
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    fig, ax = _morph_axes()
    (line,) = ax.plot(
        *np.vstack([_circle(1.0, 50)] * 2)[:51].T, color="#ff0000", alpha=0.5
    )
    patch = PathPatch(
        Path(np.vstack([_square(1.0), _square(1.0)[:1]]), [1, 2, 2, 2, 79]),
        facecolor="#0000ff",
        edgecolor="k",
        alpha=0.8,
    )
    ax.add_patch(patch)
    fig.canvas.draw()
    move = ep.morph(line, patch)
    ghost = move.artists["fill"]
    move.update(0.0)
    assert move.ax is ax
    fig.canvas.draw()
    # The line was closed by repeating its first point: one closed ring.
    ((ring, closed),) = ghost.shape(0.5)
    assert closed and len(ring) == 200
    # Alpha rides along: the line's 0.5 and the patch's 0.8 blend.
    move.update(0.5)
    fig.canvas.draw()
    assert ghost.get_edgecolor()[3] == pytest.approx(0.5 * 0.5 + 0.5 * 0.8)
    assert ghost.get_facecolor()[3] == pytest.approx(0.5 * 0.8)


def test_morph_open_path_keeps_its_order():
    fig, ax = _morph_axes()
    x = np.linspace(-1.5, 1.5, 11)
    a = np.c_[x, np.zeros_like(x)]
    b = np.c_[x, np.sin(2.0 * x)]
    move = ep.morph(a, b, ax=ax, n=60)
    ghost = move.artists["fill"]
    ((start, closed),) = ghost.shape(0.0)
    ((stop, _),) = ghost.shape(1.0)
    assert not closed
    # Ends of the open paths stay pinned to ends.
    px = ax.transData.transform
    assert np.allclose(start[0], px(a[0])) and np.allclose(start[-1], px(a[-1]))
    assert np.allclose(stop[0], px(b[0])) and np.allclose(stop[-1], px(b[-1]))
    # Interior points follow the arc-length order: x never runs backward.
    mid = ghost.shape(0.5)[0][0]
    assert np.all(np.diff(mid[:, 0]) > 0.0)
    # Drawing an open ghost has no closing segment or fill.
    move.update(0.5)
    fig.canvas.draw()
    assert _has_no_close(ghost)


def _has_no_close(ghost):
    from matplotlib.path import Path

    return Path.CLOSEPOLY not in list(ghost.get_path().codes)


def test_morph_closed_alignment_takes_the_short_way():
    _fig, ax = _morph_axes()
    a = _circle(1.0, 80)
    # The same circle, drawn the other way round and started a quarter turn on.
    b = np.roll(_circle(1.0, 80, clockwise=True), 20, axis=0)
    move = ep.morph(a, b, ax=ax, closed=True, n=80)
    ghost = move.artists["fill"]
    ((mid, _),) = ghost.shape(0.5)
    radius = np.hypot(*_data(ax, mid).T)
    # A wrong direction or start would pull the ring through the middle.
    assert radius.min() > 0.99
    # Without alignment the same pair does collapse.
    raw = ep.morph(a, b, ax=ax, closed=True, n=80, align=False).artists["fill"]
    ((bad, _),) = raw.shape(0.5)
    assert np.hypot(*_data(ax, bad).T).min() < 0.5


def test_morph_handles_mismatched_point_counts():
    _fig, ax = _morph_axes()
    a = _circle(1.0, 7)
    b = _circle(1.5, 311)
    move = ep.morph(a, b, ax=ax, closed=True, n=100)
    ghost = move.artists["fill"]
    for u in (0.0, 0.5, 1.0):
        ((ring, _),) = ghost.shape(u)
        assert ring.shape == (100, 2)
    radius = np.hypot(*_data(ax, ghost.shape(1.0)[0][0]).T)
    assert np.allclose(radius, 1.5, atol=1e-3)
    with pytest.raises(ValueError, match="at least 3"):
        ep.morph(a, b, ax=ax, n=2)


def test_morph_pairs_subpaths_by_size_and_grows_the_missing_one():
    _fig, ax = _morph_axes()
    big, small = _circle(1.0, 40), _circle(0.3, 40) + np.array([1.2, 1.2])
    two = np.vstack([small, [[np.nan, np.nan]], big])
    one = _square(1.0)
    move = ep.morph(two, one, ax=ax, n=48, closed=True)
    ghost = move.artists["fill"]
    # NaN split makes two closed parts; the larger pairs with the square.
    parts_start = ghost.shape(0.0)
    parts_stop = ghost.shape(1.0)
    assert len(parts_start) == len(parts_stop) == 2
    # The small circle shrinks to a point at its own center.
    last = _data(ax, parts_stop[1][0])
    assert np.allclose(last, [1.2, 1.2], atol=1e-6)
    # And the other way round it grows from that point.
    back = ep.morph(one, two, ax=ax, n=48, closed=True)
    grown = _data(ax, back.artists["fill"].shape(0.0)[1][0])
    assert np.allclose(grown, grown[0], atol=1e-6)
    ((ring_one, _),) = ep.morph(one, one, ax=ax).artists["fill"].shape(0.5)[:1]
    assert len(ring_one) == 200


def test_morph_rejects_bad_ends():
    _fig, ax = _morph_axes()
    with pytest.raises(ValueError, match="ax="):
        ep.morph(_circle(), _square())
    with pytest.raises(TypeError, match="PathPatch"):
        ep.morph(ax.text(0, 0, "x"), _square(), ax=ax)
    with pytest.raises(ValueError, match="no finite"):
        ep.morph(np.full((4, 2), np.nan), _square(), ax=ax)
    with pytest.raises(TypeError, match="unknown morph style"):
        ep.morph(_circle(), _square(), ax=ax, a_kw={"color": "r"})


def test_morph_is_pixel_neutral_in_both_modes_and_draws_between():
    for mode in MODES:
        hwostyle.use(mode)
        fig, ax = _morph_axes()
        move = ep.morph(_circle(), _square(), ax=ax)
        rest = _pixels(fig)
        move.update(0.5)
        assert not np.array_equal(rest, _pixels(fig))
        move.update(1.0)
        assert np.array_equal(rest, _pixels(fig))
        plt.close(fig)


# Readout


def _readout_panel():
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax


def _shown(res):
    """The number text of a readout: the integer part and the fraction."""
    return "".join(
        t.get_text() for t in res.artists["text"] if t.role in ("int", "frac")
    )


def _number(res):
    return float(_shown(res).replace("\u2212", "-").replace("%", "").replace(",", ""))


def _union(res, fig):
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in res.artists["text"]]
    return (
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
    )


@pytest.mark.parametrize(
    ("fmt", "first", "last", "first_text", "last_text"),
    [
        (".1f", 0.0, 3.2, "0.0", "3.2"),
        (".2f", 0.41, 12.345, "0.41", "12.35"),
        (".0f", 4.0, 1250.0, "4", "1250"),
        (",.0f", 4.0, 1250.0, "4", "1,250"),
        (".1%", 0.0, 0.4125, "0.0%", "41.2%"),
        ("+.1f", 1.0, -2.5, "+1.0", "\u22122.5"),
        ("d", 3, 17, "3", "17"),
    ],
)
def test_readout_end_frames_show_the_exact_formatted_values(
    fmt, first, last, first_text, last_text
):
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), first, fmt=fmt)
    assert _shown(res) == first_text
    res.set(last)
    res.update(0.0)
    assert _shown(res) == first_text
    res.update(1.0)
    assert _shown(res) == last_text
    fig.canvas.draw()


def test_readout_prefix_suffix_and_unit_are_laid_beside_the_number():
    fig, ax = _readout_panel()
    res = ep.readout(
        ep.Anchor.from_data(ax, (0.1, 0.5)), 3.2, prefix="t = ", suffix="~", unit="h"
    )
    texts = res.artists["text"]
    assert [t.role for t in texts] == ["prefix", "int", "frac", "tail"]
    assert "".join(t.get_text() for t in texts) == "t = 3.2~ h"
    boxes = _boxes(res, fig)
    for left, right in itertools.pairwise(boxes):
        assert right.x0 == pytest.approx(left.x1, abs=1e-6)
    start = ep.Anchor.from_data(ax, (0.1, 0.5)).display()[0]
    assert boxes[0].x0 == pytest.approx(start)


@pytest.mark.parametrize("mode", MODES)
def test_readout_box_does_not_move_more_than_half_a_pixel_through_a_count(mode):
    hwostyle.use(mode)
    fig, ax = _readout_panel()
    at = ep.Anchor.from_data(ax, (0.1, 0.5))
    res = ep.readout(at, -9.96, prefix="t = ", unit="h")
    res.set(1234.5)  # one digit up to four, and across a sign change
    boxes = []
    for u in np.linspace(0.0, 1.0, 81):
        res.update(u)
        boxes.append(_union(res, fig))
    boxes = np.array(boxes)
    assert np.ptp(boxes, axis=0).max() <= 0.5


def test_readout_point_is_pixel_steady_while_the_integer_digits_change():
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), 1.5, prefix="x = ", digits=3)
    frac = next(t for t in res.artists["text"] if t.role == "frac")
    fig.canvas.draw()
    point = frac.get_window_extent().x0
    frames = {}
    for value in (1.5, 12.5, 123.5):
        res.set(value)
        res.update(1.0)
        frames[value] = _pixels(fig)
    col = int(np.ceil(point))
    # Everything from the decimal point on is the same picture.
    for value in (12.5, 123.5):
        np.testing.assert_array_equal(frames[value][:, col:], frames[1.5][:, col:])
    # And the digits of the integer part end at the point, not past it.
    assert not np.array_equal(frames[12.5][:, :col], frames[1.5][:, :col])


@pytest.mark.parametrize("start", [-3.0, 3.0, 0.0])
@pytest.mark.parametrize("fmt", [".1f", ".0f", ".2f"])
def test_readout_count_is_monotone_up_and_down(start, fmt):
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), start, fmt=fmt)
    res.set(start + 7.3)
    us = np.linspace(0.0, 1.0, 301)
    up = []
    for u in us:
        res.update(u)
        up.append(_number(res))
    assert up == sorted(up)
    assert up[0] == pytest.approx(round(start, 2))
    res.set(start - 5.1)
    down = []
    for u in us:
        res.update(u)
        down.append(_number(res))
    assert down == sorted(down, reverse=True)
    fig.canvas.draw()


def test_readout_shows_no_negative_zero_and_keeps_alignment_through_a_sign_change():
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), -1.0, prefix="v = ")
    res.set(1.0)
    point = None
    right = None
    seen = set()
    for u in np.linspace(0.0, 1.0, 101):
        res.update(u)
        fig.canvas.draw()
        text = _shown(res)
        seen.add(text)
        assert "\u22120.0" not in text
        _, frac = res.artists["text"][1:3]
        box = frac.get_window_extent()
        int_box = res.artists["text"][1].get_window_extent()
        point = box.x0 if point is None else point
        right = int_box.x1 if right is None else right
        assert box.x0 == pytest.approx(point, abs=0.5)
        assert int_box.x1 == pytest.approx(right, abs=0.5)
    assert {"\u22121.0", "0.0", "1.0"} <= seen
    # With a sign flag the rounded zero keeps the plus.
    plus = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.3)), -0.04, fmt="+.1f")
    assert _shown(plus) == "+0.0"


def test_readout_with_a_sign_slot_holds_its_box_across_zero():
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), -25.0, signed=True)
    box = _union(res, fig)
    for value in (-3.0, 0.0, 3.0, 25.0):
        res.set(value)
        res.update(1.0)
        assert _union(res, fig) == pytest.approx(box, abs=0.5)


def test_readout_set_chains_moves_from_the_last_value_and_rests_on_it():
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), 1.0)
    fig.canvas.draw()
    rest = _pixels(fig)
    res.update(1.0)
    np.testing.assert_array_equal(_pixels(fig), rest)  # no move set: rest
    res.set(5.0)
    assert _shown(res) == "1.0"
    res.update(0.5)
    assert _shown(res) == "3.0"
    res.set(2.0)  # chains from 5.0, the value last set
    assert _shown(res) == "5.0"
    res.update(0.5)
    assert _shown(res) == "3.5"
    res.update(1.0)
    assert _shown(res) == "2.0"
    # Progress outside [0, 1] is clipped.
    res.update(7.0)
    assert _shown(res) == "2.0"
    res.update(-1.0)
    assert _shown(res) == "5.0"


def test_readout_ease_kind_shapes_the_count():
    _fig, ax = _readout_panel()
    at = ep.Anchor.from_data(ax, (0.1, 0.5))
    linear = ep.readout(at, 0.0, ease="linear")
    smooth = ep.readout(at, 0.0)
    for res in (linear, smooth):
        res.set(10.0)
        res.update(0.25)
    assert _shown(linear) == "2.5"
    assert _shown(smooth) == "1.6"
    with pytest.raises(ValueError, match="easing"):
        ep.readout(at, 0.0, ease="bounce")


def test_readout_ha_places_it_by_its_end_or_its_point():
    fig, ax = _readout_panel()
    at = ep.Anchor.from_data(ax, (0.5, 0.5))
    x = float(at.display()[0])
    for ha in ("left", "center", "right", "point"):
        res = ep.readout(at, 12.5, prefix="p ", unit="h", ha=ha)
        box = _union(res, fig)
        fig.canvas.draw()
        frac = next(t for t in res.artists["text"] if t.role == "frac")
        if ha == "left":
            assert box[0] == pytest.approx(x, abs=0.5)
        elif ha == "right":
            assert box[2] == pytest.approx(x, abs=0.5)
        elif ha == "center":
            assert 0.5 * (box[0] + box[2]) == pytest.approx(x, abs=0.5)
        else:
            assert frac.get_window_extent().x0 == pytest.approx(x, abs=0.5)


def test_readout_is_a_figure_level_stage_member_and_dims_with_level():
    fig, ax = _readout_panel()
    ax.plot([0.0, 1.0], [0.2, 0.8])
    limits = (ax.get_xlim(), ax.get_ylim())
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), 1.0, digits=2)
    parts = res.artists["text"]
    assert all(t.get_figure() is fig and t not in ax.texts for t in parts)
    res.set(4.0)
    res.update(0.5)
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    stage = ep.Stage({"row": parts})
    stage.set(shown=[])
    # A running value leaves the stage's levels alone.
    res.update(0.7)
    assert all(to_rgba(t.get_color())[3] == 0.0 for t in parts)
    stage.rest()
    res.update(1.0, level=0.4)
    assert all(t.get_alpha() == pytest.approx(0.4) for t in parts)
    res.update(1.0)
    assert all(t.get_alpha() is None for t in parts)
    res.update(1.0, level=0.0)
    assert not any(t.get_visible() for t in parts)
    res.update(1.0)
    assert all(t.get_visible() for t in parts)
    # The parts are anchors that stay put while the value runs.
    anchor = ep.Anchor.from_artist(parts[1], "right")
    first = anchor.display().copy()
    res.set(99.0)
    res.update(0.6)
    np.testing.assert_allclose(anchor.display(), first, atol=0.5)


def test_readout_rest_state_is_the_figure_as_drawn_in_both_modes():
    for mode in MODES:
        hwostyle.use(mode)
        fig, ax = _readout_panel()
        res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), 2.41, fmt=".2f")
        rest = _pixels(fig)
        res.set(7.0)
        res.update(0.5)
        assert not np.array_equal(rest, _pixels(fig))
        res.set(2.41)
        res.update(1.0)
        np.testing.assert_array_equal(rest, _pixels(fig))
        plt.close(fig)


def test_readout_grows_its_slots_only_when_set_and_can_reserve_in_advance():
    fig, ax = _readout_panel()
    at = ep.Anchor.from_data(ax, (0.1, 0.5))
    small = ep.readout(at, 1.0)
    narrow = _union(small, fig)
    small.set(1000.0)
    wide = _union(small, fig)
    assert wide[2] - wide[0] > narrow[2] - narrow[0] + 1.0
    reserved = ep.readout(at, 1.0, digits=4, signed=True)
    box = _union(reserved, fig)
    reserved.set(-1000.0)
    reserved.update(1.0)
    assert _union(reserved, fig) == pytest.approx(box, abs=0.5)


def test_readout_rejects_bad_inputs():
    _fig, ax = _readout_panel()
    at = ep.Anchor.from_data(ax, (0.1, 0.5))
    for bad in (".2g", "e", "5.1f", "x", ".1fz"):
        with pytest.raises(ValueError, match="fixed-point"):
            ep.readout(at, 1.0, fmt=bad)
    with pytest.raises(ValueError, match="ha"):
        ep.readout(at, 1.0, ha="top")
    with pytest.raises(ValueError, match="finite"):
        ep.readout(at, float("nan"))
    res = ep.readout(at, 1.0)
    with pytest.raises(ValueError, match="finite"):
        res.set(float("inf"))
    with pytest.raises(TypeError, match="Anchor"):
        ep.readout("here", 1.0)


def _ink_columns(res, fig, roles):
    """The columns that show ink when only the parts of `roles` are drawn."""
    parts = res.artists["text"]
    for t in parts:
        t.set_visible(False)
    empty = _pixels(fig)
    for t in parts:
        t.set_visible(t.role in roles)
    shown = _pixels(fig)
    for t in parts:
        t.set_visible(True)
    cols = np.nonzero(np.any(shown != empty, axis=(0, 2)))[0]
    return cols


def _ink_gap(res, fig):
    """Pixels from the integer part's last ink column to the fraction's first."""
    return (
        _ink_columns(res, fig, {"frac"}).min() - _ink_columns(res, fig, {"int"}).max()
    )


@pytest.mark.parametrize("mode", MODES)
def test_readout_integer_ink_ends_at_the_decimal_point(mode):
    hwostyle.use(mode)
    fig, ax = _readout_panel()
    res = ep.readout(
        ep.Anchor.from_data(ax, (0.1, 0.5)), 1.5, prefix="x = ", unit="h", digits=3
    )
    prefix_cols = None
    for value in (1.5, 12.5, 123.5, -4.5):
        res.set(value)
        res.update(1.0)
        # The integer part ends within the glyph bearings (4 px) of the point.
        assert 0 <= _ink_gap(res, fig) <= 4
        cols = tuple(_ink_columns(res, fig, {"prefix"}))
        prefix_cols = cols if prefix_cols is None else prefix_cols
        assert cols == prefix_cols
    plt.close(fig)


def test_readout_ink_test_catches_a_left_justified_integer():
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), 123.5, prefix="x = ")
    assert 0 <= _ink_gap(res, fig) <= 4
    # Positive control: justify the integer part left, as a broken layout would.
    integer = next(t for t in res.artists["text"] if t.role == "int")
    integer.set_ha("left")
    assert _ink_gap(res, fig) < 0 or _ink_gap(res, fig) > 4


@pytest.mark.parametrize("kind", ["back_out", "running_start"])
@pytest.mark.parametrize("old, new", [(0.0, 10.0), (10.0, 0.0), (-3.0, 7.0)])
def test_readout_count_stays_between_its_values_under_the_real_overshoot_eases(
    kind, old, new
):
    fig, ax = _readout_panel()
    res = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.5)), old, ease=kind)
    res.set(new)
    shown = []
    for u in np.linspace(0.0, 1.0, 201):
        res.update(u)
        shown.append(_number(res))
    lo, hi = sorted((old, new))
    assert min(shown) >= lo and max(shown) <= hi
    assert shown == sorted(shown, reverse=new < old)  # a count never goes back
    assert shown[0] == old and shown[-1] == new
    fig.canvas.draw()


def test_readout_signed_false_rejects_a_negative_value():
    _fig, ax = _readout_panel()
    at = ep.Anchor.from_data(ax, (0.1, 0.5))
    with pytest.raises(ValueError, match="signed=False"):
        ep.readout(at, -1.0, signed=False)
    res = ep.readout(at, 1.0, signed=False)
    with pytest.raises(ValueError, match="signed=False"):
        res.set(-2.0)
    res.set(0.0)  # a value that is not negative is fine


# Overshoot eases


def test_flight_with_back_out_overshoots_its_destination_and_lands_exactly():
    fig, ax = plt.subplots()
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    a, b = ep.Anchor.from_data(ax, (0.2, 0.5)), ep.Anchor.from_data(ax, (0.8, 0.5))
    carry = ep.flight(a, b, fig=fig, mark=dot, ease="back_out")
    ghost = carry.artists["line"]
    reach = b.display()[0]
    xs = []
    for u in np.linspace(0.0, 1.0, 101)[1:-1]:
        carry.update(u)
        fig.canvas.draw()
        xs.append(ghost.get_xydata()[0][0])
    travel = reach - a.display()[0]
    assert max(xs) > reach + 0.05 * travel  # visibly past the destination
    assert max(xs) == pytest.approx(reach + 0.1 * travel, abs=0.01 * travel)
    # The last frames come back toward it, and the rest state is the destination.
    assert xs[-1] < max(xs)
    carry.update(1.0)
    assert not ghost.get_visible()
    carry.update(1.0 - 1e-9)
    fig.canvas.draw()
    assert ghost.get_xydata()[0][0] == pytest.approx(reach, abs=1e-3)


@pytest.mark.parametrize("kind", ["back_out", "running_start"])
def test_link_never_draws_past_its_ends_under_an_overshoot_ease(kind):
    fig, image, phasor = _image_and_phasor()
    src = ep.Anchor.from_data(image, (12, 4))
    dst = ep.Anchor.from_data(phasor, (0.0, 0.0))
    tie = ep.link(src, dst, ease=kind)
    line = tie.artists["line"]
    p, q = src.display(), dst.display()
    along = q - p
    for u in np.linspace(0.01, 0.99, 25):
        tie.update(u)
        fig.canvas.draw()
        if not line.get_visible() or len(line.get_xydata()) == 0:
            continue
        tip = line.get_xydata()[-1]
        reach = float((tip - p) @ along / (along @ along))
        assert -1e-3 <= reach <= 1.0 + 1e-3


@pytest.mark.parametrize("kind", ["back_out", "running_start"])
def test_circumscribe_draws_a_part_of_its_outline_under_an_overshoot_ease(kind):
    fig, _ax, label = _labelled_panel()
    ring = ep.circumscribe(label, ease=kind)
    outline = ring.artists["line"]
    full = outline.outline(fig.canvas.get_renderer())
    total = np.hypot(*np.diff(full, axis=0).T).sum()
    for u in np.linspace(0.01, 0.99, 40):
        ring.update(u)
        fig.canvas.draw()
        pts = outline.get_xydata()
        drawn = np.hypot(*np.diff(pts, axis=0).T).sum()
        assert drawn <= total + 0.5
        alpha = outline.get_alpha()
        assert alpha is None or 0.0 <= alpha <= 1.0


def test_a_move_takes_a_callable_ease_to_choose_a_strength():
    import functools

    fig, ax = plt.subplots()
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    a, b = ep.Anchor.from_data(ax, (0.2, 0.5)), ep.Anchor.from_data(ax, (0.8, 0.5))
    reach = {}
    for strength in (0.1, 0.3):
        how = functools.partial(ep.ease, kind="back_out", strength=strength)
        carry = ep.flight(a, b, fig=fig, mark=dot, ease=how)
        ghost = carry.artists["line"]
        xs = []
        for u in np.linspace(0.0, 1.0, 201)[1:-1]:
            carry.update(u)
            fig.canvas.draw()
            xs.append(ghost.get_xydata()[0][0])
        reach[strength] = (max(xs) - b.display()[0]) / (b.display()[0] - a.display()[0])
    assert reach[0.1] == pytest.approx(0.1, abs=0.01)
    assert reach[0.3] == pytest.approx(0.3, abs=0.01)


@pytest.mark.parametrize("kind", ["back_out", "running_start"])
def test_morph_keeps_colors_and_widths_valid_under_an_overshoot_ease(kind):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    fig, ax = _morph_axes()
    a = PathPatch(
        Path.make_compound_path(Path(_circle(1.0), closed=True)),
        ec="tab:red",
        fc="none",
        lw=1.0,
    )
    b = PathPatch(
        Path(_square(1.2), closed=True), ec="tab:blue", fc="tab:green", lw=4.0
    )
    for patch in (a, b):
        ax.add_patch(patch)
    move = ep.morph(a, b, ax=ax, closed=True, n=40, ease=kind)
    ghost = move.artists["fill"]
    for u in np.linspace(0.0, 1.0, 21):
        move.update(u)
        fig.canvas.draw()
        assert ghost.get_linewidth() >= 0.0
        for color in (ghost.get_edgecolor(), ghost.get_facecolor()):
            assert np.all((0.0 <= np.asarray(color)) & (np.asarray(color) <= 1.0))
