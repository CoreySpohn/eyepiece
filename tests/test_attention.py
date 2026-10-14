"""Attention moves: what the viewer sees, not how it is drawn.

A flight lands on its destination and leaves no trace at either end; a
leader keeps the connector rules; a callout sits beside its subject and is
absent before its beat; all of it in light and dark.
"""

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
        ep.flight(ax.patch, ep.Anchor.from_data(ax, (1, 1)), fig=fig)
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


def test_strict_leader_refuses_to_draw_a_broken_leader():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.05], [0.05], "o")
    ep.leader(ax.text(0.95, 0.95, "far away"), dot)
    with pytest.raises(ValueError, match="quarter"):
        fig.canvas.draw()


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
