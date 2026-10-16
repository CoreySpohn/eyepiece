"""Motion faults: a control for each kind, built from small synthetic clips.

Each fault kind has a clip that shows it and a clip that must stay clean, so a
tuned threshold cannot drift into either a miss or a false alarm. The clips
re-create the defects found by eye in earlier moves (a final frame that jumps,
glyphs that travel or tear, a hop on a near-still glyph, two terms that cross,
text off the edge); none of them uses the old code.
"""

import math

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

import eyepiece.testing as ept

DPI = 100
SPAN = ept.Span(0.0, 2.0)  # 61 frames at 30 fps
DURATION = 2.0


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _figure():
    fig = plt.figure(figsize=(4.0, 2.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig


def _box(fig, x, y, w=30.0, h=20.0, color="k", alpha=1.0):
    patch = Rectangle(
        (x, y),
        w,
        h,
        transform=IdentityTransform(),
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    patch.set_clip_on(False)
    fig.add_artist(patch)
    return patch


def _glyph(fig, x, y, w=24.0, h=30.0):
    """A filled path in pixel coordinates, the way an equation draws a glyph."""
    patch = PathPatch(
        Path(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        ),
        transform=IdentityTransform(),
        facecolor="k",
        edgecolor="none",
    )
    patch.set_clip_on(False)
    fig.add_artist(patch)
    return patch


def _move(patch, x, y):
    patch.set_xy((x, y))


def _shift_glyph(patch, x, y, w=24.0, h=30.0):
    patch.set_path(
        Path(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
            [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
        )
    )


def _smooth(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _kinds(found):
    return sorted({f.kind for f in found})


# jump


def _ramp_clip(final_step):
    """A big box that fades in at a steady rate, then takes `final_step` steps."""
    fig = _figure()
    box = _box(fig, 40, 40, 300, 120, alpha=0.0)
    slope = 0.01

    def at(t):
        k = round(t * 30.0)
        alpha = slope * k if k < 60 else slope * 59 + final_step * slope
        box.set_alpha(min(1.0, alpha))

    return fig, at


def test_jump_into_the_rest_state():
    fig, at = _ramp_clip(final_step=40.0)
    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["jump"]
    assert found[0].time == pytest.approx(2.0)
    assert "rest state" in found[0].detail
    assert found[0].value > 4.0


def test_steady_fade_into_rest_is_clean():
    fig, at = _ramp_clip(final_step=1.0)
    assert ept.motion_faults(fig, at, SPAN) == []


def test_step_into_rest_larger_than_the_one_before_is_reported_below_the_ratio():
    # 2.5 times the steady step: under jump_ratio of 4, over its half.
    fig, at = _ramp_clip(final_step=2.5)
    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["jump"]
    fig, at = _ramp_clip(final_step=1.5)
    assert ept.motion_faults(fig, at, SPAN) == []


def test_jump_in_the_middle_of_a_move():
    fig = _figure()
    box = _box(fig, 40, 40, 300, 120, alpha=0.0)

    def at(t):
        k = round(t * 30.0)
        box.set_alpha(0.01 * k + (0.3 if k >= 30 else 0.0))

    found = ept.motion_faults(fig, at, SPAN)
    assert [f.kind for f in found] == ["jump"]
    assert found[0].time == pytest.approx(1.0)
    assert "one frame" in found[0].detail


def test_eased_travel_has_no_jump():
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 20 + 280 * _smooth(t / DURATION), 80)

    assert ept.motion_faults(fig, at, SPAN) == []


def test_a_hold_then_a_move_has_no_jump():
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 20 + 280 * _smooth((t - 0.5) / 1.0), 80)

    assert ept.motion_faults(fig, at, SPAN) == []


# dead


def test_dead_move_is_reported():
    fig = _figure()
    _box(fig, 20, 80, 60, 40)
    found = ept.motion_faults(
        fig,
        lambda t: None,
        SPAN,
        moves=[{"name": "slide", "span": (0.5, 1.5)}],
    )
    assert _kinds(found) == ["dead"]
    assert "slide" in found[0].detail
    assert found[0].time == pytest.approx(0.5, abs=1.0 / 30.0)


def test_move_that_moves_is_not_dead():
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 20 + 100 * _smooth(t / DURATION), 80)

    found = ept.motion_faults(fig, at, SPAN, moves=[{"name": "slide"}])
    assert found == []


def test_dead_looks_only_where_the_move_declares_its_artists():
    fig = _figure()
    still = _box(fig, 20, 20, 40, 40)
    other = _box(fig, 200, 100, 40, 40)

    def at(t):
        _move(other, 200 + 100 * _smooth(t / DURATION), 100)

    moves = [{"name": "still one", "artists": [still]}]
    assert _kinds(ept.motion_faults(fig, at, SPAN, moves=moves)) == ["dead"]
    moves = [{"name": "moving one", "artists": [other]}]
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []


# drift


def _slider(shift_px):
    fig = _figure()
    steady = _glyph(fig, 60, 80)
    mover = _glyph(fig, 200, 80)

    def at(t):
        u = _smooth(t / DURATION)
        _shift_glyph(steady, 60 + shift_px * u, 80)
        _shift_glyph(mover, 200 + 120 * u, 80)

    return fig, at, steady


def test_drift_of_an_artist_that_should_stay_still():
    fig, at, steady = _slider(shift_px=14.0)
    found = ept.motion_faults(fig, at, SPAN, rest=[steady])
    assert _kinds(found) == ["drift"]
    assert found[0].artist is steady
    assert found[0].value == pytest.approx(14.0, abs=0.6)


def test_a_still_artist_and_a_sub_tolerance_shift_are_clean():
    fig, at, steady = _slider(shift_px=0.0)
    assert ept.motion_faults(fig, at, SPAN, rest=[steady]) == []
    fig, at, steady = _slider(shift_px=0.3)
    assert ept.motion_faults(fig, at, SPAN, rest=[steady]) == []


def test_rest_may_come_from_a_declared_move():
    fig, at, steady = _slider(shift_px=14.0)
    found = ept.motion_faults(fig, at, SPAN, moves=[{"rest": [steady]}])
    assert _kinds(found) == ["drift"]


# tear


def _row(stagger, scale=False):
    fig = _figure()
    parts = [_glyph(fig, 50 + 40 * i, 80) for i in range(3)]

    def at(t):
        u = t / DURATION
        for i, part in enumerate(parts):
            if scale:
                x = 50 + 40 * i * (1.0 + 0.5 * _smooth(u))
            else:
                lag = stagger * i
                x = 50 + 40 * i + 150 * _smooth((u - lag) / (1.0 - 2 * stagger))
            _shift_glyph(part, x, 80)

    return fig, at, parts


def test_tear_of_a_group_that_travels_with_a_stagger():
    fig, at, parts = _row(stagger=0.15)
    found = ept.motion_faults(fig, at, SPAN, groups=[parts])
    assert _kinds(found) == ["tear"]
    assert found[0].value > 10.0


def test_a_rigid_group_does_not_tear():
    fig, at, parts = _row(stagger=0.0)
    assert ept.motion_faults(fig, at, SPAN, groups=[parts]) == []


def test_a_group_that_scales_tears_only_unless_it_is_declared_to_scale():
    fig, at, parts = _row(stagger=0.0, scale=True)
    assert _kinds(ept.motion_faults(fig, at, SPAN, groups=[parts])) == ["tear"]
    declared = [{"artists": parts, "scales": True}]
    assert ept.motion_faults(fig, at, SPAN, groups=declared) == []


# overlap


def _crossing(patches=True, declare=True):
    fig = _figure()
    if patches:
        a, b = _glyph(fig, 40, 80), _glyph(fig, 320, 90)
    else:
        a = fig.text(0.05, 0.5, "left term", fontsize=14)
        b = fig.text(0.65, 0.45, "right term", fontsize=14)

    def at(t):
        u = _smooth(t / DURATION)
        if patches:
            _shift_glyph(a, 40 + 240 * u, 80)
            _shift_glyph(b, 320 - 240 * u, 90)
        else:
            a.set_position((0.05 + 0.6 * u, 0.5))
            b.set_position((0.65 - 0.6 * u, 0.45))

    moves = [{"name": "swap", "artists": [a, b]}] if declare else None
    return fig, at, a, b, moves


def test_glyphs_that_cross_while_moving_overlap():
    fig, at, a, b, moves = _crossing()
    found = ept.motion_faults(fig, at, SPAN, moves=moves)
    assert _kinds(found) == ["overlap"]
    assert {found[0].artist, found[0].other} == {a, b}


def test_undeclared_glyph_paths_are_not_checked_for_overlap():
    fig, at, *_ = _crossing()
    assert ept.motion_faults(fig, at, SPAN) == []


def test_texts_that_cross_while_moving_overlap_with_no_declaration():
    fig, at, a, b, _ = _crossing(patches=False, declare=False)
    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["overlap"]
    assert {found[0].artist, found[0].other} == {a, b}


def test_a_declared_pair_may_overlap():
    fig, at, a, b, moves = _crossing()
    assert ept.motion_faults(fig, at, SPAN, moves=moves, may_overlap=[(a, b)]) == []
    moves[0]["may_overlap"] = [(a, b)]
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []


def test_a_faint_pair_crossing_is_not_an_overlap_fault():
    fig, at, a, b, moves = _crossing()
    a.set_alpha(0.1)
    b.set_alpha(0.1)
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []


def test_a_mover_over_a_still_neighbour_is_not_an_overlap_fault():
    fig = _figure()
    still = fig.text(0.4, 0.5, "static label", fontsize=14)
    mover = fig.text(0.05, 0.5, "mover", fontsize=14)

    def at(t):
        mover.set_position((0.05 + 0.6 * _smooth(t / DURATION), 0.5))

    assert ept.motion_faults(fig, at, SPAN) == []
    assert still.get_visible()


# outside


def test_glyph_path_off_the_edge_is_outside_though_readability_passes():
    fig = _figure()
    glyph = _glyph(fig, 300, 80, w=60.0)
    note = fig.text(0.05, 0.1, "label", fontsize=14)

    def at(t):
        _shift_glyph(glyph, 300 + 70 * _smooth(t / DURATION), 80, w=60.0)

    # The readability check reads text artists and axes marks, and this glyph is
    # a figure-level path, so it never looks at it.
    at(DURATION)
    assert ept.readability_faults(fig) == []
    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["outside"]
    assert found[0].artist is glyph
    assert found[0].value == pytest.approx(30.0, abs=1.0)
    assert note.get_visible()


def test_text_cut_by_its_axes_is_outside_but_a_clipped_line_is_not():
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=DPI)
    ax.set(xlim=(0, 10), ylim=(0, 1), xticks=[], yticks=[])
    (line,) = ax.plot([0, 20], [0.2, 0.2], color="C0")
    label = ax.text(1, 0.6, "a label that slides", clip_on=True)

    def at(t):
        label.set_x(1 + 7.0 * _smooth(t / DURATION))

    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["outside"]
    assert found[0].artist is label
    assert line.get_visible()


def test_a_growing_reveal_mask_is_not_a_cut():
    fig = _figure()
    label = fig.text(0.1, 0.5, "revealed left to right", fontsize=14)
    label.set_clip_on(True)

    def at(t):
        right = 20.0 + 300.0 * _smooth(t / DURATION)
        label.set_clip_box(
            TransformedBbox(Bbox([[0, 0], [right, 200]]), IdentityTransform())
        )

    assert ept.motion_faults(fig, at, SPAN) == []


def test_a_clean_clip_inside_the_frame_has_nothing_outside():
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 20 + 300 * _smooth(t / DURATION), 80)

    assert ept.motion_faults(fig, at, SPAN) == []


# spike


def _hopper(hop_px, travel_px):
    fig = _figure()
    glyph = _glyph(fig, 100, 80)

    def at(t):
        u = _smooth(t / DURATION)
        _shift_glyph(glyph, 100 + travel_px * u, 80 + hop_px * math.sin(math.pi * u))

    return fig, at, glyph


def test_hop_on_a_near_still_artist_is_a_spike():
    fig, at, glyph = _hopper(hop_px=9.0, travel_px=2.0)
    found = ept.motion_faults(fig, at, SPAN, spikes=True)
    assert _kinds(found) == ["spike"]
    assert found[0].artist is glyph
    assert found[0].value == pytest.approx(9.0, abs=1.0)


def test_spike_is_off_unless_asked_for():
    fig, at, _ = _hopper(hop_px=9.0, travel_px=2.0)
    assert ept.motion_faults(fig, at, SPAN) == []
    assert _kinds(ept.motion_faults(fig, at, SPAN, kinds=["spike"])) == ["spike"]


def test_an_arc_on_a_real_move_is_not_a_spike():
    fig, at, _ = _hopper(hop_px=40.0, travel_px=240.0)
    assert ept.motion_faults(fig, at, SPAN, spikes=True) == []


def test_a_small_marker_orbiting_a_wide_loop_is_not_a_spike():
    fig = _figure()
    marker = _box(fig, 0, 0, 8, 8)

    def at(t):
        a = 2.0 * math.pi * t / DURATION
        _move(marker, 190 + 80 * math.cos(a), 96 + 60 * math.sin(a))

    assert ept.motion_faults(fig, at, SPAN, spikes=True) == []


def test_tick_labels_that_trade_places_are_not_spikes():
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=DPI)
    fig.subplots_adjust(bottom=0.3, left=0.2)

    def at(t):
        ax.set_xlim(0.0, 10.0 + 7.3 * _smooth(t / DURATION))

    assert ept.motion_faults(fig, at, SPAN, spikes=True) == []


def test_an_artist_that_changes_shape_is_not_a_spike():
    fig = _figure()
    bar = _box(fig, 40, 80, 20, 20)

    def at(t):
        k = round(t * 30.0)
        bar.set_width(20.0 + (60.0 if k == 30 else 2.0 * k))

    assert "spike" not in _kinds(ept.motion_faults(fig, at, SPAN, spikes=True))


def test_one_frame_teleport_is_a_spike():
    fig = _figure()
    block = _box(fig, 20, 80, 40, 40)

    def at(t):
        k = round(t * 30.0)
        _move(block, 20 + 4.0 * k + (60.0 if k == 30 else 0.0), 80)

    found = ept.motion_faults(fig, at, SPAN, spikes=True)
    assert "spike" in _kinds(found)
    assert "velocity" in next(f for f in found if f.kind == "spike").detail


def test_a_figure_with_a_canvas_that_has_no_pixels_is_still_checked():
    from matplotlib.backend_bases import FigureCanvasBase
    from matplotlib.figure import Figure

    fig = Figure(figsize=(4.0, 2.0), dpi=DPI)
    FigureCanvasBase(fig)
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 20 + 400 * _smooth(t / DURATION), 80)

    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["outside"]
    assert type(fig.canvas) is FigureCanvasBase


# the call


def test_span_samples_every_frame_and_ends_on_stop():
    seen = []
    fig = _figure()
    _box(fig, 10, 10)
    ept.motion_faults(fig, seen.append, ept.Span(0.0, 1.0))
    assert len(seen) == 31
    assert seen[-1] == pytest.approx(1.0)
    seen.clear()
    ept.motion_faults(fig, seen.append, ept.Span(0.0, 1.0), fps=10)
    assert len(seen) == 11
    seen.clear()
    ept.motion_faults(fig, seen.append, ept.Span(0.0, 0.95))
    assert seen[-1] == pytest.approx(0.95)


def test_explicit_times_are_used_as_given():
    seen = []
    fig = _figure()
    _box(fig, 10, 10)
    ept.motion_faults(fig, seen.append, [0.0, 0.4, 0.9, 1.0])
    assert seen == [0.0, 0.4, 0.9, 1.0]


def test_figure_keeps_its_dpi_and_is_left_posed():
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)
    poses = []

    def at(t):
        poses.append(t)
        _move(block, 20 + 100 * t, 80)

    ept.motion_faults(fig, at, SPAN, dpi=50)
    assert fig.dpi == DPI
    assert poses[-1] == pytest.approx(2.0)
    assert block.get_x() == pytest.approx(220.0)


def test_the_result_is_deterministic_and_ordered_by_time():
    fig, at, parts = _row(stagger=0.15)
    first = ept.motion_faults(fig, at, SPAN, groups=[parts], spikes=True)
    again = ept.motion_faults(fig, at, SPAN, groups=[parts], spikes=True)
    assert [str(f) for f in first] == [str(f) for f in again]
    times = [f.time for f in first]
    assert times == sorted(times)


def test_assert_motion_raises_with_every_fault_named():
    fig, at, steady = _slider(shift_px=14.0)
    with pytest.raises(ept.MotionError, match="drift") as raised:
        ept.assert_motion(fig, at, SPAN, rest=[steady])
    assert len(raised.value.faults) == 1
    assert "should stay still" in str(raised.value)
    fig, at, steady = _slider(shift_px=0.0)
    ept.assert_motion(fig, at, SPAN, rest=[steady])


def test_bad_arguments_are_rejected():
    fig = _figure()
    block = _box(fig, 20, 20)
    with pytest.raises(ValueError, match="unknown fault kinds"):
        ept.motion_faults(fig, lambda t: None, SPAN, kinds=["wobble"])
    with pytest.raises(ValueError, match="unknown venue"):
        ept.motion_faults(fig, lambda t: None, SPAN, venue="cinema")
    with pytest.raises(ValueError, match="three samples"):
        ept.motion_faults(fig, lambda t: None, [0.0, 1.0])
    ghost = Rectangle((0, 0), 5, 5, alpha=0.0)
    with pytest.raises(ValueError, match="not visible"):
        ept.motion_faults(fig, lambda t: None, SPAN, rest=[ghost])
    assert block.get_visible()


def test_public_names():
    for name in (
        "MOTION_KINDS",
        "VENUE_FPS",
        "MotionError",
        "MotionFault",
        "Span",
        "assert_motion",
        "motion_faults",
    ):
        assert name in ept.__all__
        assert hasattr(ept, name)


def test_a_clip_with_text_and_a_move_runs_clean_and_fast():
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=DPI)
    ax.set(xlim=(0, 10), ylim=(0, 10), xticks=range(0, 11, 2), yticks=range(0, 11, 2))
    (dot,) = ax.plot([1], [5], "o", color="C0")
    title = ax.set_title("A point crossing the field", fontsize=16)
    fig.canvas.draw()
    fig.set_layout_engine("none")

    def at(t):
        dot.set_data([1 + 8 * _smooth(t / DURATION)], [5])

    found = ept.motion_faults(fig, at, SPAN, rest=[title], spikes=True)
    assert found == []
