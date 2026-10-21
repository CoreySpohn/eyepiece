"""Motion faults: a control for each kind, built from small synthetic clips.

Each fault kind has a clip that shows it and a clip that must stay clean, so a
tuned threshold cannot drift into either a miss or a false alarm. The clips
re-create the defects found by eye in earlier moves (a final frame that jumps,
glyphs that travel or tear, a hop on a near-still glyph, two terms that cross,
text off the edge); none of them uses the old code.
"""

import math
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, IdentityTransform, TransformedBbox

import eyepiece.testing as ept

DPI = 100
SPAN = ept.Window(0.0, 2.0)  # 61 frames at 30 fps
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
    assert found[0].value > 4.0


@pytest.mark.parametrize("fade", [False, True])
def test_declared_steps_suppress_only_the_named_jump(fade):
    fig = _figure()
    box = _box(fig, 20, 80, 60, 40)

    def at(t):
        k = round(t * 30.0)
        offset = (60 if k >= 20 else 0) + (60 if k >= 40 else 0)
        _move(box, 20 + offset, 80)
        box.set_alpha(0.2 + 0.005 * k if fade else 1.0)

    assert [f.time for f in ept.motion_faults(fig, at, SPAN)] == pytest.approx(
        [20 / 30, 40 / 30]
    )
    found = ept.motion_faults(fig, at, SPAN, steps=[20 / 30 + 5e-10])
    assert [f.kind for f in found] == ["jump"]
    assert found[0].time == pytest.approx(40 / 30)


@pytest.mark.parametrize("fade", [False, True])
@pytest.mark.parametrize("offset", [-1, 1])
def test_declared_step_does_not_suppress_a_jump_one_frame_away(fade, offset):
    fig = _figure()
    box = _box(fig, 20, 80, 60, 40)

    def at(t):
        k = round(t * 30.0)
        _move(box, 20 + (60 if k >= 30 else 0), 80)
        box.set_alpha(0.2 + 0.005 * k if fade else 1.0)

    found = ept.motion_faults(fig, at, SPAN, steps=[(30 + offset) / 30])
    assert [f.kind for f in found] == ["jump"]
    assert found[0].time == pytest.approx(1.0)


def test_a_declared_time_must_match_the_jump_sample_closely():
    fig = _figure()
    box = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(box, 20 + (60 if round(t * 30.0) >= 30 else 0), 80)

    found = ept.motion_faults(fig, at, SPAN, steps=[1.0 + 1e-6])
    assert [f.kind for f in found] == ["jump"]
    assert ept.motion_faults(fig, at, SPAN, steps=[1.0 + 1e-10]) == []


def test_declared_steps_suppress_the_rest_entry_only_at_its_destination():
    fig, at = _ramp_clip(final_step=2.5)
    assert ept.motion_faults(fig, at, SPAN, steps=[2.0]) == []
    found = ept.motion_faults(fig, at, SPAN, steps=[59 / 30])
    assert [f.kind for f in found] == ["jump"]
    assert found[0].time == pytest.approx(2.0)


def test_declared_steps_leave_other_motion_faults_unchanged():
    fig = _figure()
    box = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(box, 380 if t >= 1.0 else 20, 80)

    found = ept.motion_faults(fig, at, SPAN, rest=[box], steps=[1.0])
    assert _kinds(found) == ["drift", "outside"]


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


def test_a_pair_allowed_over_a_span_is_allowed_where_it_starts_to_move():
    # Overlapping labels ride together over the second half of the clip. A
    # sample just before the span counts as moving, because its look-ahead
    # window reaches the span's first frames, so the allowance covers it too.
    fig = _figure()
    a = fig.text(0.05, 0.5, "riding label", fontsize=14)
    b = fig.text(0.08, 0.5, "riding label", fontsize=14)

    def at(t):
        u = min(max(t - 1.0, 0.0), 1.0)
        a.set_position((0.05 + 0.5 * u, 0.5))
        b.set_position((0.08 + 0.5 * u, 0.5))

    move = {"name": "ride", "span": (1.0, 2.0), "artists": [a, b]}
    assert _kinds(ept.motion_faults(fig, at, SPAN, moves=[move])) == ["overlap"]
    move["may_overlap"] = [(a, b)]
    assert ept.motion_faults(fig, at, SPAN, moves=[move]) == []


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


def test_one_frame_teleport_is_a_spike():
    fig = _figure()
    block = _box(fig, 20, 80, 40, 40)

    def at(t):
        k = round(t * 30.0)
        _move(block, 20 + 4.0 * k + (60.0 if k == 30 else 0.0), 80)

    found = ept.motion_faults(fig, at, SPAN, spikes=True)
    assert "spike" in _kinds(found)
    spike = next(f for f in found if f.kind == "spike")
    assert spike.value == pytest.approx(120.0, abs=2.0)  # out by 64, back by 56
    assert spike.time == pytest.approx(1.0, abs=1.0 / 30.0)


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


# review findings: each test names the defect it reproduces


def _clip_to(artist, x0, x1):
    artist.set_clip_on(True)
    artist.set_clip_box(
        TransformedBbox(Bbox([[x0, 0], [x1, 200]]), IdentityTransform())
    )


def _rotate(points, degrees, about=(100.0, 100.0)):
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    out = []
    for x, y in points:
        dx, dy = x - about[0], y - about[1]
        out.append((about[0] + c * dx - s * dy, about[1] + s * dx + c * dy))
    return out


def test_contract_dicts_on_objects_are_unwrapped():
    fig, at, steady = _slider(shift_px=14.0)
    holder = SimpleNamespace(contract={"rest": [steady]})
    found = ept.motion_faults(fig, at, SPAN, moves=[holder])
    assert _kinds(found) == ["drift"]
    assert ept.motion_faults(fig, at, SPAN, moves=[SimpleNamespace(contract={})]) == []

    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)
    contract = {"name": "slide", "span": (0.5, 1.5), "artists": [block]}
    found = ept.motion_faults(
        fig, lambda t: None, SPAN, moves=[SimpleNamespace(contract=contract)]
    )
    assert _kinds(found) == ["dead"]
    assert "slide" in found[0].detail

    fig, at, parts = _row(stagger=0.15)
    holder = SimpleNamespace(contract={"groups": [parts]})
    assert _kinds(ept.motion_faults(fig, at, SPAN, moves=[holder])) == ["tear"]

    fig, at, a, b, _ = _crossing()
    contract = {"artists": [a, b]}
    assert _kinds(
        ept.motion_faults(fig, at, SPAN, moves=[SimpleNamespace(contract=contract)])
    ) == ["overlap"]
    contract["may_overlap"] = [(a, b)]
    holder = SimpleNamespace(contract=contract)
    assert ept.motion_faults(fig, at, SPAN, moves=[holder]) == []


def _scatter_clip():
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=DPI)
    ax.set(xlim=(0, 10), ylim=(0, 10), xticks=[], yticks=[])
    return fig, ax


def test_a_moving_scatter_is_traced_and_its_alpha_array_does_not_crash():
    fig, ax = _scatter_clip()
    dots = ax.scatter([2.0, 3.0], [5.0, 5.0], s=200, alpha=[0.5, 0.8])

    def at(t):
        u = _smooth(t / DURATION)
        dots.set_offsets([[2.0 + 5.0 * u, 5.0], [3.0 + 5.0 * u, 5.0]])

    found = ept.motion_faults(fig, at, SPAN, rest=[dots])
    assert _kinds(found) == ["drift"]
    assert found[0].artist is dots
    assert found[0].value == pytest.approx(5.0 / 10.0 * fig.bbox.width * 0.775, rel=0.1)
    dots.set_offsets([[2.0, 5.0], [3.0, 5.0]])
    assert ept.motion_faults(fig, lambda t: None, SPAN, rest=[dots]) == []


def test_a_moving_line_collection_is_traced():
    fig, ax = _scatter_clip()
    lines = LineCollection([[(1, 1), (3, 1)], [(1, 2), (3, 2)]], colors="C0")
    ax.add_collection(lines)

    def at(t):
        d = 4.0 * _smooth(t / DURATION)
        lines.set_segments([[(1 + d, 1), (3 + d, 1)], [(1 + d, 2), (3 + d, 2)]])

    found = ept.motion_faults(fig, at, SPAN, rest=[lines])
    assert _kinds(found) == ["drift"]
    assert found[0].value > 100.0


def test_a_scatter_pushed_off_the_figure_is_outside():
    fig = _figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 10), ylim=(0, 10))
    ax.axis("off")
    dots = ax.scatter([5.0], [5.0], s=400, clip_on=False)

    def at(t):
        dots.set_offsets([[5.0 + 7.0 * _smooth(t / DURATION), 5.0]])

    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["outside"]
    assert found[0].artist is dots


def test_a_pure_teleport_is_a_jump():
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 280.0 if t >= 1.0 else 20.0, 80)

    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["jump"]
    assert found[0].time == pytest.approx(1.0, abs=1.0 / 30.0)
    assert found[0].value > 1.0
    # The same block holding still has nothing to report.
    assert ept.motion_faults(fig, lambda t: _move(block, 20, 80), SPAN) == []


def test_a_one_frame_excursion_of_a_stationary_artist_is_flagged():
    fig = _figure()
    block = _box(fig, 100, 80, 30, 30)

    def at(t):
        _move(block, 230.0 if round(t * 30.0) == 30 else 100.0, 80)

    assert "jump" in _kinds(ept.motion_faults(fig, at, SPAN))
    found = ept.motion_faults(fig, at, SPAN, spikes=True)
    spikes = [f for f in found if f.kind == "spike"]
    assert spikes
    assert spikes[0].artist is block
    assert spikes[0].value == pytest.approx(130.0, abs=2.0)
    # A one-pixel flutter of the same artist is not a spike.
    fig = _figure()
    block = _box(fig, 100, 80, 30, 30)

    def flutter(t):
        _move(block, 101.0 if round(t * 30.0) == 30 else 100.0, 80)

    assert ept.motion_faults(fig, flutter, SPAN, spikes=True) == []


def _cut_label(clip_right):
    fig = _figure()
    label = fig.text(0.1, 0.5, "a long label that is cut", fontsize=14)
    label.set_clip_on(True)

    def at(t):
        _clip_to(label, 0.0, clip_right(t))

    return fig, at, label


def test_a_clip_box_that_barely_changes_is_not_a_reveal():
    fig, at, label = _cut_label(lambda t: 120.0 + 1e-5 * t)
    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["outside"]
    assert found[0].artist is label
    assert found[0].value > 50.0
    # A real reveal, growing to uncover the whole label, is not a cut.
    fig, at, _ = _cut_label(lambda t: 20.0 + 300.0 * _smooth(t / DURATION))
    assert ept.motion_faults(fig, at, SPAN) == []


def test_a_label_clipped_away_entirely_is_outside():
    fig = _figure()
    label = fig.text(0.3, 0.5, "gone", fontsize=14)
    _clip_to(label, 0.0, 300.0)

    def at(t):
        label.set_position((0.3 + 0.8 * _smooth(t / DURATION), 0.5))

    found = ept.motion_faults(fig, at, SPAN)
    assert _kinds(found) == ["outside"]
    assert found[0].artist is label
    # The same label that never leaves the clip has nothing to report.
    label.set_position((0.3, 0.5))
    assert ept.motion_faults(fig, lambda t: None, SPAN) == []


def _faded_pair(**style):
    fig = _figure()
    a = _box(fig, 100, 80, **style)
    b = _box(fig, 140, 90, **style)

    def at(t):
        _move(a, 100 + 20 * t, 80)
        _move(b, 140 - 20 * t, 90)

    return fig, at, [{"artists": [a, b]}]


def test_patch_alpha_is_not_squared():
    for style in ({"alpha": 0.5}, {"color": (0.0, 0.0, 0.0, 0.5)}):
        fig, at, moves = _faded_pair(**style)
        found = ept.motion_faults(fig, at, SPAN, moves=moves)
        assert _kinds(found) == ["overlap"], style
    # Under the 0.3 cutoff stays clean, over it reports.
    fig, at, moves = _faded_pair(alpha=0.2)
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []
    fig, at, moves = _faded_pair(alpha=0.35)
    assert _kinds(ept.motion_faults(fig, at, SPAN, moves=moves)) == ["overlap"]


def test_a_faint_patch_is_still_traced_for_the_frame_edge():
    fig = _figure()
    block = _box(fig, 300, 80, 60, 40, alpha=0.2)

    def at(t):
        _move(block, 300 + 70 * _smooth(t / DURATION), 80)

    found = ept.motion_faults(fig, at, SPAN, moves=[{"artists": [block]}])
    assert _kinds(found) == ["outside"]


def _crossing_boxes(fps_clip=None):
    fig = _figure()
    a = _box(fig, 100, 80, 30, 20)
    b = _box(fig, 140, 85, 30, 20)

    def at(t):
        _move(a, 100 + 20 * t, 80)
        _move(b, 140 - 20 * t, 85)

    return fig, at, a, b


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_overlap_does_not_depend_on_the_frame_rate(fps):
    fig, at, a, b = _crossing_boxes()
    found = ept.motion_faults(fig, at, SPAN, fps=fps, moves=[{"artists": [a, b]}])
    assert _kinds(found) == ["overlap"]
    assert {found[0].artist, found[0].other} == {a, b}


def test_overlap_uses_only_what_the_clip_leaves_visible():
    fig, at, a, b = _crossing_boxes()
    moves = [{"artists": [a, b]}]
    _clip_to(a, 0.0, 120.0)
    _clip_to(b, 125.0, 400.0)
    assert ept.motion_faults(fig, at, SPAN, moves=moves, kinds=["overlap"]) == []
    _clip_to(a, 0.0, 400.0)
    _clip_to(b, 0.0, 400.0)
    found = ept.motion_faults(fig, at, SPAN, moves=moves, kinds=["overlap"])
    assert _kinds(found) == ["overlap"]


def test_overlap_indexes_allowances_once_and_preserves_fault_records():
    fig = _figure()
    a, b, c = artists = [fig.text(0, 0, name) for name in ("a", "b", "c")]
    times = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.61, 0.7]
    trace = ept._Trace(times, keep_frames=False)
    trace.order = artists
    for j, artist in enumerate(artists):
        trace.boxes[artist] = {
            k: ((k + 2 * j, 0, k + 2 * j + 20, 10), None, 1, None)
            for k in range(len(times))
        }

    class Allowances(list):
        visits = 0

        def __iter__(self):
            for item in super().__iter__():
                self.visits += 1
                yield item

    allowed = Allowances(
        [
            (frozenset((a, b)), 0.2, 0.3),
            (frozenset((b, a)), 0.5, 0.5),
            (frozenset((a, c)), -math.inf, math.inf),
        ]
    )
    found = ept._check_overlap(trace, set(artists), allowed, 2, 0.5)
    # Exact records from the original checker: both allowance endpoints
    # include the 0.1-second motion window, and separate spans form a union.
    assert found == [
        (
            0,
            "overlap",
            a,
            b,
            (2, 0, 20, 10),
            "text 'a' and text 'b' overlap by 18.0 x 10.0 px while moving, "
            "at 3 sample(s)",
            10,
        ),
        (
            0,
            "overlap",
            b,
            c,
            (4, 0, 22, 10),
            "text 'b' and text 'c' overlap by 18.0 x 10.0 px while moving, "
            "at 9 sample(s)",
            10,
        ),
    ]
    assert allowed.visits == len(allowed)


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_a_smooth_accelerating_move_has_no_jump_at_any_frame_rate(fps):
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40)

    def at(t):
        _move(block, 20 + 250 * (t / DURATION) ** 5, 80)

    assert ept.motion_faults(fig, at, SPAN, fps=fps) == []


def test_a_step_into_rest_is_found_with_rest_appended_after_the_move():
    fig, at = _ramp_clip(final_step=2.5)
    found = ept.motion_faults(fig, at, ept.Window(0.0, 2.5))
    assert _kinds(found) == ["jump"]
    assert found[0].time == pytest.approx(2.0, abs=1.0 / 30.0)
    fig, at = _ramp_clip(final_step=1.5)
    assert ept.motion_faults(fig, at, ept.Window(0.0, 2.5)) == []


def test_a_declared_move_boundary_decides_where_rest_begins():
    fig, at = _ramp_clip(final_step=2.5)
    moves = [{"name": "fade", "span": (0.0, 2.0)}]
    found = ept.motion_faults(fig, at, ept.Window(0.0, 2.5), moves=moves)
    assert _kinds(found) == ["jump"]
    assert found[0].time == pytest.approx(2.0, abs=1.0 / 30.0)


def _teleporting(width_growth_px=0.0, normalized=False):
    fig = _figure()
    if normalized:
        box = Rectangle((0.05, 0.4), 0.1, 0.2, transform=fig.transFigure, facecolor="k")
        box.set_clip_on(False)
        fig.add_artist(box)
    else:
        box = _box(fig, 20, 80, 40, 40)

    def at(t):
        k = round(t * 30.0)
        jump = 60.0 if k >= 30 else 0.0
        if normalized:
            box.set_x(0.05 + (4.0 * k + jump) / 400.0)
            box.set_width(0.1 + width_growth_px * k / 60.0 / 400.0)
        else:
            _move(box, 20 + 4.0 * k + jump, 80)
            box.set_width(40.0 + width_growth_px * k / 60.0)

    return fig, at, box


def test_a_slight_shape_change_does_not_hide_a_teleport():
    for growth in (0.0, 2.0):
        fig, at, _ = _teleporting(width_growth_px=growth)
        found = ept.motion_faults(fig, at, SPAN, spikes=True)
        assert "spike" in _kinds(found), growth
    for dpi in (80, 100):
        fig, at, _ = _teleporting(width_growth_px=1.2, normalized=True)
        found = ept.motion_faults(fig, at, SPAN, spikes=True, dpi=dpi)
        assert "spike" in _kinds(found), dpi


def test_a_bar_that_grows_a_lot_is_not_a_spike():
    fig = _figure()
    bar = _box(fig, 40, 40, 20, 20)

    def at(t):
        k = round(t * 30.0)
        bar.set_width(20.0 + (160.0 if k >= 30 else 0.0))
        bar.set_y(40.0 + 2.0 * k)

    # Its width jumps, which takes the center sideways: not a position fault.
    assert "spike" not in _kinds(ept.motion_faults(fig, at, SPAN, spikes=True))
    # The same sideways jump of a bar that keeps its shape is one.
    fig = _figure()
    block = _box(fig, 40, 40, 20, 20)

    def hop(t):
        k = round(t * 30.0)
        _move(block, 40.0 + (80.0 if k >= 30 else 0.0), 40.0 + 2.0 * k)

    assert "spike" in _kinds(ept.motion_faults(fig, hop, SPAN, spikes=True))


def test_stable_tick_labels_with_a_one_frame_excursion_are_a_spike():
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=DPI)
    fig.subplots_adjust(bottom=0.35, left=0.2)
    ax.set(xlim=(0, 10), ylim=(0, 1), xticks=[2, 4, 6, 8], yticks=[])

    def at(t):
        ax.tick_params(axis="x", pad=30.0 if round(t * 30.0) == 30 else 3.5)

    found = ept.motion_faults(fig, at, SPAN, kinds=["spike"])
    assert _kinds(found) == ["spike"]
    assert found[0].value > 20.0


def test_velocity_is_judged_over_elapsed_time():
    fig = _figure()
    block = _box(fig, 20, 80, 30, 30)
    times = [0.0, 0.01, 0.02, 0.5, 0.51, 0.52, 1.0, 1.01, 1.02]

    def at(t):
        _move(block, 20 + 100 * t, 80)

    assert ept.motion_faults(fig, at, times, kinds=["spike"]) == []

    # A genuine change of velocity at the same sampling is still reported.
    def kinked(t):
        _move(block, 20 + 100 * t + (60.0 if t >= 0.51 else 0.0), 80)

    assert _kinds(ept.motion_faults(fig, kinked, times, kinds=["spike"])) == ["spike"]


def _pair_of_polygons(rect_lag):
    fig = _figure()
    tri_pts = [(40.0, 80.0), (70.0, 80.0), (40.0, 110.0)]
    rect_pts = [(130.0, 80.0), (160.0, 80.0), (160.0, 110.0), (130.0, 110.0)]
    tri = Polygon(tri_pts, transform=IdentityTransform(), facecolor="k")
    rect = Polygon(rect_pts, transform=IdentityTransform(), facecolor="k")
    for patch in (tri, rect):
        patch.set_clip_on(False)
        fig.add_artist(patch)

    def at(t):
        u = _smooth(t / DURATION)
        tri.set_xy(_rotate(tri_pts, 90.0 * u))
        rect.set_xy(_rotate(rect_pts, 90.0 * u * (1.0 - rect_lag)))

    return fig, at, [tri, rect]


def test_a_rigid_rotation_does_not_tear_and_a_lagging_part_does():
    fig, at, pair = _pair_of_polygons(rect_lag=0.0)
    assert ept.motion_faults(fig, at, SPAN, groups=[pair], tol_px=0.5) == []
    fig, at, pair = _pair_of_polygons(rect_lag=0.3)
    found = ept.motion_faults(fig, at, SPAN, groups=[pair])
    assert _kinds(found) == ["tear"]
    assert {found[0].artist, found[0].other} == set(pair)


def test_a_resting_line_whose_middle_point_moves_drifts():
    fig = _figure()
    line = Line2D(
        [50, 100, 150], [50, 100, 50], transform=IdentityTransform(), color="k"
    )
    line.set_clip_on(False)
    fig.add_artist(line)

    def at(t):
        line.set_data([50, 100 + 40 * _smooth(t / DURATION), 150], [50, 100, 50])

    found = ept.motion_faults(fig, at, SPAN, rest=[line])
    assert _kinds(found) == ["drift"]
    assert found[0].value > 5.0
    assert ept.motion_faults(fig, lambda t: None, SPAN, rest=[line]) == []


def _two_stage(first_moves_b, second_moves_b=True):
    fig = _figure()
    a = _glyph(fig, 60, 80)
    b = _glyph(fig, 200, 80)

    def at(t):
        _shift_glyph(a, 60 + 50 * _smooth(t / DURATION), 80)
        x = 200.0
        if first_moves_b:
            x += 40 * _smooth(t)
        if second_moves_b:
            x += 60 * _smooth(t - 1.0)
        _shift_glyph(b, x, 80)

    return fig, at, a, b


def test_rest_declared_by_a_move_holds_only_over_its_span():
    fig, at, _, b = _two_stage(first_moves_b=False)
    moves = [
        {"name": "a", "span": (0.0, 1.0), "rest": [b]},
        {"name": "b", "span": (1.0, 2.0), "artists": [b]},
    ]
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []
    fig, at, _, b = _two_stage(first_moves_b=True)
    moves[0]["rest"] = [b]
    moves[1]["artists"] = [b]
    found = ept.motion_faults(fig, at, SPAN, moves=moves)
    assert _kinds(found) == ["drift"]
    assert found[0].artist is b
    assert found[0].time < 1.0


def test_a_group_declared_by_a_move_holds_only_over_its_span():
    fig = _figure()
    left, right = _glyph(fig, 60, 80), _glyph(fig, 120, 80)

    def at(t):
        u = _smooth(t - 1.0)
        _shift_glyph(left, 60 + 80 * _smooth(t), 80)
        _shift_glyph(right, 120 + 80 * _smooth(t) + 40 * u, 80)

    moves = [{"span": (0.0, 1.0), "groups": [[left, right]]}]
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []
    moves = [{"span": (0.0, 2.0), "groups": [[left, right]]}]
    assert _kinds(ept.motion_faults(fig, at, SPAN, moves=moves)) == ["tear"]


def test_an_allowed_overlap_holds_only_over_its_span():
    fig, at, a, b, _ = _crossing()
    inside = [{"span": (0.0, 2.0), "artists": [a, b], "may_overlap": [(a, b)]}]
    assert ept.motion_faults(fig, at, SPAN, moves=inside) == []
    early = [{"span": (0.0, 0.2), "artists": [a, b], "may_overlap": [(a, b)]}]
    assert _kinds(ept.motion_faults(fig, at, SPAN, moves=early)) == ["overlap"]


def test_an_alpha_only_move_is_not_dead_on_a_transparent_figure():
    fig = _figure()
    fig.patch.set_alpha(0.0)
    block = _box(fig, 20, 80, 60, 40, color="w", alpha=0.0)

    def at(t):
        block.set_alpha(min(1.0, max(0.1, t / DURATION)))

    moves = [{"name": "fade", "span": (0.0, 2.0), "artists": [block]}]
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []
    block.set_alpha(0.6)
    assert _kinds(ept.motion_faults(fig, lambda t: None, SPAN, moves=moves)) == ["dead"]


def test_a_declared_move_whose_artists_never_show_is_dead():
    fig = _figure()
    ghost = Rectangle((10, 10), 30, 30, transform=IdentityTransform(), alpha=0.0)
    fig.add_artist(ghost)
    other = _box(fig, 200, 100, 40, 40)

    def at(t):
        _move(other, 200 + 100 * _smooth(t / DURATION), 100)

    found = ept.motion_faults(
        fig, at, SPAN, moves=[{"name": "ghost", "artists": [ghost]}]
    )
    assert _kinds(found) == ["dead"]
    assert "ghost" in found[0].detail


@pytest.mark.parametrize("stop", [0.1 + 0.2, 0.95, 1.0 / 3.0, 2.0])
def test_the_last_sample_is_exactly_stop(stop):
    seen = []
    fig = _figure()
    _box(fig, 10, 10)
    ept.motion_faults(fig, seen.append, ept.Window(0.0, stop))
    assert seen[-1] == stop
    assert seen == sorted(seen)


def test_a_callback_branching_on_stop_reaches_its_final_state():
    stop = 0.1 + 0.2
    fig = _figure()
    block = _box(fig, 20, 80, 60, 40, alpha=0.0)

    def at(t):
        block.set_alpha(1.0 if t >= stop else 0.5 * t / stop)

    found = ept.motion_faults(fig, at, ept.Window(0.0, stop))
    assert block.get_alpha() == 1.0
    assert _kinds(found) == ["jump"]
    assert found[0].time == stop


def test_pixels_are_kept_only_for_the_kinds_that_read_them(monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    calls = []
    real = FigureCanvasAgg.buffer_rgba
    monkeypatch.setattr(
        FigureCanvasAgg, "buffer_rgba", lambda self: calls.append(1) or real(self)
    )
    fig, at, steady = _slider(shift_px=14.0)
    ept.motion_faults(fig, at, SPAN, kinds=["drift"], rest=[steady])
    assert calls == []
    ept.motion_faults(fig, at, SPAN, kinds=["jump"])
    assert len(calls) == 61


def test_a_tear_names_both_artists_and_empty_groups_are_ignored():
    fig, at, parts = _row(stagger=0.15)
    found = ept.motion_faults(fig, at, SPAN, groups=[parts])
    assert found[0].artist in parts
    assert found[0].other in parts
    assert found[0].other is not found[0].artist
    assert ept.motion_faults(fig, at, SPAN, groups=[[], [parts[0]]]) == []


def test_a_spike_is_dated_at_its_first_faulty_sample():
    fig, at, _ = _hopper(hop_px=9.0, travel_px=2.0)
    found = ept.motion_faults(fig, at, SPAN, spikes=True)
    assert found[0].kind == "spike"
    assert found[0].time < 0.9  # the peak of the hop is at t = 1
    assert found[0].value == pytest.approx(9.0, abs=1.0)  # the peak is the value


# the call


def test_span_samples_every_frame_and_ends_on_stop():
    seen = []
    fig = _figure()
    _box(fig, 10, 10)
    ept.motion_faults(fig, seen.append, ept.Window(0.0, 1.0))
    assert len(seen) == 31
    assert seen[-1] == pytest.approx(1.0)
    seen.clear()
    ept.motion_faults(fig, seen.append, ept.Window(0.0, 1.0), fps=10)
    assert len(seen) == 11
    seen.clear()
    ept.motion_faults(fig, seen.append, ept.Window(0.0, 0.95))
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
    assert _kinds(first) == ["tear"]
    assert [str(f) for f in first] == [str(f) for f in again]
    assert [f.value for f in first] == [f.value for f in again]
    times = [f.time for f in first]
    assert times == sorted(times)


def test_assert_motion_raises_with_every_fault_named():
    fig, at, steady = _slider(shift_px=14.0)
    with pytest.raises(ept.MotionError, match="drift") as raised:
        ept.assert_motion(fig, at, SPAN, rest=[steady])
    assert [f.kind for f in raised.value.faults] == ["drift"]
    assert raised.value.faults[0].value == pytest.approx(14.0, abs=0.6)
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
        "Window",
        "assert_motion",
        "motion_faults",
    ):
        assert name in ept.__all__
        assert hasattr(ept, name)


def test_a_clip_with_text_and_a_move_runs_clean():
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


# second review round: counterexamples, each with a clean control


def test_collection_boxes_pair_paths_offsets_and_sizes_like_the_renderer():
    _, ax = _scatter_clip()
    lines = LineCollection(
        [[(0, 0), (10, 0)], [(0, 0), (10, 0)]],
        offsets=[(50, 100), (150, 100)],
        offset_transform=IdentityTransform(),
        transform=IdentityTransform(),
    )
    ax.add_collection(lines)
    box = ept._collection_box(lines, DPI)
    assert (box.x0, box.x1) == pytest.approx((50.0, 160.0), abs=2.0)
    dots = ax.scatter([100, 390], [50, 50], s=[10000, 4], transform=IdentityTransform())
    box = ept._collection_box(dots, DPI)
    assert box.x1 == pytest.approx(392.1, abs=0.5)  # the big marker is not on the right
    assert box.x0 == pytest.approx(100 - 100 * DPI / 144.0, abs=1.0)


def test_steady_travel_of_a_tiny_faint_artist_is_not_a_jump():
    fig = _figure()
    dot = _box(fig, 30, 80, 2, 2, alpha=0.2)

    def at(t):
        _move(dot, 30 + 120 * t, 80)

    assert ept.motion_faults(fig, at, SPAN, kinds=["jump"]) == []


def test_taking_the_clip_off_at_the_end_of_a_reveal_is_not_outside():
    # (clip removed at the end, how far the clip opens, faults expected)
    for remove, reach, expected in ((True, 100.0, []), (False, 40.0, ["outside"])):
        fig = _figure()
        bar = _box(fig, 50, 80, 100, 20)
        clip = Bbox([[0, 0], [50, 200]])
        bar.set_clip_box(TransformedBbox(clip, IdentityTransform()))

        def at(t, bar=bar, clip=clip, remove=remove, reach=reach):
            clip.x1 = 50 + reach * min(t / 1.8, 1.0)
            bar.set_clip_on(not (remove and t > 1.9))

        moves = [{"artists": [bar]}]
        found = ept.motion_faults(fig, at, SPAN, kinds=["outside"], moves=moves)
        assert _kinds(found) == expected, (remove, reach)


def test_a_far_away_move_span_does_not_hide_the_step_into_rest():
    fig, at = _ramp_clip(final_step=2.5)
    moves = [{"name": "elsewhere", "span": (3.0, 4.0)}]
    found = ept.motion_faults(
        fig, at, ept.Window(0.0, 2.5), moves=moves, kinds=["jump"]
    )
    assert _kinds(found) == ["jump"]


def test_a_rest_declaration_outside_the_clip_is_ignored():
    fig = _figure()
    block = _box(fig, 100, 80)
    other = _box(fig, 20, 80)

    def at(t):
        _move(other, 20 + 100 * t / DURATION, 80)

    moves = [{"span": (2.5, 3.0), "rest": [block]}]
    assert ept.motion_faults(fig, at, SPAN, moves=moves) == []
    with pytest.raises(ValueError, match="not visible"):
        ept.motion_faults(fig, at, SPAN, rest=[_box(fig, 0, 0, alpha=0.0)])


def test_a_width_wobble_of_a_fixed_bar_is_not_a_spike():
    fig = _figure()
    bar = _box(fig, 40, 80, 100, 20)

    def at(t):
        bar.set_width(100 + 20 * math.sin(math.pi * t / DURATION) ** 2)

    assert "spike" not in _kinds(ept.motion_faults(fig, at, SPAN, spikes=True))


def test_pixel_rectangles_translating_together_do_not_tear():
    fig = _figure()
    a = _box(fig, 50, 80, 20, 20)
    b = _box(fig, 120, 80, 40, 20)

    def at(t):
        _move(a, 50 + 10 * t, 80)
        _move(b, 120 + 10 * t, 80)

    assert ept.motion_faults(fig, at, SPAN, groups=[[a, b]], kinds=["tear"]) == []

    def apart(t):
        _move(a, 50 + 10 * t, 80)
        _move(b, 120 + 10 * t * 21 / 41 + 10 * t, 80)

    found = ept.motion_faults(fig, apart, SPAN, groups=[[a, b]], kinds=["tear"])
    assert _kinds(found) == ["tear"]


def test_resampling_a_straight_line_does_not_drift():
    fig = _figure()
    line = fig.add_artist(Line2D([30, 130], [50, 50], transform=IdentityTransform()))
    line.set_clip_on(False)

    def at(t):
        mid = 30 + 100 * t / DURATION
        line.set_data([30, mid, 130], [50, 50, 50])

    assert ept.motion_faults(fig, at, SPAN, rest=[line], kinds=["drift"]) == []


def test_a_declared_image_that_changes_is_not_dead():
    fig, ax = plt.subplots(figsize=(4.0, 2.0), dpi=DPI)
    image = ax.imshow([[0.0, 1.0], [1.0, 0.0]], vmin=0, vmax=1)

    def at(t):
        image.set_data([[t / DURATION, 1.0], [1.0, 0.0]])

    moves = [{"name": "picture", "artists": [image]}]
    assert ept.motion_faults(fig, at, SPAN, moves=moves, kinds=["dead"]) == []
    frozen = ept.motion_faults(fig, lambda t: None, SPAN, moves=moves, kinds=["dead"])
    assert _kinds(frozen) == ["dead"]
    assert "never visible" not in frozen[0].detail


def test_a_velocity_change_is_dated_at_its_first_sample():
    fig = _figure()
    block = _box(fig, 20, 80, 20, 20)

    def at(t):
        _move(block, 20 + 20 * t + 20 * (t >= 0.5) + 60 * (t >= 1.5), 80)

    found = [f for f in ept.motion_faults(fig, at, SPAN, kinds=["spike"])]
    assert found and found[0].time == pytest.approx(0.5, abs=1.0 / 30.0)


@pytest.mark.parametrize("fps", [24, 30, 60])
def test_slow_overlap_is_found_at_every_frame_rate(fps):
    fig = _figure()
    a = _box(fig, 100, 80, 30, 20)
    b = _box(fig, 110, 85, 30, 20)

    def at(t):
        _move(a, 100 + 5.5 * t, 80)
        _move(b, 110 + 5.5 * t, 85)

    found = ept.motion_faults(fig, at, SPAN, fps=fps, moves=[{"artists": [a, b]}])
    assert _kinds(found) == ["overlap"]


def test_many_identical_declarations_cost_about_one(monkeypatch):
    fig = _figure()
    block = _box(fig, 20, 80)

    def at(t):
        _move(block, 20 + 100 * t / DURATION, 80)

    calls = []
    real = ept.np.abs

    def counting(x, *a, **k):
        calls.append(1)
        return real(x, *a, **k)

    monkeypatch.setattr(ept.np, "abs", counting)
    one = [{"name": "m", "artists": [block]}]
    ept.motion_faults(fig, at, SPAN, moves=one, kinds=["dead"])
    few = len(calls)
    calls.clear()
    ept.motion_faults(fig, at, SPAN, moves=one * 20, kinds=["dead"])
    assert len(calls) <= few + 2


# Shape class: a piece declared rectilinear stays an axis-aligned rectangle.


def _rect_clip(corners_at):
    """A figure with one polygon whose corners (display pixels) depend on u."""
    fig = _figure()
    poly = Polygon(
        corners_at(0.0), closed=True, transform=IdentityTransform(), fc="C0", ec="none"
    )
    fig.add_artist(poly)

    def at(t):
        poly.set_xy(corners_at(_smooth(t / DURATION)))

    return fig, at, poly


def _corners_of(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def test_a_bar_that_stretches_as_a_rectangle_keeps_its_shape():
    fig, at, bar = _rect_clip(lambda u: _corners_of(40, 90, 80 + 160 * u, 94))
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[bar], kinds=["shape-class"])
    assert found == []


def test_a_bar_whose_ends_slant_in_the_middle_of_a_move_is_reported():
    def corners(u):
        lean = 6.0 * np.sin(np.pi * u)  # the right end leans, then straightens
        return np.array(
            [[40, 90], [80 + 160 * u + lean, 90], [80 + 160 * u, 94], [40, 94]]
        )

    fig, at, bar = _rect_clip(corners)
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[bar], kinds=["shape-class"])
    assert _kinds(found) == ["shape-class"]
    assert found[0].artist is bar
    # A slanted side is off axis by its smaller offset, here the 4 px height.
    assert found[0].value == pytest.approx(4.0, abs=0.5)
    assert 0.0 < found[0].time < DURATION


def test_a_rectangle_that_turns_is_not_axis_aligned():
    def corners(u):
        a = 0.5 * np.pi * 0.3 * u
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return (_corners_of(-40, -3, 40, 3) @ rot.T) + [120, 100]

    fig, at, bar = _rect_clip(corners)
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[bar], kinds=["shape-class"])
    assert _kinds(found) == ["shape-class"]
    # Without the declaration the same clip is not a fault of this kind.
    assert ept.motion_faults(fig, at, SPAN, kinds=["shape-class"]) == []


def test_a_piece_that_gains_a_corner_is_no_longer_a_rectangle():
    def corners(u):
        notch = 4.0 * u
        return np.array(
            [
                [40, 90],
                [200, 90],
                [200, 100],
                [120 + notch, 100],
                [120 + notch, 94],
                [40, 94],
            ]
        )

    fig, at, piece = _rect_clip(corners)
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[piece], kinds=["shape-class"])
    assert _kinds(found) == ["shape-class"]
    assert "no longer a rectangle" in found[0].detail


def test_extra_points_along_a_straight_side_do_not_count_as_corners():
    def corners(u):
        x1 = 80 + 160 * u
        return np.array(
            [[40, 90], [60, 90], [x1 / 2 + 30, 90], [x1, 90], [x1, 94], [40, 94]]
        )

    fig, at, bar = _rect_clip(corners)
    assert (
        ept.motion_faults(fig, at, SPAN, rectilinear=[bar], kinds=["shape-class"]) == []
    )


def test_a_declared_move_carries_its_rectilinear_pieces():
    def corners(u):
        return np.array([[40, 90], [160 + 6 * u, 90], [160, 94], [40, 94]])

    fig, at, bar = _rect_clip(corners)
    move = {"name": "bar", "rectilinear": [bar]}
    found = ept.motion_faults(fig, at, SPAN, moves=[move], kinds=["shape-class"])
    assert _kinds(found) == ["shape-class"]


def test_a_rectilinear_artist_that_never_shows_is_an_error():
    fig, at, bar = _rect_clip(lambda u: _corners_of(40, 90, 80 + 160 * u, 94))
    bar.set_visible(False)
    with pytest.raises(ValueError, match="not visible"):
        ept.motion_faults(fig, at, SPAN, rectilinear=[bar], kinds=["shape-class"])


def test_a_collection_cannot_be_declared_rectilinear():
    fig = _figure()
    mark = fig.add_artist(LineCollection([[(10, 10), (50, 10)]]))
    with pytest.raises(TypeError, match="rectilinear"):
        ept.motion_faults(
            fig, lambda t: None, SPAN, rectilinear=[mark], kinds=["shape-class"]
        )


def _line_clip(end_at):
    fig = _figure()
    line = Line2D([40, 200], [90, 90], transform=IdentityTransform(), lw=2)
    fig.add_artist(line)

    def at(t):
        x, y = end_at(_smooth(t / DURATION))
        line.set_data([40, x], [90, y])

    return fig, at, line


def test_a_horizontal_line_rule_passes_while_it_stays_axis_aligned():
    fig, at, line = _line_clip(lambda u: (120 + 80 * u, 90))
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[line], kinds=["shape-class"])
    assert found == []


def test_a_line_rule_that_tilts_is_reported():
    fig, at, line = _line_clip(lambda u: (200, 90 + 8 * u))
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[line], kinds=["shape-class"])
    assert _kinds(found) == ["shape-class"]
    assert found[0].value == pytest.approx(8.0, abs=0.5)


def test_a_bent_polyline_is_not_a_rectangle():
    fig = _figure()
    line = Line2D([40, 120, 120], [90, 90, 130], transform=IdentityTransform())
    fig.add_artist(line)
    found = ept.motion_faults(
        fig, lambda t: None, SPAN, rectilinear=[line], kinds=["shape-class"]
    )
    assert "no longer a rectangle" in found[0].detail


def test_a_hairline_rectangle_keeps_its_corners():
    fig, at, bar = _rect_clip(lambda u: _corners_of(40, 90, 80 + 160 * u, 90.4))
    found = ept.motion_faults(fig, at, SPAN, rectilinear=[bar], kinds=["shape-class"])
    assert found == []


def test_a_slow_large_fade_is_not_an_eight_bit_jump():
    fig = _figure()
    box = _box(fig, 20, 20, 360, 160)
    dot = _box(fig, 10, 5, 4, 4)

    def at(t):
        box.set_alpha(0.1 * _smooth(t / 6.0))
        _move(dot, 10 + 10 * t, 5)

    assert ept.motion_faults(fig, at, ept.Window(0, 6), kinds=["jump"]) == []
