"""Regressions for check noise and explicit, auditable exceptions."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Rectangle
from matplotlib.transforms import IdentityTransform

import eyepiece as ep


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_arrow_on_marks_uses_drawn_extent():
    fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    arrow = ep.motion.arrow((0.8, 0.7), (0.8, 0.9), ax=ax, color="C0")
    mark = arrow.artists["arrow"]
    if isinstance(mark, list):
        mark = mark[0]
    fig.canvas.draw()
    # CLOSEPOLY's dummy vertex must not draw a stroke to the origin.
    label = fig.text(0, 0, "note", color="C0", fontsize=16)
    label.set_transform(IdentityTransform())
    label.set_position((80, 180))
    assert not ep.testing.readability_faults(fig, kinds=("on-marks",))
    x0, y0, x1, y1 = mark.get_window_extent().extents
    label.set_position(((x0 + x1) / 2, (y0 + y1) / 2))
    assert ep.testing.readability_faults(fig, kinds=("on-marks",))


@pytest.mark.parametrize(
    "hidden",
    [
        {"alpha": 0},
        {"color": "none"},
        {"linestyle": "none", "marker": "none"},
        {"linewidth": 0},
    ],
)
def test_no_ink_line_is_not_a_mark(hidden):
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0.5, 0.5], color="C0", lw=3)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.5, 0.5, "value", color="C0", fontsize=16, va="center")
    line.set(**hidden)
    assert ep.testing.readability_faults(fig, kinds=("on-marks",)) == []
    line.set(alpha=1, color="C0", linestyle="-", linewidth=3)
    assert len(ep.testing.readability_faults(fig, kinds=("on-marks",))) == 1


@pytest.mark.parametrize("alpha", [0, 1])
def test_group_member_with_no_ink_in_window_reports_fault(alpha):
    fig, ax = plt.subplots()
    a = ax.text(0.2, 0.5, "a")
    b = ax.text(0.7, 0.5, "b")

    def at(t):
        b.set_alpha(alpha if t < 1 else 1)

    move = {"span": (0, 0.5), "groups": [[a, b]]}
    found = ep.testing.motion_faults(
        fig, at, [0, 0.25, 0.5, 1], moves=[move], kinds=("tear",)
    )
    assert len(found) == (1 if alpha == 0 else 0)
    if found:
        assert found[0].artist is b
        assert found[0].kind == "tear"
        assert "no ink" in found[0].detail


def test_identical_text_crossfade_is_not_overlap():
    fig = plt.figure()
    a = fig.text(0.2, 0.5, "same", alpha=0.5)
    b = fig.text(0.2, 0.5, "same", alpha=0.5)
    assert ep.testing.readability_faults(fig, kinds=("overlap",)) == []
    b.set_text("different")
    assert ep.testing.readability_faults(fig, kinds=("overlap",))
    b.set_text("same")
    b.set_position((0.205, 0.5))
    assert ep.testing.readability_faults(fig, kinds=("overlap",))
    b.set_position(a.get_position())
    b.set_rotation(10)
    assert ep.testing.readability_faults(fig, kinds=("overlap",))


@pytest.mark.parametrize("reserve", [False, True])
def test_stamp_is_small_furniture_but_still_checked_for_collisions(reserve):
    fig = plt.figure()
    stamp = ep.render.stamp(
        fig, note="simulated", date="2000-01-01", sha="abc", reserve=reserve
    )
    if reserve:
        stamp = stamp.artists["text"]
    assert ep.testing.readability_faults(fig, kinds=("small",)) == []
    fig.text(*stamp.get_position(), "collision")
    assert ep.testing.readability_faults(fig, kinds=("overlap",))


def test_duplicate_samples_are_reported_once():
    fig = plt.figure()
    fig.text(0.2, 0.5, "tiny", fontsize=3)
    found = ep.testing.readability_faults(
        fig, at=lambda t: None, times=[0, 0, 1], kinds=("small",)
    )
    assert [(f.time, f.kind) for f in found] == [(0, "small"), (1, "small")]


def test_repeated_move_declarations_produce_one_fault():
    fig = plt.figure()
    text = fig.text(0.2, 0.5, "tiny", fontsize=3)
    move = {"name": "still", "artists": [text]}
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.5, 1], moves=[move, move], kinds=("dead",)
    )
    assert len(found) == 1


def test_build_axes_registers_partial_wipe_without_hiding_other_cuts():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="a long axis label")
    build = ep.motion.build_axes(ax, axis="x")
    found = ep.testing.motion_faults(
        fig, lambda t: build.update(0.72 + t * 0.02), [0, 0.1, 0.2], kinds=("outside",)
    )
    assert found == []
    build.update(1)
    mask = Rectangle((0, 0), 1, 1, transform=IdentityTransform())
    ax.xaxis.label.set_clip_path(mask)
    ax.xaxis.label.set_clip_on(True)
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.1, 0.2], kinds=("outside",)
    )
    assert any(f.artist is ax.xaxis.label for f in found)


def test_waivers_match_kind_and_artist_and_preserve_audit():
    fig = plt.figure()
    a = fig.text(0.2, 0.5, "small", fontsize=3)
    b = fig.text(0.8, 0.5, "other", fontsize=3)
    waiver = ep.testing.Waive("small", artist=a, reason="footer by design")
    found = ep.testing.readability_faults(fig, waive=[waiver], kinds=("small",))
    assert [f.artist for f in found if not f.waived] == [b]
    assert len(found.waived) == 1
    fault = found.waived[0]
    reason = fault.waived
    assert fault.artist is a and reason == "footer by design"
    assert "waived: 1" in str(found) and reason in str(found)
    with pytest.raises(ep.testing.ReadabilityError, match="waived: 1"):
        ep.testing.assert_readable(fig, waive=[waiver], kinds=("small",))
    with pytest.raises(ValueError, match="reason"):
        ep.testing.Waive("small", artist=a, reason=" ")
    with pytest.raises(ValueError, match="kind"):
        ep.testing.Waive("typo", artist=a, reason="why")


def test_clip_report_retains_waived_findings():
    fig = plt.figure()
    a = fig.text(0.2, 0.5, "small", fontsize=3)
    tl = ep.Timeline()
    tl.hold(1)
    plan = tl.at_fps(3)
    waiver = ep.testing.Waive("small", artist=a, reason="footer")
    report = ep.testing.check_clip(fig, lambda k: None, plan, waive=[waiver])
    assert report.ok
    assert report.waived
    assert "waived:" in str(report) and "footer" in str(report)


def test_invisible_group_reports_fault_and_outside_span_is_ignored():
    fig = plt.figure()
    a = fig.text(0.2, 0.5, "a")
    b = fig.text(0.7, 0.5, "b", alpha=0)
    declaration = {"span": (0, 1), "groups": [[a, b]]}
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.5, 1], moves=[declaration], kinds=("tear",)
    )
    assert len(found) == 1 and found[0].artist is b
    declaration["span"] = (3, 4)
    assert (
        ep.testing.motion_faults(
            fig, lambda t: None, [0, 0.5, 1], moves=[declaration], kinds=("tear",)
        )
        == []
    )


def test_motion_waiver_is_reported_and_does_not_excuse_other_kinds():
    fig = plt.figure(figsize=(4, 3), dpi=100)
    a = fig.text(0.9, 0.5, "moves outside", fontsize=16)

    def at(t):
        a.set_position((0.9 + 0.05 * t, 0.5))

    waiver = ep.testing.Waive("outside", artist=a, reason="intentional departure")
    found = ep.testing.motion_faults(
        fig, at, [0, 0.5, 1], rest=[a], kinds=("outside", "drift"), waive=[waiver]
    )
    assert {f.kind for f in found if not f.waived} == {"drift"}
    assert {f.kind for f in found.waived} == {"outside"}
    assert "intentional departure" in str(found)
    with pytest.raises(ep.testing.MotionError, match="waived: 1"):
        ep.testing.assert_motion(
            fig, at, [0, 0.5, 1], rest=[a], kinds=("outside", "drift"), waive=[waiver]
        )


def test_pair_waiver_does_not_cover_a_third_artist():
    fig = plt.figure()
    a, b, c = [
        fig.text(0.2, 0.5, word, fontsize=16) for word in ("one", "two", "three")
    ]
    waiver = ep.testing.Waive("overlap", artist=a, other=b, reason="intentional pair")
    found = ep.testing.readability_faults(fig, kinds=("overlap",), waive=[waiver])
    assert len(found) == 3
    assert len(found.waived) == 1
    assert {found.waived[0].artist, found.waived[0].other} == {a, b}
    assert all(c in (f.artist, f.other) for f in found if not f.waived)
    with pytest.raises(ValueError, match="artist"):
        ep.testing.Waive("small", artist=None, reason="too broad")


def test_distinct_unnamed_dead_moves_keep_both_regions():
    fig = plt.figure(figsize=(4, 2), dpi=80)
    left = fig.text(0.1, 0.3, "left")
    right = fig.text(0.7, 0.6, "right")
    found = ep.testing.motion_faults(
        fig,
        lambda t: None,
        [0, 0.5, 1],
        moves=[{"artists": [left]}, {"artists": [right]}],
        kinds=("dead",),
    )
    assert len(found) == 2
    assert found[0].box != found[1].box
