"""Regressions for check noise and explicit, auditable exceptions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import patheffects
from matplotlib.patches import Rectangle
from matplotlib.text import Text
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


def test_repeated_move_indices_remain_distinct():
    fig = plt.figure()
    text = fig.text(0.2, 0.5, "tiny", fontsize=3)
    move = {"name": "still", "artists": [text]}
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.5, 1], moves=[move, move], kinds=("dead",)
    )
    assert len(found) == 2
    assert found[0].declaration != found[1].declaration


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


def test_crossfade_different_colors():
    fig = plt.figure(figsize=(4, 2), dpi=100)
    fig.text(0.2, 0.5, "same", fontsize=20, color="red", alpha=0.5)
    fig.text(0.2, 0.5, "same", fontsize=20, color="blue", alpha=0.5)
    found = ep.testing.readability_faults(fig, kinds=("overlap",))
    assert len(found) == 1


def _assert_visible_mark_collision(fig, line):
    fig.canvas.draw()
    shown = np.asarray(fig.canvas.buffer_rgba()).copy()
    line.set_visible(False)
    fig.canvas.draw()
    hidden = np.asarray(fig.canvas.buffer_rgba()).copy()
    assert np.count_nonzero(np.any(shown != hidden, axis=2)) > 100
    line.set_visible(True)
    found = ep.testing.readability_faults(fig, kinds=("on-marks",))
    assert any(f.other is line for f in found)


@pytest.mark.parametrize("color_alpha", [0, 0.01, 0.06])
def test_visible_line_with_overridden_alpha(color_alpha):
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    (line,) = ax.plot(
        [0, 1], [0.5, 0.5], color=(0, 0, 0, color_alpha), alpha=0.75, lw=5
    )
    ax.text(0.5, 0.5, "value", color="black", fontsize=16, va="center")
    _assert_visible_mark_collision(fig, line)


@pytest.mark.parametrize("staged", [False, True])
def test_alpha_zero_with_visible_path_effect(staged):
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    (line,) = ax.plot(
        [0, 1],
        [0.5, 0.5],
        color="black",
        alpha=1,
        lw=5,
        path_effects=[
            patheffects.Stroke(linewidth=8, foreground="black", alpha=1),
            patheffects.Normal(),
        ],
    )
    ax.text(0.5, 0.5, "value", color="black", fontsize=16, va="center")
    if staged:
        ep.motion.Stage({"line": line}).weigh({"line": 0})
    else:
        line.set_alpha(0)
    _assert_visible_mark_collision(fig, line)


def test_distinct_dead_moves_same_region():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    (a,) = ax.plot([0.2, 0.8], [0.4, 0.6], color="black", lw=6)
    (b,) = ax.plot([0.2, 0.8], [0.4, 0.6], color="blue", lw=2, ls="--")
    found = ep.testing.motion_faults(
        fig,
        lambda t: None,
        [0, 0.5, 1],
        moves=[{"artists": [a]}, {"artists": [b]}],
        kinds=("dead",),
    )
    assert len(found) == 2


def test_dead_move_waiver_matches_artist():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    (line,) = ax.plot([0.2, 0.8], [0.4, 0.6], color="black", lw=6)
    found = ep.testing.motion_faults(
        fig,
        lambda t: None,
        [0, 0.5, 1],
        moves=[{"artists": [line]}],
        kinds=("dead",),
        waive=[ep.testing.Waive("dead", artist=line, reason="static design")],
    )
    assert len(found) == 1
    assert found[0].waived == "static design"


@pytest.mark.parametrize("hidden", [False, True])
def test_dead_move_waiver_matches_any_participant_and_pair(hidden):
    fig = plt.figure()
    a, b, c, other = [fig.text(0.2, 0.5, name) for name in "abcd"]
    for text in (a, b, c):
        text.set_visible(not hidden)
    waiver = ep.testing.Waive("dead", artist=c, other=b, reason="static group")
    found = ep.testing.motion_faults(
        fig,
        lambda t: None,
        [0, 0.5, 1],
        moves=[{"artists": [a, b, c]}, {"artists": [other]}],
        kinds=("dead",),
        waive=[waiver],
    )
    assert len(found) == 2
    assert found[0].artists == (a, b, c)
    assert found[0].waived == "static group"
    assert found[1].waived is None


@pytest.mark.parametrize("nested", [False, True])
def test_distinct_groups_keep_identical_findings(nested):
    fig = plt.figure()
    a = fig.text(0.2, 0.5, "a")
    b = fig.text(0.7, 0.5, "b", alpha=0)
    groups = [[a, b], [a, b]]
    kwargs = {"moves": [{"groups": groups}]} if nested else {"groups": groups}
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.5, 1], kinds=("tear",), **kwargs
    )
    assert len(found) == 2
    assert found[0].declaration != found[1].declaration


def test_distinct_hop_groups_keep_identical_findings():
    fig = plt.figure()
    a = fig.text(0.2, 0.5, "a")
    found = ep.testing.motion_faults(
        fig,
        lambda t: None,
        [0, 0.5, 1],
        kinds=("hop-rate",),
        hops=[{"artists": [a], "rate": 10}, {"artists": [a], "rate": 10}],
    )
    assert len(found) == 2
    assert found[0].declaration != found[1].declaration


@pytest.mark.parametrize("effect", [False, True])
def test_conservative_mark_finding_does_not_hide_another_collision(effect):
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    (faint,) = ax.plot([0, 1], [0.5, 0.5], color="black", alpha=0.01)
    if effect:
        faint.set_path_effects([patheffects.Normal()])
    (solid,) = ax.plot([0, 1], [0.5, 0.5], color="black", lw=5)
    text = ax.text(0.5, 0.5, "value", color="black", va="center")
    found = ep.testing.readability_faults(
        fig,
        kinds=("on-marks",),
        waive=[ep.testing.Waive("on-marks", artist=faint, reason="faint scenery")],
    )
    assert any(f.artist is text and f.other is solid and not f.waived for f in found)


def test_hidden_group_member_must_not_hide_visible_tear():
    fig = plt.figure(figsize=(4, 2), dpi=100)
    a = fig.text(0.2, 0.5, "a")
    b = fig.text(0.5, 0.5, "b")
    c = fig.text(0.8, 0.5, "c", alpha=0)

    def at(t):
        b.set_position((0.5 + 0.2 * t, 0.5))

    with pytest.raises(ValueError, match=r"tear.*representative pair"):
        ep.testing.Waive("tear", artist=c, reason="c intentionally absent")
    found = ep.testing.motion_faults(
        fig, at, [0, 0.5, 1], groups=[[a, b, c]], kinds=("tear",)
    )
    assert any(f.waived is None and {f.artist, f.other} == {a, b} for f in found)
    assert not found.waived
    assert any(f.artist is c and "no ink" in f.detail for f in found)


def test_check_clip_does_not_pass_a_visible_tear():
    fig = plt.figure(figsize=(4, 2), dpi=100)
    a = fig.text(0.1, 0.5, "A", fontsize=18)
    b = fig.text(0.5, 0.5, "B", fontsize=18)
    c = fig.text(0.9, 0.5, "C", fontsize=18, alpha=0)
    tl = ep.Timeline({"x": 0})
    tl.run("move", 1, to={"x": 1})
    plan = tl.at_fps(5)

    def draw(k):
        b.set_position((0.5 + 0.2 * plan.value(k, "x"), 0.5))

    report = ep.testing.check_clip(
        fig,
        draw,
        plan,
        moves=[{"artists": [b], "groups": [[a, b, c]]}],
    )
    assert not report.ok


def test_distant_normal_effect_is_not_a_collision():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    ax.plot([0.1, 0.9], [0.1, 0.1], color="red", path_effects=[patheffects.Normal()])
    ax.text(0.1, 0.8, "far away", color="black")
    assert not ep.testing.readability_faults(fig, kinds=("on-marks",))


def test_pixel_jump_artist_waiver_raises():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    patch = Rectangle((0.1, 0.3), 0.2, 0.3, color="black")
    ax.add_patch(patch)

    def at(t):
        patch.set_x(0.1 + 0.01 * t if t < 5 else 0.65)

    found = ep.testing.motion_faults(fig, at, list(range(6)), kinds=("jump",))
    assert found and all(f.artist is None and not f.artists for f in found)
    with pytest.raises(ValueError, match=r"jump.*no attributable artist"):
        waiver = ep.testing.Waive("jump", artist=patch, reason="intentional jump")
        ep.testing.motion_faults(
            fig, at, list(range(6)), kinds=("jump",), waive=[waiver]
        )


@pytest.mark.parametrize("kind", ["line", "patch", "arrow"])
def test_path_effect_stroke_and_offset_geometry(kind):
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    if kind == "line":
        (mark,) = ax.plot([0.2, 0.8], [0.3, 0.3], alpha=0)
    elif kind == "patch":
        mark = Rectangle((0.2, 0.3), 0.6, 0.02, alpha=0)
        ax.add_patch(mark)
    else:
        from matplotlib.patches import FancyArrowPatch

        mark = FancyArrowPatch((0.2, 0.3), (0.8, 0.3), alpha=0)
        ax.add_patch(mark)
    mark.set_path_effects(
        [patheffects.Stroke(linewidth=20, foreground="black", alpha=1, offset=(0, 20))]
    )
    label = ax.text(0.5, 0.3, "label", va="center", fontsize=10)
    fig.canvas.draw()
    x, y = ax.transData.transform((0.5, 0.3))
    label.set_transform(IdentityTransform())
    # The offset and half-width put ink here, beyond the original geometry.
    label.set_position((x, y + 38))
    assert any(
        f.other is mark for f in ep.testing.readability_faults(fig, kinds=("on-marks",))
    )
    label.set_position((x, y + 75))
    assert not ep.testing.readability_faults(fig, kinds=("on-marks",))


def test_dead_move_without_participants_rejects_artist_waiver():
    fig = plt.figure()
    text = fig.text(0.2, 0.5, "still")
    with pytest.raises(ValueError, match=r"dead.*no attributable artist"):
        ep.testing.motion_faults(
            fig,
            lambda t: None,
            [0, 0.5, 1],
            moves=[{"name": "whole frame"}],
            kinds=("dead",),
            waive=[ep.testing.Waive("dead", artist=text, reason="static design")],
        )


def test_normal_effect_does_not_create_zero_width_stroke():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.plot([0, 1], [0.5, 0.5], lw=0, path_effects=[patheffects.Normal()])
    ax.text(0.5, 0.5, "value", color="black", va="center")
    assert not ep.testing.readability_faults(fig, kinds=("on-marks",))


@pytest.mark.parametrize("pair", [True, False], ids=["pair", "artist"])
def test_pair_waiver_preserves_other_group_tears(pair):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    a = fig.text(0.1, 0.5, "A", fontsize=18)
    b = fig.text(0.4, 0.5, "B", fontsize=18)
    c = fig.text(0.8, 0.5, "C", fontsize=18)
    timeline = ep.Timeline({"u": 0})
    timeline.run("move", 1, to={"u": 1})
    plan = timeline.at_fps(5)

    def draw(k):
        t = plan.value(k, "u")
        b.set_position((0.4 + 0.1 * t, 0.5))
        c.set_position((0.8 - 0.2 * t, 0.5))

    report = ep.testing.check_clip(
        fig, draw, plan, moves=[{"artists": [b, c], "groups": [[a, b, c]]}]
    )
    assert not report.ok
    assert any(f.kind == "tear" for f in report.motion)
    # A tear reports only a representative pair. Neither waiver scope can
    # safely excuse it without also hiding the unwaived A-B or A-C tears.
    with pytest.raises(ValueError, match=r"tear.*representative pair"):
        ep.testing.Waive(
            "tear", artist=b, other=c if pair else None, reason="Only B-C may separate"
        )


def test_annotation_bounds_do_not_establish_identical_text():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    measure = ax.text(0.3, 0.5, "same", fontsize=24)
    fig.canvas.draw()
    width = measure.get_window_extent().width
    measure.remove()
    delta = (
        ax.transData.transform((0.35, 0.5))[0] - ax.transData.transform((0.3, 0.5))[0]
    )
    annotations = []
    for x, relative in ((0.3, -0.5), (0.35, -0.5 - delta / width)):
        annotations.append(
            ax.annotate(
                "same",
                xy=(0.95, 0.95),
                xytext=(x, 0.5),
                fontsize=24,
                alpha=0.5,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "none",
                    "relpos": (relative, -1),
                    "patchA": None,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
        )
    timeline = ep.Timeline()
    timeline.hold(1)
    report = ep.testing.check_clip(fig, lambda k: None, timeline.at_fps(3))
    assert not report.ok, "The words visibly overprint at positions 15.5 px apart"
    assert any(
        f.kind == "overlap" and {f.artist, f.other} == set(annotations)
        for f in report.readability
    )


def test_normal_effect_does_not_make_an_invisible_referent_visible():
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    (line,) = ax.plot(
        [0.1, 0.9],
        [0.2, 0.2],
        color="C2",
        alpha=0,
        path_effects=[patheffects.Normal()],
    )
    equation = ep.motion.equation(ax, r"${{sigma:\sigma_x}}$", fontsize=30)
    equation.bind("sigma", line)
    equation.update(1)
    fig.canvas.draw()
    shown = np.asarray(fig.canvas.buffer_rgba()).copy()
    line.set_visible(False)
    fig.canvas.draw()
    hidden = np.asarray(fig.canvas.buffer_rgba()).copy()
    line.set_visible(True)
    assert np.array_equal(shown, hidden), "The referent paints exactly zero pixels"
    assert any(f.kind == "no-referent" for f in ep.testing.grounding_faults(equation))


@pytest.mark.parametrize("hidden", [{"alpha": 0}, {"color": "none"}])
def test_normal_effect_does_not_make_an_invisible_group_member_present(hidden):
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    (line,) = ax.plot([0.1, 0.9], [0.2, 0.2], path_effects=[patheffects.Normal()])
    line.set(**hidden)
    label = ax.text(0.5, 0.5, "label")
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.5, 1], groups=[[line, label]], kinds=("tear",)
    )
    assert any(f.artist is line and "no ink" in f.detail for f in found)


@pytest.mark.parametrize("first_plain", [False, True])
def test_text_subclasses_never_get_the_crossfade_exemption(first_plain):
    class CustomText(Text):
        pass

    fig = plt.figure()
    for cls in (Text if first_plain else CustomText, CustomText):
        fig.add_artist(cls(0.3, 0.5, "same", alpha=0.5, transform=fig.transFigure))
    assert ep.testing.readability_faults(fig, kinds=("overlap",))


@pytest.mark.parametrize("kind", ["line", "patch", "text", "collection"])
@pytest.mark.parametrize(
    "base_alpha, effect_kw, painted_kinds",
    [
        (0, {"foreground": "black"}, ("line", "collection")),
        (0, {"foreground": "black", "alpha": 0}, ()),
        (1, {"foreground": "black", "alpha": 0}, ()),
        (None, {"foreground": (0, 0, 0, 0)}, ("patch", "text", "collection")),
        (
            0,
            {"foreground": "black", "alpha": 1},
            ("line", "patch", "text", "collection"),
        ),
        (
            None,
            {"foreground": (0, 0, 0, 0), "alpha": 1},
            ("line", "patch", "text", "collection"),
        ),
    ],
)
def test_path_effect_presence_uses_effect_alpha_and_color(
    kind, base_alpha, effect_kw, painted_kinds
):
    paints = kind in painted_kinds
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    if kind == "line":
        (mark,) = ax.plot([0.1, 0.9], [0.2, 0.2])
    elif kind == "patch":
        mark = ax.add_patch(Rectangle((0.1, 0.1), 0.8, 0.1))
    elif kind == "text":
        mark = ax.text(0.1, 0.2, "referent", fontsize=18)
    else:
        mark = ax.scatter([0.3, 0.6], [0.2, 0.2], s=50)
    mark.set_alpha(base_alpha)
    mark.set_path_effects([patheffects.Stroke(linewidth=4, **effect_kw)])
    label = ax.text(0.5, 0.8, "label")
    fig.canvas.draw()
    shown = np.asarray(fig.canvas.buffer_rgba()).copy()
    mark.set_visible(False)
    fig.canvas.draw()
    hidden = np.asarray(fig.canvas.buffer_rgba()).copy()
    mark.set_visible(True)
    assert bool(np.any(shown != hidden)) is paints
    found = ep.testing.motion_faults(
        fig, lambda t: None, [0, 0.5, 1], groups=[[mark, label]], kinds=("tear",)
    )
    assert any(f.artist is mark and "no ink" in f.detail for f in found) is not paints


@pytest.mark.parametrize("cached", [False, True])
def test_effect_collision_ink_does_not_establish_group_presence(cached):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    labels = [
        fig.text(0.2, 0.5, word, alpha=0, path_effects=[patheffects.Normal()])
        for word in ("one", "two")
    ]
    timeline = ep.Timeline({"u": 0})
    timeline.run("move", 1, to={"u": 1})
    plan = timeline.at_fps(5)

    def at(t):
        for label in labels:
            label.set_x(0.2 + 0.5 * t)

    move = {"groups": [labels]}
    if cached:
        report = ep.testing.check_clip(
            fig, lambda k: at(plan.value(k, "u")), plan, moves=[move]
        )
        found = report.motion
        assert not report.ok
    else:
        found = ep.testing.motion_faults(
            fig, at, [0, 0.5, 1], moves=[move], kinds=("tear", "overlap")
        )
    assert {f.artist for f in found if f.kind == "tear"} == set(labels)
    assert any(f.kind == "overlap" for f in found)


@pytest.mark.parametrize("effect_alpha", [None, 0, 1])
def test_path_effect_readout_requires_real_ink(effect_alpha):
    fig = plt.figure(figsize=(4, 2), dpi=100)
    effect = (
        patheffects.Normal()
        if effect_alpha is None
        else patheffects.Stroke(foreground="black", alpha=effect_alpha, linewidth=2)
    )
    readout = fig.text(0.5, 0.5, "8", alpha=0, path_effects=[effect])
    line = ep.narration.Line("number", "8")
    narration = ep.narration.Narration(
        ep.narration.Script((line,)),
        {"number": ep.narration.Spoken("number", 1, (ep.narration.Word("8", 0, 1),))},
    )
    found = ep.testing.timing_faults(
        narration, at=lambda t: None, readouts={("number", 0): readout}
    )
    assert any(f.kind == "number-late" for f in found) is (effect_alpha != 1)
