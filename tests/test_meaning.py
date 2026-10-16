"""Meaning checks: one positive and one negative control per kind of fault."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, Rectangle

import eyepiece as ep
import eyepiece.testing as ept

CYAN = "#08F7FE"
PINK = "#FE53BB"
YELLOW = "#F5D300"
GREEN = "#00FF41"
BLUE = "#08F7F0"  # a hair from cyan: reads as the same hue
NEAR = "#20E0F0"  # 6.4 dE from cyan: close, but past the hard floor
ORANGE = "#FFA040"  # apart from PINKISH under normal vision, 6.4 dE under tritanomaly
PINKISH = "#FF80C0"


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _axes():
    fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
    fig.set_facecolor("#101010")
    ax.set_facecolor("#101010")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return fig, ax


def _line(ax, y, color, gid, **kw):
    (art,) = ax.plot([1, 9], [y, y], color=color, lw=3, gid=gid, **kw)
    return art


def _fault_kinds(faults, severity=None):
    return [f.kind for f in faults if severity is None or f.severity == severity]


def test_same_frame_flags_two_keys_in_one_color():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, BLUE, "b")
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "library"})
    same = [f for f in found if f.kind == "same-frame"]
    assert len(same) == 1
    assert same[0].severity == "fault"
    assert set(same[0].keys) == {"planet", "library"}
    assert same[0].de < 10.0
    with pytest.raises(ept.MeaningError, match="same-frame"):
        ept.assert_meaning(fig, keys={"a": "planet", "b": "library"})


def test_same_frame_passes_distinct_hues():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, YELLOW, "b")
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "star"})
    assert "same-frame" not in _fault_kinds(found)


def test_same_frame_warns_on_the_cyan_pink_pair_under_deuteranopia():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, PINK, "b")
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "model"})
    (same,) = found
    assert same.severity == "warning"
    assert 15.0 < same.de < 20.0


def test_same_frame_downgrades_a_second_channel():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, NEAR, "b", linestyle="--")
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "library"})
    same = [f for f in found if f.kind == "same-frame"]
    assert [f.severity for f in same] == ["warning"]
    assert "line style" in same[0].detail


def test_a_second_channel_does_not_save_a_pair_under_the_hard_floor():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, BLUE, "b", linestyle="--")  # under 5 dE: a dash cannot help
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "library"})
    assert [f.severity for f in found if f.kind == "same-frame"] == ["fault"]


def test_same_frame_exempts_a_sharing_set():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, BLUE, "b")
    found = ept.meaning_faults(
        fig, keys={"a": "disk", "b": "prior"}, sharing=[("disk", "prior")]
    )
    assert "same-frame" not in _fault_kinds(found)


def test_a_hidden_artist_is_not_drawn():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, BLUE, "b", alpha=0.02)
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "library"})
    assert found == []


def _two_clip_figure():
    """Planet line on for t < 2, a library outline in the same hue for t >= 3."""
    fig, ax = _axes()
    planet = _line(ax, 3, CYAN, "planet")
    badge = _line(ax, 6, BLUE, "badge")

    def at(t):
        planet.set_alpha(1.0 if t < 2 else 0.0)
        badge.set_alpha(1.0 if t >= 3 else 0.0)

    return fig, at


KEYS = {"planet": "planet", "badge": "library"}


def test_rebound_reports_the_gap_when_the_hue_changes_meaning():
    fig, at = _two_clip_figure()
    found = ept.meaning_faults(fig, at, [0.0, 1.0, 2.0, 3.0, 4.0], keys=KEYS)
    assert "same-frame" not in _fault_kinds(found)
    (rebound,) = [f for f in found if f.kind == "rebound"]
    assert rebound.severity == "warning"
    assert "gap 2" in rebound.detail
    strict = ept.meaning_faults(
        fig, at, [0.0, 1.0, 2.0, 3.0, 4.0], keys=KEYS, strict=True
    )
    assert [f.severity for f in strict if f.kind == "rebound"] == ["fault"]


def test_rebound_is_silent_for_one_key():
    fig, ax = _axes()
    line = _line(ax, 3, CYAN, "planet")

    def at(t):
        line.set_alpha(1.0 if t < 2 else 0.0)

    found = ept.meaning_faults(fig, at, [0.0, 1.0, 3.0, 4.0], keys={"planet": "p"})
    assert found == []


def test_a_declared_rebind_releases_the_hue():
    fig, at = _two_clip_figure()
    times = [0.0, 1.0, 2.0, 3.0, 4.0]
    found = ept.meaning_faults(fig, at, times, keys=KEYS, rebind=[2.5])
    assert "rebound" not in _fault_kinds(found)


def test_sharing_pair_in_one_clip_is_still_reported():
    fig, at = _two_clip_figure()
    found = ept.meaning_faults(
        fig,
        at,
        [0.0, 1.0, 3.0, 4.0],
        keys={"planet": "disk", "badge": "prior"},
        sharing=[("disk", "prior")],
    )
    (rebound,) = [f for f in found if f.kind == "rebound"]
    assert "never to share a clip" in rebound.detail
    assert rebound.severity == "fault"  # a declared pair is a fault, even unstrict


def test_an_undeclared_rebound_is_a_warning_until_strict():
    fig, at = _two_clip_figure()
    times = [0.0, 1.0, 3.0, 4.0]
    (loose,) = ept.meaning_faults(fig, at, times, keys=KEYS)
    (tight,) = ept.meaning_faults(fig, at, times, keys=KEYS, strict=True)
    assert (loose.severity, tight.severity) == ("warning", "fault")


def test_chain_finds_the_hue_across_clips():
    fig1, ax1 = _axes()
    _line(ax1, 3, CYAN, "planet")
    fig2, ax2 = _axes()
    _line(ax2, 3, BLUE, "badge")
    clips = [
        (fig1, lambda t: None, [0.0, 4.0], {"planet": "planet"}),
        (fig2, lambda t: None, [0.0, 4.0], {"badge": "library"}),
    ]
    found = ept.meaning_faults_chain(clips)
    assert [f.kind for f in found] == ["rebound"]
    assert found[0].times == (0.0, 4.0, 8.0)
    assert ept.meaning_faults_chain(clips, rebind=[4.0]) == []
    assert _fault_kinds(ept.meaning_faults_chain(clips, rebind=[6.0])) == ["rebound"]


def test_cue_as_a_line_is_flagged():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    _line(ax, 6, "#f0f0f0", "cue")
    found = ept.meaning_faults(
        fig, keys={"planet": "planet", "cue": "cue"}, cue_keys=["cue"]
    )
    cue = [f for f in found if f.kind == "cue-form"]
    assert len(cue) == 1
    assert "line" in cue[0].detail


def test_cue_as_a_translucent_band_in_the_text_color_passes():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    ax.add_patch(
        Rectangle((0.5, 2), 9, 2, color="#f0f0f0", alpha=0.25, gid="cue", zorder=0.5)
    )
    found = ept.meaning_faults(
        fig, keys={"planet": "planet", "cue": "cue"}, cue_keys=["cue"]
    )
    assert "cue-form" not in _fault_kinds(found)


def test_cue_ring_in_a_hue_another_key_uses_is_flagged():
    fig, ax = _axes()
    _line(ax, 3, PINK, "model")
    ax.add_patch(Circle((5, 5), 2, fill=False, ec=PINK, lw=3, gid="cue"))
    found = ept.meaning_faults(
        fig, keys={"model": "model", "cue": "cue"}, cue_keys=["cue"]
    )
    cue = [f for f in found if f.kind == "cue-form"]
    assert len(cue) == 1
    assert cue[0].keys == ("cue",)
    assert "hue" in cue[0].detail


def test_a_cue_in_a_hue_no_key_uses_is_still_flagged():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    ax.add_patch(Circle((5, 5), 2, fill=False, ec=PINK, lw=3, gid="cue"))
    found = ept.meaning_faults(
        fig, keys={"planet": "planet", "cue": "cue"}, cue_keys=["cue"]
    )
    assert [f.keys for f in found if f.kind == "cue-form"] == [("cue",)]


def test_a_cue_ring_in_the_text_color_passes():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    ax.add_patch(Circle((5, 5), 2, fill=False, ec="#f0f0f0", lw=3, gid="cue"))
    found = ept.meaning_faults(
        fig, keys={"planet": "planet", "cue": "cue"}, cue_keys=["cue"]
    )
    assert "cue-form" not in _fault_kinds(found)


def test_opaque_filled_cue_above_the_subject_is_flagged():
    fig, ax = _axes()
    _line(ax, 5, CYAN, "planet", zorder=1)
    ax.add_patch(Rectangle((0.5, 4), 9, 2, color="#f0f0f0", gid="cue", zorder=3))
    found = ept.meaning_faults(
        fig, keys={"planet": "planet", "cue": "cue"}, cue_keys=["cue"]
    )
    assert (
        "opaque filled patch" in next(f for f in found if f.kind == "cue-form").detail
    )


def test_unkeyed_is_advisory_then_strict():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    _line(ax, 6, GREEN, "stray")
    found = ept.meaning_faults(fig, keys={"planet": "planet"})
    (unkeyed,) = [f for f in found if f.kind == "unkeyed"]
    assert unkeyed.severity == "advisory"
    assert unkeyed.keys == ("stray",)
    ept.assert_meaning(fig, keys={"planet": "planet"}, strict=False)
    with pytest.raises(ept.MeaningError, match="unkeyed"):
        ept.assert_meaning(fig, keys={"planet": "planet"})


def test_unkeyed_skips_neutral_gray_and_keyed_artists():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    _line(ax, 6, "#9a9a9a", "grid")
    found = ept.meaning_faults(fig, keys={"planet": "planet"})
    assert found == []


def test_a_waiver_keeps_the_finding_in_the_report():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, BLUE, "b")
    keys = {"a": "planet", "b": "library"}
    waive = {"same-frame:library+planet": "badge names itself, Corey 2026-10-15"}
    found = ept.meaning_faults(fig, keys=keys, waive=waive)
    assert [f.waived for f in found] == ["badge names itself, Corey 2026-10-15"]
    report = ept.meaning_report(found)
    assert "waived (badge names itself" in report
    assert "no unwaived tier 1 or 2 findings" in report
    ept.assert_meaning(fig, keys=keys, waive=waive)


def test_report_never_says_clean():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    text = ept.meaning_report(ept.meaning_faults(fig, keys={"a": "planet"}))
    assert "no tier 1 or 2 findings" in text
    assert "clean" not in text


def test_at_and_times_come_together():
    fig, _ = _axes()
    with pytest.raises(ValueError, match="together"):
        ept.meaning_faults(fig, lambda t: None, None, keys={})


def test_stage_dim_enters_the_measured_color():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a", alpha=0.5)
    _line(ax, 6, YELLOW, "b", alpha=0.5)
    found = ept.meaning_faults(fig, keys={"a": "planet", "b": "star"})
    assert found == []


def test_a_colored_outline_around_a_name_is_read():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "planet")
    badge = fig.text(
        0.05,
        0.05,
        "orbix",
        color="white",
        bbox={"facecolor": "none", "edgecolor": BLUE, "linewidth": 2},
    )
    badge.set_gid("badge")
    found = ept.meaning_faults(fig, keys={"planet": "planet", "badge": "library"})
    (same,) = [f for f in found if f.kind == "same-frame"]
    assert set(same.keys) == {"planet", "library"}


def _pose_pair(a, b):
    def pose(t):
        a.set_visible(t < 0.5)
        b.set_visible(t >= 0.5)

    return pose


def test_dimmed_pair_in_two_clips_is_advisory_not_a_rebound():
    fig, ax = _axes()
    a = _line(ax, 3, CYAN, "a", alpha=0.06)
    b = _line(ax, 6, YELLOW, "b", alpha=0.06)
    found = ept.meaning_faults(
        fig,
        _pose_pair(a, b),
        [0.0, 1.0],
        keys={"a": "planet", "b": "star"},
        fault_de=25.0,
    )
    assert _fault_kinds(found) == ["dimmed"]
    assert found[0].severity == "advisory"
    assert ept.assert_meaning(
        fig,
        _pose_pair(a, b),
        [0.0, 1.0],
        keys={"a": "planet", "b": "star"},
        fault_de=25.0,
    )


def test_full_strength_pair_in_two_clips_has_no_dimmed_finding():
    fig, ax = _axes()
    a = _line(ax, 3, CYAN, "a")
    b = _line(ax, 6, YELLOW, "b")
    found = ept.meaning_faults(
        fig, _pose_pair(a, b), [0.0, 1.0], keys={"a": "planet", "b": "star"}
    )
    assert found == []


def test_a_faded_hue_still_clusters_on_its_own_color():
    fig, ax = _axes()
    a = _line(ax, 3, CYAN, "a", alpha=0.3)
    b = _line(ax, 6, BLUE, "b", alpha=1.0)
    found = ept.meaning_faults(
        fig, _pose_pair(a, b), [0.0, 1.0], keys={"a": "planet", "b": "library"}
    )
    assert _fault_kinds(found) == ["rebound"]


def _stage_figure(dim):
    fig, ax = _axes()
    a = _line(ax, 3, CYAN, "a")
    b = _line(ax, 6, YELLOW, "b")
    stage = ep.Stage({"a": a, "b": b}, dim=dim)
    return fig, stage


def test_a_stage_pair_fine_at_full_color_fails_when_dimmed():
    keys = {"a": "planet", "b": "star"}
    fig, stage = _stage_figure(0.05)
    stage.set(shown=["a", "b"])
    assert ept.meaning_faults(fig, keys=keys) == []
    stage.set(dimmed=["a", "b"])
    found = ept.meaning_faults(fig, keys=keys)
    assert [(f.kind, f.severity) for f in found] == [("same-frame", "fault")]


def test_a_pair_apart_only_under_normal_vision_is_flagged_for_tritanomaly():
    fig, ax = _axes()
    _line(ax, 3, ORANGE, "a")
    _line(ax, 6, PINKISH, "b")
    (found,) = ept.meaning_faults(fig, keys={"a": "planet", "b": "star"})
    assert (found.kind, found.severity) == ("same-frame", "fault")
    assert 5.0 < found.de < 10.0  # the tritanomaly distance; the others are over 30


def test_a_pair_far_under_every_vision_type_is_not_flagged():
    fig, ax = _axes()
    _line(ax, 3, ORANGE, "a")
    _line(ax, 6, "#2060FF", "b")
    assert ept.meaning_faults(fig, keys={"a": "planet", "b": "star"}) == []


def test_opacity_under_the_shown_floor_is_not_drawn():
    keys = {"a": "planet", "b": "star"}
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a", alpha=0.04)
    _line(ax, 6, BLUE, "b", alpha=0.04)
    assert ept.meaning_faults(fig, keys=keys) == []
    fig2, ax2 = _axes()
    _line(ax2, 3, CYAN, "a", alpha=0.06)
    _line(ax2, 6, BLUE, "b", alpha=0.06)
    assert _fault_kinds(ept.meaning_faults(fig2, keys=keys)) == ["same-frame"]


def test_a_sub_pixel_artist_is_not_drawn():
    keys = {"a": "planet", "b": "star"}
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    ax.add_patch(Rectangle((5, 6), 0.001, 0.001, color=BLUE, lw=0, gid="b"))
    assert ept.meaning_faults(fig, keys=keys) == []
    fig2, ax2 = _axes()
    _line(ax2, 3, CYAN, "a")
    ax2.add_patch(Rectangle((5, 6), 1.0, 1.0, color=BLUE, lw=0, gid="b"))
    assert _fault_kinds(ept.meaning_faults(fig2, keys=keys)) == ["same-frame"]


def test_the_axes_face_is_the_background_a_color_sits_on():
    keys = {"a": "planet", "b": "star"}
    fig, ax = _axes()
    _line(ax, 3, GREEN, "a")
    _line(ax, 6, "#00FF40", "b")
    assert _fault_kinds(ept.meaning_faults(fig, keys=keys)) == ["same-frame"]
    ax.set_facecolor(GREEN)  # both lines now sit on their own color: no ink
    assert ept.meaning_faults(fig, keys=keys) == []


def test_a_marker_less_line_is_not_padded_by_marker_size():
    keys = {"a": "planet", "b": "star"}
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    (b,) = ax.plot([5, 5.002], [6, 6], color=BLUE, lw=0.2, markersize=40, gid="b")
    assert ept.meaning_faults(fig, keys=keys) == []  # a tiny thin line has no ink
    b.set_marker("o")
    assert _fault_kinds(ept.meaning_faults(fig, keys=keys)) == ["same-frame"]


def test_a_line_alpha_replaces_its_color_alpha():
    keys = {"a": "planet", "b": "star"}
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, (*to_rgb(BLUE), 0.5), "b", alpha=1.0)  # alpha 1 wins over 0.5
    (found,) = ept.meaning_faults(fig, keys=keys)
    assert found.de < 5.0  # drawn at full strength, not blended to half


def test_a_report_with_every_finding_waived_says_so():
    fig, ax = _axes()
    _line(ax, 3, CYAN, "a")
    _line(ax, 6, BLUE, "b")
    found = ept.meaning_faults(
        fig,
        keys={"a": "planet", "b": "library"},
        waive={"same-frame:library+planet": "reason"},
    )
    assert "no unwaived tier 1 or 2 findings; 1 waived" in ept.meaning_report(found)
    open_ = ept.meaning_faults(fig, keys={"a": "planet", "b": "library"})
    assert "waived" not in ept.meaning_report(open_).splitlines()[0]
