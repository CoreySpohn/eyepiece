"""A declared draw group: its cuts at the draws are excused, and nothing else's."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _run(n=6, glide=False):
    rng = np.random.default_rng(5)
    draws = rng.standard_normal((n, 25, 2)).cumsum(axis=1) * 0.1
    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_axes([0.05, 0.1, 0.4, 0.8])
    ax.set_axis_off()
    side = fig.add_axes([0.55, 0.1, 0.4, 0.8])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.set_axis_off()
    mark = ep.hop(draws, ax=ax, rate=4.0, ghosts=3)
    times = np.arange(round(mark.duration * 30) + 1) / 30.0
    cuts = [times[i] for i in mark.steps(times / mark.duration)]
    label = side.text(0.1, 0.5, "label", fontsize=12)
    marker = None
    if glide:
        (marker,) = side.plot([0.1], [0.15], "s", color="0.5", ms=5)
    return fig, mark, times, cuts, label, marker


def _poser(mark, label=None, jump_at=None, marker=None):
    def at(t):
        mark.update(t / mark.duration)
        if label is not None and jump_at is not None:
            label.set_x(0.6 if t >= jump_at - 1e-9 else 0.1)
        if marker is not None:
            marker.set_data([0.1 + 0.8 * t / mark.duration], [0.15])

    return at


def _moves(mark, *, with_draws=True):
    contract = mark.contract
    move = {"artists": contract.moves, "may_overlap": contract.may_overlap}
    if with_draws:
        move["draws"] = contract.draws
    return [move]


def test_hop_declares_its_draw_group():
    _, mark, *_ = _run()
    assert set(mark.contract.draws) == set(mark.contract.moves)
    assert mark.artists["line"] in mark.contract.draws


def test_draw_cuts_alone_are_clean():
    fig, mark, times, cuts, *_ = _run()
    faults = ept.motion_faults(fig, _poser(mark), times, moves=[mark], steps=cuts)
    assert faults == []


def test_another_artist_cutting_with_a_draw_is_still_reported():
    fig, mark, times, cuts, label, _ = _run()
    at = _poser(mark, label, jump_at=cuts[2])
    faults = ept.motion_faults(fig, at, times, moves=[mark], steps=cuts)
    jumps = [f for f in faults if f.kind == "jump"]
    assert len(jumps) == 1
    assert jumps[0].artist is label
    assert jumps[0].time == pytest.approx(cuts[2])
    # Without the group, a declared step excuses the whole picture: missed.
    blind = ept.motion_faults(
        fig, at, times, moves=_moves(mark, with_draws=False), steps=cuts
    )
    assert [f for f in blind if f.kind == "jump"] == []


def test_against_a_pixel_baseline_the_other_cut_is_reported():
    fig, mark, times, cuts, label, marker = _run(glide=True)
    glides = {"artists": [marker]}
    clean = ept.motion_faults(
        fig, _poser(mark, marker=marker), times, moves=[mark, glides], steps=cuts
    )
    assert clean == []
    at = _poser(mark, label, jump_at=cuts[3], marker=marker)
    faults = ept.motion_faults(fig, at, times, moves=[mark, glides], steps=cuts)
    jumps = [f for f in faults if f.kind == "jump"]
    assert len(jumps) == 1
    assert jumps[0].time == pytest.approx(cuts[3])
    x0 = jumps[0].box[0]
    assert x0 >= 0.5 * fig.bbox.width  # the label's half of the frame


def test_draws_argument_matches_the_contract():
    fig, mark, times, cuts, label, _ = _run()
    at = _poser(mark, label, jump_at=cuts[1])
    by_argument = ept.motion_faults(
        fig,
        at,
        times,
        moves=_moves(mark, with_draws=False),
        steps=cuts,
        draws=mark.contract.draws,
    )
    by_contract = ept.motion_faults(fig, at, times, moves=_moves(mark), steps=cuts)
    assert [(f.kind, f.time) for f in by_argument] == [
        (f.kind, f.time) for f in by_contract
    ]
    assert [f.kind for f in by_contract] == ["jump"]


def test_draw_group_may_overlap_itself():
    fig, mark, times, cuts, *_ = _run()
    move = {"artists": mark.contract.moves, "draws": mark.contract.draws}
    faults = ept.motion_faults(fig, _poser(mark), times, moves=[move], steps=cuts)
    assert [f for f in faults if f.kind == "overlap"] == []
    no_group = {"artists": mark.contract.moves}
    faults = ept.motion_faults(fig, _poser(mark), times, moves=[no_group], steps=cuts)
    assert any(f.kind == "overlap" for f in faults)


def test_a_step_where_the_group_holds_still_excuses_the_picture():
    fig, mark, times, cuts, label, _ = _run()
    # A scene cut between draws: the label jumps where no draw happens.
    between = times[int(np.searchsorted(times, cuts[1])) + 3]
    at = _poser(mark, label, jump_at=between)
    faults = ept.motion_faults(fig, at, times, moves=[mark], steps=[*cuts, between])
    assert faults == []


def test_check_clip_reads_the_group_from_the_mark():
    fig, mark, _, _, label, _ = _run()
    tl = ep.Timeline({"u": 0.0}, tempo=ep.Tempo(factor=1.0))
    tl.run("hop", mark.duration, to={"u": 1.0})
    tl.hold(0.5, name="rest")
    plan = tl.at_fps(30)
    knob = [plan.value(k, "u") for k in range(plan.n_frames)]
    steps = mark.steps(knob)

    def draw(k):
        mark.update(plan.value(k, "u"))
        label.set_x(0.6 if k >= steps[2] else 0.1)

    report = ept.check_clip(fig, draw, plan, moves=[mark], steps=steps)
    assert [f.kind for f in report.motion] == ["jump"]
    assert report.motion[0].artist is label


def test_a_cut_inside_the_group_boxes_is_still_reported():
    fig, mark, times, cuts, _, marker = _run(glide=True)
    # A label inside the sky axes, where the lead and ghost boxes mask the
    # pixel baseline at every draw.
    inside = mark.ax.text(0.0, 0.0, "inside", fontsize=10)
    x0, x1 = mark.ax.get_xlim()
    y0, y1 = mark.ax.get_ylim()

    def at(t):
        _poser(mark, marker=marker)(t)
        right = t >= cuts[3] - 1e-9
        inside.set_position((x0 + (0.7 if right else 0.1) * (x1 - x0), 0.5 * (y0 + y1)))

    glides = {"artists": [marker]}
    faults = ept.motion_faults(fig, at, times, moves=[mark, glides], steps=cuts)
    jumps = [f for f in faults if f.kind == "jump"]
    assert [f.artist for f in jumps] == [inside]
    assert jumps[0].time == pytest.approx(cuts[3])


def _recolored(where):
    """A clip whose square in `where` axes turns from dark to light at a cut."""
    fig, mark, times, cuts, _, marker = _run(glide=True)
    host = mark.ax if where == "group" else fig.axes[1]
    square = plt.Rectangle(
        (0.4, 0.4), 0.2, 0.2, transform=host.transAxes, color="0.1", zorder=0
    )
    host.add_patch(square)

    def at(t):
        _poser(mark, marker=marker)(t)
        square.set_color("0.9" if t >= cuts[3] - 1e-9 else "0.1")

    faults = ept.motion_faults(
        fig, at, times, moves=[mark, {"artists": [marker]}], steps=cuts
    )
    return [f for f in faults if f.kind == "jump"], cuts


def test_a_recolor_outside_the_group_boxes_is_reported():
    jumps, cuts = _recolored("side")
    assert [f.time for f in jumps] == [pytest.approx(cuts[3])]


@pytest.mark.xfail(
    strict=True,
    reason="known gap: inside the group's boxes the pixel baseline is masked at "
    "a cut, and an artist that changes color or alpha without moving its box "
    "is not followed on its own",
)
def test_a_recolor_inside_the_group_boxes_is_reported():
    jumps, cuts = _recolored("group")
    assert [f.time for f in jumps] == [pytest.approx(cuts[3])]


@pytest.mark.parametrize("cut_kind", ["draws", "hops"])
def test_a_cut_move_covers_its_own_caption_by_default(cut_kind):
    fig, mark, times, cuts, label, _ = _run()
    move = {"artists": [*mark.contract.moves, label]}
    if cut_kind == "draws":
        move["draws"] = mark.contract.draws
    else:
        move["hops"] = [{"artists": mark.contract.draws, "rate": 4}]
    at = _poser(mark, label, jump_at=cuts[2])
    faults = ept.motion_faults(fig, at, times, moves=[move], steps=cuts, kinds=["jump"])
    assert faults == []
