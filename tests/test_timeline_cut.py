"""An explicit zero-duration cut works on the strict clip clock."""

import matplotlib.pyplot as plt
import pytest

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import narration as nr


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_cut_builds_strictly_changes_on_its_frame_and_checks_cleanly():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    (dot,) = ax.plot([0.2], [0.5], "o", color="black", ms=10)
    tl = ep.Timeline({"x": 0.2})
    tl.hold(0.5)
    cut = tl.cut("switch", to={"x": 0.8})
    assert cut.start_s == cut.end_s == 0.5
    assert tl.now == 0.5
    tl.hold(0.5)
    plan = tl.at_fps(10)
    resolved = plan.spans[cut.name]
    assert resolved.kind == "cut" and resolved.n == 0 and not resolved.collapsed
    assert plan.value(4, "x") == 0.2 and plan.value(5, "x") == 0.8
    assert plan.progress(4, cut.name) == 0 and plan.progress(5, cut.name) == 1
    assert plan.step_frames() == (5,)
    assert {plan.frame_of(cut.name, where) for where in ["start", "mid", "last"]} == {5}
    assert {e.frame for e in plan.events(kind="cut")} == {5}
    assert {4, 5} <= set(plan.review_frames())

    def draw(k):
        dot.set_xdata([plan.value(k, "x")])

    assert ept.check_clip(fig, draw, plan).ok


def test_cut_at_explicit_time_with_numeric_target():
    tl = ep.Timeline({"switch": 0})
    tl.cut("switch", to=1, at=0.5)
    tl.hold(1)
    assert tl.at_fps(10).value(5, "switch") == 1


def test_cut_can_declare_an_external_state_change_without_a_knob():
    tl = ep.Timeline()
    tl.cut("swap", at=0.5)
    tl.hold(1)
    plan = tl.at_fps(10)
    assert plan.step_frames() == (5,)
    assert plan.progress(5, "swap") == 1


def test_cut_on_a_narration_cue_does_not_take_time():
    script = nr.Script((nr.Line("line", "The {move:state}state changes."),))
    narration = nr.Narration.estimated(script)
    tl = ep.Timeline({"state": 0}, narration=narration)
    tl.line("line")
    cut = tl.cut("change", to={"state": 1}, cue="move:state")
    assert cut.start_s == cut.end_s == narration.cue("line", "move", "state")
    plan = tl.at_fps(20)
    assert plan.faults == ()
    assert plan.step_frames() == (plan.spans[cut.name].k0,)
