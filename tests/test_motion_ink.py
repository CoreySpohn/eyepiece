"""The motion-ink ratio: how much of what changes on screen is the declared knob's."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.lines import Line2D
from matplotlib.transforms import IdentityTransform

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _scene():
    fig = plt.figure(figsize=(4, 2), dpi=100)
    dot = Line2D(
        [40],
        [50],
        marker="o",
        markersize=14,
        linestyle="none",
        color="black",
        transform=IdentityTransform(),
    )
    fig.add_artist(dot)
    block = fig.text(0.5, 0.6, "furniture", fontsize=18)
    return fig, dot, block


def _poser(dot, block, *, slide=True, blink=False):
    def at(t):
        if slide:
            dot.set_xdata([40 + 200 * t])
        if blink:
            block.set_text("furniture" if round(t * 10) % 2 else "FURNITURE")

    return at


TIMES = [k / 10 for k in range(11)]


def test_only_the_driven_artist_moving_is_all_ink():
    fig, dot, block = _scene()
    ink = ept.motion_ink(fig, _poser(dot, block), TIMES, driven=[dot])
    assert ink == pytest.approx(1.0, abs=0.02)


def test_only_furniture_moving_is_no_ink():
    fig, dot, block = _scene()
    ink = ept.motion_ink(
        fig, _poser(dot, block, slide=False, blink=True), TIMES, driven=[dot]
    )
    assert ink == pytest.approx(0.0, abs=0.02)


def test_both_moving_gives_a_share_between_and_declaring_the_other_flips_it():
    fig, dot, block = _scene()
    at = _poser(dot, block, blink=True)
    mine = ept.motion_ink(fig, at, TIMES, driven=[dot])
    theirs = ept.motion_ink(fig, at, TIMES, driven=[block])
    assert 0.1 < mine < 0.9
    assert mine + theirs == pytest.approx(1.0, abs=0.05)


def test_a_clip_that_never_changes_has_no_ratio():
    fig, dot, block = _scene()
    at = _poser(dot, block, slide=False)
    assert ept.motion_ink(fig, at, TIMES, driven=[dot]) is None


def test_a_driven_artist_that_is_never_visible_is_refused():
    fig, dot, block = _scene()
    ghost = fig.text(0.5, 0.5, "x", alpha=0.0)
    with pytest.raises(ValueError, match="not visible"):
        ept.motion_ink(fig, _poser(dot, block), TIMES, driven=[ghost])


def _clip():
    fig, dot, block = _scene()
    tl = ep.Timeline({"x": 0.0}, tempo=ep.Tempo(factor=1.0))
    tl.hold(0.2)
    tl.run("slide", 0.8, to={"x": 8.0})
    tl.hold(0.2)
    plan = tl.at_fps(10)

    def draw(k):
        dot.set_xdata([40 + 20 * plan.value(k, "x")])
        block.set_text("furniture" if k % 2 else "FURNITURE")

    return fig, draw, plan, dot


def test_check_clip_reports_the_ratio_when_driven_is_given_and_it_is_not_a_fault():
    fig, draw, plan, dot = _clip()
    report = ept.check_clip(fig, draw, plan, driven=[dot])
    assert 0.0 <= report.motion_ink < 1.0
    assert report.ok
    assert "motion ink" in str(report)
    assert ept.check_clip(fig, draw, plan).motion_ink is None
