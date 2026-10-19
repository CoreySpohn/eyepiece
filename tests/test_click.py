"""The click: a composite lands in one 0.6 s beat, then the picture holds still."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep
import eyepiece.testing as ept

LW = 2.0


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _scene():
    fig = plt.figure(figsize=(6.4, 3.2), dpi=100, facecolor="white")
    ax = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal")
    ax.set_axis_off()
    th = np.linspace(0.0, 2.0 * np.pi, 241)
    (ring,) = ax.plot(np.cos(th), np.sin(th), color="black", lw=LW)
    count = ep.readout(
        ep.Anchor.from_data(ax, (2.0, 1.0)), 123456, fmt=",d", digits=7, fontsize=14
    )
    rule = ep.ruler(ax, (0.0, 0.0), (-1.0, 0.0), "r = 1.00", head_scale=0)
    end = ep.Anchor.from_data(ax, (1.0, 0.0))
    fig.canvas.draw()
    return fig, ring, count, rule, end


def _land(ring, count, rule, end):
    return ep.click(ring, dot=end, readout=count, ruler=rule)


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def test_the_beat_is_0_6_s_and_a_1_5_s_stillness_follows_at_any_tempo():
    for factor in (0.5, 1.0, 2.0):
        tl = ep.Timeline({"a": 0.0}, tempo=ep.Tempo(factor=factor))
        tl.run("a", 1.0, to={"a": 1.0})
        beat = tl.click("ring")
        after = tl.run("b", 0.5, to={"a": 0.0})
        assert float(beat.end) - float(beat.start) == pytest.approx(0.6)
        assert float(after.start) - float(beat.end) >= 1.5 - 1e-12
        plan = tl.at_fps(60)
        moving = [
            k for k in range(plan.n_frames) if 0.0 < plan.value(k, "click.ring") < 1.0
        ]
        assert len(moving) == 35  # 36 frames from 0 to 1 at 60 fps


def test_start_is_the_figure_before_the_beat():
    fig, ring, count, rule, end = _scene()
    before = _pixels(fig)
    landed = _land(ring, count, rule, end)
    landed.update(0.0)
    np.testing.assert_array_equal(_pixels(fig), before)


def test_rest_is_the_landed_still_whatever_the_path():
    fig, *parts = _scene()
    landed = _land(*parts)
    for u in np.linspace(0.0, 1.0, 37):
        landed.update(u)
        _pixels(fig)
    traversed = _pixels(fig)
    fig2, *parts2 = _scene()
    _land(*parts2).update(1.0)
    np.testing.assert_array_equal(_pixels(fig2), traversed)


def test_landed_state_dot_set_number_bold_ruler_gone_width_back():
    _, ring, count, rule, end = _scene()
    landed = _land(ring, count, rule, end)
    landed.update(1.0)
    dot = landed.artists["dot"]
    assert dot.get_visible() and dot.get_markersize() == pytest.approx(3.2 * LW)
    assert all(t.get_fontweight() == "bold" for t in count.artists["text"])
    assert not rule.artists["arrow"].get_visible()
    assert not rule.artists["text"].get_visible()
    assert ring.get_linewidth() == pytest.approx(LW)


def test_swell_holds_its_plateau_and_returns():
    _, ring, count, rule, end = _scene()
    landed = _land(ring, count, rule, end)
    us = np.linspace(0.0, 1.0, 121)
    widths = []
    for u in us:
        landed.update(u)
        widths.append(ring.get_linewidth())
    widths = np.array(widths)
    held = (us >= 0.35) & (us <= 0.65)
    np.testing.assert_allclose(widths[held], 1.55 * LW)
    assert np.all(np.diff(widths[us <= 0.35]) >= 0.0)
    assert np.all(np.diff(widths[us >= 0.65]) <= 0.0)
    assert widths[0] == LW and widths[-1] == LW


def test_nothing_moves_position_and_the_number_sets_once():
    fig, ring, count, rule, end = _scene()
    xy = ring.get_xydata().copy()
    slots = [t.get_position() for t in count.artists["text"]]
    landed = _land(ring, count, rule, end)
    dots, bold = set(), []
    for u in np.linspace(0.0, 1.0, 61):
        landed.update(u)
        fig.canvas.draw()
        if landed.artists["dot"].get_visible():
            dots.add(tuple(np.round(landed.artists["dot"].get_xydata()[0], 9)))
        bold.append(count.artists["text"][0].get_fontweight() == "bold")
        assert [t.get_position() for t in count.artists["text"]] == slots
    np.testing.assert_array_equal(ring.get_xydata(), xy)
    assert len(dots) == 1
    assert np.count_nonzero(np.diff(np.array(bold, dtype=int))) == 1
    assert bold.index(True) / 60 == pytest.approx(0.55, abs=1 / 60)


def test_ruler_recedes_after_the_number_sets():
    _, ring, count, rule, end = _scene()
    landed = _land(ring, count, rule, end)
    landed.update(0.3)
    assert rule.artists["arrow"].get_alpha() in (None, 1.0)
    alphas = []
    for u in np.linspace(0.36, 0.99, 30):
        landed.update(u)
        alphas.append(rule.artists["arrow"].get_alpha())
    assert np.all(np.diff(alphas) <= 0.0) and alphas[-1] < 0.05


def test_click_clip_has_no_motion_fault():
    fig, *parts = _scene()
    landed = _land(*parts)
    frames = np.linspace(0.0, 1.0, 37)
    assert ept.motion_faults(fig, landed.update, frames, moves=[landed]) == []


def test_click_validation():
    _, ring, *_ = _scene()
    with pytest.raises(ValueError, match="geometry"):
        ep.click([])
    with pytest.raises(TypeError, match="line or a patch"):
        ep.click(ring.axes.text(0, 0, "x"))
    with pytest.raises(ValueError, match="swell"):
        ep.click(ring, swell=0.9)
