"""Underline emphasis follows the text and leaves the still untouched."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("word", [False, True])
def test_underline_rest_peak_and_following_text(mode, word):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    ax.set_axis_off()
    if word:
        target = ep.words(ep.Anchor.from_data(ax, (0.2, 0.5)), "one word").artists[
            "text"
        ][1]
    else:
        target = ax.text(0.2, 0.5, "one word", va="baseline", fontsize=20)
    still = _pixels(fig)
    move = ep.underline(target)
    np.testing.assert_array_equal(_pixels(fig), still)
    for u in [0, 0.5, 1, 0.5, 0]:
        move.update(u)
        pixels = _pixels(fig)
        if u in (0, 1):
            np.testing.assert_array_equal(pixels, still)
        else:
            line = move.artists["line"]
            box = target.get_window_extent()
            np.testing.assert_allclose(line.get_xdata(), [box.x0, box.x1])
            assert line.get_ydata()[0] < box.y0
            assert to_rgba(line.get_color()) == to_rgba(target.get_color())
            move.update(u)
            np.testing.assert_array_equal(_pixels(fig), pixels)
    if not word:
        move.update(0.5)
        _pixels(fig)
        before = np.array(move.artists["line"].get_xydata())
        target.set_position((0.3, 0.6))
        _pixels(fig)
        delta = ax.transData.transform((0.3, 0.6)) - ax.transData.transform((0.2, 0.5))
        np.testing.assert_allclose(move.artists["line"].get_xydata(), before + delta)


def test_underline_scales_with_font_and_accepts_ease_and_color():
    fig, ax = plt.subplots()
    target = ax.text(0.3, 0.5, "baseline", va="baseline", fontsize=20)
    move = ep.underline(target, color="C1", ease="linear")
    move.update(0.2)
    fig.canvas.draw()
    line = move.artists["line"]
    assert np.diff(line.get_xdata())[0] == pytest.approx(
        target.get_window_extent().width / 2
    )
    width = line.get_linewidth()
    gap = ax.transData.transform(target.get_position())[1] - line.get_ydata()[0]
    target.set_fontsize(40)
    fig.canvas.draw()
    assert line.get_linewidth() == 2 * width
    assert ax.transData.transform(target.get_position())[1] - line.get_ydata()[
        0
    ] == pytest.approx(2 * gap)
    assert to_rgba(line.get_color()) == to_rgba("C1")


def test_underline_motion_contract_and_timeline():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=80)
    ax.set_axis_off()
    target = ax.text(0.3, 0.5, "baseline", fontsize=20)
    move = ep.underline(target)
    assert isinstance(move.contract, ep.MotionContract)
    assert (
        ept.motion_faults(
            fig, lambda t: move.update(t / 2), ept.Window(0, 2), moves=[move]
        )
        == []
    )
    tl = ep.Timeline()
    span = tl.underline("word", at=0.2)
    tl.hold(2)
    plan = tl.at_fps(20)
    assert span.kind == "underline"
    assert plan.value(0, "underline.word") == 0
    assert plan.value(39, "underline.word") == 1


def test_default_underline_color_follows_text_between_draws():
    fig, ax = plt.subplots()
    target = ax.text(0.2, 0.5, "color", color="red")
    mark = ep.underline(target)
    mark.update(0.5)
    fig.canvas.draw()
    target.set_color("blue")
    fig.canvas.draw()
    assert to_rgba(mark.artists["line"].get_color()) == to_rgba("blue")


def test_skipped_underline_knob_is_optional_like_other_attention():
    tl = ep.Timeline()
    tl.underline("word")
    tl.hold(2)
    assert tl.at_fps(20).unused_knobs() == ((), ())


def test_custom_duration_underline_span_has_attention_defaults():
    tl = ep.Timeline(tempo=ep.Tempo(factor=2))
    span = tl.span("word", 1.3, kind="underline", at=0)
    tl.hold(2)
    assert span.end_s == pytest.approx(1.3)
    assert span.ease == "linear" and not span.extends
    assert tl.at_fps(20).faults == ()


def test_underline_tracks_subpixel_vertical_motion_without_pixel_jumps():
    from matplotlib.transforms import IdentityTransform

    fig, ax = plt.subplots(figsize=(5, 3), dpi=80, facecolor="white")
    ax.set_axis_off()
    target = ax.text(
        100,
        150.2,
        "ABC",
        va="baseline",
        fontsize=20,
        color="black",
        transform=IdentityTransform(),
    )
    move = ep.underline(target)
    move.update(0.5)

    def center():
        pixels = _pixels(fig)[::-1]
        y = move.artists["line"].get_ydata()[0]
        lo, hi = int(y) - 2, int(y) + 3
        ink = (255 - pixels[lo:hi, 110:120, 0]).astype(float).sum(axis=1)
        return np.dot(np.arange(lo, hi), ink) / ink.sum()

    before = center()
    target.set_y(150.35)
    # The antialiased stroke moves by less than half a pixel instead of snapping.
    assert 0 < center() - before < 0.5
