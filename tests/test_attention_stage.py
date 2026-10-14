"""Stage, axis builds, and collapses: what the viewer sees, not how.

A stage at rest is the figure as drawn, and a dimmed group keeps the
fraction of its contrast the stage names, on a light figure and a dark one;
an axis build ends on the finished axes and only ever adds ink; a collapse
draws nothing at rest and lands its bar where the caller's bar is drawn.
"""

from itertools import pairwise

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import eyepiece as ep

MODES = ["light", "dark"]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _scene():
    """One axes with a thick line, a labeled marker, and a colormapped mesh."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100, layout="constrained")
    x = np.linspace(0.0, 1.0, 50)
    (line,) = ax.plot(x, 0.5 + 0.0 * x, lw=12, color="tab:blue", solid_capstyle="butt")
    (dot,) = ax.plot([0.2], [0.8], "o", ms=10, color="tab:red")
    note = ax.text(0.6, 0.8, "the dot", bbox=ep.backing())
    mesh = ax.pcolormesh(
        np.linspace(0.0, 1.0, 6), [0.05, 0.25], np.arange(5.0)[None, :], cmap="viridis"
    )
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig, ax, {"line": [line], "marks": [dot, note], "mesh": [mesh]}


def _ink(image, background):
    """Pixels that differ from the background color."""
    bg = np.round(255 * np.asarray(to_rgba(background))[:3])
    return int(np.any(np.abs(image[..., :3].astype(float) - bg) > 2, axis=-1).sum())


# Callout pieces and rise


@pytest.mark.parametrize("mode", MODES)
def test_callout_pieces_draw_each_term_in_its_color_on_one_line(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    res = ep.callout(
        dot, ["add up the ", ("row", "tab:red"), " of cells"], side="right"
    )
    caption = res.artists["text"]
    fig.canvas.draw()
    assert caption.get_text() == "add up the row of cells"
    pieces = caption.get_children()
    assert [p.get_text() for p in pieces] == ["add up the ", "row", " of cells"]
    assert to_rgba(pieces[1].get_color()) == to_rgba("tab:red")
    assert to_rgba(pieces[0].get_color()) == to_rgba(caption.get_color())
    boxes = [p.get_window_extent() for p in pieces]
    for a, b in pairwise(boxes):
        assert abs(b.x0 - a.x1) <= 1.0
    whole = caption.get_window_extent()
    assert abs(boxes[0].x0 - whole.x0) <= 1.0
    assert abs(boxes[-1].x1 - whole.x1) <= 2.0
    with pytest.raises(ValueError, match="words"):
        ep.callout(dot, ["one two three ", ("four five six seven eight nine", "k")])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("side", ["right", "left"])
def test_a_heavier_key_term_grows_the_caption_away_from_its_subject(mode, side):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=100)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.5], [0.5], "o")
    text = "the posterior: planets the data allow"
    res = ep.callout(
        dot,
        [(text[:13], None, "bold"), text[13:]],
        side=side,
        leader=False,
        box={"boxstyle": "square,pad=0.2", "fc": "w", "ec": "k"},
    )
    caption = res.artists["text"]
    fig.canvas.draw()
    assert caption.get_text() == text
    key, rest = caption.get_children()
    assert key.get_fontweight() in ("bold", 700)
    assert rest.get_fontweight() == caption.get_fontweight()
    assert to_rgba(key.get_color()) == to_rgba(caption.get_color())
    plain = caption.get_window_extent()
    left, right = key.get_window_extent().x0, rest.get_window_extent().x1
    patch = caption.get_bbox_patch().get_window_extent()
    # The box holds the whole line.
    assert patch.x0 <= left and patch.x1 >= right
    # The extra width goes away from the subject; the near edge stays put.
    if side == "right":
        assert left == pytest.approx(plain.x0, abs=0.5) and right > plain.x1
    else:
        assert right == pytest.approx(plain.x1, abs=0.5) and left < plain.x0


def test_callout_rises_into_place_as_it_fades_in():
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=100)
    ax.set(xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    (dot,) = ax.plot([0.2], [0.5], "o")
    res = ep.callout(dot, "the dot", side="right", leader=False, rise_pt=6.0)
    caption = res.artists["text"]
    fig.canvas.draw()
    rest = caption.get_window_extent().y0
    res.update(0.3)
    fig.canvas.draw()
    assert caption.get_window_extent().y0 < rest - 1.0
    res.update(1.0)
    fig.canvas.draw()
    assert caption.get_window_extent().y0 == pytest.approx(rest)
