"""bracket: a square bracket over a span, labeled on its far side."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

import eyepiece as ep


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_above_ticks_point_down_and_label_sits_over_the_bar(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 10), ylim=(0, 5))
    res = ep.bracket(ax, 1.0, 6.0, 3.0, "upstream", depth=0.25)
    assert set(res.artists) == {"line", "text"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    line, text = res.artists["line"], res.artists["text"]
    assert isinstance(line, Line2D)
    np.testing.assert_allclose(
        line.get_xydata(), [(1.0, 2.75), (1.0, 3.0), (6.0, 3.0), (6.0, 2.75)]
    )
    assert to_rgb(line.get_color()) == pytest.approx(to_rgb(plt.rcParams["text.color"]))
    assert text.xy == pytest.approx((3.5, 3.0))
    assert (text.get_ha(), text.get_va()) == ("center", "bottom")
    assert text.get_bbox_patch() is None
    fig.canvas.draw()
    plt.close(fig)


def test_below_ticks_point_up_and_label_sits_under_the_bar():
    fig, ax = plt.subplots()
    res = ep.bracket(ax, 0.0, 2.0, 1.0, "frame", depth=0.5, side="below")
    np.testing.assert_allclose(res.artists["line"].get_ydata(), [1.5, 1.0, 1.0, 1.5])
    assert res.artists["text"].get_va() == "top"
    plt.close(fig)


def test_bracket_leaves_the_limits_alone_and_takes_overrides():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    res = ep.bracket(
        ax,
        -3.0,
        4.0,
        9.0,
        depth=1.0,
        color="tab:green",
        line_kw={"lw": 2.0, "gid": "span"},
    )
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == ((0, 1), (0, 1))
    assert "text" not in res.artists
    line = res.artists["line"]
    assert (line.get_linewidth(), line.get_gid()) == (2.0, "span")
    assert to_rgb(line.get_color()) == to_rgb("tab:green")
    assert res.update is None
    plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"side": "left"}, "side"), ({"depth": 0.0}, "positive")],
)
def test_bad_arguments_raise(kwargs, match):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match=match):
        ep.bracket(ax, 0, 1, 0, **{"depth": 0.2, **kwargs})
    plt.close(fig)
