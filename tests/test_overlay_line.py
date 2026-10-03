"""overlay_line: a light dash over a dark underlay along any path."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

import eyepiece as ep


def _lum(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_light_dash_over_a_solid_dark_underlay_in_either_mode(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    ax.imshow(np.random.default_rng(0).random((8, 8)), extent=(-4, 4, -4, 4))
    res = ep.overlay_line(ax, [-3.0, 3.0], [-1.0, 1.0], label="cut")
    assert set(res.artists) == {"lines", "text"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    under, dash = res.artists["lines"]
    assert isinstance(dash, Line2D) and isinstance(under, Line2D)
    np.testing.assert_allclose(dash.get_xydata(), [(-3, -1), (3, 1)])
    np.testing.assert_allclose(under.get_xydata(), dash.get_xydata())
    assert dash.get_linestyle() not in ("-", "solid")
    assert under.get_linestyle() == "-"
    assert _lum(dash.get_color()) > _lum(under.get_color())
    assert under.get_linewidth() == pytest.approx(dash.get_linewidth() + 1.0)
    order = ax.get_children()
    assert order.index(dash) > order.index(under)
    text = res.artists["text"]
    assert text.get_position() == pytest.approx((0.0, 0.0))
    assert text.get_bbox_patch() is not None
    assert text.get_zorder() > dash.get_zorder()
    fig.canvas.draw()
    plt.close(fig)


def test_a_path_beyond_the_view_leaves_the_limits_alone():
    fig, ax = plt.subplots()
    ax.imshow(np.ones((4, 4)), extent=(-1, 1, -1, 1))
    limits = ax.get_xlim(), ax.get_ylim()
    ep.overlay_line(ax, [-5.0, 5.0], [0.0, 3.0])
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    plt.close(fig)


def test_label_sits_at_a_fraction_of_the_path_length():
    fig, ax = plt.subplots()
    # An L: 3 units right, then 1 up, so three quarters of the way is the corner.
    res = ep.overlay_line(
        ax, [0.0, 3.0, 3.0], [0.0, 0.0, 1.0], label="L", label_at=0.75
    )
    assert res.artists["text"].get_position() == pytest.approx((3.0, 0.0))
    res_end = ep.overlay_line(ax, [0.0, 2.0], [0.0, 2.0], label="end", label_at=1.0)
    assert res_end.artists["text"].get_position() == pytest.approx((2.0, 2.0))
    plt.close(fig)


def test_underlay_off_or_colored_and_overrides_reach_each_line():
    fig, ax = plt.subplots()
    res = ep.overlay_line(ax, [0, 1], [0, 1], underlay=False, ls=":")
    (dash,) = res.artists["lines"]
    assert dash.get_linestyle() == ":"
    assert "text" not in res.artists
    res = ep.overlay_line(
        ax,
        [0, 1],
        [0, 1],
        underlay="navy",
        color="gold",
        line_kw={"lw": 0.9, "zorder": 7, "gid": "cut"},
        underlay_kw={"lw": 2.2, "zorder": 6},
    )
    under, dash = res.artists["lines"]
    assert to_rgb(under.get_color()) == to_rgb("navy")
    assert to_rgb(dash.get_color()) == to_rgb("gold")
    assert (dash.get_linewidth(), dash.get_zorder(), dash.get_gid()) == (0.9, 7, "cut")
    assert (under.get_linewidth(), under.get_zorder()) == (2.2, 6)
    plt.close(fig)


def test_update_moves_lines_and_label_without_new_artists():
    fig, ax = plt.subplots()
    res = ep.overlay_line(ax, [0, 2], [0, 0], label="cut", label_at=1.0)
    n_children = len(ax.get_children())
    phi = np.linspace(0.0, np.pi / 2, 30)
    res.update(np.cos(phi), np.sin(phi))
    assert len(ax.get_children()) == n_children
    for line in res.artists["lines"]:
        np.testing.assert_allclose(line.get_xdata(), np.cos(phi))
    assert res.artists["text"].get_position() == pytest.approx((0.0, 1.0), abs=1e-12)
    with pytest.raises(ValueError, match="matching"):
        res.update([0, 1], [0])
    plt.close(fig)


def test_mismatched_path_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="matching"):
        ep.overlay_line(ax, [0, 1, 2], [0, 1])
    plt.close(fig)
