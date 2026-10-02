"""overlay_circle: a light dash over a dark underlay that never moves the view."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle

import eyepiece as ep


def _lum(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_light_dash_over_a_solid_dark_underlay_in_either_mode(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    ax.imshow(np.random.default_rng(0).random((8, 8)), extent=(-4, 4, -4, 4))
    res = ep.overlay_circle(ax, (1.0, -0.5), 2.0)
    under, dash = res.artists["ellipse"]
    assert isinstance(dash, Circle) and isinstance(under, Circle)
    assert dash.center == under.center == (1.0, -0.5)
    assert dash.radius == under.radius == 2.0
    assert dash.get_linestyle() not in ("-", "solid")
    assert under.get_linestyle() in ("-", "solid")
    assert _lum(dash.get_edgecolor()) > _lum(under.get_edgecolor())
    assert under.get_linewidth() > dash.get_linewidth()
    # The dash is drawn after the underlay, on the same layer.
    order = ax.get_children()
    assert order.index(dash) > order.index(under)
    assert dash.get_zorder() == under.get_zorder()
    plt.close(fig)


def test_a_circle_larger_than_the_view_leaves_the_limits_alone():
    fig, ax = plt.subplots()
    ax.imshow(np.ones((4, 4)), extent=(-1, 1, -1, 1))
    limits = ax.get_xlim(), ax.get_ylim()
    ep.overlay_circle(ax, (0.0, 0.0), 5.0)
    fig.canvas.draw()
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    plt.close(fig)


def test_overrides_reach_the_dash_and_the_underlay_follows_its_width():
    fig, ax = plt.subplots()
    res = ep.overlay_circle(
        ax, (0, 0), 1.0, color="red", circle_kw={"lw": 2.0, "gid": "aperture"}
    )
    under, dash = res.artists["ellipse"]
    assert to_rgb(dash.get_edgecolor()) == to_rgb("red")
    assert dash.get_gid() == "aperture"
    assert under.get_linewidth() == pytest.approx(3.0)
    plt.close(fig)


def test_underlay_can_be_colored_or_omitted():
    fig, ax = plt.subplots()
    colored = ep.overlay_circle(ax, (0, 0), 1.0, underlay="blue")
    assert to_rgb(colored.artists["ellipse"][0].get_edgecolor()) == to_rgb("blue")
    bare = ep.overlay_circle(ax, (0, 0), 1.0, underlay=False)
    assert len(bare.artists["ellipse"]) == 1
    assert set(bare.artists) <= ep.ARTIST_KEYS
    plt.close(fig)


def test_underlay_kw_sets_the_underlay_width_and_tags_it():
    fig, ax = plt.subplots()
    res = ep.overlay_circle(
        ax,
        (0, 0),
        1.0,
        circle_kw={"lw": 1.5, "gid": "aperture"},
        underlay_kw={"lw": 3.6, "gid": "aperture"},
    )
    under, dash = res.artists["ellipse"]
    assert under.get_linewidth() == pytest.approx(3.6)
    assert dash.get_linewidth() == pytest.approx(1.5)
    assert under.get_gid() == dash.get_gid() == "aperture"
    # Only what was passed changes: the underlay stays solid, on the dash's layer.
    assert under.get_linestyle() in ("-", "solid")
    assert under.get_zorder() == dash.get_zorder()
    plt.close(fig)


def test_underlay_kw_is_ignored_without_an_underlay():
    fig, ax = plt.subplots()
    res = ep.overlay_circle(ax, (0, 0), 1.0, underlay=False, underlay_kw={"lw": 9.0})
    (dash,) = res.artists["ellipse"]
    assert dash.get_linewidth() == pytest.approx(1.0)
    plt.close(fig)


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_nonpositive_radius_raises(radius):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="radius"):
        ep.overlay_circle(ax, (0, 0), radius)
    plt.close(fig)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_default_circles_are_pinned(mode):
    """The default call draws exactly these two circles and nothing else."""
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    ax.imshow(np.ones((4, 4)), extent=(-2, 2, -2, 2))
    n_children = len(ax.get_children())
    res = ep.overlay_circle(ax, (0.5, -0.25), 1.5)
    assert set(res.artists) == {"ellipse"}
    assert len(ax.get_children()) == n_children + 2
    under, dash = res.artists["ellipse"]
    light, dark = ep.images._light_and_dark()
    assert dash.get_fill() is False and under.get_fill() is False
    assert to_rgb(dash.get_edgecolor()) == pytest.approx(light)
    assert to_rgb(under.get_edgecolor()) == pytest.approx(dark)
    assert dash.get_linewidth() == pytest.approx(1.0)
    assert under.get_linewidth() == pytest.approx(2.0)
    assert dash.get_linestyle() == (0, (3, 2))
    assert under.get_linestyle() in ("-", "solid")
    assert dash.get_zorder() == under.get_zorder() == 4
    assert dash.get_gid() is None and under.get_gid() is None
    assert res.update is None
    plt.close(fig)
