"""Following zoom: a magnified inset that tracks a moving point."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.artist import Artist
from matplotlib.collections import Collection
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import eyepiece as ep


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _panel(inverted=False):
    """An image under a marker on a track, with unequal axis scales."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=100)
    rng = np.random.default_rng(3)
    ax.imshow(rng.random((30, 40)), extent=(0.0, 8.0, 0.0, 3.0), aspect="auto")
    (dot,) = ax.plot([2.0], [1.0], "o", ms=5, color="white")
    ax.set(xlim=(8.0, 0.0) if inverted else (0.0, 8.0), ylim=(0.0, 3.0))
    fig.set_layout_engine("none")
    return fig, ax, dot


def _track(t):
    return 1.0 + 6.0 * t, 1.0 + 0.8 * np.sin(4.0 * t)


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _center(inset):
    return 0.5 * sum(inset.get_xlim()), 0.5 * sum(inset.get_ylim())


@pytest.mark.parametrize("t", [0.0, 0.2, 0.5, 0.9])
def test_inset_limits_are_centered_on_the_target_at_several_times(t):
    fig, ax, _ = _panel()
    res = ep.follow_zoom(ax, _track, loc="upper left")
    res.update(1.0, time=t)
    fig.canvas.draw()
    cx, cy = _center(res.insets[0])
    assert (cx, cy) == pytest.approx(_track(t), abs=1e-9)


@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("zoom", [2.0, 4.0, 7.5])
def test_the_zoom_factor_is_exact_on_both_axes(zoom, inverted):
    fig, ax, _ = _panel(inverted)
    res = ep.follow_zoom(ax, _track, zoom=zoom, size=0.3)
    res.update(1.0, time=0.4)
    fig.canvas.draw()
    inset = res.insets[0]
    for axis in (0, 1):
        outer = ax.transData.transform([(1, 1), (2, 2)])
        inner = inset.transData.transform([(1, 1), (2, 2)])
        per_outer = outer[1, axis] - outer[0, axis]
        per_inner = inner[1, axis] - inner[0, axis]
        assert per_inner / per_outer == pytest.approx(zoom, rel=1e-9)
    assert inset.bbox.width == pytest.approx(inset.bbox.height)


def test_the_box_and_the_inset_limits_agree():
    fig, ax, _ = _panel()
    res = ep.follow_zoom(ax, _track)
    for t in (0.1, 0.6):
        res.update(1.0, time=t)
        fig.canvas.draw()
        box = res.artists["ellipse"].get_window_extent()
        corners = ax.transData.inverted().transform(
            [(box.x0, box.y0), (box.x1, box.y1)]
        )
        inset = res.insets[0]
        assert sorted(corners[:, 0]) == pytest.approx(sorted(inset.get_xlim()))
        assert sorted(corners[:, 1]) == pytest.approx(sorted(inset.get_ylim()))


def test_zero_is_the_figure_without_the_zoom_and_one_is_the_rest_state():
    plain = _pixels(_panel()[0])
    fig, ax, _ = _panel()
    res = ep.follow_zoom(ax, _track)
    drawn = _pixels(fig)
    assert not np.array_equal(drawn, plain)
    res.update(0.0)
    assert np.array_equal(_pixels(fig), plain)
    res.update(0.4)
    res.update(1.0, level=0.3)
    res.update(1.0)
    assert np.array_equal(_pixels(fig), drawn)


def test_drawing_hands_the_parents_artists_back_exactly():
    fig, ax, dot = _panel()
    ax.plot([0, 8], [0, 3], alpha=0.5)
    ax.add_patch(Rectangle((1, 1), 2, 1, alpha=0.4))
    ax.scatter([3, 4], [1, 2])
    before = [
        (a, Artist.get_transform(a), a.get_clip_box(), a.get_clip_path(), a.get_alpha())
        for a in ax.get_children()
        if isinstance(a, (Line2D, Patch, Collection, AxesImage))
    ]
    res = ep.follow_zoom(ax, dot)
    res.update(0.5)
    fig.canvas.draw()
    for artist, tf, box, path, alpha in before:
        assert Artist.get_transform(artist) is tf
        assert artist.get_clip_box() is box
        assert artist.get_clip_path() is path
        assert artist.get_alpha() == alpha


def test_level_is_an_opacity_and_never_geometry():
    fig, ax, _ = _panel()
    res = ep.follow_zoom(ax, _track)
    res.update(1.0, time=0.3)
    fig.canvas.draw()
    full = (
        res.insets[0].get_xlim(),
        res.insets[0].get_ylim(),
        res.insets[0].bbox.bounds,
    )
    res.update(1.0, level=0.4)
    fig.canvas.draw()
    half = (
        res.insets[0].get_xlim(),
        res.insets[0].get_ylim(),
        res.insets[0].bbox.bounds,
    )
    assert full == half
    assert res.artists["ellipse"].get_alpha() == pytest.approx(0.4)
    res.update(1.0, level=0.0)
    assert not res.insets[0].get_visible()
    assert not res.artists["ellipse"].get_visible()


def test_an_overshooting_ease_never_pushes_opacity_outside_zero_to_one():
    _, ax, _ = _panel()
    res = ep.follow_zoom(ax, _track, ease=lambda r: 1.4 * r)
    res.update(0.9)
    assert res.artists["ellipse"].get_alpha() in (None, 1.0)


def test_an_artist_target_is_followed_with_no_update_call():
    fig, ax, dot = _panel()
    res = ep.follow_zoom(ax, dot, loc="lower left")
    for x in (1.0, 3.5, 6.0):
        dot.set_data([x], [1.5])
        fig.canvas.draw()
        assert _center(res.insets[0]) == pytest.approx((x, 1.5), abs=1e-9)


def test_an_anchor_target_survives_resizing():
    fig, ax, _ = _panel()
    res = ep.follow_zoom(ax, ep.Anchor.from_data(ax, (5.0, 2.0)))
    fig.set_size_inches(8.0, 3.0)
    fig.canvas.draw()
    assert _center(res.insets[0]) == pytest.approx((5.0, 2.0), abs=1e-6)


def test_a_fixed_pair_is_a_target():
    fig, ax, _ = _panel()
    res = ep.follow_zoom(ax, (4.0, 1.0))
    fig.canvas.draw()
    assert _center(res.insets[0]) == pytest.approx((4.0, 1.0))


def test_the_inset_shows_the_parents_marks_not_a_replot():
    fig, ax, dot = _panel()
    res = ep.follow_zoom(ax, dot, zoom=6.0, size=0.4)
    inset = res.insets[0]
    assert not inset.lines and not inset.images
    dot.set_markersize(30)
    fig.canvas.draw()
    inside = _pixels(fig)[
        int(fig.bbox.height - inset.bbox.y1) : int(fig.bbox.height - inset.bbox.y0),
        int(inset.bbox.x0) : int(inset.bbox.x1),
    ]
    assert (inside[..., :3] > 250).all(axis=-1).sum() > 100


def test_audit_passes_the_connectors_and_a_crossed_line_is_a_fault():
    fig, ax, dot = _panel()
    res = ep.follow_zoom(ax, dot, loc="upper right")
    dot.set_data([1.0], [1.0])
    fig.canvas.draw()
    assert ep.audit(fig) == {}
    for line in res.artists["lines"]:
        assert line.faults() == []
    # Positive control: a data line laid across the connectors is named.
    ax.plot([0.0, 8.0], [2.4, 2.4], color="red", lw=2)
    ax.plot([6.0, 6.0], [0.0, 3.0], color="red", lw=2)
    fig.canvas.draw()
    found = ep.audit(fig, raise_on_fault=False)
    assert found and all("crosses data" in m for ms in found.values() for m in ms)
    with pytest.raises(ep.AttentionFault):
        ep.audit(fig)


def test_connectors_are_neutral_and_vanish_when_the_box_meets_the_inset():
    fig, ax, dot = _panel()
    res = ep.follow_zoom(ax, dot, loc="upper right", size=0.4)
    dot.set_data([7.0], [2.5])
    fig.canvas.draw()
    for line in res.artists["lines"]:
        assert line._ends(None) is None
        assert line.faults() == []


def test_a_stage_can_weigh_the_inset():
    fig, ax, dot = _panel()
    res = ep.follow_zoom(ax, dot)
    stage = ep.Stage({"zoom": [res.insets[0], res.artists["ellipse"]]})
    stage.weigh({"zoom": 0.0})
    assert fig.canvas.draw() is None
    stage.rest()


def test_bad_inputs_are_named():
    fig, ax, dot = _panel()
    with pytest.raises(ValueError, match="above 1"):
        ep.follow_zoom(ax, dot, zoom=1.0)
    with pytest.raises(ValueError, match="size"):
        ep.follow_zoom(ax, dot, size=0.0)
    with pytest.raises(ValueError, match="loc"):
        ep.follow_zoom(ax, dot, loc="middle")
    with pytest.raises(TypeError, match="target"):
        ep.follow_zoom(ax, "x")
    with pytest.raises(TypeError, match="Axes"):
        ep.follow_zoom(fig, dot)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 3.0)
    with pytest.raises(ValueError, match="linear"):
        ep.follow_zoom(ax, dot)


def test_a_patch_and_a_scatter_land_where_the_window_says():
    fig, ax, _ = _panel()
    ax.add_patch(Rectangle((5.0, 1.0), 0.5, 0.4, color="red", lw=0, zorder=5))
    ax.scatter([2.0], [2.0], s=60, color="lime", zorder=5)
    res = ep.follow_zoom(ax, (5.25, 1.2), zoom=8.0, size=0.4, loc="upper left")
    fig.canvas.draw()
    inset = res.insets[0]
    img = np.asarray(fig.canvas.buffer_rgba())
    cx, cy = (
        inset.bbox.x0 + 0.5 * inset.bbox.width,
        inset.bbox.y0 + 0.5 * inset.bbox.height,
    )
    assert tuple(img[int(fig.bbox.height - cy), int(cx), :3]) == (255, 0, 0)
