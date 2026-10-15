"""Morphs: an image drawn as movable pixel quads, and linked brushing."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

import eyepiece as ep


def _grid(ny=3, nx=4):
    y, x = np.mgrid[0 : ny + 1, 0 : nx + 1].astype(float)
    corners = np.stack([x, y], axis=-1)
    values = np.arange(ny * nx, dtype=float).reshape(ny, nx) + 1.0
    return values, corners


# ---------------------------------------------------------------------------
# quad_image


def test_quad_image_draws_one_quad_per_pixel():
    values, corners = _grid()
    res = ep.quad_image(values, corners)
    coll = res.artists["collection"]
    assert isinstance(coll, PolyCollection)
    assert len(coll.get_paths()) == 12
    assert np.allclose(coll.get_array(), values.ravel())
    assert res.ax.get_xlim() == pytest.approx((0.0, 4.0), abs=0.5)
    plt.close(res.fig)


def test_quad_image_default_norm_spans_the_finite_values():
    values, corners = _grid()
    values[0, 0] = np.nan
    res = ep.quad_image(values, corners)
    norm = res.artists["collection"].norm
    assert (norm.vmin, norm.vmax) == (2.0, 12.0)
    plt.close(res.fig)


def test_quad_image_keep_selects_pixels_and_accepts_a_norm():
    values, corners = _grid()
    keep = values > 6.0
    fig, ax = plt.subplots()
    res = ep.quad_image(values, corners, ax=ax, keep=keep, norm=LogNorm(1.0, 12.0))
    coll = res.artists["collection"]
    assert res.ax is ax
    assert len(coll.get_paths()) == int(keep.sum())
    assert isinstance(coll.norm, LogNorm)
    plt.close(fig)


def test_quad_image_update_moves_and_recolors_the_same_artist():
    values, corners = _grid()
    res = ep.quad_image(values, corners, collection_kw={"linewidths": 0.0})
    coll = res.artists["collection"]
    xlim = res.ax.get_xlim()
    quads = ep.pixel_quads(corners) + np.array([10.0, 0.0])
    res.update(quads=quads, values=values.ravel()[::-1])
    assert res.artists["collection"] is coll
    assert np.allclose(coll.get_paths()[0].vertices[0], (10.0, 0.0))
    assert coll.get_array()[0] == pytest.approx(12.0)
    assert res.ax.get_xlim() == xlim  # an update never rescales
    assert coll.get_linewidths()[0] == pytest.approx(0.0)
    plt.close(res.fig)


def test_quad_image_update_with_the_wrong_count_raises_and_changes_nothing():
    values, corners = _grid()
    res = ep.quad_image(values, corners)
    coll = res.artists["collection"]
    before = coll.get_paths()[0].vertices.copy()
    with pytest.raises(ValueError, match="12"):
        res.update(quads=np.zeros((5, 4, 2)))
    with pytest.raises(ValueError, match="12"):
        res.update(quads=ep.pixel_quads(corners) + 1.0, values=np.ones(5))
    assert np.allclose(coll.get_paths()[0].vertices, before)
    plt.close(res.fig)


def test_quad_image_rejects_mismatched_values_and_corners():
    values, corners = _grid()
    with pytest.raises(ValueError, match="corners"):
        ep.quad_image(values, corners[:-1])


# ---------------------------------------------------------------------------
# brush


def _two_panels():
    values, corners = _grid()
    fig, axes = plt.subplots(1, 2)
    left = ep.pixel_quads(corners)
    right = left + np.array([5.0, 0.0])
    for ax, q in zip(axes, (left, right), strict=True):
        ep.quad_image(values, corners, ax=ax).update(quads=q)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 4)
    return fig, axes, [left, right], values.ravel()


def test_brush_starts_hidden_and_shows_the_masked_items_over_a_veil():
    fig, axes, quads, values = _two_panels()
    res = ep.brush(axes, quads, values)
    veils, overlays = res.artists["fill"], res.artists["collection"]
    assert len(veils) == len(overlays) == 2
    assert all(isinstance(v, Rectangle) for v in veils)
    assert not any(v.get_visible() for v in veils)
    mask = values > 9.0
    res.update(mask)
    assert all(v.get_visible() for v in veils)
    assert veils[0].get_alpha() == pytest.approx(0.72)
    for overlay, q in zip(overlays, quads, strict=True):
        assert len(overlay.get_paths()) == int(mask.sum())
        assert np.allclose(overlay.get_paths()[0].vertices[0], q[mask][0, 0])
    assert overlays[0].zorder > veils[0].zorder
    plt.close(fig)


def test_brush_follows_moving_quads_and_clears():
    fig, axes, quads, values = _two_panels()
    res = ep.brush(axes, quads, values)
    mask = np.zeros(values.size, bool)
    mask[3] = True
    moved = [q + 1.0 for q in quads]
    res.update(mask, quads=moved)
    assert np.allclose(
        res.artists["collection"][1].get_paths()[0].vertices[0], moved[1][3, 0]
    )
    res.update(None)
    assert not any(v.get_visible() for v in res.artists["fill"])
    assert not any(o.get_visible() for o in res.artists["collection"])
    plt.close(fig)


def test_brush_rejects_mismatched_inputs():
    fig, axes, quads, values = _two_panels()
    with pytest.raises(ValueError, match="one quads array per axes"):
        ep.brush(axes, quads[:1], values)
    with pytest.raises(ValueError, match="12"):
        ep.brush(axes, [quads[0], quads[1][:5]], values)
    res = ep.brush(axes, quads, values)
    res.update(values > 9.0)
    overlay = res.artists["collection"][0]
    before = [p.vertices.copy() for p in overlay.get_paths()]
    with pytest.raises(ValueError, match="mask"):
        res.update(np.ones(5, bool))
    with pytest.raises(ValueError, match="12"):
        res.update(np.ones(12, bool), quads=[quads[0], quads[1][:5]])
    with pytest.raises(TypeError, match="boolean"):
        res.update(np.arange(12))
    after = [p.vertices for p in overlay.get_paths()]
    assert len(after) == len(before)
    assert all(np.array_equal(a, b) for a, b in zip(after, before, strict=True))
    plt.close(fig)


def test_brush_veils_each_panel_in_its_own_background():
    fig, axes, quads, values = _two_panels()
    axes[0].set_facecolor("black")
    axes[1].set_facecolor("none")
    fig.set_facecolor("#334455")
    res = ep.brush(axes, quads, values)
    from matplotlib.colors import to_rgba

    veils = res.artists["fill"]
    assert veils[0].get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
    assert veils[1].get_facecolor()[:3] == pytest.approx(to_rgba("#334455")[:3])
    plt.close(fig)
