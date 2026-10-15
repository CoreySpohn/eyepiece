"""Plane warp: a matrix or a pointwise map applied to drawn artists."""

import functools

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path

import eyepiece as ep


def _axes():
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.set(xlim=(-3.0, 3.0), ylim=(-3.0, 3.0))
    return fig, ax


def _ring(k=48):
    t = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    return np.c_[np.cos(t), np.sin(t)]


def _xy(line):
    return np.c_[line.get_xdata(), line.get_ydata()]


def _rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def test_warp_is_exported_and_returns_a_result():
    _fig, ax = _axes()
    (line,) = ax.plot([0, 1], [0, 1])
    move = ep.warp(line, np.eye(2))
    assert isinstance(move, ep.PlotResult)
    assert set(move.artists) <= ep.ARTIST_KEYS
    assert move.artists["line"] == [line]


def test_matrix_endpoints_are_exact():
    _fig, ax = _axes()
    pts = _ring()
    (line,) = ax.plot(pts[:, 0], pts[:, 1])
    m = np.array([[1.0, 0.7], [-0.2, 0.5]])
    move = ep.warp(line, m)
    move.update(0.0)
    assert np.array_equal(_xy(line), pts)
    move.update(1.0)
    assert np.allclose(_xy(line), pts @ m.T, atol=1e-14)
    move.update(0.0)
    assert np.array_equal(_xy(line), pts)


def test_midpoint_is_the_blended_matrix():
    _fig, ax = _axes()
    pts = _ring()
    (line,) = ax.plot(pts[:, 0], pts[:, 1])
    m = np.array([[2.0, 0.0], [0.0, 0.5]])
    move = ep.warp(line, m, ease="linear")
    move.update(0.5)
    assert np.allclose(_xy(line), pts @ (np.eye(2) + 0.5 * (m - np.eye(2))).T)


def test_rotation_path_keeps_lengths_through_the_move():
    _fig, ax = _axes()
    pts = _ring()
    (line,) = ax.plot(pts[:, 0], pts[:, 1])
    m = _rot(np.pi * 0.9)
    move = ep.warp(line, m, path="rotation", ease="linear")
    for u in np.linspace(0.0, 1.0, 11):
        move.update(u)
        assert np.allclose(np.hypot(*_xy(line).T), 1.0)
    move.update(1.0)
    assert np.allclose(_xy(line), pts @ m.T, atol=1e-12)


def test_linear_path_shrinks_a_rotation_and_rotation_path_does_not():
    _fig, ax = _axes()
    pts = _ring()
    (a,) = ax.plot(pts[:, 0], pts[:, 1])
    (b,) = ax.plot(pts[:, 0], pts[:, 1])
    ep.warp(a, _rot(np.pi), ease="linear").update(0.5)
    ep.warp(b, _rot(np.pi), path="rotation", ease="linear").update(0.5)
    assert np.hypot(*_xy(a).T).max() < 1e-9
    assert np.allclose(np.hypot(*_xy(b).T), 1.0)


def test_rotation_path_with_a_stretch_ends_on_the_matrix():
    _fig, ax = _axes()
    pts = _ring()
    (line,) = ax.plot(pts[:, 0], pts[:, 1])
    m = _rot(0.8) @ np.array([[2.0, 0.3], [0.3, 0.7]])
    ep.warp(line, m, path="rotation").update(1.0)
    assert np.allclose(_xy(line), pts @ m.T, atol=1e-12)


def test_rotation_path_rejects_a_reflection():
    _fig, ax = _axes()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="determinant"):
        ep.warp(line, [[1.0, 0.0], [0.0, -1.0]], path="rotation")


def test_about_fixes_a_point():
    _fig, ax = _axes()
    (line,) = ax.plot([1.0, 2.0], [1.0, 1.0])
    ep.warp(line, 2.0 * np.eye(2), about=(1.0, 1.0)).update(1.0)
    assert np.allclose(_xy(line), [[1.0, 1.0], [3.0, 1.0]])


def test_pointwise_map_hits_f_at_the_end_and_the_original_at_the_start():
    _fig, ax = _axes()
    pts = np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, 0.0]])
    (line,) = ax.plot(pts[:, 0], pts[:, 1])

    def f(p):
        return np.c_[p[:, 0] + 0.3 * p[:, 1] ** 2, np.sin(p[:, 1])]

    move = ep.warp(line, f)
    move.update(0.0)
    assert np.array_equal(_xy(line), pts)
    move.update(1.0)
    out = _xy(line)
    assert len(out) > len(pts)
    assert np.allclose(out[0], f(pts[:1])[0])
    assert np.allclose(out[-1], f(pts[-1:])[0])
    for p in pts:
        assert np.isclose(out, f(p[None])[0]).all(axis=1).any()


def test_pointwise_midpoint_is_the_straight_blend():
    _fig, ax = _axes()
    pts = _ring(16)
    (line,) = ax.plot(pts[:, 0], pts[:, 1])
    ep.warp(line, lambda p: 3.0 * p, ease="linear").update(0.5)
    out = _xy(line)
    for p in pts:
        assert np.isclose(out, 2.0 * p).all(axis=1).any()


def test_pointwise_map_bends_a_straight_line_smoothly():
    _fig, ax = _axes()
    (line,) = ax.plot([-1.0, 1.0], [1.0, 1.0])
    ep.warp(line, lambda p: np.c_[p[:, 0] * np.cos(p[:, 1] * p[:, 0]), p[:, 0]]).update(
        1.0
    )
    assert len(line.get_xdata()) > 50


def test_artists_are_reused():
    _fig, ax = _axes()
    (line,) = ax.plot([0.0, 1.0], [0.0, 1.0])
    poly = ax.add_patch(Polygon([[0, 0], [1, 0], [1, 1]], closed=True))
    path = ax.add_patch(
        PathPatch(Path([[0, 0], [1, 0], [1, 1], [0, 0]], [1, 2, 2, 79]))
    )
    arrow = ax.add_patch(FancyArrowPatch((0, 0), (1, 1)))
    dots = ax.scatter([0.0, 1.0], [0.0, 1.0])
    seg = ax.add_collection(LineCollection([[(0, 0), (1, 1)]]))
    text = ax.text(0.5, 0.5, "x")
    items = [line, poly, path, arrow, dots, seg, text]
    before = list(ax.get_children())
    for mapping in (np.array([[1.0, 1.0], [0.0, 1.0]]), lambda p: p**2):
        move = ep.warp(items, mapping)
        for u in (0.3, 1.0, 0.0):
            move.update(u)
        assert list(ax.get_children()) == before
        assert [a for v in move.artists.values() for a in v] == items


def test_every_kind_reaches_its_image():
    fig, ax = _axes()
    m = np.array([[1.0, 1.0], [0.0, 2.0]])
    poly = ax.add_patch(Polygon([[0, 0], [1, 0], [1, 1]], closed=True))
    path = ax.add_patch(
        PathPatch(Path([[0, 0], [1, 0], [1, 1], [0, 0]], [1, 2, 2, 79]))
    )
    arrow = ax.add_patch(FancyArrowPatch((0, 0), (1, 1)))
    dots = ax.scatter([0.0, 1.0], [0.0, 1.0])
    seg = ax.add_collection(LineCollection([[(0, 0), (1, 1)]]))
    text = ax.text(0.5, 0.25, "x")
    ann = ax.annotate("a", xy=(1.0, 0.5), xytext=(2.0, 1.0))
    fill = ax.fill_between([0.0, 1.0], [0.0, 0.0], [1.0, 1.0])
    fill_before = fill.get_paths()[0].vertices.copy()
    ep.warp([poly, path, arrow, dots, seg, text, ann, fill], m).update(1.0)
    assert np.allclose(poly.get_xy()[:3], np.array([[0, 0], [1, 0], [1, 1]]) @ m.T)
    assert np.allclose(
        path.get_path().vertices[:3], np.array([[0, 0], [1, 0], [1, 1]]) @ m.T
    )
    assert np.allclose(arrow._posA_posB, [[0, 0], [2, 2]])
    assert np.allclose(dots.get_offsets(), [[0, 0], [2, 2]])
    assert np.allclose(seg.get_segments()[0], [[0, 0], [2, 2]])
    assert np.allclose(text.get_position(), [0.75, 0.5])
    assert np.allclose(ann.xy, [1.5, 1.0])
    assert np.allclose(ann.xyann, [3.0, 2.0])
    assert np.allclose(fill.get_paths()[0].vertices, fill_before @ m.T)
    fig.canvas.draw()


def test_level_never_changes_the_geometry():
    fig, ax = _axes()
    (line,) = ax.plot([1.0, 2.0], [0.0, 0.0])
    move = ep.warp(line, 3.0 * np.eye(2), ease="linear")
    move.update(1.0, level=0.5)
    assert np.allclose(_xy(line), [[3.0, 0.0], [6.0, 0.0]])
    fig.canvas.draw()


def test_nan_rows_survive_and_are_not_passed_to_the_map():
    _fig, ax = _axes()
    (line,) = ax.plot([0.0, 1.0, np.nan, 2.0, 3.0], [0.0, 1.0, np.nan, 0.0, 1.0])
    seen = []

    def f(p):
        seen.append(np.isfinite(p).all())
        return p + 1.0

    move = ep.warp(line, f)
    move.update(1.0)
    assert all(seen)
    assert np.isnan(_xy(line)).any()
    move.update(0.0)
    assert len(line.get_xdata()) == 5


def test_closed_polygon_is_densified_around_its_closing_edge():
    _fig, ax = _axes()
    poly = ax.add_patch(Polygon([[0, 0], [1, 0], [1, 1]], closed=True))
    ep.warp(poly, lambda p: p + 1.0, n=30).update(1.0)
    xy = poly.get_xy()
    assert np.allclose(xy[0], xy[-1])
    assert np.hypot(*np.diff(xy, axis=0).T).max() < 0.2


def test_patch_with_parametric_shape_is_refused():
    _fig, ax = _axes()
    disk = ax.add_patch(Circle((0, 0), 1.0))
    with pytest.raises(TypeError, match="PathPatch"):
        ep.warp(disk, np.eye(2))


def test_artist_off_data_coordinates_is_refused():
    _fig, ax = _axes()
    (line,) = ax.plot([0, 1], [0, 1], transform=ax.transAxes)
    with pytest.raises(ValueError, match="data coordinates"):
        ep.warp(line, np.eye(2))


@pytest.mark.parametrize(
    "bad",
    [np.eye(3), "x", [[1.0, np.nan], [0.0, 1.0]]],
)
def test_bad_matrix_is_refused(bad):
    _fig, ax = _axes()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises((ValueError, TypeError)):
        ep.warp(line, bad)


def test_callable_with_rotation_path_or_about_is_refused():
    _fig, ax = _axes()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError):
        ep.warp(line, lambda p: p, path="rotation")
    with pytest.raises(ValueError):
        ep.warp(line, lambda p: p, about=(0, 0))


def test_wrong_shaped_map_is_refused():
    _fig, ax = _axes()
    (line,) = ax.plot([0, 1], [0, 1])
    with pytest.raises(ValueError, match="N, 2"):
        ep.warp(line, lambda p: p[:, 0])


def test_warp_grid_is_two_collections_and_leaves_limits():
    _fig, ax = _axes()
    result = ep.warp_grid(ax, spacing=1.0)
    grid, heavy = result.artists["collection"]
    assert isinstance(grid, LineCollection) and len(heavy.get_segments()) == 2
    assert ax.get_xlim() == (-3.0, 3.0)
    assert len(grid.get_segments()) > 20


def test_warp_grid_through_warp_ends_on_the_image_and_restores():
    fig, ax = _axes()
    grid = ep.warp_grid(ax, extent=(-2, 2, -2, 2))
    before = [s.copy() for c in grid.artists["collection"] for s in c.get_segments()]
    m = np.array([[1.0, 0.5], [0.0, 1.0]])
    move = ep.warp(grid, m)
    move.update(1.0)
    after = [s for c in grid.artists["collection"] for s in c.get_segments()]
    for b, a in zip(before, after, strict=True):
        assert np.allclose(a, b @ m.T)
    fig.canvas.draw()
    move.update(0.0)
    again = [s for c in grid.artists["collection"] for s in c.get_segments()]
    assert all(np.array_equal(b, a) for b, a in zip(before, again, strict=True))


def test_warp_grid_rejects_bad_spacing_and_extent():
    _fig, ax = _axes()
    with pytest.raises(ValueError):
        ep.warp_grid(ax, spacing=0.0)
    with pytest.raises(ValueError):
        ep.warp_grid(ax, extent=(1, 1, 0, 1))


def test_scatter_group_key():
    _fig, ax = _axes()
    dots = ax.scatter([0.0], [1.0])
    assert isinstance(dots, PathCollection)
    assert ep.warp(dots, np.eye(2)).artists["scatter"] == [dots]


def _extreme(line, move, us, pick):
    best = None
    for u in us:
        move.update(u)
        value = pick(_xy(line)[:, 0])
        best = value if best is None else pick([best, value])
    return best


def _lands_like_smoothstep(make, path_kwargs):
    """The final and first frames of an overshoot warp equal a plain one's."""
    _fig, ax = _axes()
    (ref_line,) = ax.plot(*_ring().T)
    ref = ep.warp(ref_line, **path_kwargs)
    ref.update(1.0)
    want_end = _xy(ref_line).copy()
    ref.update(0.0)
    want_start = _xy(ref_line).copy()
    _fig, ax = _axes()
    (line,) = ax.plot(*_ring().T)
    move = make(line)
    return line, move, want_start, want_end


TURN_AND_STRETCH = _rot(np.pi / 3.0) @ np.diag([2.0, 1.0])


@pytest.mark.parametrize("path", ["linear", "rotation"])
def test_back_out_warp_passes_its_image_and_lands_exactly(path):
    kw = {"mapping": TURN_AND_STRETCH, "path": path}
    line, move, start, end = _lands_like_smoothstep(
        lambda ln: ep.warp(ln, ease="back_out", **kw), kw
    )
    ref_lines = np.linspace(0.0, 1.0, 201)[:-1]
    # An overshoot carries the ring's rightmost point past the image's.
    top = _extreme(line, move, ref_lines, np.max)
    assert top > end[:, 0].max() + 0.05
    move.update(1.0)
    np.testing.assert_allclose(_xy(line), end, atol=1e-12)
    move.update(0.0)
    np.testing.assert_allclose(_xy(line), start, atol=1e-12)


def test_back_out_pointwise_warp_passes_its_image_and_lands_exactly():
    def shift(p):
        return p + np.array([1.0, 0.0])

    kw = {"mapping": shift}
    line, move, _start, end = _lands_like_smoothstep(
        lambda ln: ep.warp(ln, ease="back_out", **kw), kw
    )
    top = _extreme(line, move, np.linspace(0.0, 1.0, 201)[:-1], np.max)
    assert top > end[:, 0].max() + 0.05
    move.update(1.0)
    np.testing.assert_allclose(_xy(line), end, atol=1e-12)


@pytest.mark.parametrize("path", ["linear", "rotation"])
def test_running_start_warp_pulls_back_before_it_goes(path):
    kw = {"mapping": TURN_AND_STRETCH, "path": path}
    line, move, start, end = _lands_like_smoothstep(
        lambda ln: ep.warp(ln, ease="running_start", **kw), kw
    )
    pull = _extreme(line, move, np.linspace(0.0, 1.0, 201)[1:-1], np.min)
    # Early in the move the ring's leftmost point sits left of where it began.
    assert pull < start[:, 0].min() - 0.02
    move.update(1.0)
    np.testing.assert_allclose(_xy(line), end, atol=1e-12)


def test_rotation_warp_snaps_on_progress_not_on_the_eased_value():
    # A strong back_out reaches an eased value of 1 well before u = 1, so a
    # snap on the eased value would jump to the goal early.
    fast = functools.partial(ep.ease, kind="back_out", strength=0.5)
    assert fast(0.95) > 1.0
    kw = {"mapping": TURN_AND_STRETCH, "path": "rotation"}
    _fig, ax = _axes()
    (ref_line,) = ax.plot(*_ring().T)
    ref = ep.warp(ref_line, **kw)
    ref.update(1.0)
    goal = _xy(ref_line).copy()
    _fig, ax = _axes()
    (line,) = ax.plot(*_ring().T)
    move = ep.warp(line, ease=fast, **kw)
    move.update(0.95)
    near = _xy(line).copy()
    assert near.shape == goal.shape
    assert np.abs(near - goal).max() > 1e-3  # still on the way, not at the goal
    move.update(1.0)
    np.testing.assert_allclose(_xy(line), goal, atol=1e-12)
