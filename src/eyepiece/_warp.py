"""Plane warp: carry drawn artists through a linear or pointwise map of the plane.

`warp` moves the geometry of artists that already exist, in the data
coordinates of their axes, so the viewer watches a shape become its image
under a map rather than see one figure replaced by another. `warp_grid`
draws the usual subject, a coordinate grid.
"""

import math

import numpy as np
from matplotlib.artist import Artist
from matplotlib.collections import (
    Collection,
    LineCollection,
    PathCollection,
    PolyCollection,
)
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, PathPatch, Polygon
from matplotlib.path import Path
from matplotlib.text import Annotation, Text

from eyepiece._motion import _apply_ease as _ease
from eyepiece._result import PlotResult
from eyepiece.attention import _check_u, _line_parts, _path_parts, _resample

_WARP_POINTS = 200


def _densify(parts, n):
    """Subdivide every segment of `parts` so none is much longer than a step.

    The step is the total arc length of a path divided by `n`. Each segment
    is split by `_resample` (the arc-length sampling `morph` uses), so every
    original vertex survives and a corner stays a corner.
    """
    out = []
    for verts, closed in parts:
        ring = np.vstack([verts, verts[:1]]) if closed and len(verts) > 1 else verts
        if len(ring) < 2:
            out.append((verts, closed))
            continue
        seg = np.hypot(*np.diff(ring, axis=0).T)
        step = float(seg.sum()) / max(int(n), 1)
        if step <= 0.0:
            out.append((verts, closed))
            continue
        pieces = [ring[:1]]
        for i, length in enumerate(seg):
            k = max(1, math.ceil(length / step))
            pieces.append(_resample(ring[i : i + 2], False, k + 1)[1:])
        dense = np.vstack(pieces)
        out.append((dense[:-1] if closed else dense, closed))
    return out


def _codes(parts):
    codes = []
    for verts, closed in parts:
        codes += [Path.MOVETO, *[Path.LINETO] * (len(verts) - 1)]
        if closed:
            codes.append(Path.CLOSEPOLY)
    return codes


def _stack(parts):
    """Vertices of `parts` as one array, with a repeated first point per ring."""
    return np.vstack([np.vstack([v, v[:1]]) if c else v for v, c in parts])


def _check_in_data(artist, transform, what):
    if artist.axes is None:
        raise ValueError(f"warp needs {what} on an axes")
    if not transform.contains_branch(artist.axes.transData):
        raise ValueError(
            f"warp moves data coordinates, but {what} is not drawn in them"
        )


class _Target:
    """One artist's points, how to rebuild it from moved points, how to restore it.

    `points` is the (N, 2) array the map acts on; `build(q)` sets the artist
    from the moved array; `restore()` puts back exactly what the artist had.
    """

    def __init__(self, points, build, restore):
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.build = build
        self.restore = restore
        self.goal = None


def _path_target(get, put, dense, n):
    path = get()
    original = Path(
        path.vertices.copy(), None if path.codes is None else path.codes.copy()
    )
    if not dense:
        codes = original.codes

        def build(q):
            put(Path(q, codes))

        return _Target(original.vertices, build, lambda: put(original))
    parts = _densify(_path_parts(original), n)
    codes = _codes(parts)

    def build(q):
        put(Path(q, codes))

    return _Target(_stack(parts), build, lambda: put(original))


def _line_target(line, dense, n):
    x, y = (
        np.array(line.get_xdata(), dtype=float),
        np.array(line.get_ydata(), dtype=float),
    )
    xy = np.c_[x, y]
    if dense:
        pieces = []
        for verts, _ in _densify(_line_parts(xy, False), n):
            pieces += [verts, np.full((1, 2), np.nan)]
        xy = np.vstack(pieces[:-1]) if pieces else xy

    def build(q):
        line.set_data(q[:, 0], q[:, 1])

    return _Target(xy, build, lambda: line.set_data(x, y))


def _polygon_target(patch, dense, n):
    xy = np.array(patch.get_xy(), dtype=float)
    closed = patch.get_closed()
    if closed and len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        ring = xy[:-1]
    else:
        ring = xy
    points = _densify([(ring, closed)], n)[0][0] if dense else xy

    def build(q):
        patch.set_xy(q)

    return _Target(points, build, lambda: patch.set_xy(xy))


def _arrow_target(arrow, dense, n):
    if arrow._posA_posB is not None:
        a, b = (np.array(p, dtype=float) for p in arrow._posA_posB)

        def build(q):
            arrow.set_positions(q[0], q[1])

        return _Target(np.array([a, b]), build, lambda: arrow.set_positions(a, b))
    return _path_target(
        lambda: arrow._path_original,
        lambda p: setattr(arrow, "_path_original", p),
        dense,
        n,
    )


def _segments_target(collection, dense, n):
    segments = [np.array(s, dtype=float) for s in collection.get_segments()]
    parts = [(s, False) for s in segments]
    if dense:
        parts = _densify(parts, n)
    cuts = np.cumsum([len(v) for v, _ in parts])[:-1]

    def build(q):
        collection.set_segments(np.split(q, cuts))

    return _Target(
        np.vstack([v for v, _ in parts]),
        build,
        lambda: collection.set_segments(segments),
    )


def _polys_target(collection, dense, n):
    paths = [np.array(p.vertices, dtype=float) for p in collection.get_paths()]
    parts = [(v, False) for v in paths]
    if dense:
        parts = _densify(parts, n)
    cuts = np.cumsum([len(v) for v, _ in parts])[:-1]

    def build(q):
        collection.set_verts(np.split(q, cuts), closed=False)

    return _Target(
        np.vstack([v for v, _ in parts]),
        build,
        lambda: collection.set_verts(paths, closed=False),
    )


def _offsets_target(collection):
    offsets = np.array(collection.get_offsets(), dtype=float)
    return _Target(
        offsets,
        lambda q: collection.set_offsets(q),
        lambda: collection.set_offsets(offsets),
    )


def _text_target(text):
    names, points = [], []
    if isinstance(text, Annotation):
        if text.xycoords == "data":
            names.append("xy")
            points.append(np.array(text.xy, dtype=float))
        if text.get_anncoords() == "data":
            names.append("xyann")
            points.append(np.array(text.xyann, dtype=float))
        if not names:
            raise ValueError("warp moves data coordinates, but an annotation has none")
    else:
        _check_in_data(text, text.get_transform(), "a text")
        names.append(None)
        points.append(np.array(text.get_position(), dtype=float))

    def put(q):
        for name, p in zip(names, q, strict=True):
            if name is None:
                text.set_position(p)
            else:
                setattr(text, name, tuple(p))

    return _Target(np.array(points), put, lambda: put(np.array(points)))


def _key(artist):
    if isinstance(artist, Line2D):
        return "line"
    if isinstance(artist, Text):
        return "text"
    if isinstance(artist, FancyArrowPatch):
        return "arrow"
    if isinstance(artist, Patch):
        return "fill"
    if isinstance(artist, PathCollection):
        return "scatter"
    return "collection"


def _target(artist, dense, n):
    if isinstance(artist, Line2D):
        _check_in_data(artist, artist.get_transform(), "a line")
        return _line_target(artist, dense, n)
    if isinstance(artist, Text):
        return _text_target(artist)
    if isinstance(artist, FancyArrowPatch):
        _check_in_data(artist, artist.get_transform(), "an arrow")
        return _arrow_target(artist, dense, n)
    if isinstance(artist, PathPatch):
        _check_in_data(artist, artist.get_transform(), "a patch")
        return _path_target(artist.get_path, artist.set_path, dense, n)
    if isinstance(artist, Polygon):
        _check_in_data(artist, artist.get_transform(), "a patch")
        return _polygon_target(artist, dense, n)
    if isinstance(artist, Patch):
        raise TypeError(
            f"warp cannot reshape a {type(artist).__name__}, whose shape is a "
            f"few parameters; draw it as a PathPatch or a Polygon instead"
        )
    if isinstance(artist, LineCollection):
        _check_in_data(artist, artist.get_transform(), "a line collection")
        return _segments_target(artist, dense, n)
    if isinstance(artist, PolyCollection):
        _check_in_data(artist, artist.get_transform(), "a polygon collection")
        return _polys_target(artist, dense, n)
    if isinstance(artist, Collection) and len(artist.get_offsets()):
        _check_in_data(artist, artist.get_offset_transform(), "a collection")
        return _offsets_target(artist)
    raise TypeError(
        f"warp moves lines, patches, collections and texts, got {type(artist).__name__}"
    )


def _expand(items):
    if isinstance(items, (Artist, PlotResult)):
        items = [items]
    out = []
    for item in items:
        if isinstance(item, PlotResult):
            for value in item.artists.values():
                out += list(value) if isinstance(value, (list, tuple)) else [value]
        elif isinstance(item, Artist):
            out.append(item)
        else:
            raise TypeError(f"warp moves artists, got {type(item).__name__}")
    return out


def _rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _matrix_path(m, kind):
    """The function from eased progress to the 2x2 matrix applied."""
    ident = np.eye(2)
    if kind == "linear":
        return lambda e: ident + e * (m - ident)
    if np.linalg.det(m) <= 0.0:
        raise ValueError(
            "path='rotation' needs a matrix with positive determinant; a "
            "reflection or a collapse has no rotation to turn through"
        )
    u, sv, vt = np.linalg.svd(m)
    rot = u @ vt
    stretch = vt.T @ np.diag(sv) @ vt
    angle = math.atan2(rot[1, 0], rot[0, 0])
    # The stretch is clipped so a strong shrink cannot flip under an overshoot.
    return lambda e: (
        _rotation(e * angle) @ (ident + min(1.0, max(0.0, e)) * (stretch - ident))
    )


def warp(
    artists,
    mapping,
    *,
    path="linear",
    about=None,
    n=_WARP_POINTS,
    ease="smoothstep",
):
    """Carry drawn artists through a map of the plane, continuously.

    The move is a matrix applied to the plane (Manim's ``ApplyMatrix``) or
    a pointwise map (Manim's ``Homotopy``). Every artist given is reshaped
    in place, in the data coordinates of its axes: lines, polygons and path
    patches, arrows, line, polygon and point collections, and the positions
    of texts. Nothing is created or replaced, so the artists keep their
    place in the axes, their style, their zorder, and their membership in a
    `Stage`. Arrows made by ``ep.arrow`` are supported as soon as that
    function is merged; they are `FancyArrowPatch` artists, which are
    handled here, as is any other patch that is a path or a polygon.

    For a 2x2 matrix `M` the plane is mapped by ``I + u (M - I)`` with
    ``path="linear"``, which is a straight blend of the matrices and passes
    through the origin's neighborhood shrunken when `M` is a rotation. With
    ``path="rotation"`` the matrix is split into a rotation `R` and a
    symmetric stretch `S` (polar decomposition, ``M = R S``) and the move is
    ``R(u * angle) (I + u (S - I))``, so a rotation turns through its angle
    without changing any length on the way. The split needs a positive
    determinant. A linear map keeps straight lines straight, so paths keep
    their own vertices and curves.

    For a callable `mapping` taking an (N, 2) array of points and returning
    an (N, 2) array, every path is first subdivided so that no segment is
    much longer than one `n`-th of its length (the original vertices are
    kept, so corners survive; the subdivision uses the arc-length sampling
    of `morph`), then each point moves along the straight line from `p` to
    ``f(p)``: the point is ``p + u (f(p) - p)``. A curve that the map bends
    therefore bends smoothly into place. Curves of a patch are flattened
    first, and an arrow given by its two end points moves only those.

    The artists are left as drawn when the move is made, which is the state
    at ``u = 0``. At ``u = 0`` the geometry is exactly what it was
    (a subdivided path is put back as it was drawn), and at ``u = 1`` it is
    exactly ``M p`` or ``f(p)``. The move does not touch axis limits; set
    them to hold the image.

    Args:
        artists: An artist, a `PlotResult` (its artists are taken), or a
            list of either. Each must be drawn in data coordinates.
        mapping: A (2, 2) matrix, or a callable from an (N, 2) float array
            to an (N, 2) float array. NaN rows are not passed to it.
        path: ``"linear"`` or ``"rotation"``; only for a matrix.
        about: The fixed point of a matrix, ``(x, y)``; default the origin.
            Not for a callable map.
        n: Subdivision of a path under a callable map, at least 1.
        ease: An easing kind for `eyepiece.ease`, applied to the progress; a callable
            from progress to eased progress also works, for a strength.

    Returns:
        A `PlotResult` on the axes of the first artist, whose ``artists``
        group the moved artists by kind (``line``, ``text``, ``arrow``,
        ``fill`` for patches, ``scatter``, ``collection``), and with
        ``update(u, level=1.0)``, `u` from 0 to 1 (values outside are
        clipped). `level` is accepted for the signature every move shares
        and does not change the geometry: a warp edits the caller's own
        artists, whose opacity belongs to the caller or a `Stage`, so
        dimming a group never undoes its warp.

    Raises:
        TypeError: If an artist is not of a kind listed above, including a
            `Rectangle`, `Circle` or other patch with parametric shape.
        ValueError: If `mapping` is neither a (2, 2) matrix nor a callable,
            `path` is unknown or does not suit the mapping, an artist is
            not drawn in data coordinates, or the map returns the wrong
            shape.

    Example::

        grid = ep.warp_grid(ax)
        arrow = ax.arrow(0, 0, 1, 0, ...)
        shear = ep.warp([grid, arrow], [[1.0, 1.0], [0.0, 1.0]])
        for u in np.linspace(0.0, 1.0, 40):
            shear.update(u)
            rec.frame()
    """
    if path not in ("linear", "rotation"):
        raise ValueError(f"warp path must be 'linear' or 'rotation', got {path!r}")
    n = int(n)
    if n < 1:
        raise ValueError(f"warp needs n of at least 1, got {n}")
    items = _expand(artists)
    if not items:
        raise ValueError("warp has no artist to move")
    pointwise = callable(mapping)
    if pointwise:
        if path != "linear":
            raise ValueError("path='rotation' is for a matrix, not a callable map")
        if about is not None:
            raise ValueError(
                "about= is for a matrix; a callable map has its own center"
            )
    else:
        m = np.asarray(mapping, dtype=float)
        if m.shape != (2, 2) or not np.isfinite(m).all():
            raise ValueError(f"warp needs a finite (2, 2) matrix, got shape {m.shape}")
    targets, groups = [], {}
    for item in items:
        targets.append(_target(item, pointwise, n))
        groups.setdefault(_key(item), []).append(item)

    if pointwise:
        for t in targets:
            t.goal = np.full_like(t.points, np.nan)
            good = np.isfinite(t.points).all(axis=1)
            if good.any():
                image = np.asarray(mapping(t.points[good]), dtype=float)
                if image.shape != (int(good.sum()), 2):
                    raise ValueError(
                        f"the map must return an (N, 2) array for N points, "
                        f"got shape {image.shape}"
                    )
                t.goal[good] = image
    else:
        center = np.zeros(2) if about is None else np.asarray(about, dtype=float)
        moved = _matrix_path(m, path)

    def update(u, level=1.0):
        u = _check_u(u)
        del level
        e = float(_ease(u, ease))
        # Snap to the ends on the progress, not the eased value: an overshoot
        # ease passes 1 before the move is over.
        for t in targets:
            if u <= 0.0:
                t.restore()
            elif pointwise:
                t.build(t.goal if u >= 1.0 else t.points + e * (t.goal - t.points))
            else:
                a = m if u >= 1.0 else moved(e)
                t.build((t.points - center) @ a.T + center)

    owner = next((i.axes for i in items if i.axes is not None), None)
    return PlotResult(ax=owner, artists=groups, update=update)


def warp_grid(
    ax,
    *,
    extent=None,
    spacing=1.0,
    pad=1.0,
    color=None,
    lw=0.6,
    axis_lw=1.4,
    alpha=0.6,
    zorder=1.5,
):
    """Draw a coordinate grid of straight lines to carry through `warp`.

    The grid is two line collections in data coordinates, one for the lines
    at multiples of `spacing` and one for the two coordinate axes through
    the origin, drawn heavier. They are added without touching the axes'
    limits, so the grid cannot change the scale of the figure. By default
    it reaches `pad` times the current span beyond each edge of the view,
    because a map that stretches the plane would otherwise show the grid's
    edge.

    Args:
        ax: The axes to draw on.
        extent: ``(xmin, xmax, ymin, ymax)`` of the grid; default the
            current limits widened by `pad`.
        spacing: Distance between grid lines, positive.
        pad: Fraction of the view span added on each side when `extent` is
            not given.
        color: Color of the grid lines; default the axes' tick color.
        lw: Line width of the grid lines.
        axis_lw: Line width of the two lines through the origin.
        alpha: Opacity of the grid lines (the axes lines are opaque).
        zorder: Drawing order.

    Returns:
        A `PlotResult` on `ax` with ``artists["collection"]`` the list
        ``[grid, axes_lines]`` of `LineCollection`; pass it to `warp`
        directly. There is no `update`.

    Raises:
        ValueError: If `spacing` is not positive or `extent` is empty.
    """
    spacing = float(spacing)
    if not spacing > 0.0:
        raise ValueError(f"warp_grid spacing must be positive, got {spacing}")
    if extent is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        px, py = pad * abs(x1 - x0), pad * abs(y1 - y0)
        extent = (
            min(x0, x1) - px,
            max(x0, x1) + px,
            min(y0, y1) - py,
            max(y0, y1) + py,
        )
    xmin, xmax, ymin, ymax = (float(v) for v in extent)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError(f"warp_grid extent is empty: {extent}")
    color = color if color is not None else ax.xaxis.get_gridlines()[0].get_color()
    xs = np.arange(math.ceil(xmin / spacing), math.floor(xmax / spacing) + 1) * spacing
    ys = np.arange(math.ceil(ymin / spacing), math.floor(ymax / spacing) + 1) * spacing
    grid = [[(x, ymin), (x, ymax)] for x in xs if x != 0.0]
    grid += [[(xmin, y), (xmax, y)] for y in ys if y != 0.0]
    heavy = [[(0.0, ymin), (0.0, ymax)], [(xmin, 0.0), (xmax, 0.0)]]
    lines = LineCollection(
        grid, colors=color, linewidths=lw, alpha=alpha, zorder=zorder
    )
    axes_lines = LineCollection(
        heavy, colors=color, linewidths=axis_lw, zorder=zorder + 0.1
    )
    for coll in (lines, axes_lines):
        ax.add_collection(coll, autolim=False)
    return PlotResult(ax=ax, artists={"collection": [lines, axes_lines]})
