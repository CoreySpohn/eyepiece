"""Cubic Bezier outlines aligned so that one can morph into another.

A morph needs two outlines with the same structure: the same number of
contours, and in each pair of contours the same number of curves. The
alignment here follows ManimCE's point alignment (`VMobject.align_points`,
`VMobject.insert_n_curves_to_point_list` and `Mobject.add_n_more_submobjects`),
reimplemented in NumPy on matplotlib paths:

- every segment becomes a cubic, a line with its handles at the thirds and a
  quadratic raised in degree;
- contours pair by index, and the side with fewer is padded with null
  contours, each one point repeated, at the last point of its last contour, so
  a hole grows from (or shrinks to) a point;
- in each pair the side with fewer curves is split until the counts match,
  every curve cut evenly in its parameter, with the cuts spread by repeating
  curve indices;
- of two lists of outlines the shorter is padded with copies of its own
  outlines, spread the same way and stacked on the originals, and a copy
  starts (or ends) invisible.

An outline is a list of contours; a contour is an array of shape ``(k, 4, 2)``
of ``k`` cubic curves, four control points each. A contour is closed when its
last point meets its first; an open one is left open.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

_CLOSE_TOL = 1e-9


def _line(p0, p1):
    return np.array([p0, p0 + (p1 - p0) / 3.0, p0 + 2.0 * (p1 - p0) / 3.0, p1])


def _quad(p0, p1, p2):
    return np.array([p0, p0 + 2.0 / 3.0 * (p1 - p0), p2 + 2.0 / 3.0 * (p1 - p2), p2])


def _null(point, count=1):
    return np.tile(np.asarray(point, dtype=float), (count, 4, 1))


def to_cubic(path):
    """The contours of a matplotlib `Path` as cubic Beziers.

    Lines and quadratics are raised to cubics exactly, so the curve is the
    same set of points. A closing segment is a line back to the start of its
    contour, added only when the contour does not already end there. A contour
    that is a lone point becomes one null curve.

    Args:
        path: A `matplotlib.path.Path` (or any object with ``vertices`` and
            ``codes``). ``codes=None`` is a polyline through the vertices.

    Returns:
        A list of arrays of shape ``(k, 4, 2)``, one per contour.
    """
    vertices = np.asarray(path.vertices, dtype=float)
    codes = path.codes
    if codes is None:
        codes = np.full(len(vertices), Path.LINETO)
        codes[:1] = Path.MOVETO
    scale = float(np.ptp(vertices, axis=0).max()) if len(vertices) else 0.0
    tol = _CLOSE_TOL * max(1.0, scale)
    contours, curves = [], []
    start = last = None

    def finish():
        nonlocal curves
        if curves:
            contours.append(np.array(curves))
        elif start is not None:
            contours.append(_null(start))
        curves = []

    i = 0
    n = len(vertices)
    while i < n:
        code = codes[i]
        if code == Path.STOP:
            break
        if code == Path.MOVETO:
            finish()
            start = last = vertices[i]
            i += 1
        elif code == Path.LINETO:
            curves.append(_line(last, vertices[i]))
            last = vertices[i]
            i += 1
        elif code == Path.CURVE3:
            curves.append(_quad(last, vertices[i], vertices[i + 1]))
            last = vertices[i + 1]
            i += 2
        elif code == Path.CURVE4:
            curves.append(
                np.array([last, vertices[i], vertices[i + 1], vertices[i + 2]])
            )
            last = vertices[i + 2]
            i += 3
        else:  # CLOSEPOLY
            if start is not None and np.hypot(*(last - start)) > tol:
                curves.append(_line(last, start))
            last = start
            i += 1
    finish()
    return contours


def _split(curve, t):
    """The two cubics that a cubic is cut into at parameter `t` (de Casteljau)."""
    p0, p1, p2, p3 = curve
    a = p0 + t * (p1 - p0)
    b = p1 + t * (p2 - p1)
    c = p2 + t * (p3 - p2)
    d = a + t * (b - a)
    e = b + t * (c - b)
    f = d + t * (e - d)
    return np.array([p0, a, d, f]), np.array([f, e, c, p3])


def _subdivide(curve, parts):
    """`parts` cubics that trace `curve`, each over an equal span of its parameter."""
    pieces = []
    rest = curve
    for i in range(parts - 1):
        first, rest = _split(rest, 1.0 / (parts - i))
        pieces.append(first)
    pieces.append(rest)
    return np.array(pieces)


def _spread(count, target):
    """How many pieces each of `count` items becomes when `target` are wanted.

    Item indices repeat as ``(arange(target) * count) // target``, so the
    extra pieces fall evenly along the list.
    """
    repeat = (np.arange(target) * count) // target
    return np.bincount(repeat, minlength=count)


def _remap(curves, target):
    """`curves` cut up so that exactly `target` curves trace the same contour."""
    count = len(curves)
    if target == count:
        return curves
    return np.concatenate(
        [
            _subdivide(c, int(f))
            for c, f in zip(curves, _spread(count, target), strict=True)
        ]
    )


def _trim(contour):
    """Drop trailing curves that sit on the point before them and add nothing."""
    while len(contour) > 1:
        last = contour[-1]
        if np.all(np.abs(last - contour[-2][-1]) <= _CLOSE_TOL):
            contour = contour[:-1]
        else:
            break
    return contour


def _bounds_center(outline):
    if not outline:
        return None
    points = np.concatenate([c.reshape(-1, 2) for c in outline])
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def align(path_a, path_b, point=None):
    """Give two outlines the same contours, with the same curves in each pair.

    As Manim's `VMobject.align_points`: contours pair by index, the shorter
    list is padded with null contours at the last point of its last contour,
    and the shorter contour of each pair is split evenly until the curve
    counts match. An outline with no contours at all becomes one null contour
    at `point`.

    Args:
        path_a: Contours of the first outline, from `to_cubic`.
        path_b: Contours of the second.
        point: Where an empty outline sits, an ``(x, y)``. Default: the center
            of the other outline's bounding box.

    Returns:
        ``(path_a, path_b)`` with equal contour counts and, contour by
        contour, equal curve counts. Neither input is changed.
    """
    sides = []
    for own, other in ((path_a, path_b), (path_b, path_a)):
        if own:
            sides.append([_trim(c) for c in own])
        else:
            where = point if point is not None else _bounds_center(other)
            sides.append([_null(np.zeros(2) if where is None else where)])
    first, second = sides
    total = max(len(first), len(second))
    for side in sides:
        while len(side) < total:
            side.append(_null(side[-1][-1, -1]))
    out_a, out_b = [], []
    for ca, cb in zip(first, second, strict=True):
        count = max(len(ca), len(cb))
        out_a.append(_remap(ca, count))
        out_b.append(_remap(cb, count))
    return out_a, out_b


@dataclass(frozen=True)
class Aligned:
    """Two lists of outlines paired one to one, each pair aligned.

    Attributes:
        a: Outlines of the first side, as lists of ``(k, 4, 2)`` contours.
        b: Outlines of the second side, pair for pair with `a`.
        index_a: For each pair, the index of the first side's original outline,
            or -1 for a null point.
        index_b: The same for the second side.
        copy_a: True where the first side's outline is a stacked copy (or a
            null point) that begins invisible; the original is the first of its
            run.
        copy_b: The same for the second side, where the copy ends invisible.
    """

    a: list
    b: list
    index_a: tuple
    index_b: tuple
    copy_a: tuple
    copy_b: tuple


def _pad_group(outlines, target, point):
    """`outlines` stretched to `target` entries by stacked copies.

    As Manim's `Mobject.add_n_more_submobjects`: the extra entries repeat
    outline indices evenly along the list, each outline followed by its copies.
    An empty list becomes `target` null points.

    Returns:
        ``(outlines, index, copy)``.
    """
    count = len(outlines)
    if count == 0:
        return [[_null(point)] for _ in range(target)], [-1] * target, [True] * target
    out, index, copy = [], [], []
    for k, factor in enumerate(_spread(count, target)):
        for rep in range(int(factor)):
            out.append(outlines[k])
            index.append(k)
            copy.append(rep > 0)
    return out, index, copy


def align_groups(paths_a, paths_b, point=None):
    """Pair two lists of outlines, padding the shorter with copies, and align each pair.

    As Manim's `Mobject.align_submobjects`: the side with fewer outlines gets
    stacked copies of its own, spread evenly along the list, so every outline
    on the larger side has a partner; a side with none gets null points. The
    pairs then go in reading order, the order of the lists.

    Args:
        paths_a: Outlines of the first side, each a list of contours.
        paths_b: Outlines of the second side.
        point: Where null points sit, an ``(x, y)``. Default: the center of the
            bounding box of the other side.

    Returns:
        An `Aligned`.
    """
    target = max(len(paths_a), len(paths_b))
    if point is None:
        boxes = [
            _bounds_center([c for outline in side for c in outline])
            for side in (paths_b, paths_a)
        ]
        point = next((b for b in boxes if b is not None), np.zeros(2))
    pad_a, index_a, copy_a = _pad_group(paths_a, target, point)
    pad_b, index_b, copy_b = _pad_group(paths_b, target, point)
    out_a, out_b = [], []
    for pa, pb in zip(pad_a, pad_b, strict=True):
        aa, bb = align(pa, pb, point)
        out_a.append(aa)
        out_b.append(bb)
    return Aligned(
        out_a, out_b, tuple(index_a), tuple(index_b), tuple(copy_a), tuple(copy_b)
    )


def flatten(outline):
    """One outline as drawable vertices and codes.

    Each contour is a ``MOVETO`` and then three ``CURVE4`` points per curve.
    Two aligned outlines flatten to arrays of the same shape and the same
    codes, so their vertices can be mixed point for point.
    """
    vertices, codes = [], []
    for contour in outline:
        vertices.append(contour[0, 0][None])
        codes.append([Path.MOVETO])
        vertices.append(contour[:, 1:].reshape(-1, 2))
        codes.append([Path.CURVE4] * (3 * len(contour)))
    return np.concatenate(vertices), np.concatenate(codes).astype(Path.code_type)
