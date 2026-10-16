"""Cubic outline alignment: the pieces a morph is built from."""

import numpy as np
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

from eyepiece import _bezier


def _glyph(char):
    prop = FontProperties(family="DejaVu Sans")
    return _bezier.to_cubic(TextPath((0, 0), char, size=10, prop=prop))


def _sample(contour, per_curve=64):
    """Points along a (k, 4, 2) contour, `per_curve` per curve."""
    t = np.linspace(0.0, 1.0, per_curve, endpoint=False)[None, :, None]
    c = contour[:, None]
    pts = (
        (1 - t) ** 3 * c[:, :, 0]
        + 3 * (1 - t) ** 2 * t * c[:, :, 1]
        + 3 * (1 - t) * t**2 * c[:, :, 2]
        + t**3 * c[:, :, 3]
    )
    return pts.reshape(-1, 2)


def _area(contour):
    x = _sample(contour, 80)
    return 0.5 * np.sum(x[:, 0] * np.roll(x[:, 1], -1) - np.roll(x[:, 0], -1) * x[:, 1])


def _is_null(contour):
    return bool(np.all(contour == contour[0, 0]))


def _distance_to(points, curve_points):
    d = np.linalg.norm(points[:, None] - curve_points[None], axis=-1)
    return d.min(axis=1).max()


def test_to_cubic_round_trips_lines_quadratics_and_cubics():
    p = np.array(
        [
            [0.0, 0.0],
            [3.0, 1.0],  # line
            [4.0, 3.0],
            [6.0, 2.0],  # quadratic
            [7.0, 0.0],
            [8.0, -2.0],
            [9.5, 1.0],  # cubic
        ]
    )
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]
    (contour,) = _bezier.to_cubic(Path(p, codes))
    assert contour.shape == (3, 4, 2)
    t = np.linspace(0.0, 1.0, 41)[:, None]
    line = (1 - t) * p[0] + t * p[1]
    quad = (1 - t) ** 2 * p[1] + 2 * (1 - t) * t * p[2] + t**2 * p[3]
    cubic = (
        (1 - t) ** 3 * p[3]
        + 3 * (1 - t) ** 2 * t * p[4]
        + 3 * (1 - t) * t**2 * p[5]
        + t**3 * p[6]
    )
    for curve, original in zip(contour, (line, quad, cubic), strict=True):
        tt = t[:, :, None]
        c = curve[None]
        got = (
            (1 - tt) ** 3 * c[:, 0]
            + 3 * (1 - tt) ** 2 * tt * c[:, 1]
            + 3 * (1 - tt) * tt**2 * c[:, 2]
            + tt**3 * c[:, 3]
        )[:, 0]
        assert np.abs(got - original).max() < 1e-9


def test_to_cubic_closes_with_a_line_only_when_the_contour_is_open():
    square = Path(
        [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)],
        [Path.MOVETO, *[Path.LINETO] * 3, Path.CLOSEPOLY],
    )
    (contour,) = _bezier.to_cubic(square)
    assert len(contour) == 4  # the closing line is added, ending at the start
    assert np.array_equal(contour[-1, -1], contour[0, 0])
    already = Path(
        [(0, 0), (2, 0), (2, 2), (0, 0), (0, 0)],
        [Path.MOVETO, *[Path.LINETO] * 3, Path.CLOSEPOLY],
    )
    assert len(_bezier.to_cubic(already)[0]) == 3


def test_a_lone_point_is_a_null_contour_and_an_empty_path_has_none():
    lone = _bezier.to_cubic(Path([(1.0, 2.0)], [Path.MOVETO]))
    assert len(lone) == 1 and _is_null(lone[0]) and lone[0].shape == (1, 4, 2)
    assert _bezier.to_cubic(Path(np.zeros((0, 2)))) == []


def test_zero_against_one_pads_the_missing_hole_at_the_last_point():
    zero, one = _glyph("0"), _glyph("1")
    assert (len(zero), len(one)) == (2, 1)
    a, b = _bezier.align(zero, one)
    assert len(a) == len(b) == 2
    assert not _is_null(a[0]) and not _is_null(a[1])
    assert not _is_null(b[0]) and _is_null(b[1])
    # The null contour sits at the last point of the last contour of "1".
    assert np.array_equal(b[1][0, 0], one[-1][-1, -1])
    for ca, cb in zip(a, b, strict=True):
        assert len(ca) == len(cb)


def test_eight_against_zero_pads_the_third_contour_of_zero():
    eight, zero = _glyph("8"), _glyph("0")
    assert (len(eight), len(zero)) == (3, 2)
    a, b = _bezier.align(eight, zero)
    assert len(a) == len(b) == 3
    assert [_is_null(c) for c in a] == [False, False, False]
    assert [_is_null(c) for c in b] == [False, False, True]
    assert np.array_equal(b[2][0, 0], zero[-1][-1, -1])


def test_the_shorter_contour_is_split_until_the_curve_counts_match():
    for first, second in (("0", "1"), ("8", "0"), ("g", "i"), ("B", "o")):
        a, b = _bezier.align(_glyph(first), _glyph(second))
        assert len(a) == len(b)
        for ca, cb in zip(a, b, strict=True):
            assert len(ca) == len(cb)


def test_splitting_traces_the_original_outline():
    original = _glyph("g")
    other = _glyph("o")
    aligned, _ = _bezier.align(original, other)
    for before, after in zip(original, aligned, strict=True):
        assert len(after) >= len(before)
        dense = _sample(before, 4000)
        assert _distance_to(_sample(after, 200), dense) < 5e-3
        assert _distance_to(dense[::50], _sample(after, 600)) < 5e-3
        assert np.allclose(after[0, 0], before[0, 0], atol=1e-12)
        assert np.allclose(after[-1, -1], before[-1, -1], atol=1e-12)
        # Pieces join: each curve ends where the next begins.
        assert np.array_equal(after[:-1, -1], after[1:, 0])
        assert abs(_area(after) - _area(before)) < 1e-6 * abs(_area(before))


def test_an_empty_outline_becomes_one_null_contour_at_the_other_center():
    zero = _glyph("0")
    a, b = _bezier.align([], zero)
    box = np.concatenate([c.reshape(-1, 2) for c in zero])
    center = 0.5 * (box.min(0) + box.max(0))
    assert len(a) == len(b) == 2
    assert all(_is_null(c) for c in a)
    assert np.allclose(a[0][0, 0], center)
    a, _ = _bezier.align([], zero, point=(3.0, 4.0))
    assert np.array_equal(a[0][0, 0], [3.0, 4.0])


def test_trailing_curves_that_add_nothing_are_dropped():
    path = Path(
        [(0, 0), (4, 0), (4, 4), (4, 4), (4, 4)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO],
    )
    triangle = Path(
        [(0, 0), (4, 0), (2, 3), (0, 0)],
        [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO],
    )
    contour = _bezier.to_cubic(path)
    assert len(contour[0]) == 4
    a, b = _bezier.align(contour, _bezier.to_cubic(triangle))
    assert len(a[0]) == len(b[0]) == 3


def test_twelve_glyphs_into_four_stack_copies_of_the_four():
    sources = [_glyph(c) for c in "Pr(E|H)Pr(H)"[:12]]
    targets = [_glyph(c) for c in "0.24"]
    al = _bezier.align_groups(sources, targets)
    assert len(al.a) == len(al.b) == 12
    assert al.index_a == tuple(range(12)) and not any(al.copy_a)
    # Each target is repeated three times, the first the original.
    assert al.index_b == (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    assert al.copy_b == (False, True, True) * 4
    # Every copy traces its original outline exactly.
    for k, idx in enumerate(al.index_b):
        pts = np.concatenate([_sample(c, 200) for c in al.b[k]])
        ref = np.concatenate([_sample(c, 4000) for c in targets[idx]])
        assert _distance_to(pts, ref) < 5e-3
        back = np.concatenate([_sample(c, 800) for c in al.b[k]])
        assert _distance_to(ref[::200], back) < 5e-3
    for pair_a, pair_b in zip(al.a, al.b, strict=True):
        assert [len(c) for c in pair_a] == [len(c) for c in pair_b]


def test_four_glyphs_into_twelve_stack_copies_of_the_four():
    al = _bezier.align_groups([_glyph(c) for c in "0.24"], [_glyph("a")] * 12)
    assert al.index_a == (0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3)
    assert al.copy_a == (False, True, True) * 4
    assert al.index_b == tuple(range(12)) and not any(al.copy_b)


def test_uneven_counts_spread_the_copies_as_manim_does():
    al = _bezier.align_groups(
        [_glyph("a"), _glyph("b"), _glyph("c")], [_glyph("1")] * 5
    )
    # arange(5) * 3 // 5 is [0, 0, 1, 1, 2]
    assert al.index_a == (0, 0, 1, 1, 2)
    assert al.copy_a == (False, True, False, True, False)


def test_three_into_none_collapse_to_null_points():
    sources = [_glyph(c) for c in "abc"]
    al = _bezier.align_groups(sources, [])
    assert len(al.a) == len(al.b) == 3
    assert al.index_b == (-1, -1, -1) and all(al.copy_b)
    box = np.concatenate([c.reshape(-1, 2) for s in sources for c in s])
    center = 0.5 * (box.min(0) + box.max(0))
    for pair in al.b:
        assert all(_is_null(c) for c in pair)
        assert np.allclose(pair[0][0, 0], center)
    al = _bezier.align_groups([], sources, point=(1.0, 2.0))
    assert al.index_a == (-1, -1, -1)
    assert np.array_equal(al.a[0][0][0, 0], [1.0, 2.0])


def test_flatten_gives_matching_vertices_and_codes_for_aligned_pairs():
    a, b = _bezier.align(_glyph("8"), _glyph("1"))
    va, ca = _bezier.flatten(a)
    vb, cb = _bezier.flatten(b)
    assert va.shape == vb.shape and np.array_equal(ca, cb)
    assert np.isfinite(va).all() and np.isfinite(vb).all()
    assert ca[0] == Path.MOVETO and (ca == Path.MOVETO).sum() == 3


def test_alignment_matches_manim_when_it_is_installed():
    manim = pytest.importorskip("manim")

    def to_vmobject(contours):
        vm = manim.VMobject()
        vm.set_points(
            np.vstack(
                [np.hstack([curve, np.zeros((4, 1))]) for c in contours for curve in c]
            )
        )
        return vm

    for first, second in (("0", "1"), ("8", "0"), ("1", "8"), ("g", "i"), ("B", "o")):
        ca, cb = _glyph(first), _glyph(second)
        ma, mb = to_vmobject(ca), to_vmobject(cb)
        ma.align_points(mb)
        a, b = _bezier.align(ca, cb)
        got_a = np.vstack([c.reshape(-1, 2) for c in a])
        got_b = np.vstack([c.reshape(-1, 2) for c in b])
        assert np.allclose(got_a, ma.points[:, :2], atol=1e-12)
        assert np.allclose(got_b, mb.points[:, :2], atol=1e-12)
