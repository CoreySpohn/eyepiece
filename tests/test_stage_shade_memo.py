"""The memoized single-color shade equals the array computation bit for bit."""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array

from eyepiece import attention


def a_look(background):
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1])
    look = attention._Look(line, background)
    plt.close(fig)
    return look


COLORS = ["C0", "C3", "red", "#1f77b480", (0.2, 0.4, 0.6), (0.9, 0.1, 0.3, 0.5)]
LEVELS = [0.0, 0.1, 1 / 3, 0.5, 0.97, 1.0, np.float32(0.25), np.float64(0.7)]


def test_single_colors_match_the_array_path_exactly():
    for background in ("black", "#0b1020", (1.0, 1.0, 1.0)):
        look = a_look(background)
        for color, presence, contrast, own in itertools.product(
            COLORS, LEVELS, LEVELS, (False, True)
        ):
            memo = look._shade(color, presence, contrast, own)
            array = look._shade(
                np.asarray(to_rgba_array(color)), presence, contrast, own
            )
            assert memo == tuple(array[0]), (background, color, presence, contrast)
            assert all(type(v) is type(array[0][0]) for v in memo)


def test_none_lists_and_arrays_keep_their_original_paths():
    look = a_look("black")
    assert look._shade("none", 0.5, 0.5, False) == "none"
    pair = [(1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
    # A plain list keeps the original result: the first color, shaded.
    first = look._shade(np.asarray(to_rgba_array(pair)), 0.5, 0.5, False)[0]
    assert look._shade(pair, 0.5, 0.5, False) == tuple(first)
    grid = np.asarray(to_rgba_array(pair))
    assert look._shade(grid, 0.5, 0.5, False).shape == (2, 4)


def test_cycle_colors_resolve_at_call_time():
    look = a_look("black")
    first = look._shade("C0", 1.0, 0.5, False)
    with plt.rc_context({"axes.prop_cycle": plt.cycler(color=["#00ff00"])}):
        assert look._shade("C0", 1.0, 0.5, False) != first
    assert look._shade("C0", 1.0, 0.5, False) == first


def bits(values):
    return [np.float64(v).tobytes() for v in values]


def test_signed_zero_levels_and_colors_keep_their_bits():
    look = a_look("black")
    for warm, probe in (
        (("red", 0.0, 0.5, False), ("red", -0.0, 0.5, False)),
        (
            ((0.1, 0.2, 0.3, 0.0), 0.5, 0.5, False),
            ((0.1, 0.2, 0.3, -0.0), 0.5, 0.5, False),
        ),
    ):
        look._shade(*warm)
        color, presence, contrast, own = probe
        grid = np.asarray(to_rgba_array(color))
        expected = look._shade(grid, presence, contrast, own)
        assert bits(look._shade(*probe)) == bits(expected[0])


def test_unusual_levels_and_pairs_take_the_original_path():
    from decimal import Decimal

    import pytest

    look = a_look("black")
    look._shade("red", 0.5, 0.5, False)  # warm the memo for the equal float key
    with pytest.raises(TypeError):
        look._shade("red", Decimal("0.5"), 0.5, False)
    with pytest.raises(IndexError):
        look._shade(("none", 0.5), 0.5, 0.5, False)
    inf = look._shade("red", 0.5, float("inf"), False)
    assert np.isnan(inf[1]) or np.isinf(inf[0])
