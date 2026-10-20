"""A head reveal staggered by per-line offsets, such as the lines' coordinates."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _lines(n=3):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    return fig, ax.plot([0.1, 0.9], np.full((2, n), 0.5) * [[1], [1]], color="black")


def _fractions(move, u):
    move.update(u)
    return [c._fraction for c in move.contract.moves]


def test_offsets_start_each_line_at_its_own_place_in_the_beat():
    _, lines = _lines()
    move = ep.head_draw(lines, lag=[0.0, 0.5, 2.0], ease="linear")
    # span = 1 / (1 + 2): the lines start at 0, 1/6 and 2/3 of the beat.
    assert _fractions(move, 0.25) == pytest.approx([0.75, 0.25, 0.0])
    assert _fractions(move, 5 / 6) == pytest.approx([1.0, 1.0, 0.5])


def test_offsets_that_equal_the_scalar_lag_draw_the_same():
    _, lines = _lines()
    scalar = ep.head_draw(lines, lag=0.4, ease="linear")
    listed = ep.head_draw(lines, lag=[0.0, 0.4, 0.8], ease="linear")
    for u in np.linspace(0.0, 1.0, 11):
        assert _fractions(listed, u) == pytest.approx(_fractions(scalar, u))


def test_offsets_follow_the_lines_not_their_order():
    _, lines = _lines()
    move = ep.head_draw(lines, lag=[2.0, 0.0, 1.0], ease="linear")
    # span = 1/3; the middle line goes first, the first one last.
    assert _fractions(move, 0.2) == pytest.approx([0.0, 0.6, 0.0])


def test_only_the_spacing_of_the_offsets_matters():
    _, lines = _lines()
    a = ep.head_draw(lines, lag=[0.0, 0.5, 2.0], ease="linear")
    b = ep.head_draw(lines, lag=[10.0, 10.5, 12.0], ease="linear")
    for u in np.linspace(0.0, 1.0, 9):
        assert _fractions(b, u) == pytest.approx(_fractions(a, u))


def test_equal_offsets_draw_together():
    _, lines = _lines()
    move = ep.head_draw(lines, lag=[3.0, 3.0, 3.0], ease="linear")
    assert _fractions(move, 0.4) == pytest.approx([0.4, 0.4, 0.4])


@pytest.mark.parametrize("bad", [[0.0, 1.0], [0.0, -1.0, 1.0], [0.0, np.nan, 1.0]])
def test_bad_offsets_raise(bad):
    _, lines = _lines()
    with pytest.raises(ValueError, match="lag"):
        ep.head_draw(lines, lag=bad)
