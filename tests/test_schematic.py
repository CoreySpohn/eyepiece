"""Optical-train rail smoke + highlight."""

import matplotlib.pyplot as plt
import pytest

from eyepiece.schematic import schematic


def test_schematic_draws_and_highlights():
    res = schematic("imager", highlight="focal")
    assert res.ax.patches or res.ax.lines
    lines = res.artists["lines"]
    # "imager" is (pupil, focal); only "focal" is highlighted, so its
    # marker color must differ from the unhighlighted pupil plane's.
    assert lines[0].get_color() != lines[1].get_color()
    plt.close(res.fig)


def test_schematic_unknown_highlight_raises():
    with pytest.raises(ValueError, match="highlight"):
        schematic("imager", highlight="nope")
