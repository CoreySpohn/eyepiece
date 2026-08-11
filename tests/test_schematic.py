"""Optical-train rail smoke + highlight."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
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


def _envelope_facecolor():
    res = schematic("coronagraph")
    facecolor = tuple(np.ravel(res.artists["fill"].get_facecolor()))
    plt.close(res.fig)
    return facecolor


def test_schematic_neutrals_follow_the_mode():
    hwostyle.use("dark")
    dark_envelope = _envelope_facecolor()
    with hwostyle.light():
        light_envelope = _envelope_facecolor()
    assert dark_envelope != light_envelope
