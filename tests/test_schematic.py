"""Optical-train rail smoke + highlight."""

import matplotlib.pyplot as plt

from eyepiece.schematic import schematic


def test_schematic_draws_and_highlights():
    res = schematic("imager", highlight="focal")
    assert res.ax.patches or res.ax.lines
    plt.close(res.fig)
