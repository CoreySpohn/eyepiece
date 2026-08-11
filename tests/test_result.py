"""PlotResult/MosaicResult contract."""

import dataclasses

import matplotlib.pyplot as plt
import pytest

from eyepiece._result import ARTIST_KEYS, MosaicResult, PlotResult


def test_plotresult_fig_and_frozen():
    fig, ax = plt.subplots()
    res = PlotResult(ax=ax, artists={})
    assert res.fig is fig
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.ax = None
    plt.close(fig)


def test_mosaicresult_fig():
    fig, axes = plt.subplots(2, 2)
    res = MosaicResult(axes=axes, artists={})
    assert res.fig is fig
    plt.close(fig)


def test_artist_keys_vocabulary():
    assert {"image", "cbar", "line", "title"} <= ARTIST_KEYS
