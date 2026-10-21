"""Axes furniture can be quiet without changing global spine policy."""

import hwostyle
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba

import eyepiece as ep


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_quiet_spines_hides_only_unkept_and_resolves_neutral(mode):
    hwostyle.use(mode)
    fig, ax = plt.subplots()
    before = dict(mpl.rcParams)
    ax.spines["left"].set_color("C1")
    ep.quiet_spines(ax)
    assert {key for key, spine in ax.spines.items() if spine.get_visible()} == {
        "left",
        "bottom",
    }
    assert to_rgba(ax.spines["left"].get_edgecolor()) == to_rgba("C1")
    ep.quiet_spines(ax, keep=("top", "right"), level=0.4)
    assert {key for key, spine in ax.spines.items() if spine.get_visible()} == {
        "top",
        "right",
    }
    for key in ("top", "right"):
        assert ax.spines[key].get_edgecolor() == to_rgba(ep.neutral(0.4))
    assert dict(mpl.rcParams) == before
    plt.close(fig)


def test_quiet_spines_rejects_unknown_spine_before_mutating():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="spine"):
        ep.quiet_spines(ax, keep=("left", "typo"))
    assert all(spine.get_visible() for spine in ax.spines.values())
    plt.close(fig)
