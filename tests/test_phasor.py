"""phasor: arrows, chains, resultants, and dials on the complex plane."""

import itertools

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

import eyepiece as ep
from eyepiece import _style
from eyepiece._phasor import HEAD_LENGTH, HEAD_PER_MARKER


def _ends(patch):
    """Start and end of an arrow patch as complex numbers."""
    (x0, y0), (x1, y1) = patch._ends
    return complex(x0, y0), complex(x1, y1)


def _slit_steps(n=24):
    """Equal phasors whose phases wind through one full turn: a closed chain.

    This is the slit at its first dark fringe: the wavelets from the two
    edges differ by one wavelength, so their sum is zero.
    """
    return np.exp(2j * np.pi * (np.arange(n) + 0.5) / n) / n


def test_keys_are_drawn_from_the_vocabulary():
    res = ep.phasor([1 + 1j, -0.5j], show_sum=True, ring=True)
    assert set(res.artists) <= ep.ARTIST_KEYS
    assert set(res.artists) == {"arrow", "lines", "collection", "text"}
    assert all(isinstance(a, FancyArrowPatch) for a in res.artists["arrow"])
    assert isinstance(res.artists["collection"], LineCollection)
    plt.close(res.fig)


def test_ax_none_creates_a_constrained_figure():
    res = ep.phasor([1j])
    assert res.fig.get_layout_engine() is not None
    assert type(res.fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
    assert res.ax.get_aspect() == 1.0
    plt.close(res.fig)


def test_separate_arrows_share_the_origin():
    values = np.array([1 + 2j, -1j, 0.5])
    res = ep.phasor(values, origin=0.25 + 0.25j)
    for patch, value in zip(res.artists["arrow"], values, strict=True):
        start, end = _ends(patch)
        assert start == pytest.approx(0.25 + 0.25j)
        assert end == pytest.approx(0.25 + 0.25j + value)
    plt.close(res.fig)


def test_chain_tips_are_the_cumulative_sum():
    rng = np.random.default_rng(0)
    values = rng.normal(size=10) + 1j * rng.normal(size=10)
    origin = 1.0 - 2.0j
    res = ep.phasor(values, origin=origin, chain=True)
    patches = res.artists["arrow"]
    tips = np.array([_ends(p)[1] for p in patches])
    np.testing.assert_allclose(tips, origin + np.cumsum(values), atol=1e-12)
    for prev, nxt in itertools.pairwise(patches):
        assert _ends(nxt)[0] == pytest.approx(_ends(prev)[1])
    plt.close(res.fig)


def test_a_uniform_full_turn_chain_closes():
    res = ep.phasor(_slit_steps(24), chain=True)
    last_tip = _ends(res.artists["arrow"][-1])[1]
    assert abs(last_tip) < 1e-12
    plt.close(res.fig)


def test_the_resultant_is_drawn_last_to_the_sum():
    values = np.array([1.0, 1j, -0.25 + 0.5j])
    res = ep.phasor(values, chain=True, show_sum=True, origin=0.5j)
    arrows = res.artists["arrow"]
    assert len(arrows) == len(values) + 1
    start, end = _ends(arrows[-1])
    assert start == pytest.approx(0.5j)
    assert end == pytest.approx(0.5j + values.sum())
    assert to_rgba(arrows[-1].get_edgecolor()) == to_rgba(
        matplotlib.rcParams["text.color"]
    )
    assert arrows[-1].get_linewidth() > arrows[0].get_linewidth()
    # Listed last, layered beneath: the whole sits under its parts.
    assert all(arrows[-1].get_zorder() < a.get_zorder() for a in arrows[:-1])
    plt.close(res.fig)


def test_update_moves_the_same_patches_and_keeps_styles():
    res = ep.phasor(
        [1.0, 1j],
        chain=True,
        show_sum=True,
        colors=["red", "blue"],
        linestyles=["-", "--"],
    )
    arrows = list(res.artists["arrow"])
    styles = [(a.get_edgecolor(), a.get_linestyle(), a.get_linewidth()) for a in arrows]
    n_patches = len(res.ax.patches)
    res.update([2j, -1.0], origin=1.0)
    assert res.artists["arrow"] == arrows
    assert len(res.ax.patches) == n_patches
    assert [
        (a.get_edgecolor(), a.get_linestyle(), a.get_linewidth()) for a in arrows
    ] == styles
    assert _ends(arrows[0]) == pytest.approx((1.0, 1.0 + 2j))
    assert _ends(arrows[1]) == pytest.approx((1.0 + 2j, 2j))
    assert _ends(arrows[2]) == pytest.approx((1.0, 2j))
    res.update([1.0, 1.0])
    assert _ends(arrows[2]) == pytest.approx((1.0, 3.0))
    plt.close(res.fig)


def test_update_with_a_different_count_raises():
    res = ep.phasor([1.0, 1j], chain=True)
    with pytest.raises(ValueError, match="3 vectors for 2 arrows"):
        res.update([1.0, 1j, -1.0])
    plt.close(res.fig)


def test_per_arrow_colors_and_linestyles_are_applied():
    res = ep.phasor(
        [1.0, 1j, -1.0],
        colors=["red", "green", "blue"],
        linestyles=["-", "--", ":"],
        widths=[1.0, 2.0, 3.0],
    )
    arrows = res.artists["arrow"]
    for patch, color in zip(arrows, ["red", "green", "blue"], strict=True):
        assert to_rgba(patch.get_edgecolor()) == to_rgba(color)
    styles = [a.get_linestyle() for a in arrows]
    assert styles[0] in ("-", "solid")
    assert styles[1] in ("--", "dashed")
    assert styles[2] in (":", "dotted")
    assert [a.get_linewidth() for a in arrows] == [1.0, 2.0, 3.0]
    plt.close(res.fig)


def test_a_single_color_and_the_default_color():
    res = ep.phasor([1.0, 1j], colors=(1.0, 0.0, 0.0))
    assert all(
        to_rgba(a.get_edgecolor()) == to_rgba("red") for a in res.artists["arrow"]
    )
    plt.close(res.fig)
    res = ep.phasor([1.0, 1j])
    for patch in res.artists["arrow"]:
        assert to_rgba(patch.get_edgecolor()) == to_rgba(_style.color(0))
    plt.close(res.fig)


def test_a_per_arrow_sequence_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match="colors has 2 entries for 3 vectors"):
        ep.phasor([1.0, 1j, -1.0], colors=["red", "blue"])


def test_dial_is_centered_and_leaves_the_parent_in_place():
    fig, ax = plt.subplots(layout="constrained")
    ax.imshow(np.ones((8, 8)), extent=(-4, 4, -4, 4))
    fig.canvas.draw()
    position = ax.get_position().bounds
    limits = ax.get_xlim(), ax.get_ylim()
    res = ep.phasor([0.5 + 0.5j], ax=ax, at=(1.5, -1.0), size=2.0, ring=True)
    fig.canvas.draw()
    assert res.ax is not ax
    assert res.ax in ax.child_axes
    assert ax.get_position().bounds == pytest.approx(position)
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    box = res.ax.get_window_extent()
    center = ax.transData.inverted().transform(
        [0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)]
    )
    np.testing.assert_allclose(center, (1.5, -1.0), atol=1e-6)
    width = np.diff(ax.transData.inverted().transform([(box.x0, 0), (box.x1, 0)])[:, 0])
    assert width[0] == pytest.approx(2.0, rel=1e-6)
    assert "text" not in res.artists
    plt.close(fig)


def test_dial_needs_a_parent():
    with pytest.raises(ValueError, match="pass ax"):
        ep.phasor([1j], at=(0.0, 0.0))


def test_small_heads_thin_their_default_shafts():
    res = ep.phasor([1.0, 1j, -1.0], head_scale=[1.0, 0.5, 0.0])
    lw = matplotlib.rcParams["lines.linewidth"]
    widths = [a.get_linewidth() for a in res.artists["arrow"]]
    assert widths == pytest.approx([lw, 0.7 * lw, 0.4 * lw])
    plt.close(res.fig)


def test_head_size_scales_linearly_with_the_marker_size():
    scales = []
    for ms in (6.0, 12.0):
        with matplotlib.rc_context({"lines.markersize": ms}):
            res = ep.phasor([1.0], head_scale=0.5, lim=1.5)
        scales.append(res.artists["arrow"][0].get_mutation_scale())
        plt.close(res.fig)
    assert scales[0] == pytest.approx(HEAD_PER_MARKER * 6.0 * 0.5)
    assert scales[1] == pytest.approx(2.0 * scales[0])


def test_a_short_arrow_shrinks_its_head_to_fit():
    res = ep.phasor([1.0, 1e-3], lim=1.5)
    long_arrow, short_arrow = res.artists["arrow"]
    nominal = HEAD_PER_MARKER * matplotlib.rcParams["lines.markersize"]
    assert long_arrow.get_mutation_scale() == pytest.approx(nominal)
    assert short_arrow.get_mutation_scale() < nominal
    a, b = short_arrow.get_transform().transform([(0.0, 0.0), (1e-3, 0.0)])
    length_pt = (b[0] - a[0]) * 72.0 / res.fig.dpi
    assert HEAD_LENGTH * short_arrow.get_mutation_scale() <= length_pt + 1e-9
    res.fig.canvas.draw()
    plt.close(res.fig)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_style_switch_between_calls_is_honored(mode):
    other = "dark" if mode == "light" else "light"
    hwostyle.use(other)
    plt.close(ep.phasor([1.0]).fig)
    hwostyle.use(mode)
    res = ep.phasor([1.0, 1j], show_sum=True)
    *pieces, total = res.artists["arrow"]
    for patch in pieces:
        assert to_rgba(patch.get_edgecolor()) == to_rgba(hwostyle.palette[0])
    assert to_rgba(total.get_edgecolor()) == to_rgba(matplotlib.rcParams["text.color"])
    plt.close(res.fig)


def test_ring_is_colored_by_the_phase_colormap():
    res = ep.phasor([1.0], ring=True, ring_radius=2.0)
    ring = res.artists["collection"]
    segments = ring.get_segments()
    mid = np.array([0.5 * (s[0] + s[1]) for s in segments])
    np.testing.assert_allclose(np.hypot(mid[:, 0], mid[:, 1]), 2.0, rtol=1e-3)
    phi = np.arctan2(mid[:, 1], mid[:, 0])
    expected = _style.cmap("phase")((phi + np.pi) / (2.0 * np.pi))
    np.testing.assert_allclose(ring.get_colors(), expected, atol=1e-6)
    plt.close(res.fig)


def test_limits_symmetric_explicit_and_automatic():
    res = ep.phasor([1j], lim=2.0)
    assert res.ax.get_xlim() == (-2.0, 2.0)
    assert res.ax.get_ylim() == (-2.0, 2.0)
    plt.close(res.fig)
    res = ep.phasor([1j], lim=(-1, 3, -2, 2))
    assert res.ax.get_xlim() == (-1.0, 3.0)
    plt.close(res.fig)
    res = ep.phasor(_slit_steps(), chain=True, cross=False)
    tips = np.cumsum(_slit_steps())
    x0, x1 = res.ax.get_xlim()
    y0, y1 = res.ax.get_ylim()
    assert x0 < tips.real.min() and x1 > tips.real.max()
    assert y0 < tips.imag.min() and y1 > tips.imag.max()
    assert x1 - x0 == pytest.approx(y1 - y0)
    plt.close(res.fig)


def test_axis_labels_and_routed_kwargs():
    res = ep.phasor(
        [1j],
        axis_labels=("Re E", "Im E"),
        arrow_kw={"zorder": 9, "gid": "field"},
        line_kw={"color": "red"},
    )
    assert [t.get_text().strip() for t in res.artists["text"]] == ["Re E", "Im E"]
    assert res.artists["arrow"][0].get_zorder() == 9
    assert res.artists["arrow"][0].get_gid() == "field"
    assert all(
        to_rgba(line.get_color()) == to_rgba("red") for line in res.artists["lines"]
    )
    plt.close(res.fig)
    res = ep.phasor([1j], axis_labels=False)
    assert "text" not in res.artists
    plt.close(res.fig)
    res = ep.phasor([1j], cross=False)
    assert "lines" not in res.artists and "text" not in res.artists
    plt.close(res.fig)


def test_a_closed_chain_draws_without_error_and_skips_the_zero_resultant():
    res = ep.phasor(_slit_steps(200), chain=True, show_sum=True)
    total = res.artists["arrow"][-1]
    assert abs(_ends(total)[1]) < 1e-12
    assert total._length_pt() < 0.05
    res.fig.canvas.draw()
    plt.close(res.fig)
