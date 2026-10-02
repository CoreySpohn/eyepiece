"""phasor: arrows, chains, resultants, and dials on the complex plane."""

import io
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


def _centered_steps(turns, n=24):
    """Slit wavelets with phases centered on zero, winding `turns` full turns."""
    k = np.arange(n) + 0.5
    return np.exp(1j * (2.0 * np.pi * turns * k / n - np.pi * turns)) / n


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_a_closed_chain_raises_no_warning_built_or_updated():
    """A vanishing resultant never reaches the arrowhead geometry (a 0/0 there).

    Matplotlib asks for an arrow's path when it is added and for its extents,
    not only when drawing, so construction, layout, saving, and an update
    into and out of the closed state are all exercised.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), layout="constrained")
    results = [
        ep.phasor(
            _centered_steps(turns),
            ax=ax,
            chain=True,
            show_sum=True,
            head_scale=0.3,
            lim=0.65,
        )
        for ax, turns in zip(axes, (0.0, 0.5, 1.0), strict=True)
    ]
    fig.savefig(io.BytesIO(), format="png", dpi=110)
    res = results[0]
    total = res.artists["arrow"][-1]
    for turns in (1.0, 0.5, 1.0, 0.0):
        res.update(_centered_steps(turns))
        fig.savefig(io.BytesIO(), format="png", dpi=110)
        total.get_window_extent()
    assert total.get_visible()
    assert total.get_window_extent().width > 0.0
    plt.close(fig)


_PIN_RC = {"lines.linewidth": 1.5, "lines.markersize": 6.0}


def test_default_geometry_matches_the_released_layout():
    """The default path, pinned as numbers, so new options cannot move it."""
    with matplotlib.rc_context(_PIN_RC):
        res = ep.phasor([1 + 2j, -1j], show_sum=True, head_scale=[1.0, 0.5], ring=True)
    arrows = res.artists["arrow"]
    assert [_ends(a) for a in arrows] == [(0j, 1 + 2j), (0j, -1j), (0j, 1 + 1j)]
    assert [a.get_linewidth() for a in arrows] == pytest.approx([1.5, 1.05, 2.25])
    assert [a.get_zorder() for a in arrows] == [5, 5, 4]
    nominal = [super(type(a), a).get_mutation_scale() for a in arrows]
    assert nominal == pytest.approx([9.6, 4.8, 9.6])
    # Square limits around every point, 0, and the ring's bounding box.
    assert res.ax.get_xlim() == pytest.approx((-1.74, 1.74))
    assert res.ax.get_ylim() == pytest.approx((-1.24, 2.24))
    assert [ln.get_linewidth() for ln in res.artists["lines"]] == pytest.approx(
        [1.05, 1.05]
    )
    assert [ln.get_zorder() for ln in res.artists["lines"]] == [1, 1]
    small = matplotlib.font_manager.FontProperties(size="small").get_size_in_points()
    texts = res.artists["text"]
    assert [t.get_text() for t in texts] == [" Re", " Im"]
    assert [t.get_position() for t in texts] == [(1.0, 0.0), (0.0, 1.0)]
    assert [(t.get_ha(), t.get_va()) for t in texts] == [
        ("right", "bottom"),
        ("left", "top"),
    ]
    assert all(t.get_fontsize() == pytest.approx(small) for t in texts)
    assert all(t.get_zorder() == 1 and t.get_bbox_patch() is None for t in texts)
    ring = res.artists["collection"]
    assert list(ring.get_linewidths()) == pytest.approx([4.5])
    assert ring.get_zorder() == 2
    assert len(ring.get_segments()) == 180
    assert res.ax.get_aspect() == 1.0
    assert not res.ax.get_xticks().size and not res.ax.get_yticks().size
    assert not any(s.get_visible() for s in res.ax.spines.values())
    plt.close(res.fig)


def test_default_dial_is_a_data_unit_inset():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set(xlim=(-4, 4), ylim=(-4, 4))
    res = ep.phasor([1j], ax=ax, at=(1.5, -1.0), size=2.0)
    locator = res.ax.get_axes_locator()
    assert tuple(locator._bounds) == pytest.approx((0.5, -2.0, 2.0, 2.0))
    assert locator._transform is ax.transData
    assert res.ax.get_zorder() == 6
    assert not res.ax.get_in_layout()
    assert res.ax.patch.get_alpha() == 0.0
    assert res.ax.get_anchor() == "C"
    plt.close(fig)


def test_starts_place_each_arrow_and_the_resultant_keeps_the_origin():
    res = ep.phasor([1.0, 0.5j], starts=[0j, 1.0], show_sum=True, origin=0j)
    e1, e2, total = res.artists["arrow"]
    assert _ends(e1) == pytest.approx((0j, 1.0))
    assert _ends(e2) == pytest.approx((1.0, 1.0 + 0.5j))
    assert _ends(total) == pytest.approx((0j, 1.0 + 0.5j))
    plt.close(res.fig)


def test_automatic_limits_include_the_starts():
    res = ep.phasor([0.5], starts=[3.0 + 3.0j], cross=False)
    x0, x1 = res.ax.get_xlim()
    y0, y1 = res.ax.get_ylim()
    assert x0 < 3.0 and x1 > 3.5 and y0 < 3.0 < y1
    plt.close(res.fig)


def test_starts_on_a_chain_or_of_the_wrong_length_raise():
    with pytest.raises(ValueError, match="not chained"):
        ep.phasor([1.0, 1j], starts=[0j, 1.0], chain=True)
    with pytest.raises(ValueError, match="1 starts for 2 vectors"):
        ep.phasor([1.0, 1j], starts=[0j])


def test_update_takes_new_starts_and_otherwise_keeps_the_last():
    res = ep.phasor([1.0, 1j], starts=[0j, 1.0], show_sum=True)
    arrows = res.artists["arrow"]
    res.update([2.0, -1j], starts=[1j, 2.0 + 1j])
    assert _ends(arrows[0]) == pytest.approx((1j, 2.0 + 1j))
    assert _ends(arrows[1]) == pytest.approx((2.0 + 1j, 2.0))
    assert _ends(arrows[2]) == pytest.approx((0j, 2.0 - 1j))
    res.update([1.0, 1.0], origin=0.5)
    assert _ends(arrows[0]) == pytest.approx((1j, 1.0 + 1j))
    assert _ends(arrows[1]) == pytest.approx((2.0 + 1j, 3.0 + 1j))
    assert _ends(arrows[2]) == pytest.approx((0.5, 2.5))
    with pytest.raises(ValueError, match="3 starts for 2 vectors"):
        res.update([1.0, 1.0], starts=[0j, 0j, 0j])
    plt.close(res.fig)


def test_update_can_place_arrows_drawn_from_the_origin_but_not_a_chain():
    res = ep.phasor([1.0, 1j])
    res.update([1.0, 1j], starts=[0j, 1.0])
    assert _ends(res.artists["arrow"][1]) == pytest.approx((1.0, 1.0 + 1j))
    plt.close(res.fig)
    res = ep.phasor([1.0, 1j], chain=True)
    with pytest.raises(ValueError, match="not chained"):
        res.update([1.0, 1j], starts=[0j, 1.0])
    plt.close(res.fig)


def _dial_box(dial, parent):
    """The dial's on-screen box: center in parent data units, side in pixels."""
    box = dial.get_window_extent()
    center = parent.transData.inverted().transform(
        [0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)]
    )
    return center, box.width, box.height


def test_axes_units_size_the_dial_by_the_parents_shorter_side():
    fig, ax = plt.subplots(figsize=(6.0, 3.0), layout="constrained")
    ax.set(xlim=(0, 10), ylim=(0, 2))
    res = ep.phasor([1j], ax=ax, at=(7.0, 1.5), size=0.4, size_units="axes", ring=True)
    for figsize in ((6.0, 3.0), (4.0, 5.0), (8.0, 2.5)):
        fig.set_size_inches(figsize)
        fig.canvas.draw()
        parent = ax.get_window_extent()
        center, width, height = _dial_box(res.ax, ax)
        np.testing.assert_allclose(center, (7.0, 1.5), atol=1e-6)
        side = 0.4 * min(parent.width, parent.height)
        assert width == pytest.approx(side, rel=1e-6)
        assert height == pytest.approx(side, rel=1e-6)
    # Moving the parent's limits keeps the dial on its point.
    ax.set_xlim(5, 9)
    fig.canvas.draw()
    np.testing.assert_allclose(_dial_box(res.ax, ax)[0], (7.0, 1.5), atol=1e-6)
    assert not res.ax.get_in_layout()
    fig.savefig(io.BytesIO(), format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)


def test_unknown_size_units_raise():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="size_units"):
        ep.phasor([1j], ax=ax, at=(0.5, 0.5), size_units="inches")
    plt.close(fig)


def test_separate_moves_the_resultant_of_a_straight_chain_beneath_it():
    res = ep.phasor([1.0, 0.5], chain=True, show_sum=True, lim=1.0, separate=0.14)
    e1, e2, total = res.artists["arrow"]
    # Consecutive links touch at a point only and stay on the line.
    assert _ends(e1) == pytest.approx((0j, 1.0))
    assert _ends(e2) == pytest.approx((1.0, 1.5))
    # The rightward resultant moves clockwise from its direction: below.
    assert _ends(total) == pytest.approx((-0.14j, 1.5 - 0.14j))
    plt.close(res.fig)


def test_separate_draws_a_folded_back_link_beside_the_one_before_it():
    res = ep.phasor([1.0, -0.5], chain=True, show_sum=True, lim=1.0, separate=0.14)
    e1, e2, total = res.artists["arrow"]
    assert _ends(e1) == pytest.approx((0j, 1.0))
    # Leftward, so clockwise from its direction is up.
    assert _ends(e2) == pytest.approx((1.0 + 0.14j, 0.5 + 0.14j))
    assert _ends(total) == pytest.approx((-0.14j, 0.5 - 0.14j))
    plt.close(res.fig)


def test_separate_leaves_arrows_that_only_touch_or_cross_alone():
    values = [1.0, -1.0, 1j, 1 + 1j]
    plain = ep.phasor(values, lim=2.0)
    moved = ep.phasor(values, lim=2.0, separate=0.3)
    assert [_ends(a) for a in moved.artists["arrow"]] == [
        _ends(a) for a in plain.artists["arrow"]
    ]
    plt.close(plain.fig)
    plt.close(moved.fig)


def test_separate_moves_a_shorter_arrow_along_a_longer_one_from_the_origin():
    res = ep.phasor([1j, 0.5j], lim=(-1.0, 3.0, -1.0, 1.0), separate=0.1)
    first, second = res.artists["arrow"]
    assert _ends(first) == pytest.approx((0j, 1j))
    # Half the larger span of the limits is 2.0; upward, so clockwise is right.
    assert _ends(second) == pytest.approx((0.2, 0.2 + 0.5j))
    plt.close(res.fig)


def test_update_separates_with_the_first_draws_offset():
    res = ep.phasor([1.0, 0.5], chain=True, show_sum=True, lim=1.0, separate=0.14)
    total = res.artists["arrow"][-1]
    res.update([1.0, 0.5j])
    assert _ends(total) == pytest.approx((0j, 1.0 + 0.5j))
    res.update([0.5j, 0.5j])
    assert _ends(total) == pytest.approx((0.14, 0.14 + 1j))
    plt.close(res.fig)


def test_text_kw_reaches_both_axis_labels():
    bbox = {"facecolor": "white", "edgecolor": "none", "pad": 1.0}
    res = ep.phasor(
        [1j],
        text_kw={"fontsize": 15.0, "zorder": 9, "color": "red", "bbox": bbox},
    )
    for text in res.artists["text"]:
        assert text.get_fontsize() == 15.0
        assert text.get_zorder() == 9
        assert to_rgba(text.get_color()) == to_rgba("red")
        assert text.get_bbox_patch() is not None
    plt.close(res.fig)


def test_phase_ring_alone_draws_one_collection():
    res = ep.phase_ring(radius=0.5, center=1.0 - 2.0j)
    assert set(res.artists) == {"collection"} <= ep.ARTIST_KEYS
    assert res.update is None
    assert res.ax.get_aspect() == 1.0
    assert type(res.fig.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
    ring = res.artists["collection"]
    assert isinstance(ring, LineCollection)
    mid = np.array([0.5 * (s[0] + s[1]) for s in ring.get_segments()])
    np.testing.assert_allclose(
        np.hypot(mid[:, 0] - 1.0, mid[:, 1] + 2.0), 0.5, rtol=1e-3
    )
    phi = np.arctan2(mid[:, 1] + 2.0, mid[:, 0] - 1.0)
    expected = _style.cmap("phase")((phi + np.pi) / (2.0 * np.pi))
    np.testing.assert_allclose(ring.get_colors(), expected, atol=1e-6)
    lw = matplotlib.rcParams["lines.linewidth"]
    assert list(ring.get_linewidths()) == pytest.approx([3.0 * lw])
    assert ring.get_zorder() == 2
    plt.close(res.fig)


def test_phase_ring_options_and_no_side_effects_on_a_given_axes():
    fig, ax = plt.subplots()
    ax.set(xlim=(-3, 3), ylim=(-2, 2))
    ticks = list(ax.get_xticks()), list(ax.get_yticks())
    res = ep.phase_ring(ax, cmap="twilight", width=2.5, zorder=7)
    assert res.ax is ax
    assert ax.get_xlim() == (-3.0, 3.0) and ax.get_ylim() == (-2.0, 2.0)
    assert ax.get_aspect() == "auto"
    assert (list(ax.get_xticks()), list(ax.get_yticks())) == ticks
    assert all(s.get_visible() for s in ax.spines.values())
    ring = res.artists["collection"]
    assert list(ring.get_linewidths()) == [2.5]
    assert ring.get_zorder() == 7
    mid = np.array([0.5 * (s[0] + s[1]) for s in ring.get_segments()])
    phi = np.arctan2(mid[:, 1], mid[:, 0])
    expected = matplotlib.colormaps["twilight"]((phi + np.pi) / (2.0 * np.pi))
    np.testing.assert_allclose(ring.get_colors(), expected, atol=1e-6)
    plt.close(fig)


def test_phasor_ring_is_the_phase_ring():
    res = ep.phasor([1.0], ring=True, ring_radius=0.8, ring_cmap="twilight")
    alone = ep.phase_ring(radius=0.8, cmap="twilight")
    ring, ref = res.artists["collection"], alone.artists["collection"]
    np.testing.assert_array_equal(
        np.asarray(ring.get_segments()), np.asarray(ref.get_segments())
    )
    np.testing.assert_array_equal(ring.get_colors(), ref.get_colors())
    assert list(ring.get_linewidths()) == list(ref.get_linewidths())
    plt.close(res.fig)
    plt.close(alone.fig)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_a_zero_head_draws_a_plain_shaft_without_warning():
    res = ep.phasor(
        [1.0, 1j, -1.0],
        head_scale=[1.0, 0.0, 0.0],
        show_sum=True,
        sum_head_scale=0.0,
    )
    arrows = res.artists["arrow"]
    assert type(arrows[0].get_arrowstyle()).__name__ == "CurveFilledB"
    assert all(type(a.get_arrowstyle()).__name__ == "Curve" for a in arrows[1:])
    res.fig.savefig(io.BytesIO(), format="png", dpi=72)
    res.update([0.5, 0.5j, -0.5])
    res.fig.savefig(io.BytesIO(), format="png", dpi=72)
    for patch in arrows[1:]:
        box = patch.get_window_extent()
        assert max(box.width, box.height) > 0.0
    plt.close(res.fig)


def test_resultant_head_and_width_overrides_survive_update():
    with matplotlib.rc_context(_PIN_RC):
        res = ep.phasor(
            [1.0, 1j],
            chain=True,
            show_sum=True,
            head_scale=0.3,
            sum_head_scale=0.55,
            sum_width=3.0,
        )
    total = res.artists["arrow"][-1]
    nominal = super(type(total), total).get_mutation_scale()
    assert nominal == pytest.approx(HEAD_PER_MARKER * 6.0 * 0.55)
    assert total.get_linewidth() == 3.0
    res.update([0.5, -1j])
    assert super(type(total), total).get_mutation_scale() == pytest.approx(nominal)
    assert total.get_linewidth() == 3.0
    plt.close(res.fig)


def test_resultant_head_alone_sets_its_default_width():
    with matplotlib.rc_context(_PIN_RC):
        res = ep.phasor([1.0, 1j], show_sum=True, sum_head_scale=0.5)
    total = res.artists["arrow"][-1]
    assert total.get_linewidth() == pytest.approx(1.5 * 1.5 * 0.7)
    plt.close(res.fig)


def test_colors_may_mix_names_hex_and_rgba_tuples():
    mixed = ["red", (0.0, 0.0, 1.0, 0.5), "#00ff00", (1.0, 1.0, 0.0)]
    res = ep.phasor([1, 1j, -1, -1j], colors=mixed)
    got = [patch.get_edgecolor() for patch in res.artists["arrow"]]
    expected = [to_rgba(c) for c in mixed]
    np.testing.assert_allclose(got, expected)
    plt.close(res.fig)


def test_a_single_rgba_tuple_colors_every_arrow():
    # Four arrows and a four-number tuple: one RGBA color, not four colors.
    res = ep.phasor([1, 1j, -1, -1j], colors=(0.2, 0.4, 0.6, 0.8))
    for patch in res.artists["arrow"]:
        np.testing.assert_allclose(patch.get_edgecolor(), (0.2, 0.4, 0.6, 0.8))
    rgb = ep.phasor([1, 1j, -1], colors=(0.2, 0.4, 0.6))
    for patch in rgb.artists["arrow"]:
        np.testing.assert_allclose(patch.get_edgecolor(), (0.2, 0.4, 0.6, 1.0))
    plt.close("all")


def test_a_two_color_tuple_is_two_colors_not_a_dash_pattern():
    res = ep.phasor([1, 1j], colors=("red", (0.0, 0.0, 1.0)))
    got = [patch.get_edgecolor() for patch in res.artists["arrow"]]
    np.testing.assert_allclose(got, [to_rgba("red"), to_rgba((0.0, 0.0, 1.0))])
    plt.close(res.fig)


def test_linestyles_may_mix_names_and_dash_patterns():
    res = ep.phasor([1, 1j], linestyles=["--", (0, (3, 2))])
    first, second = res.artists["arrow"]
    assert first.get_linestyle() == "--"
    assert second.get_linestyle() == (0, (3, 2))
    single = ep.phasor([1, 1j], linestyles=(0, (3, 2)))
    assert all(p.get_linestyle() == (0, (3, 2)) for p in single.artists["arrow"])
    plt.close("all")


def test_widths_and_head_scales_take_numpy_scalars_and_sequences():
    res = ep.phasor([1, 1j], widths=np.float64(2.5), head_scale=np.array(0.5))
    assert all(p.get_linewidth() == pytest.approx(2.5) for p in res.artists["arrow"])
    per = ep.phasor([1, 1j], widths=np.array([1.0, 3.0]), head_scale=[0.5, 1.0])
    widths = [p.get_linewidth() for p in per.artists["arrow"]]
    assert widths == pytest.approx([1.0, 3.0])
    plt.close("all")
