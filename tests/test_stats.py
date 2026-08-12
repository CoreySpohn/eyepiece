"""corner parity essentials, hist-vs-pdf, cov ellipse."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from eyepiece import _style
from eyepiece.stats import corner, corner_overlay, cov_ellipse, hist_vs_pdf


def _samples(n=500):
    rng = np.random.default_rng(2)
    return {
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "c": rng.uniform(size=n),
    }


def test_corner_triangle_shape():
    res = corner(_samples(), ["a", "b", "c"], truths={"a": 0.0, "b": 5.0})
    assert res.axes.shape == (3, 3)
    assert not res.axes[0, 2].get_visible()  # upper triangle hidden
    plt.close(res.fig)


def test_corner_labels_dict_applied():
    res = corner(_samples(), ["a", "b"], labels={"a": "alpha [au]"})
    assert res.axes[1, 0].get_xlabel() == "alpha [au]"
    plt.close(res.fig)


def test_corner_overlay_into_existing_axes():
    base = corner(_samples(), ["a", "b"])
    n_before = len(base.axes[1, 0].collections) + len(base.axes[1, 0].lines)
    corner_overlay([_samples()], ["a", "b"], axes=base.axes)
    n_after = len(base.axes[1, 0].collections) + len(base.axes[1, 0].lines)
    assert n_after > n_before
    plt.close(base.fig)


def test_corner_overlay_wraps_the_palette_past_its_end():
    rng = np.random.default_rng(4)
    datasets = [
        {"a": rng.normal(size=30), "b": rng.normal(size=30)} for _ in range(8)
    ]  # palettes hold six colors
    res = corner_overlay(datasets, ["a", "b"])
    scatters = res.artists["scatter"]
    assert len(scatters) == 8
    assert np.allclose(scatters[6].get_facecolor(), scatters[0].get_facecolor())
    plt.close(res.fig)


def test_corner_axes_draws_into_handed_axes():
    fig, axes = plt.subplots(2, 2, layout="constrained")
    axes[0, 1].set_visible(True)  # opposite of corner's hidden-upper contract
    n_axes_before = len(fig.axes)
    res = corner(_samples(), ["a", "b"], axes=axes)
    assert res.fig is fig
    assert len(fig.axes) == n_axes_before
    assert len(res.artists["hist"]) == 2
    assert len(axes[1, 0].collections) == 1  # the 2D density mesh
    # corner hides the upper triangle unconditionally, even when it did not
    # create the axes and the cell was pre-set visible.
    assert not axes[0, 1].get_visible()
    plt.close(fig)


def test_corner_axes_wrong_shape_raises():
    fig, axes = plt.subplots(2, 3, layout="constrained")
    with pytest.raises(ValueError, match="cannot reshape"):
        corner(_samples(), ["a", "b"], axes=axes)
    plt.close(fig)


def test_corner_axes_none_truths_scenario_structure():
    # Structural regression lock for the axes=None path (replaces a
    # one-time PNG-hash proof against the pre-axes= implementation; see the
    # F7 fix report for that proof's evidence). Hash equality is brittle
    # across platforms/matplotlib versions, so this asserts on the artist
    # counts, visibility, and geometry a regression would actually break.
    res = corner(_samples(), ["a", "b", "c"], truths={"a": 0.0, "b": 5.0})
    assert res.axes.shape == (3, 3)
    assert len(res.artists["hist"]) == 3  # one diagonal histogram per param
    assert len(res.artists["collection"]) == 3  # 3 lower-triangle density cells
    # the QuadMesh's array is (bins, bins) -- catches a mesh-resolution
    # regression that a mere collection count would miss.
    assert res.artists["collection"][0].get_array().shape == (30, 30)
    # truth lines: diagonal cells for a, b (1 each); off-diagonal (a, b) both
    # in truths (2); off-diagonal (a, c) only a in truths (1); off-diagonal
    # (b, c) only b in truths (1); diagonal c not in truths (0) -> 6 total.
    assert len(res.artists["line"]) == 6
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert not res.axes[i, j].get_visible()  # upper triangle hidden
    for i in range(3):
        assert res.axes[i, i].get_visible()  # diagonal shown
    for i, j in [(1, 0), (2, 0), (2, 1)]:
        assert res.axes[i, j].get_visible()  # lower triangle shown
    expected_hist_color = to_rgba(_style.color(0))
    expected_truth_color = to_rgba(_style.color(1))
    assert to_rgba(res.artists["hist"][0][0].get_edgecolor()) == expected_hist_color
    assert to_rgba(res.artists["line"][0].get_color()) == expected_truth_color
    assert res.artists["collection"][0].get_cmap().name == _style.cmap("intensity").name
    plt.close(res.fig)


def test_corner_axes_none_labels_scenario_structure():
    res = corner(_samples(), ["a", "b"], labels={"a": "alpha [au]"})
    assert res.axes[1, 0].get_xlabel() == "alpha [au]"  # overridden
    assert res.axes[1, 0].get_ylabel() == "b"  # falls back to its own name
    assert res.axes[1, 1].get_xlabel() == "b"  # bottom-row diagonal cell
    # non-bottom-row cells get their tick text cleared, not the ticks removed
    assert all(t.get_text() == "" for t in res.axes[0, 0].get_xticklabels())
    plt.close(res.fig)


def test_corner_title_applies_to_own_created_figure():
    # Also serves as the structural regression lock for the title scenario
    # that used to be covered by a PNG-hash comparison.
    res = corner(_samples(), ["a", "b"], title="posterior")
    assert res.fig.get_suptitle() == "posterior"
    plt.close(res.fig)


def test_corner_title_with_handed_axes_raises():
    fig, axes = plt.subplots(2, 2, layout="constrained")
    with pytest.raises(ValueError, match="title is not supported when axes"):
        corner(_samples(), ["a", "b"], axes=axes, title="posterior")
    assert fig.get_suptitle() == ""  # the figure is untouched by the raise
    plt.close(fig)


def test_corner_handed_axes_does_not_steal_sibling_space():
    fig, axes = plt.subplots(2, 3, layout="constrained")
    corner(_samples(), ["a", "b"], axes=axes[:, :2])
    fig.canvas.draw()
    # original=True reports the gridspec slot BEFORE aspect adjustments --
    # see the analogous compare_row geometry test for why.
    w_sibling = axes[0, 2].get_position(original=True).width
    w_target = axes[0, 0].get_position(original=True).width
    assert w_sibling == pytest.approx(w_target, rel=0.01)
    plt.close(fig)


def test_hist_vs_pdf_keys():
    s = np.random.default_rng(3).normal(size=1000)
    pdf = lambda x: np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)  # noqa: E731
    res = hist_vs_pdf(s, pdf)
    assert {"hist", "line"} <= set(res.artists)
    plt.close(res.fig)


def test_cov_ellipse_scales_with_sigma():
    r1 = cov_ellipse((0, 0), np.eye(2), n_sigma=1)
    r2 = cov_ellipse((0, 0), np.eye(2), ax=r1.ax, n_sigma=2)
    assert r2.artists["ellipse"].width == pytest.approx(2 * r1.artists["ellipse"].width)
    plt.close(r1.fig)
