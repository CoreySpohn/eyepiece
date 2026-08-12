"""corner parity essentials, hist-vs-pdf, cov ellipse."""

import hashlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece.stats import corner, corner_overlay, cov_ellipse, hist_vs_pdf


def _samples(n=500):
    rng = np.random.default_rng(2)
    return {
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "c": rng.uniform(size=n),
    }


def _hash_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    return hashlib.sha256(buf.getvalue()).hexdigest()


# Captured from corner() on commit 15cead4 (the last commit before axes=
# existed), via _hash_fig() on these exact scenarios in a detached git
# worktree checked out at that commit. Proves axes=None is byte-identical
# to the pre-change implementation. Regenerate only if corner's rendered
# output is intentionally changing.
_GOLDEN_TRUTHS = "bd2f85c984890e453832cc99ef9daaf7e403467e427b0664556cabdd3dfbe4b6"
_GOLDEN_LABELS = "a94f1aab75d2ad958eb1f7af7f5abacabfc44d002df6f620c9ecd77be0db7c87"
_GOLDEN_TITLE = "1509a7ebe890c1f6e1ff3c8dc60bb049e2f04343e07ba14fb167960b883c42db"


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
    n_axes_before = len(fig.axes)
    res = corner(_samples(), ["a", "b"], axes=axes)
    assert res.fig is fig
    assert len(fig.axes) == n_axes_before
    assert len(res.artists["hist"]) == 2
    assert len(axes[1, 0].collections) == 1  # the 2D density mesh
    plt.close(fig)


def test_corner_axes_wrong_shape_raises():
    fig, axes = plt.subplots(2, 3, layout="constrained")
    with pytest.raises(ValueError, match="cannot reshape"):
        corner(_samples(), ["a", "b"], axes=axes)
    plt.close(fig)


def test_corner_axes_none_matches_pre_change_implementation():
    # Proves the additive guarantee empirically: these hashes were captured
    # by rendering the same scenarios against commit 15cead4 (pre-axes=) in
    # a detached git worktree. A byte-identical PNG means axes=None is
    # exactly what corner() drew before this change, not merely "looks the
    # same".
    res = corner(_samples(), ["a", "b", "c"], truths={"a": 0.0, "b": 5.0})
    assert _hash_fig(res.fig) == _GOLDEN_TRUTHS
    plt.close(res.fig)

    res = corner(_samples(), ["a", "b"], labels={"a": "alpha [au]"})
    assert _hash_fig(res.fig) == _GOLDEN_LABELS
    plt.close(res.fig)

    res = corner(_samples(), ["a", "b"], title="posterior")
    assert _hash_fig(res.fig) == _GOLDEN_TITLE
    plt.close(res.fig)


def test_corner_title_applies_to_own_created_figure():
    res = corner(_samples(), ["a", "b"], title="posterior")
    assert res.fig.get_suptitle() == "posterior"
    plt.close(res.fig)


def test_corner_title_ignored_with_handed_axes():
    fig, axes = plt.subplots(2, 2, layout="constrained")
    corner(_samples(), ["a", "b"], axes=axes, title="posterior")
    assert fig.get_suptitle() == ""
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
