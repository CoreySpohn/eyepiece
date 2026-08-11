"""corner parity essentials, hist-vs-pdf, cov ellipse."""

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
