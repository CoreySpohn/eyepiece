"""End labels clear each other with the least vertical displacement."""

from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _labels(ax, ys):
    for y in ys:
        ax.plot([0, 0.8], [0.2, y])
    return [ax.text(0.8, y, str(i), va="center", fontsize=14) for i, y in enumerate(ys)]


def _boxes(fig, labels):
    fig.canvas.draw()
    return [label.get_window_extent() for label in labels]


def test_three_close_curve_labels_spread_symmetrically_and_keep_order():
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    labels = _labels(ax, [0.5, 0.501, 0.502])
    before = _boxes(fig, labels)
    result = ep.spread_labels(labels, min_gap=3)
    after = _boxes(fig, labels)
    centers = [box.y0 + box.height / 2 for box in after]
    gap = fig.dpi * 3 / 72
    for a, b in pairwise(after):
        assert b.y0 - a.y1 == pytest.approx(gap)
    assert np.mean(centers) == pytest.approx(
        np.mean([b.y0 + b.height / 2 for b in before])
    )
    assert [box.x0 for box in after] == [box.x0 for box in before]
    assert result.artists["text"] == labels
    ep.spread_labels(labels, min_gap=3)
    np.testing.assert_allclose(
        [label.get_position() for label in labels],
        [
            ax.transData.inverted().transform((b.x0, c))
            for b, c in zip(after, centers, strict=True)
        ],
    )


@pytest.mark.parametrize("ys", [[0.999, 0.998, 0.997], [0.001, 0.002, 0.003]])
@pytest.mark.parametrize("scale", ["linear", "log"])
def test_spread_respects_axes_bounds_with_unequal_heights_and_transforms(ys, scale):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    ax.set(xlim=(0, 1), ylim=(0.01, 1), yscale=scale)
    labels = _labels(ax, ys)
    labels[1].set_fontsize(22)
    ep.spread_labels(labels, min_gap=2)
    after = sorted(_boxes(fig, labels), key=lambda box: box.y0)
    assert after[0].y0 >= ax.bbox.y0 - 1e-8
    assert after[-1].y1 <= ax.bbox.y1 + 1e-8
    assert all(b.y0 - a.y1 >= fig.dpi * 2 / 72 - 1e-8 for a, b in pairwise(after))


def test_separated_labels_do_not_move_and_impossible_fit_raises():
    _fig, ax = plt.subplots(figsize=(5, 3))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    labels = _labels(ax, [0.3, 0.5, 0.7])
    before = [label.get_position() for label in labels]
    ep.spread_labels(labels)
    np.testing.assert_allclose([label.get_position() for label in labels], before)
    for label in labels:
        label.set_fontsize(100)
    with pytest.raises(ValueError, match="fit"):
        ep.spread_labels(labels)
