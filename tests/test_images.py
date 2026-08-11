"""Image primitives: floor, geometry rule, update path, shared-norm row."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece.images import compare_row, imshow_diverging, imshow_log


def _img(lo=1e-12, hi=1e-6):
    rng = np.random.default_rng(0)
    return rng.uniform(lo, hi, (8, 8))


def test_imshow_log_floor_and_keys():
    img = _img()
    img[0, 0] = 0.0  # would break LogNorm without the floor
    res = imshow_log(img, floor=1e-14)
    assert {"image", "cbar"} <= set(res.artists)
    assert res.artists["image"].get_array().min() >= 1e-14
    assert res.artists["image"].get_interpolation() == "nearest"
    plt.close(res.fig)


def test_imshow_log_extent_applied():
    res = imshow_log(_img(), extent=(-2, 2, -2, 2))
    assert tuple(res.artists["image"].get_extent()) == (-2, 2, -2, 2)
    plt.close(res.fig)


def test_colorbar_does_not_steal_sibling_space():
    fig, axes = plt.subplots(1, 2)
    imshow_log(_img(), ax=axes[0])
    fig.canvas.draw()
    w0 = axes[0].get_position().width
    w1 = axes[1].get_position().width
    assert w0 == pytest.approx(w1, rel=0.01)
    plt.close(fig)


def test_update_no_artist_leak():
    res = imshow_log(_img())
    n0 = len(res.ax.images)
    for _ in range(3):
        res.update(_img())
    assert len(res.ax.images) == n0
    plt.close(res.fig)


def test_compare_row_shared_norm():
    a, b = _img(), _img(1e-10, 1e-5)
    res = compare_row([a, b], titles=["A", "B"])
    ims = [res.artists["image"][0], res.artists["image"][1]]
    assert ims[0].norm is ims[1].norm
    plt.close(res.fig)


def test_diverging_symmetric():
    res = imshow_diverging(np.array([[-3.0, 1.0], [2.0, -0.5]]))
    im = res.artists["image"]
    assert im.norm.vmin == -im.norm.vmax
    plt.close(res.fig)


def test_compare_row_single_image():
    res = compare_row([_img()])
    assert res.axes.shape == (1,)
    assert len(res.artists["image"]) == 1
    plt.close(res.fig)


def test_compare_row_handed_axes_does_not_steal_sibling_space():
    fig, axes = plt.subplots(1, 3)
    compare_row([_img(), _img()], axes=axes[:2])
    fig.canvas.draw()
    w2 = axes[2].get_position().width
    w0 = axes[0].get_position().width
    assert w2 == pytest.approx(w0, rel=0.01)
    plt.close(fig)


def test_compare_row_empty_raises():
    with pytest.raises(ValueError, match="at least one image"):
        compare_row([])
