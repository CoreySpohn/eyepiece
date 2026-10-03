"""imshow_log/imshow_diverging centers=, and kymograph built on it."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

import eyepiece as ep


def _grid(nx=9, ny=5):
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(400.0, 800.0, ny)
    values = np.exp(-(x[None, :] ** 2)) * (y[:, None] / 800.0) + 1e-6
    return values, x, y


@pytest.mark.parametrize("show", [ep.imshow_log, ep.imshow_diverging])
def test_centers_give_the_pixel_edge_extent(show):
    values, x, y = _grid()
    res = show(values, centers=(x, y), colorbar=False)
    dx, dy = x[1] - x[0], y[1] - y[0]
    expected = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)
    assert res.artists["image"].get_extent() == pytest.approx(expected)
    # Physical coordinates keep their ticks.
    assert len(res.ax.get_xticks()) > 0
    plt.close(res.fig)


def test_centers_match_the_hand_built_extent_pixel_for_pixel():
    values, x, y = _grid()
    dx, dy = x[1] - x[0], y[1] - y[0]
    by_hand = ep.imshow_log(
        values,
        extent=(x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2),
        colorbar=False,
    )
    by_centers = ep.imshow_log(values, centers=(x, y), colorbar=False)
    for res in (by_hand, by_centers):
        res.ax.set_aspect("auto")
        res.fig.canvas.draw()
    a = np.asarray(by_hand.fig.canvas.buffer_rgba())
    b = np.asarray(by_centers.fig.canvas.buffer_rgba())
    np.testing.assert_array_equal(a, b)
    plt.close(by_hand.fig)
    plt.close(by_centers.fig)


def test_centers_with_origin_upper_keep_row_zero_at_y0():
    values, x, y = _grid()
    res = ep.imshow_log(
        values, centers=(x, y), colorbar=False, imshow_kw={"origin": "upper"}
    )
    _, _, bottom, top = res.artists["image"].get_extent()
    dy = y[1] - y[0]
    assert top == pytest.approx(y[0] - dy / 2)
    assert bottom == pytest.approx(y[-1] + dy / 2)
    plt.close(res.fig)


@pytest.mark.parametrize(
    ("centers", "match"),
    [
        ((np.arange(9.0), np.arange(4.0)), "4 entries for 5"),
        ((np.arange(9.0), np.array([0.0, 1.0, 3.0, 4.0, 5.0])), "evenly spaced"),
        ((np.arange(9.0),), r"\(x, y\)"),
        ((np.arange(9.0), np.zeros((5, 1))), "1D"),
    ],
)
def test_bad_centers_raise(centers, match):
    values, _, _ = _grid()
    with pytest.raises(ValueError, match=match):
        ep.imshow_log(values, centers=centers)
    plt.close("all")


def test_centers_and_extent_together_raise():
    values, x, y = _grid()
    with pytest.raises(ValueError, match="not both"):
        ep.imshow_log(values, centers=(x, y), extent=(0, 1, 0, 1))
    plt.close("all")


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_kymograph_is_a_log_image_on_an_auto_aspect(mode):
    hwostyle.use(mode)
    values, x, y = _grid()
    res = ep.kymograph(values, x, y, vmin=1e-3, vmax=1.0, cbar_label="brightness")
    image = res.artists["image"]
    assert set(res.artists) == {"image", "cbar"}
    assert isinstance(image.norm, LogNorm)
    assert (image.norm.vmin, image.norm.vmax) == (1e-3, 1.0)
    assert res.ax.get_aspect() == "auto"
    assert image.get_interpolation() == "nearest"
    dx = x[1] - x[0]
    assert image.get_extent()[:2] == pytest.approx((x[0] - dx / 2, x[-1] + dx / 2))
    res.fig.canvas.draw()
    plt.close(res.fig)


def test_kymograph_update_reuses_the_floor_and_norm():
    values, x, y = _grid()
    res = ep.kymograph(values, x, y, floor=1e-4, colorbar=False)
    norm = res.artists["image"].norm
    res.update(np.zeros_like(values))
    assert res.artists["image"].norm is norm
    assert np.asarray(res.artists["image"].get_array()).min() == 1e-4
    plt.close(res.fig)


def test_kymograph_imshow_kw_wins_over_the_auto_aspect():
    values, x, y = _grid()
    res = ep.kymograph(values, x, y, colorbar=False, imshow_kw={"aspect": "equal"})
    assert res.ax.get_aspect() == 1.0
    plt.close(res.fig)
