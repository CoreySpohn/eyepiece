"""Numeric contracts for points with absolute intervals."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import eyepiece as ep


@pytest.mark.parametrize("hollow_zero", [False, True])
def test_zero_faces_and_original_offsets(hollow_zero):
    result = ep.interval_points([1, 2, 3], [0, 2, 0], hollow_zero=hollow_zero)
    assert isinstance(result, ep.PlotResult)
    assert set(result.artists) == {"scatter", "lines"}
    scatter = result.artists["scatter"]
    np.testing.assert_array_equal(scatter.get_offsets(), [[1, 0], [2, 2], [3, 0]])
    alpha = np.broadcast_to(scatter.get_facecolors(), (3, 4))[:, 3]
    np.testing.assert_array_equal(alpha == 0, [hollow_zero, False, hollow_zero])
    assert result.artists["lines"] == []
    plt.close(result.fig)


@pytest.mark.parametrize("orient", ["v", "h"])
@pytest.mark.parametrize(
    "bounds, expected",
    [
        ({"lo": [0, 2]}, [[0, 1], [2, 4]]),
        ({"hi": [3, 7]}, [[1, 3], [4, 7]]),
        ({"lo": [0, 2], "hi": [3, 7]}, [[0, 3], [2, 7]]),
    ],
)
def test_absolute_bounds_and_orientation(orient, bounds, expected):
    result = ep.interval_points([10, 20], [1, 4], orient=orient, **bounds)
    xy = np.array([[10, 1], [20, 4]])
    if orient == "h":
        xy = xy[:, ::-1]
    np.testing.assert_array_equal(result.artists["scatter"].get_offsets(), xy)
    for line, pos, ends in zip(
        result.artists["lines"], [10, 20], expected, strict=True
    ):
        coords = [[pos, pos], ends]
        np.testing.assert_array_equal(
            line.get_data(), coords if orient == "v" else coords[::-1]
        )
    plt.close(result.fig)


def test_handed_axes_styles_and_scalar_bounds():
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    position = ax.get_position(original=True).bounds
    point_kw = {"s": 81, "alpha": 0.4}
    interval_kw = {"lw": 3, "linestyle": ":"}
    result = ep.interval_points(
        [0, 1],
        [0, 2],
        lo=-1,
        ax=ax,
        color="red",
        label="samples",
        point_kw=point_kw,
        interval_kw=interval_kw,
    )
    assert result.ax is ax
    assert fig.get_layout_engine() is None
    assert ax.get_shared_x_axes().get_siblings(ax) == [ax]
    assert ax.get_shared_y_axes().get_siblings(ax) == [ax]
    assert ax.get_position(original=True).bounds == position
    scatter = result.artists["scatter"]
    np.testing.assert_allclose(scatter.get_facecolors()[:, 3], [0, 0.4])
    np.testing.assert_allclose(scatter.get_edgecolors(), [to_rgba("red", 0.4)])
    assert scatter.get_label() == "samples"
    assert scatter.get_sizes()[0] == 81
    assert all(line.get_linewidth() == 3 for line in result.artists["lines"])
    np.testing.assert_array_equal(result.artists["lines"][1].get_ydata(), [-1, 2])
    assert point_kw == {"s": 81, "alpha": 0.4}
    assert interval_kw == {"lw": 3, "linestyle": ":"}
    plt.close(fig)


@pytest.mark.parametrize("x,y", [([[1]], [2]), ([1, 2], [3])])
def test_shape_errors(x, y):
    with pytest.raises(ValueError, match=r"expected.*received"):
        ep.interval_points(x, y)


def test_mapped_point_colors_keep_holes_after_draw():
    result = ep.interval_points(
        [0, 1], [0, 2], point_kw={"c": [0.1, 0.9], "cmap": "viridis"}
    )
    result.fig.canvas.draw()
    faces = result.artists["scatter"].get_facecolors()
    np.testing.assert_array_equal(faces[:, 3], [0, 1])
    assert not np.allclose(faces[0, :3], faces[1, :3])
    plt.close(result.fig)
