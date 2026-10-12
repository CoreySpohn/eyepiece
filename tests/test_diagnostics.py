"""Artist data for calibration and seeded jitter strips."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

import eyepiece as ep


@pytest.mark.parametrize("scale", ["linear", "log"])
def test_calibration_identity_shared_limits_and_intervals(scale):
    result = ep.calibration(
        [0.1, 0.3, 0.8],
        [0.2, 0.35, 0.9],
        lo=[0.15, 0.25, 0.7],
        hi=[0.25, 0.45, 1.1],
        scale=scale,
    )
    ax = result.ax
    assert isinstance(result, ep.PlotResult)
    assert set(result.artists) == {"scatter", "lines", "line"}
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_xscale() == ax.get_yscale() == scale
    np.testing.assert_allclose(result.artists["line"].get_xdata(), ax.get_xlim())
    np.testing.assert_allclose(result.artists["line"].get_ydata(), ax.get_ylim())
    np.testing.assert_allclose(
        result.artists["lines"][1].get_data(), [[0.3, 0.3], [0.25, 0.45]]
    )
    np.testing.assert_allclose(
        result.artists["scatter"].get_offsets(), [[0.1, 0.2], [0.3, 0.35], [0.8, 0.9]]
    )
    plt.close(result.fig)


def test_floor_maps_only_zeros_in_points_and_intervals():
    stated = np.array([0, 0.1, 0.5])
    realized = np.array([0.2, 0, 0.7])
    lo = np.array([0.1, 0, 0.4])
    result = ep.calibration(stated, realized, lo=lo, hi=[0.3, 0.02, 0.8], floor=1e-4)
    np.testing.assert_allclose(
        result.artists["scatter"].get_offsets(), [[1e-4, 0.2], [0.1, 1e-4], [0.5, 0.7]]
    )
    np.testing.assert_allclose(result.artists["lines"][1].get_ydata(), [1e-4, 0.02])
    assert len(result.artists["lines"]) == 7  # 3 intervals, 2 breaks per axis
    for axis in (result.ax.xaxis, result.ax.yaxis):
        assert axis.get_major_formatter()(1e-4) == "0"
        assert 1e-4 in axis.get_majorticklocs()
    assert result.ax.get_xlim()[0] == result.ax.get_ylim()[0] == 1e-4
    np.testing.assert_array_equal(stated, [0, 0.1, 0.5])
    np.testing.assert_array_equal(realized, [0.2, 0, 0.7])
    np.testing.assert_array_equal(lo, [0.1, 0, 0.4])
    plt.close(result.fig)


def test_hollow_mask_linear_zero_and_passthroughs():
    point_kw = {"s": 64, "alpha": 0.6, "color": "red"}
    result = ep.calibration(
        [0, 0.4, 1],
        [0.1, 0, 0.9],
        scale="linear",
        hollow=[True, False, True],
        hi=[0.2, 0.1, 1],
        point_kw=point_kw,
        interval_kw={"lw": 3},
        ref_kw={"color": "blue", "ls": ":"},
    )
    scatter = result.artists["scatter"]
    np.testing.assert_allclose(scatter.get_facecolors()[:, 3], [0, 0.6, 0])
    np.testing.assert_allclose(scatter.get_edgecolors(), [to_rgba("red", 0.6)])
    assert scatter.get_sizes()[0] == 64
    assert result.artists["line"].get_color() == "blue"
    assert result.artists["line"].get_linestyle() == ":"
    assert all(line.get_linewidth() == 3 for line in result.artists["lines"])
    assert point_kw == {"s": 64, "alpha": 0.6, "color": "red"}
    plt.close(result.fig)


def test_absolute_tolerance_band_and_identity_disabled():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    result = ep.calibration(
        [0.2, 0.8], [0.3, 0.7], ax=ax, scale="linear", tolerance=0.1, identity=False
    )
    assert "line" not in result.artists
    vertices = result.artists["fill"].get_paths()[0].vertices
    assert np.all(vertices >= 0) and np.all(vertices <= 1)
    assert any(np.allclose(v, [0.1, 0.2]) for v in vertices)
    assert any(np.allclose(v, [0.9, 0.8]) for v in vertices)
    assert np.all(np.abs(vertices[:, 0] - vertices[:, 1]) <= 0.1 + 1e-15)
    plt.close(fig)


def test_strip_jitter_reproducible_and_rng_is_local():
    state = np.random.get_state()
    groups = [np.arange(7), np.arange(5)]
    results = [
        ep.strip(groups, seed=seed, width=0.4, positions=[2, 5]) for seed in [7, 7, 8]
    ]
    for a, b, other, pos, values in zip(
        *(r.artists["scatter"] for r in results), [2, 5], groups, strict=True
    ):
        np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
        assert not np.array_equal(a.get_offsets(), other.get_offsets())
        assert np.all(np.abs(a.get_offsets()[:, 0] - pos) <= 0.2)
        np.testing.assert_array_equal(a.get_offsets()[:, 1], values)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    for result in results:
        plt.close(result.fig)


def test_strip_cap_includes_equality_and_positive_infinity():
    result = ep.strip(
        {"a": [0.2, 1, 2, np.inf, np.nan], "b": [0.4]},
        capped=1,
        positions=[3, 8],
        width=0,
        point_kw={"s": 36},
        color="red",
    )
    ordinary, capped, second, empty = result.artists["scatter"]
    np.testing.assert_array_equal(ordinary.get_offsets(), [[3, 0.2]])
    np.testing.assert_array_equal(capped.get_offsets(), [[3, 1], [3, 1], [3, 1]])
    np.testing.assert_array_equal(second.get_offsets(), [[8, 0.4]])
    assert empty.get_offsets().shape == (0, 2)
    triangle = MarkerStyle("^")
    np.testing.assert_array_equal(
        capped.get_paths()[0].vertices,
        triangle.get_path().transformed(triangle.get_transform()).vertices,
    )
    assert capped.get_sizes()[0] == 36
    np.testing.assert_allclose(capped.get_facecolors(), [to_rgba("red")])
    assert [t.get_text() for t in result.ax.get_xticklabels()] == ["a", "b"]
    plt.close(result.fig)


@pytest.mark.parametrize(
    "draw",
    [
        lambda ax: ep.calibration([0.2, 0.4], [0.3, 0.5], ax=ax, scale="linear"),
        lambda ax: ep.strip([[0.2, 0.3], [0.4]], ax=ax, labels=["a", "b"]),
    ],
)
def test_diagnostics_caller_contract(draw):
    fig, axes = plt.subplots(1, 2)
    positions = [ax.get_position(original=True).bounds for ax in axes]
    result = draw(axes[0])
    assert isinstance(result, ep.PlotResult)
    assert result.ax is axes[0]
    assert result.artists.keys() <= ep.ARTIST_KEYS
    assert fig.get_layout_engine() is None
    assert [ax.get_position(original=True).bounds for ax in axes] == positions
    assert axes[0].get_shared_x_axes().get_siblings(axes[0]) == [axes[0]]
    assert axes[0].get_shared_y_axes().get_siblings(axes[0]) == [axes[0]]
    plt.close(fig)


@pytest.mark.parametrize(
    "kw,match",
    [
        ({}, "provide floor"),
        ({"floor": 0}, "floor"),
        ({"hollow": [True, False]}, "expected.*received"),
        ({"scale": "symlog"}, "scale"),
        ({"scale": "linear", "tolerance": -1}, "tolerance"),
    ],
)
def test_calibration_invalid_arguments(kw, match):
    with pytest.raises(ValueError, match=match):
        ep.calibration([0], [0.2], **kw)


def test_strip_empty_groups_and_invalid_positions():
    result = ep.strip([[], []])
    assert all(len(s.get_offsets()) == 0 for s in result.artists["scatter"])
    plt.close(result.fig)
    with pytest.raises(ValueError, match=r"expected.*received"):
        ep.strip([[1], [2]], positions=[0])
