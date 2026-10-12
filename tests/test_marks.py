"""Numeric geometry for spans, reference labels, and logarithmic zero rows."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep


@pytest.mark.parametrize("orient", ["v", "h"])
@pytest.mark.parametrize(
    "log,x,expected",
    [
        (False, [0, 1, 2, 3], [0.5, 2.5]),
        (True, [1, 10, 100, 1000], [np.sqrt(10), np.sqrt(100000)]),
    ],
)
def test_excluded_run_midpoints(orient, log, x, expected):
    result = ep.spans(x, ~np.array([True, False, False, True]), orient=orient, log=log)
    assert isinstance(result, ep.PlotResult)
    assert set(result.artists) == {"fill"}
    assert len(result.artists["fill"]) == 1
    patch = result.artists["fill"][0]
    ends = (
        [patch.get_x(), patch.get_x() + patch.get_width()]
        if orient == "v"
        else [patch.get_y(), patch.get_y() + patch.get_height()]
    )
    np.testing.assert_allclose(ends, expected)
    plt.close(result.fig)


def test_true_mask_shades_end_runs():
    x = [0, 1, 2, 3]
    result = ep.spans(x, [True, False, False, True])
    patches = result.artists["fill"]
    np.testing.assert_allclose([p.get_x() for p in patches], [-0.5, 2.5])
    np.testing.assert_allclose([p.get_width() for p in patches], [1, 1])
    plt.close(result.fig)


@pytest.mark.parametrize("orient", ["v", "h"])
def test_infinite_intervals_clip_to_handed_limits(orient):
    fig, ax = plt.subplots()
    ax.set(xlim=(2, 8), ylim=(2, 8))
    result = ep.spans(
        intervals=[[-np.inf, 4], [6, np.inf]],
        ax=ax,
        orient=orient,
        fill_kw={"alpha": 0.3},
    )
    for patch, expected in zip(result.artists["fill"], [[2, 4], [6, 8]], strict=True):
        ends = (
            [patch.get_x(), patch.get_x() + patch.get_width()]
            if orient == "v"
            else [patch.get_y(), patch.get_y() + patch.get_height()]
        )
        np.testing.assert_allclose(ends, expected)
        assert patch.get_alpha() == 0.3
    assert ax.get_xlim() == (2, 8)
    assert ax.get_ylim() == (2, 8)
    plt.close(fig)


def test_run_endpoints_empty_and_singleton():
    result = ep.spans([0, 2, 4], [True, False, True])
    np.testing.assert_allclose([p.get_x() for p in result.artists["fill"]], [-1, 3])
    np.testing.assert_allclose([p.get_width() for p in result.artists["fill"]], [2, 2])
    plt.close(result.fig)
    for x, mask in [([], []), ([1], [False])]:
        result = ep.spans(x, mask)
        assert not result.artists["fill"]
        plt.close(result.fig)
    result = ep.spans([1], [True], log=True)
    patch = result.artists["fill"][0]
    np.testing.assert_allclose(
        [patch.get_x(), patch.get_width()], [10**-0.5, 10**0.5 - 10**-0.5]
    )
    plt.close(result.fig)


@pytest.mark.parametrize("orient", ["h", "v"])
@pytest.mark.parametrize("where,edge", [("start", 0), ("end", 1)])
def test_reference_axis_transform(orient, where, edge):
    result = ep.ref_line(
        None,
        3,
        "limit",
        orient=orient,
        where=where,
        line_kw={"lw": 4},
        text_kw={"fontsize": 13},
    )
    text = result.artists["text"]
    expected = (edge, 3) if orient == "h" else (3, edge)
    transform = (
        result.ax.get_yaxis_transform()
        if orient == "h"
        else result.ax.get_xaxis_transform()
    )
    assert text.get_position() == expected
    assert text.get_transform() is transform
    assert result.artists["line"].get_linewidth() == 4
    assert text.get_fontsize() == 13
    result.ax.set(xlim=(-10, 10), ylim=(-10, 10))
    np.testing.assert_allclose(
        text.get_transform().transform(text.get_position()),
        transform.transform(expected),
    )
    plt.close(result.fig)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("break_mark", [False, True])
def test_floor_tick_and_break(axis, break_mark):
    result = ep.log_zero_floor(
        None, 1e-4, axis=axis, break_mark=break_mark, mark_kw={"lw": 3}
    )
    ax = result.ax
    target = ax.xaxis if axis == "x" else ax.yaxis
    assert target.get_scale() == "log"
    ticks = target.get_majorticklocs()
    assert np.count_nonzero(ticks == 1e-4) == 1
    assert target.get_major_formatter()(1e-4) == "0"
    assert len(result.artists["lines"]) == (2 if break_mark else 0)
    for line in result.artists["lines"]:
        assert line.get_linewidth() == 3
        coords = line.get_xdata() if axis == "x" else line.get_ydata()
        assert np.all(np.asarray(coords) > 1e-4)
    (ax.set_xlim if axis == "x" else ax.set_ylim)(1e-4, 100)
    result.fig.canvas.draw()
    assert target.get_major_formatter()(1e-4) == "0"
    assert 10 in target.get_majorticklocs()
    assert any(t.get_text() == "0" for t in target.get_ticklabels())
    plt.close(result.fig)


@pytest.mark.parametrize(
    "draw",
    [
        lambda ax: ep.spans([0, 1], [False, True], ax=ax),
        lambda ax: ep.ref_line(ax, 1, "reference"),
        lambda ax: ep.log_zero_floor(ax, 1e-3),
    ],
)
def test_marks_preserve_caller_geometry(draw):
    fig, axes = plt.subplots(1, 2)
    positions = [ax.get_position(original=True).bounds for ax in axes]
    result = draw(axes[0])
    assert result.ax is axes[0]
    assert result.artists.keys() <= ep.ARTIST_KEYS
    assert fig.get_layout_engine() is None
    assert [ax.get_position(original=True).bounds for ax in axes] == positions
    assert axes[0].get_shared_x_axes().get_siblings(axes[0]) == [axes[0]]
    assert axes[0].get_shared_y_axes().get_siblings(axes[0]) == [axes[0]]
    plt.close(fig)


@pytest.mark.parametrize("floor", [0, -1, np.inf, np.nan])
def test_invalid_floor(floor):
    with pytest.raises(ValueError, match="finite and positive"):
        ep.log_zero_floor(None, floor)


def test_invalid_spans():
    with pytest.raises(ValueError, match="positive"):
        ep.spans([0, 1], [True, False], log=True)
    with pytest.raises(ValueError, match=r"expected.*received"):
        ep.spans([1, 2], [True])
    with pytest.raises(ValueError, match="either"):
        ep.spans([1], [True], intervals=[[0, 2]])


def test_passthrough_aliases_override_defaults():
    from matplotlib.colors import to_rgba

    result = ep.spans([0, 1], [False, True], fill_kw={"fc": "red"})
    np.testing.assert_allclose(
        result.artists["fill"][0].get_facecolor(), to_rgba("red")
    )
    line = ep.ref_line(
        result.ax,
        1,
        "limit",
        line_kw={"ls": ":"},
        text_kw={"horizontalalignment": "right"},
    )
    assert line.artists["line"].get_linestyle() == ":"
    assert line.artists["text"].get_ha() == "right"
    mark = ep.log_zero_floor(result.ax, 0.01, mark_kw={"linewidth": 5})
    assert all(stroke.get_linewidth() == 5 for stroke in mark.artists["lines"])
    plt.close(result.fig)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_log_zero_floor_before_data_leaves_far_end_free(axis):
    fig, ax = plt.subplots()
    ep.log_zero_floor(ax, 1e-4, axis=axis)
    data = ([1e-4, 1e4], [1.0, 2.0]) if axis == "x" else ([1.0, 2.0], [1e-4, 1e4])
    ax.plot(*data)
    limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
    assert limits[0] == pytest.approx(1e-4)
    assert limits[1] >= 1e4
    plt.close(fig)
