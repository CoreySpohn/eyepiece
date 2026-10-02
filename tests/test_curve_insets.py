"""curve_insets: square insets standing over marked points of a curve."""

import io

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import eyepiece as ep
from eyepiece import _style

THETA = np.linspace(0.0, 3.0, 301)
CURVE = np.sinc(THETA) ** 2


def _panel(figsize=(6.0, 3.0)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(THETA, CURVE)
    ax.set(xlim=(-0.2, 3.2), ylim=(-0.05, 2.0))
    return fig, ax


def _geometry(inset, ax):
    """Inset center x (data), bottom (display), and side (display)."""
    box = inset.get_window_extent()
    cx = ax.transData.inverted().transform([0.5 * (box.x0 + box.x1), 0.0])[0]
    return cx, box.y0, box.width, box.height


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_keys_insets_and_tones_in_either_mode(mode):
    hwostyle.use(mode)
    fig, ax = _panel()
    res = ep.curve_insets(ax, THETA, CURVE, [0.0, 0.5, 1.5])
    assert res.ax is ax
    assert set(res.artists) == {"scatter", "lines"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    assert len(res.insets) == 3
    assert all(inset in ax.child_axes for inset in res.insets)
    assert all(isinstance(line, Line2D) for line in res.artists["lines"])
    assert res.update is None
    offsets = res.artists["scatter"].get_offsets()
    np.testing.assert_allclose(offsets[:, 0], [0.0, 0.5, 1.5])
    np.testing.assert_allclose(offsets[:, 1], np.interp([0.0, 0.5, 1.5], THETA, CURVE))
    dots = tuple(res.artists["scatter"].get_facecolor()[0][:3])
    assert dots == pytest.approx(to_rgb(_style.color(0)))
    connector = res.artists["lines"][0].get_color()
    assert to_rgb(connector) == pytest.approx(_style.neutral(0.45))
    fig.canvas.draw()
    plt.close(fig)


def test_insets_stand_on_their_points_through_resizes_and_limit_changes():
    fig, ax = _panel()
    marks = [0.5, 2.0]
    res = ep.curve_insets(ax, THETA, CURVE, marks, size=0.3, above=0.1)
    tops = np.interp(marks, THETA, CURVE)
    for figsize, xlim in (((6.0, 3.0), None), ((4.0, 5.0), None), ((8, 3), (0, 4))):
        fig.set_size_inches(figsize)
        if xlim is not None:
            ax.set_xlim(*xlim)
        fig.canvas.draw()
        parent = ax.get_window_extent()
        side = 0.3 * min(parent.width, parent.height)
        for inset, mark, top, line in zip(
            res.insets, marks, tops, res.artists["lines"], strict=True
        ):
            cx, bottom, width, height = _geometry(inset, ax)
            assert cx == pytest.approx(mark, abs=1e-6)
            point_y = ax.transData.transform((mark, top))[1]
            assert bottom == pytest.approx(point_y + 0.1 * parent.height, abs=1e-6)
            assert width == pytest.approx(side, rel=1e-6)
            assert height == pytest.approx(side, rel=1e-6)
            # The connector runs from the point up to the inset's bottom.
            xs, ys = line.get_data()
            assert ys[0] == pytest.approx(point_y, abs=1e-6)
            assert ys[1] == pytest.approx(bottom, abs=1e-6)
            assert xs[0] == pytest.approx(xs[1])
    assert not any(inset.get_in_layout() for inset in res.insets)
    fig.savefig(io.BytesIO(), format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)


def test_heights_stand_insets_on_a_level_row():
    fig, ax = _panel()
    res = ep.curve_insets(ax, THETA, CURVE, [0.2, 1.0, 2.5], heights=1.2, above=0.0)
    fig.canvas.draw()
    level = ax.transData.transform((0.0, 1.2))[1]
    for inset in res.insets:
        assert _geometry(inset, ax)[1] == pytest.approx(level, abs=1e-6)
    staggered = ep.curve_insets(ax, THETA, CURVE, [0.2, 1.0], heights=[1.2, 0.6])
    fig.canvas.draw()
    bottoms = [_geometry(inset, ax)[1] for inset in staggered.insets]
    assert bottoms[0] > bottoms[1]
    plt.close(fig)


def test_data_units_size_the_insets_in_x():
    fig, ax = _panel()
    res = ep.curve_insets(ax, THETA, CURVE, [1.0], size=0.5, size_units="data")
    fig.canvas.draw()
    box = res.insets[0].get_window_extent()
    x0, x1 = ax.transData.inverted().transform([(box.x0, 0), (box.x1, 0)])[:, 0]
    assert x1 - x0 == pytest.approx(0.5, rel=1e-6)
    plt.close(fig)


def test_build_fills_each_inset_in_mark_order():
    fig, ax = _panel()
    seen = []

    def build(inset, i):
        seen.append(i)
        inset.imshow(np.full((4, 4), float(i)))

    res = ep.curve_insets(ax, THETA, CURVE, [0.0, 1.0, 2.0], build=build)
    assert seen == [0, 1, 2]
    assert [len(inset.images) for inset in res.insets] == [1, 1, 1]
    fig.canvas.draw()
    plt.close(fig)


def test_chains_draw_phasor_chains_on_one_shared_frame():
    fig, ax = _panel()
    n = 12
    chains = [
        np.exp(2j * np.pi * t * (np.arange(n) + 0.5) / n) / n for t in (0.0, 0.5, 1.0)
    ]
    res = ep.curve_insets(
        ax, THETA, CURVE, [0.0, 0.5, 1.0], chains=chains, phasor_kw={"head_scale": 0.4}
    )
    limits = {(inset.get_xlim(), inset.get_ylim()) for inset in res.insets}
    assert len(limits) == 1
    for inset in res.insets:
        arrows = [p for p in inset.patches if isinstance(p, FancyArrowPatch)]
        assert len(arrows) == n + 1
        assert inset.patch.get_alpha() == 0.0
    # The resultant of the straight chain is its full length, 1.
    resultant = [p for p in res.insets[0].patches if isinstance(p, FancyArrowPatch)][-1]
    (x0, y0), (x1, y1) = resultant._ends
    assert np.hypot(x1 - x0, y1 - y0) == pytest.approx(1.0)
    fig.canvas.draw()
    plt.close(fig)


def test_toggles_and_routed_kwargs():
    fig, ax = _panel()
    res = ep.curve_insets(ax, THETA, CURVE, [1.0], points=False, connectors=False)
    assert res.artists == {}
    assert len(res.insets) == 1
    res = ep.curve_insets(
        ax,
        THETA,
        CURVE,
        [1.0],
        color="red",
        scatter_kw={"s": 50.0},
        line_kw={"color": "blue", "lw": 3.0},
    )
    assert tuple(res.artists["scatter"].get_facecolor()[0][:3]) == to_rgb("red")
    assert res.artists["scatter"].get_sizes()[0] == pytest.approx(50.0)
    assert to_rgb(res.artists["lines"][0].get_color()) == to_rgb("blue")
    assert res.artists["lines"][0].get_linewidth() == pytest.approx(3.0)
    plt.close(fig)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ((THETA, CURVE[:-1], [1.0]), {}, "y"),
        ((THETA[::-1], CURVE, [1.0]), {}, "increasing"),
        ((THETA, CURVE, [1.0]), {"size": 0.0}, "positive"),
        ((THETA, CURVE, [1.0]), {"size_units": "inches"}, "size_units"),
        ((THETA, CURVE, [1.0, 2.0]), {"heights": [1.0]}, "heights"),
        ((THETA, CURVE, [1.0]), {"chains": [[1.0], [1.0]]}, "chains"),
        ((THETA, CURVE, [1.0]), {"chains": [[1.0]], "build": print}, "not both"),
    ],
)
def test_bad_arguments_raise(args, kwargs, match):
    fig, ax = _panel()
    with pytest.raises(ValueError, match=match):
        ep.curve_insets(ax, *args, **kwargs)
    plt.close(fig)
