"""convergence: samples, their running mean or sum, and labeled references."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import eyepiece as ep
from eyepiece import _style


def _looks(n=50):
    return np.random.default_rng(0).exponential(1.0, n)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_keys_and_tones_in_either_mode(mode):
    hwostyle.use(mode)
    res = ep.convergence(_looks(), refs=[1.0, 0.5], ref_labels=["a", "b"], band=0.1)
    assert set(res.artists) == {"scatter", "line", "lines", "fill", "text"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    assert isinstance(res.artists["scatter"], PathCollection)
    assert isinstance(res.artists["line"], Line2D)
    assert all(isinstance(r, Rectangle) for r in res.artists["fill"])
    assert len(res.artists["lines"]) == len(res.artists["text"]) == 2
    assert to_rgb(res.artists["line"].get_color()) == pytest.approx(
        to_rgb(_style.color(0))
    )
    dots = res.artists["scatter"].get_facecolor()[0][:3]
    assert tuple(dots) == pytest.approx(_style.neutral(0.55))
    band = res.artists["fill"][0].get_facecolor()[:3]
    assert band == pytest.approx(_style.neutral(0.18))
    for text in res.artists["text"]:
        assert text.get_bbox_patch() is not None
    res.fig.canvas.draw()
    plt.close(res.fig)


@pytest.mark.parametrize(
    ("running", "expected"),
    [
        ("mean", lambda v: np.cumsum(v) / np.arange(1, v.size + 1)),
        ("sum", np.cumsum),
        (None, lambda v: v),
    ],
)
def test_the_line_is_the_running_statistic(running, expected):
    values = _looks(20)
    res = ep.convergence(values, running=running)
    line = res.artists["line"]
    np.testing.assert_allclose(line.get_xdata(), np.arange(1, 21))
    np.testing.assert_allclose(line.get_ydata(), expected(values))
    plt.close(res.fig)


def test_update_reveals_the_first_k_without_new_artists_or_new_limits():
    values = _looks(40)
    res = ep.convergence(values, refs=1.0, ref_labels="expected", band=0.2)
    ax = res.ax
    res.fig.canvas.draw()
    limits = ax.get_xlim(), ax.get_ylim()
    n_children = len(ax.get_children())
    for k in (0, 1, 7, 40):
        res.update(k)
        res.fig.canvas.draw()
        assert len(ax.get_children()) == n_children
        assert (ax.get_xlim(), ax.get_ylim()) == limits
        assert len(res.artists["line"].get_xdata()) == k
        assert res.artists["scatter"].get_offsets().shape == (k, 2)
    np.testing.assert_allclose(res.artists["scatter"].get_offsets()[:, 1], values)
    plt.close(res.fig)


def test_limits_cover_the_full_data_and_leave_room_for_labels():
    values = np.array([3.0, -1.0, 2.0, 0.0])
    res = ep.convergence(values, refs=[5.0], ref_labels=["top"], band=0.5)
    (x0, x1), (y0, y1) = res.ax.get_xlim(), res.ax.get_ylim()
    assert x0 < 1.0 and x1 >= 4.0 + 0.2 * 3.0
    assert y0 < -1.0 and y1 > 5.5
    bare = ep.convergence(values, refs=[5.0])
    assert bare.ax.get_xlim()[1] < 4.0 + 0.2 * 3.0
    plt.close("all")


@pytest.mark.parametrize(
    ("side", "va", "sign"), [("above", "bottom", 1), ("below", "top", -1)]
)
def test_labels_sit_at_the_right_edge_beside_their_line(side, va, sign):
    res = ep.convergence(_looks(), refs=[1.0], ref_labels=["one"], label_side=side)
    (text,) = res.artists["text"]
    assert text.xy == (1.0, 1.0)
    assert (text.get_ha(), text.get_va()) == ("right", va)
    assert np.sign(text.get_position()[1]) == sign
    # The label is placed in axes x, data y: it stays at the edge when x moves.
    res.ax.set_xlim(-100, 100)
    res.fig.canvas.draw()
    box = text.get_window_extent()
    assert box.x1 <= res.ax.get_window_extent().x1 + 1.0
    plt.close(res.fig)


def test_update_replaces_values_and_moves_references():
    values = _looks(10)
    res = ep.convergence(
        values, refs=[1.0, 2.0], ref_labels=[None, "two"], band=[0.1, None]
    )
    assert len(res.artists["text"]) == 1
    assert len(res.artists["fill"]) == 1
    res.update(5, values=np.ones(10), refs=[1.5, 2.5])
    np.testing.assert_allclose(res.artists["line"].get_ydata(), np.ones(5))
    np.testing.assert_allclose(res.artists["lines"][0].get_ydata(), [1.5, 1.5])
    np.testing.assert_allclose(res.artists["lines"][1].get_ydata(), [2.5, 2.5])
    assert res.artists["text"][0].xy == (1.0, 2.5)
    assert res.artists["fill"][0].get_y() == pytest.approx(1.4)
    assert res.artists["fill"][0].get_height() == pytest.approx(0.2)
    # A k of None keeps the reveal.
    res.update(values=values)
    assert len(res.artists["line"].get_xdata()) == 5
    with pytest.raises(ValueError, match="values"):
        res.update(values=np.ones(3))
    with pytest.raises(ValueError, match="refs"):
        res.update(refs=[1.0])
    plt.close(res.fig)


def test_two_calls_share_one_panel():
    fig, ax = plt.subplots()
    ep.convergence(
        np.linspace(0, 1, 30), ax=ax, refs=[1.0], running=None, show_samples=False
    )
    ep.convergence(
        np.linspace(-2, 0, 30), ax=ax, refs=[-2.0], running=None, show_samples=False
    )
    y0, y1 = ax.get_ylim()
    assert y0 < -2.0 and y1 > 1.0
    plt.close(fig)


def test_options_and_routed_kwargs():
    res = ep.convergence(
        _looks(),
        x=np.arange(50) * 0.5,
        refs=[1.0],
        show_samples=False,
        color="red",
        ref_linestyles=":",
        line_kw={"marker": "o", "markevery": [-1]},
        ref_kw={"lw": 3.0},
        fill_kw={"alpha": 0.5},
        text_kw={"color": "blue"},
        ref_labels=["r"],
        band=0.2,
    )
    assert "scatter" not in res.artists
    assert to_rgb(res.artists["line"].get_color()) == to_rgb("red")
    assert res.artists["lines"][0].get_linestyle() == ":"
    assert res.artists["lines"][0].get_linewidth() == pytest.approx(3.0)
    assert res.artists["fill"][0].get_alpha() == pytest.approx(0.5)
    assert to_rgb(res.artists["text"][0].get_color()) == to_rgb("blue")
    np.testing.assert_allclose(res.artists["line"].get_xdata()[:2], [0.0, 0.5])
    res.update(0)
    res.fig.canvas.draw()
    res.update(3)
    res.fig.canvas.draw()
    plt.close(res.fig)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"running": "median"}, "running"),
        ({"label_side": "left"}, "label_side"),
        ({"x": np.arange(3)}, "x has"),
        ({"band": 0.1}, "pass refs"),
        ({"refs": [1.0, 2.0], "ref_labels": ["a"]}, "ref_labels"),
    ],
)
def test_bad_arguments_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ep.convergence(_looks(10), **kwargs)
    plt.close("all")


def test_empty_values_raise():
    with pytest.raises(ValueError, match="at least one"):
        ep.convergence([])


def test_reference_styles_may_mix_names_and_dash_patterns():
    res = ep.convergence(_looks(), refs=[1.0, 0.5], ref_linestyles=["--", (0, (3, 2))])
    first, second = res.artists["lines"]
    assert first.get_linestyle() == "--"
    assert second.get_linestyle() == "--"
    assert second._unscaled_dash_pattern == (0, (3, 2))
    single = ep.convergence(_looks(), refs=[1.0, 0.5], ref_linestyles=(0, (3, 2)))
    assert len(single.artists["lines"]) == 2
    bands = ep.convergence(_looks(), refs=[1.0, 0.5], band=np.float64(0.1))
    assert len(bands.artists["fill"]) == 2
    plt.close("all")
