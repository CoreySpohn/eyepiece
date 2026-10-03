"""signed_trace: a trace about a level, filled one color above and another below."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

import eyepiece as ep
from eyepiece import _style


def _wave(n=60):
    t = np.linspace(0.0, 10.0, n)
    return t, 1.0 + 0.4 * np.sin(t) + 0.1 * np.cos(3.0 * t)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_keys_and_fill_colors_in_either_mode(mode):
    hwostyle.use(mode)
    t, v = _wave()
    res = ep.signed_trace(v, x=t, level=1.0, level_label="static")
    assert set(res.artists) == {"line", "fill", "lines", "text"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    assert isinstance(res.artists["line"], Line2D)
    above, below = res.artists["fill"]
    assert isinstance(above, PolyCollection) and isinstance(below, PolyCollection)
    cmap = _style.cmap("residual")
    assert above.get_facecolor()[0][:3] == pytest.approx(cmap(0.85)[:3])
    assert below.get_facecolor()[0][:3] == pytest.approx(cmap(0.15)[:3])
    assert above.get_alpha() == 0.5
    (level_line,) = res.artists["lines"]
    np.testing.assert_allclose(level_line.get_ydata(), [1.0, 1.0])
    assert res.artists["text"].get_bbox_patch() is not None
    res.fig.canvas.draw()
    plt.close(res.fig)


def test_fills_stay_on_their_side_of_the_level():
    t, v = _wave(200)
    res = ep.signed_trace(v, x=t, level=1.0)
    above, below = res.artists["fill"]
    up = np.concatenate([p.vertices for p in above.get_paths()])
    down = np.concatenate([p.vertices for p in below.get_paths()])
    assert up[:, 1].min() >= 1.0 - 1e-12
    assert down[:, 1].max() <= 1.0 + 1e-12
    plt.close(res.fig)


def test_update_reveals_the_first_k_and_replaces_the_fills_in_place():
    t, v = _wave()
    res = ep.signed_trace(v, x=t, level=1.0)
    ax = res.ax
    fills = res.artists["fill"]
    res.fig.canvas.draw()
    limits = ax.get_xlim(), ax.get_ylim()
    n_children = len(ax.get_children())
    for k in (0, 1, 2, 30, 60):
        res.update(k)
        res.fig.canvas.draw()
        assert res.artists["fill"] is fills
        assert len(fills) == (2 if k >= 2 else 0)
        assert len(res.artists["line"].get_xdata()) == k
        assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert len(ax.get_children()) == n_children
    plt.close(res.fig)


def test_partial_update_matches_a_fresh_draw_of_the_prefix():
    t, v = _wave()
    res = ep.signed_trace(v, x=t, level=1.0)
    res.update(25)
    fresh = ep.signed_trace(v[:25], x=t[:25], level=1.0)
    for a, b in zip(res.artists["fill"], fresh.artists["fill"], strict=True):
        pa = np.concatenate([p.vertices for p in a.get_paths()])
        pb = np.concatenate([p.vertices for p in b.get_paths()])
        np.testing.assert_allclose(pa, pb)
    plt.close(res.fig)
    plt.close(fresh.fig)


def test_limits_cover_the_trace_and_the_level():
    v = np.array([0.2, 0.4, 0.3])
    res = ep.signed_trace(v, level=1.0)
    assert res.ax.get_xlim() == (0.0, 2.0)
    lo, hi = res.ax.get_ylim()
    assert lo < 0.2 and hi > 1.0
    plt.close(res.fig)


def test_values_update_and_overrides():
    t, v = _wave()
    res = ep.signed_trace(
        v,
        x=t,
        level=1.0,
        color="black",
        fill_colors=("tab:red", "tab:blue"),
        fill_alpha=0.3,
        line_kw={"lw": 2.0},
    )
    assert res.artists["line"].get_linewidth() == 2.0
    above, below = res.artists["fill"]
    assert tuple(above.get_facecolor()[0][:3]) == pytest.approx(to_rgba("tab:red")[:3])
    res.update(values=2.0 - v)
    np.testing.assert_allclose(res.artists["line"].get_ydata(), 2.0 - v)
    above, below = res.artists["fill"]
    assert tuple(below.get_facecolor()[0][:3]) == pytest.approx(to_rgba("tab:blue")[:3])
    assert above.get_alpha() == 0.3
    with pytest.raises(ValueError, match="values for"):
        res.update(values=v[:3])
    plt.close(res.fig)


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="at least one"):
        ep.signed_trace([])
    with pytest.raises(ValueError, match="entries for"):
        ep.signed_trace([1.0, 2.0], x=[0.0])
    plt.close("all")
