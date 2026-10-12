"""Scalar scatter colors and non-stealing colorbar geometry."""

from types import SimpleNamespace

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import Normalize

import eyepiece as ep
from eyepiece import _style


def test_inset_right_of_caller_slot():
    fig, axes = plt.subplots(1, 2)
    before = [ax.get_position(original=True).bounds for ax in axes]
    result = ep.scatter_values(
        [0, 1], [2, 3], [0.2, 0.8], ax=axes[0], vmin=0, vmax=1, cbar_label="probability"
    )
    assert isinstance(result, ep.PlotResult)
    assert result.ax is axes[0]
    assert set(result.artists) == {"scatter", "cbar"}
    assert len(result.insets) == 1
    fig.canvas.draw()
    assert result.insets[0].get_position().x0 > axes[0].get_position().x1
    assert result.insets[0] in axes[0].child_axes
    assert result.artists["cbar"].ax is result.insets[0]
    assert [ax.get_position(original=True).bounds for ax in axes] == before
    assert fig.get_layout_engine() is None
    assert axes[0].get_shared_x_axes().get_siblings(axes[0]) == [axes[0]]
    assert axes[0].get_shared_y_axes().get_siblings(axes[0]) == [axes[0]]
    scatter = result.artists["scatter"]
    np.testing.assert_allclose(scatter.get_array(), [0.2, 0.8])
    np.testing.assert_allclose(scatter.get_offsets(), [[0, 2], [1, 3]])
    assert scatter.get_clim() == (0, 1)
    assert result.insets[0].get_ylabel() == "probability"
    plt.close(fig)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_probability_uses_current_hwostyle(mode):
    hwostyle.use(mode)
    result = ep.scatter_values([0], [1], [0.5], cmap="probability", colorbar=False)
    expected = hwostyle.cmaps.probability
    if isinstance(expected, str):
        expected = matplotlib.colormaps[expected]
    np.testing.assert_allclose(
        result.artists["scatter"].cmap([0, 0.5, 1]), expected([0, 0.5, 1])
    )
    assert result.insets == ()
    assert set(result.artists) == {"scatter"}
    plt.close(result.fig)


def test_statistic_fallback_and_future_role(monkeypatch):
    for source, expected in [
        (SimpleNamespace(), "PuOr"),
        (SimpleNamespace(statistic="viridis"), "viridis"),
    ]:
        monkeypatch.setattr(_style, "_light_cmaps", lambda source=source: source)
        monkeypatch.setattr(hwostyle, "current_mode", lambda: None)
        result = ep.scatter_values([0], [1], [3], cmap="statistic")
        assert result.artists["scatter"].cmap.name == expected
        plt.close(result.fig)


def test_installed_statistic_fallback():
    hwostyle.use("light")
    expected = getattr(hwostyle.cmaps, "statistic", "PuOr")
    result = ep.scatter_values([0], [1], [3], cmap="statistic")
    assert result.artists["scatter"].cmap.name == (
        expected if isinstance(expected, str) else expected.name
    )
    plt.close(result.fig)


@pytest.mark.parametrize("cmap", ["viridis", matplotlib.colormaps["viridis"]])
def test_colormap_overrides_and_keyword_routing(cmap):
    norm = Normalize(-1, 2)
    scatter_kw = {"s": 49, "norm": norm}
    cbar_kw = {"label": "override", "ticks": [0, 1]}
    result = ep.scatter_values(
        [0, 1], [2, 3], [0.2, 0.8], cmap=cmap, scatter_kw=scatter_kw, cbar_kw=cbar_kw
    )
    assert result.artists["scatter"].cmap.name == "viridis"
    assert result.artists["scatter"].norm is norm
    assert result.artists["scatter"].get_sizes()[0] == 49
    assert result.insets[0].get_ylabel() == "override"
    np.testing.assert_array_equal(result.artists["cbar"].get_ticks(), [0, 1])
    assert scatter_kw == {"s": 49, "norm": norm}
    assert cbar_kw == {"label": "override", "ticks": [0, 1]}
    plt.close(result.fig)


def test_shape_error():
    with pytest.raises(ValueError, match=r"expected.*received"):
        ep.scatter_values([0, 1], [2], [3])
