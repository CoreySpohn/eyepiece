"""hist_fill: a histogram that fills as samples arrive, on a pinned scale."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

import eyepiece as ep
from eyepiece import _style

EDGES = np.arange(0.0, 6.01, 0.5)


def _looks(n=300):
    return np.random.default_rng(1).exponential(1.0, n)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_keys_counts_and_tones_in_either_mode(mode):
    hwostyle.use(mode)
    looks = _looks()
    res = ep.hist_fill(
        looks, EDGES, law=lambda x: np.exp(-x), law_label="exponential", rug=10
    )
    assert set(res.artists) == {"hist", "line", "text", "scatter"}
    assert set(res.artists) <= ep.ARTIST_KEYS
    assert isinstance(res.artists["hist"], BarContainer)
    assert isinstance(res.artists["line"], Line2D)
    assert isinstance(res.artists["scatter"], PathCollection)
    heights = [bar.get_height() for bar in res.artists["hist"]]
    expected, _ = np.histogram(looks, EDGES)
    np.testing.assert_array_equal(heights, expected)
    bar_color = res.artists["hist"][0].get_facecolor()[:3]
    assert bar_color == pytest.approx(_style.neutral(0.55))
    assert res.artists["text"].get_bbox_patch() is not None
    res.fig.canvas.draw()
    plt.close(res.fig)


def test_y_axis_is_pinned_to_the_final_or_expected_counts():
    looks = _looks()
    res = ep.hist_fill(looks, EDGES, law=lambda x: np.exp(-x))
    counts, _ = np.histogram(looks, EDGES)
    expected_top = looks.size * 0.5 * 1.0
    assert res.ax.get_ylim() == pytest.approx(
        (0.0, 1.08 * max(counts.max(), expected_top))
    )
    assert res.ax.get_xlim() == (0.0, 6.0)
    curve = res.artists["line"]
    np.testing.assert_allclose(
        curve.get_ydata(), looks.size * 0.5 * np.exp(-curve.get_xdata())
    )
    plt.close(res.fig)


def test_update_counts_the_first_k_without_new_artists_or_new_limits():
    looks = _looks()
    res = ep.hist_fill(looks, EDGES, rug=5, ymax=120.0)
    ax = res.ax
    res.fig.canvas.draw()
    limits = ax.get_xlim(), ax.get_ylim()
    n_children = len(ax.get_children())
    for k in (0, 1, 3, 50, 300):
        res.update(k)
        res.fig.canvas.draw()
        heights = [bar.get_height() for bar in res.artists["hist"]]
        np.testing.assert_array_equal(heights, np.histogram(looks[:k], EDGES)[0])
        assert res.artists["scatter"].get_offsets().shape == (min(k, 5), 2)
        assert len(ax.get_children()) == n_children
        assert (ax.get_xlim(), ax.get_ylim()) == limits
    assert ax.get_ylim()[1] == 120.0
    np.testing.assert_allclose(res.artists["scatter"].get_offsets()[:, 0], looks[:5])
    plt.close(res.fig)


def test_rug_alpha_and_show_law_persist_until_changed():
    res = ep.hist_fill(_looks(), EDGES, law=lambda x: np.exp(-x), law_label="e", rug=4)
    res.update(10, rug_alpha=0.25, show_law=False)
    assert res.artists["scatter"].get_alpha() == 0.25
    assert not res.artists["line"].get_visible()
    assert not res.artists["text"].get_visible()
    res.update(20)
    assert res.artists["scatter"].get_alpha() == 0.25
    assert not res.artists["line"].get_visible()
    res.update(show_law=True)
    assert res.artists["line"].get_visible() and res.artists["text"].get_visible()
    plt.close(res.fig)


def test_a_bin_count_spans_the_data_and_samples_outside_edges_are_dropped():
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    res = ep.hist_fill(values, 4)
    assert res.ax.get_xlim() == (0.0, 4.0)
    assert [b.get_height() for b in res.artists["hist"]] == [1, 1, 1, 2]
    res = ep.hist_fill(np.array([-1.0, 0.5, 9.0]), [0.0, 1.0, 2.0])
    assert [b.get_height() for b in res.artists["hist"]] == [1, 0]
    plt.close("all")


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (([],), {}, "at least one"),
        (([1.0], [0.0, 2.0, 1.0]), {}, "increasing"),
        (([1.0], [0.0, 1.0, 3.0]), {"law": np.exp}, "equal bins"),
    ],
)
def test_bad_inputs_raise(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        ep.hist_fill(*args, **kwargs)
    plt.close("all")
