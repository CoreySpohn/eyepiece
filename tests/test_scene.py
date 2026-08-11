"""trail depth cues, sky_fan semantics, fading_track alpha ramp."""

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece.scene import fading_track, sky_fan, trail


def _orbit3d(n=50):
    t = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), 0.3 * np.sin(t)])


def test_trail_3d_creates_3d_axes():
    res = trail(_orbit3d())
    assert res.ax.name == "3d"
    plt.close(res.fig)


def test_trail_2d_on_given_axes():
    fig, ax = plt.subplots()
    res = trail(_orbit3d()[:, :2], ax=ax)
    assert res.ax is ax
    plt.close(fig)


def test_trail_marker_sizes_vary_with_depth():
    res = trail(_orbit3d())
    sizes = res.artists["scatter"].get_sizes()
    assert sizes.min() < sizes.max()  # far side shrinks
    plt.close(res.fig)


def test_sky_fan_weights_scale_alpha():
    t = np.linspace(0, 1, 20)
    tracks = [(np.cos(t), np.sin(t)), (np.cos(t) + 0.1, np.sin(t))]
    res = sky_fan(tracks, weights=[1.0, 0.1])
    lines = res.artists["lines"]
    assert lines[0].get_alpha() > lines[1].get_alpha()
    plt.close(res.fig)


def test_sky_fan_data_errorbar_is_a_single_collection():
    t = np.linspace(0, 1, 20)
    tracks = [(np.cos(t), np.sin(t))]
    data = (np.array([0.1, 0.2]), np.array([0.05, 0.15]), np.array([0.01, 0.02]))
    res = sky_fan(tracks, data=data)
    collection = res.artists["collection"]
    assert isinstance(collection, matplotlib.collections.Collection)
    collection.set_alpha(0.5)  # a tuple has no setters; this must not raise
    plt.close(res.fig)


def test_trail_3d_positions_on_2d_axes_raises():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="3D"):
        trail(_orbit3d(), ax=ax)
    plt.close(fig)


def test_fading_track_alpha_ramps():
    t = np.linspace(0, 1, 30)
    res = fading_track(np.column_stack([t, t**2]))
    colors = res.artists["collection"].get_colors()
    assert colors[0, 3] < colors[-1, 3]
    plt.close(res.fig)
