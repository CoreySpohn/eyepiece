"""trail depth cues, sky_fan semantics, fading_track alpha ramp."""

import hwostyle
import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece.layout import SourceStyles
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


def test_sky_fan_wraps_the_palette_past_its_end():
    t = np.linspace(0, 1, 10)
    tracks = [(np.cos(t) + k, np.sin(t)) for k in range(8)]  # palettes hold six
    res = sky_fan(tracks)
    lines = res.artists["lines"]
    assert lines[6].get_color() == lines[0].get_color()
    plt.close(res.fig)


def _iwa_disk_facecolor():
    t = np.linspace(0, 1, 10)
    res = sky_fan([(np.cos(t), np.sin(t))], iwa=0.3)
    facecolor = tuple(np.ravel(res.artists["ellipse"].get_facecolor()))
    plt.close(res.fig)
    return facecolor


def test_sky_fan_neutrals_follow_the_mode():
    hwostyle.use("dark")
    dark_disk = _iwa_disk_facecolor()
    with hwostyle.light():
        light_disk = _iwa_disk_facecolor()
    assert dark_disk != light_disk


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


def test_trail_accepts_a_source_styles_entry():
    # the linked-views case is the whole reason this library exists, and its
    # two halves have to compose: `trail(track, style=styles[name])` was a
    # ValueError from inside matplotlib until `style` learned the mapping
    styles = SourceStyles(["star", "b", "c"])
    track = np.stack(
        [np.cos(np.linspace(0, 6, 24)), np.sin(np.linspace(0, 6, 24))], axis=-1
    )
    res = trail(track, style=styles["c"])
    assert res.artists["line"].get_color() == styles["c"]["color"]
    # the marker travels too: same source, same glyph, in every panel.
    # Compared against a reference scatter, because a PathCollection stores
    # the marker path with its transform already applied.
    reference = res.ax.scatter([0], [0], marker=styles["c"]["marker"])
    assert np.array_equal(
        res.artists["scatter"].get_paths()[0].vertices,
        reference.get_paths()[0].vertices,
    )
    plt.close(res.fig)


def test_trail_still_takes_a_plain_color():
    track = np.zeros((5, 2))
    res = trail(track, style="#123456")
    assert res.artists["line"].get_color() == "#123456"
    plt.close(res.fig)


def test_sky_fan_iwa_disk_sits_under_the_tracks():
    """The disk is context; a track crossing it must stay visible."""
    tracks = [(np.linspace(-0.2, 0.2, 50), np.zeros(50))]
    result = sky_fan(tracks, iwa=0.05)
    disk = result.artists["ellipse"]
    assert all(
        disk.get_zorder() < line.get_zorder() for line in result.artists["lines"]
    )
    plt.close(result.ax.figure)
