"""Shared prepared-view builders and narration test data."""

import wave
from pathlib import Path as FilePath

import numpy as np

from eyepiece.prepared import (
    AxisSpec,
    Label,
    Path,
    Points,
    ReferenceLine,
    Region,
    Scale,
    TrackView,
)


def prepared_axes(x_limits=(0.0, 4.0), y_limits=(0.0, 4.0)):
    return AxisSpec("x", "y", x_limits, y_limits)


def linear_scale(vmin=0.0, vmax=1.0):
    return Scale("linear", vmin, vmax)


def marked_track(view_id="track", prefix=""):
    """A TrackView with every mark kind, RA increasing to the left."""
    axes = AxisSpec("RA", "Dec", (-2.0, 2.0), (-1.0, 1.0), x_reverse=True)
    angles = np.linspace(0.0, np.pi, 5)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    return TrackView(
        view_id,
        axes,
        marks=(
            Path(f"{prefix}path/0", circle, source_id="b", weight=1.0),
            Path(f"{prefix}path/1", circle * 0.5, source_id="c", weight=0.25),
            Points(
                f"{prefix}obs",
                np.array([[0.5, 0.25], [-0.5, -0.25]]),
                source_id="b",
                xerr=np.array([0.1, 0.2]),
                yerr=np.array([0.05, 0.15]),
            ),
            Region(f"{prefix}iwa", np.array([0.0, 0.0]), 0.3),
            Region(f"{prefix}owa", np.array([0.0, 0.0]), 1.8, inner_radius=1.5),
            ReferenceLine(f"{prefix}zero", "y", 0.0),
            Label(f"{prefix}clock", "0 d", (0.02, 0.95)),
            Label(f"{prefix}tag", "b", (0.5, 0.25), space="data"),
        ),
    )


def weighted_heads(heavy=1.0, light=0.25):
    """Two weighted point sets beside two weighted paths on one panel."""
    return TrackView(
        "track",
        AxisSpec("x", "y", (0, 1), (0, 1)),
        (
            Path("path/a", np.array([[0.1, 0.1], [0.9, 0.9]]), weight=1.0),
            Path("path/b", np.array([[0.1, 0.9], [0.9, 0.1]]), weight=0.5),
            Points("head/a", np.array([[0.9, 0.9]]), weight=heavy, xerr=np.ones(1)),
            Points("head/b", np.array([[0.9, 0.1]]), weight=light),
        ),
    )


def write_narration_wav(directory, key, audio, rate):
    name = f"{key}.wav"
    data = np.round(np.asarray(audio) * 32767).astype("<i2")
    with wave.open(str(FilePath(directory) / name), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return name


def rms_dbfs(x):
    return 20.0 * np.log10(np.sqrt(np.mean(x**2)) / 32767.0)


def cached_narration_script():
    """Build the two-line script used by the cached narration tests."""
    from eyepiece import narration as nr

    return nr.Script(
        (
            nr.Line(
                "open",
                "The coronagraph blocks the {indicate:star}starlight.",
                pad_after_s=0.4,
            ),
            nr.Line(
                "leak",
                "What leaks through is {indicate:speckle}speckle, and the "
                "{circumscribe:planet}planet must stand out.",
            ),
        )
    )
