"""Headless backend, a style state that cannot leak, and shared sequences."""

import importlib
import os

import hwostyle.core
import matplotlib
import numpy as np
import pytest

from eyepiece.prepared import (
    ArrayChannel,
    AxisSpec,
    Clock,
    ImageView,
    Label,
    Path,
    PathWindow,
    Points,
    Region,
    Scale,
    Sequence,
    TrackView,
)

matplotlib.use("Agg")

# The render CI job sets EYEPIECE_REQUIRE_MANIM=1. There, the Manim tests may
# not skip: Manim must import at session start, and any skipped Manim test
# fails the session. Without it (the base job), those tests skip through
# `pytest.importorskip("manim")` when the presentation packages are absent.
_REQUIRE_MANIM = os.environ.get("EYEPIECE_REQUIRE_MANIM") == "1"
_MANIM_SKIPS = []


def pytest_configure(config):
    if _REQUIRE_MANIM:
        importlib.import_module("manim")


def _record_manim_skip(report):
    if _REQUIRE_MANIM and report.skipped and "test_manim_" in report.nodeid:
        _MANIM_SKIPS.append(report.nodeid)


def pytest_collectreport(report):
    _record_manim_skip(report)


def pytest_runtest_logreport(report):
    _record_manim_skip(report)


def pytest_sessionfinish(session, exitstatus):
    if _MANIM_SKIPS:
        print(f"\nEYEPIECE_REQUIRE_MANIM=1 but Manim tests skipped: {_MANIM_SKIPS}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


_STYLE_STATE = ("_current_mode", "_current_family", "_activated", "palette", "cmaps")


@pytest.fixture(autouse=True)
def restore_style_state():
    """Put hwostyle and the rcParams back the way the test found them.

    hwostyle.use() is global: it rebinds the mode, the palette, and the
    colormaps, and updates rcParams. A test that activates a mode would
    otherwise hand every test after it a style the suite never asked for,
    making results depend on collection order. There is no public call for
    "no mode is active", which is the state a fresh interpreter starts in
    and which the zero-style fallback tests depend on, so the module state
    itself is snapshotted and put back.
    """
    saved = {name: getattr(hwostyle.core, name) for name in _STYLE_STATE}
    saved_rc = matplotlib.rcParams.copy()
    yield
    for name, value in saved.items():
        setattr(hwostyle.core, name, value)
    matplotlib.rcParams.update(saved_rc)


def make_image_sequence():
    """Three constant (2, 3) frames at values 0, 10, 20, sampled at t = 0, 1, 10 s."""
    cube = np.array([np.full((2, 3), v) for v in (0.0, 10.0, 20.0)])
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 3), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
    )
    return Sequence(
        view,
        np.array([0.0, 1.0, 10.0]),
        "s",
        (ArrayChannel("image", "data", cube),),
    )


@pytest.fixture
def image_sequence():
    return make_image_sequence


def make_track_sequence():
    """A unit-circle track sampled at eight equal times, RA increasing left.

    The full circle is one static `Path` revealed as a growing trail, one
    `Points` head follows it through an `ArrayChannel`, a `Clock` label
    reports acquisition time, and `Region("iwa")` is a 0.5-radius circle
    at the origin. Synthetic NumPy data only.
    """
    times = np.arange(8.0)
    angles = 2 * np.pi * times / 8
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    heads = circle[:, None, :]
    view = TrackView(
        "track",
        AxisSpec("RA", "Dec", (-1.5, 1.5), (-1.5, 1.5), x_reverse=True),
        marks=(
            Region("iwa", np.array([0.0, 0.0]), 0.5),
            Path("orbit", circle, source_id="b"),
            Points("head", heads[0], source_id="b"),
            Label("clock", "0 d", (0.02, 0.95)),
        ),
    )
    return Sequence(
        view,
        times,
        "d",
        (
            ArrayChannel("head", "xy", heads),
            PathWindow("orbit"),
            Clock("clock"),
        ),
    )


@pytest.fixture
def track_sequence():
    return make_track_sequence()
