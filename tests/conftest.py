"""Headless backend, a style state that cannot leak, and shared sequences."""

import hwostyle.core
import matplotlib
import numpy as np
import pytest

from eyepiece.prepared import ArrayChannel, AxisSpec, ImageView, Scale, Sequence

matplotlib.use("Agg")

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
