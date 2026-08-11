"""Headless backend, and a style state that cannot leak between tests."""

import hwostyle.core
import matplotlib
import pytest

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
