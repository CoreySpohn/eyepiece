"""Renamed public names: each old name warns and resolves to its replacement."""

import pytest

import eyepiece.testing as ept


def test_testing_span_is_window_with_a_warning():
    with pytest.warns(DeprecationWarning, match=r"eyepiece\.testing\.Window"):
        old = ept.Span
    assert old is ept.Window
    assert "Window" in ept.__all__
    assert "Span" not in ept.__all__


def test_from_import_of_testing_span_warns():
    with pytest.warns(DeprecationWarning, match="Window"):
        from eyepiece.testing import Span
    assert Span(0.0, 1.0) == ept.Window(0.0, 1.0)
