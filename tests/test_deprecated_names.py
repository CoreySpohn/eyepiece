"""Renamed public names: each old name warns and resolves to its replacement."""

import pytest

import eyepiece as ep
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


def test_spans_is_shade_intervals_with_a_warning():
    with pytest.warns(DeprecationWarning, match=r"eyepiece\.shade_intervals"):
        old = ep.spans
    assert old is ep.shade_intervals
    assert "shade_intervals" in ep.__all__
    assert "spans" not in ep.__all__


def test_deprecated_name_warns_on_every_access():
    for _ in range(2):
        with pytest.warns(DeprecationWarning):
            ep.spans  # noqa: B018
    assert "spans" not in vars(ep)
