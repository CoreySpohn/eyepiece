"""Renamed public names: each old name warns and resolves to its replacement."""

import pytest
from matplotlib.colors import to_hex

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import motifs


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


@pytest.mark.parametrize("name", ["planet", "model", "truth", "hardware", "text"])
@pytest.mark.parametrize("mode", [None, "dark", "light", "paper"])
def test_resolve_role_is_role_with_a_warning(name, mode):
    with pytest.warns(DeprecationWarning, match=r"eyepiece\.role"):
        old = motifs.resolve_role(name, mode)
    assert old == ep.role(name, mode=mode)
    assert "resolve_role" not in motifs.__all__


def test_role_takes_a_neutral_rank_as_its_gray():
    assert ep.role("hardware") == to_hex(ep.neutral(0.7))
    assert ep.role("text") == to_hex(ep.neutral(1.0))
    with pytest.raises(KeyError, match="neutral ranks"):
        ep.role("nonsense")
