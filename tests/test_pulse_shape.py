"""A public plateau shape is the same pulse used by attention moves."""

import numpy as np
import pytest

import eyepiece as ep
from eyepiece.attention import _pulse


def test_pulse_shape_endpoints_plateau_and_monotone_edges():
    assert ep.pulse_shape(0) == ep.pulse_shape(1) == 0
    assert ep.pulse_shape(-1) == ep.pulse_shape(2) == 0
    assert all(ep.pulse_shape(u) == 1 for u in np.linspace(0.35, 0.65, 31))
    rise = [ep.pulse_shape(u) for u in np.linspace(0, 0.35, 101)]
    fall = [ep.pulse_shape(u) for u in np.linspace(0.65, 1, 101)]
    assert np.all(np.diff(rise) >= 0)
    assert np.all(np.diff(fall) <= 0)
    for u in np.linspace(0, 1, 101):
        assert ep.pulse_shape(u) == pytest.approx(_pulse(u, "plateau"))


def test_pulse_shape_relative_phase_lengths_and_custom_ease():
    assert ep.pulse_shape(0.125, rise=1, hold=1, fall=2, ease="linear") == 0.5
    assert ep.pulse_shape(0.625, rise=1, hold=1, fall=2, ease=lambda u: u) == 0.75
    assert ep.pulse_shape(0.5, hold=0) == 1


@pytest.mark.parametrize("kwargs", [{"rise": 0}, {"hold": -1}, {"fall": float("inf")}])
def test_pulse_shape_rejects_invalid_phases(kwargs):
    with pytest.raises(ValueError):
        ep.pulse_shape(0.5, **kwargs)
