"""Role colors include neutral ranks."""

import pytest
from matplotlib.colors import to_hex

import eyepiece as ep


def test_role_takes_a_neutral_rank_as_its_gray():
    assert ep.style.role("hardware") == to_hex(ep.style.neutral(0.7))
    assert ep.style.role("text") == to_hex(ep.style.neutral(1.0))
    with pytest.raises(KeyError, match="neutral ranks"):
        ep.style.role("nonsense")
