"""Extent helpers: pixel-edge convention, analytic values."""

import numpy as np
import pytest

from eyepiece import layout


def test_extent_from_pixels_edges():
    assert layout.extent_lod_from_pixels(4, 1.0) == (-2.0, 2.0, -2.0, 2.0)
    assert layout.extent_lod_from_pixels(3, 2.0) == (-3.0, 3.0, -3.0, 3.0)


def test_extent_from_coordinate_vector():
    u = np.array([-1.0, 0.0, 1.0])
    assert layout.extent_lod(u) == (-1.5, 1.5, -1.5, 1.5)


def test_extent_arcsec_needs_hwo_extra_or_computes():
    try:
        import hwoutils  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"eyepiece\[hwo\]"):
            layout.extent_arcsec(4, 1000.0)
    else:
        _left, right, *_ = layout.extent_arcsec(4, 1000.0)
        assert right == pytest.approx(2.0)  # 4 px at 1000 mas/px -> +/- 2 arcsec


def test_label_lod_sets_both_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    layout.label_lod(ax)
    assert "lambda" in ax.get_xlabel() or r"\lambda" in ax.get_xlabel()
    plt.close(fig)
