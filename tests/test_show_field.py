"""show_field: complex 2x2 panel, SubFigure embedding, phase masking."""

import matplotlib.pyplot as plt
import numpy as np

from eyepiece.images import show_field


def _field():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    f[0, :] = 0.0  # zero-amplitude region -> phase must be masked
    return f


def test_own_figure_shape_and_keys():
    res = show_field(_field())
    assert res.axes.shape == (2, 2)
    assert len(res.artists["image"]) == 4
    plt.close(res.fig)


def test_phase_masked_where_amplitude_zero():
    res = show_field(_field())
    phase = res.artists["image"][3].get_array()
    assert np.ma.is_masked(phase) or np.isnan(np.asarray(phase)).any()
    plt.close(res.fig)


def test_subfigure_embedding_leaves_sibling_alone():
    fig = plt.figure(layout="constrained")
    left, right = fig.subfigures(1, 2)
    ax_r = right.subplots()
    show_field(_field(), fig=left)
    fig.canvas.draw()
    assert ax_r.get_position().width > 0.25  # right half not squeezed
    plt.close(fig)
