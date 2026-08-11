"""show_field: complex 2x2 panel, SubFigure embedding, phase masking."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

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
    fig.canvas.draw()
    # original=True reports the gridspec slot BEFORE aspect="equal" shrinks
    # a panel's box to fit its image data -- get_position() would fold that
    # unrelated shrink in and could pass or fail for the wrong reason.
    pos_before = ax_r.get_position(original=True).bounds

    res = show_field(_field(), fig=left)
    fig.canvas.draw()

    # A numeric position check alone cannot discriminate a stolen-space bug
    # here: matplotlib lays out sibling SubFigures via independent gridspec
    # trees that constrained layout does not renegotiate in response to a
    # leaf's colorbar demand, and fig.colorbar refuses outright to span axes
    # from two different SubFigures (raises ValueError). What DOES change if
    # show_field escapes the SubFigure slot it was handed is which figure
    # ends up owning the axes it creates, so containment is asserted
    # directly rather than relying on a position threshold that cannot fail.
    assert ax_r.get_position(original=True).bounds == pytest.approx(pos_before)
    assert right.axes == [ax_r]
    assert all(ax.get_figure(root=False) is left for ax in res.axes.ravel())
    plt.close(fig)
