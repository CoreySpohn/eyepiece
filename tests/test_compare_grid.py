"""compare_grid: a 2D layout of images on one shared norm, gaps allowed."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from eyepiece import compare_grid


def _ramp(scale):
    return scale * np.linspace(-1.0, 1.0, 64).reshape(8, 8)


def test_triangle_layout_shares_one_norm_and_switches_off_gaps():
    images = [[None, _ramp(1.0), None], [_ramp(2.0), None, _ramp(3.0)]]
    res = compare_grid(images, norm="diverging")
    assert res.axes.shape == (2, 3)
    drawn = [im for row in res.artists["image"] for im in row if im is not None]
    assert len(drawn) == 3
    assert all(im.norm is drawn[0].norm for im in drawn)
    assert drawn[0].norm.vmin == -3.0 and drawn[0].norm.vmax == 3.0
    assert res.artists["image"][0][0] is None
    assert not res.axes[0, 0].axison
    assert res.axes[0, 1].axison
    plt.close(res.fig)


def test_ragged_rows_pad_to_a_rectangle():
    res = compare_grid([[_ramp(1.0)], [_ramp(1.0), _ramp(1.0)]])
    assert res.axes.shape == (2, 2)
    assert res.artists["image"][0][1] is None
    plt.close(res.fig)


def test_titles_follow_cells_and_skip_gaps():
    res = compare_grid([[_ramp(1.0), None]], titles=[["a", "ignored"]])
    assert res.axes[0, 0].get_title() == "a"
    assert res.axes[0, 1].get_title() == ""
    plt.close(res.fig)


def test_nan_outside_support_does_not_break_the_norm():
    img = _ramp(2.0)
    img[0, 0] = np.nan
    res = compare_grid([[img]], norm="diverging")
    assert res.artists["image"][0][0].norm.vmax == pytest.approx(2.0, rel=0.1)
    plt.close(res.fig)


def test_handed_axes_are_used_and_shape_checked():
    fig, axes = plt.subplots(1, 2)
    res = compare_grid([[_ramp(1.0), _ramp(1.0)]], axes=np.array([axes]))
    assert res.axes[0, 1] is axes[1]
    with pytest.raises(ValueError, match="expected axes shape"):
        compare_grid([[_ramp(1.0)]], axes=np.array([axes]))
    plt.close(fig)


def test_empty_and_all_gap_layouts_raise():
    with pytest.raises(ValueError, match="at least one image"):
        compare_grid([])
    with pytest.raises(ValueError, match="at least one image"):
        compare_grid([[None, None]])


def test_mismatched_titles_raise():
    with pytest.raises(ValueError, match="titles must match"):
        compare_grid([[_ramp(1.0), _ramp(1.0)]], titles=[["only one"]])


def _positions(fig):
    return [tuple(ax.get_position(original=True).bounds) for ax in fig.axes]


@pytest.mark.parametrize("handed", [False, True])
def test_caller_cax_holds_the_shared_colorbar_and_nothing_moves(handed):
    fig = plt.figure(figsize=(6, 4))
    axes = np.array(
        [
            [
                fig.add_axes([0.05 + 0.3 * j, 0.55 - 0.5 * i, 0.25, 0.4])
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    cax = fig.add_axes([0.7, 0.05, 0.03, 0.9])
    before = _positions(fig)
    images = [[_ramp(1.0), _ramp(2.0)], [_ramp(3.0), None]]
    if handed:
        res = compare_grid(images, axes=axes, cax=cax, cbar_label="flux")
    else:
        res = compare_grid(images, axes=axes, cax=cax)
    cb = res.artists["cbar"]
    assert cb.ax is cax
    assert cb.mappable.norm is res.artists["image"][0][0].norm
    # No inset colorbar was hung off a panel, and no slot moved.
    assert all(not ax.child_axes for ax in axes.ravel())
    assert _positions(fig) == before
    plt.close(fig)
