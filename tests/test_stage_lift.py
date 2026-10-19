"""A label arrives attached: it drifts at most 10 px into place while it fades in."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

import eyepiece as ep
import eyepiece.testing as ept


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _board(dpi=100):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=dpi, facecolor="white")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    return fig, ax


def _pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _labels(ax):
    (mark,) = ax.plot([0.2, 0.8], [0.3, 0.3], color="C0")
    plain = ax.text(0.1, 0.8, "plain", color="C0", fontsize=14)
    note = ax.annotate("note", (0.6, 0.65), xytext=(6, 6), textcoords="offset points",
                       color="C0", fontsize=14)  # fmt: skip
    call = ep.callout(mark, "callout", side="above", color="C0").artists["text"]
    return mark, {"plain": plain, "note": note, "call": call}


def _y(fig, text):
    fig.canvas.draw()
    box = text.get_window_extent()
    return box.x0, box.y0


@pytest.mark.parametrize("which", ["plain", "note", "call"])
def test_drift_is_straight_up_within_ten_pixels_and_lands_exactly(which):
    fig, ax = _board(dpi=120)
    _, labels = _labels(ax)
    text = labels[which]
    x_rest, y_rest = _y(fig, text)
    still = _pixels(fig)
    stage = ep.Stage({"label": text})
    xs, ys = [], []
    for u in np.linspace(0.0, 1.0, 31):
        stage.weigh({"label": u}, ease="smooth", lift_pt=6.0)  # 10 px at 120 dpi
        x, y = _y(fig, text)
        xs.append(x)
        ys.append(y)
    assert np.allclose(xs, x_rest)
    assert np.all(np.diff(ys) >= -1e-9)
    assert y_rest - ys[0] == pytest.approx(10.0, abs=0.01)
    assert max(y_rest - y for y in ys) <= 10.0 + 1e-6
    assert ys[-1] == y_rest
    np.testing.assert_array_equal(_pixels(fig), still)


def test_bound_is_ten_pixels_at_the_figure_dpi():
    _, ax = _board(dpi=144)
    _, labels = _labels(ax)
    stage = ep.Stage({"label": labels["plain"]})
    stage.weigh({"label": 0.0}, lift_pt=5.0)  # exactly 10 px
    with pytest.raises(ValueError, match="at most 10 px"):
        stage.weigh({"label": 0.0}, lift_pt=5.1)


def test_a_data_mark_never_drifts():
    _, ax = _board()
    mark, labels = _labels(ax)
    stage = ep.Stage({"mark": mark, "both": [labels["plain"], mark]})
    for group in ("mark", "both"):
        with pytest.raises(ValueError, match="labels and callouts only"):
            stage.weigh({group: 0.5}, lift_pt=4.0)
    assert labels["plain"].get_alpha() is None


def test_annotation_outside_offset_coordinates_raises():
    _, ax = _board()
    note = ax.annotate("note", (0.5, 0.3), xytext=(0.6, 0.4), textcoords="data")
    stage = ep.Stage({"note": note})
    with pytest.raises(ValueError, match="offset points"):
        stage.weigh({"note": 0.5}, lift_pt=4.0)


def test_weigh_without_lift_and_rest_put_labels_back():
    fig, ax = _board()
    _, labels = _labels(ax)
    still = _pixels(fig)
    stage = ep.Stage({"labels": list(labels.values())})
    stage.weigh({"labels": 0.4}, lift_pt=5.0)
    stage.weigh({"labels": 1.0})
    np.testing.assert_array_equal(_pixels(fig), still)
    stage.weigh({"labels": 0.4}, lift_pt=5.0)
    stage.rest()
    np.testing.assert_array_equal(_pixels(fig), still)


def test_colors_stay_the_labels_own_ink():
    _, ax = _board()
    _, labels = _labels(ax)
    stage = ep.Stage({"labels": list(labels.values())})
    stage.weigh({"labels": 0.5}, lift_pt=5.0)
    stage.weigh({"labels": 1.0}, lift_pt=5.0)
    for text in labels.values():
        assert to_rgba(text.get_color()) == to_rgba("C0")


def test_contract_declares_lifted_labels_as_moving_and_clip_is_clean():
    fig, ax = _board(dpi=120)
    mark, labels = _labels(ax)
    stage = ep.Stage({"mark": mark, "labels": list(labels.values())})

    def at(u):
        stage.weigh({"labels": u}, ease="smooth", lift_pt=6.0)

    at(0.0)
    moving = set(stage.contract.moves)
    assert set(labels.values()) <= moving
    assert mark in stage.contract.rest
    frames = np.linspace(0.0, 1.0, 37)
    assert ept.motion_faults(fig, at, frames, moves=[stage]) == []
