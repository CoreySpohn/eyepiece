"""rail_panels: empty axes hung under the planes of a drawn rail."""

import io

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece as ep

TRAIN = [("Pupil", "pupil"), ("FPM", "fpm"), ("Lyot", "lyot"), ("Image", "detector")]
POSITIONS = (1.0, 4.0, 7.0, 10.0)


def _data_rail(figsize=(8.0, 4.0)):
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    res = ep.rail(TRAIN, ax=ax, coords="data", positions=POSITIONS)
    ax.set(xlim=(-0.5, 11.5), ylim=(-6.0, 2.5), aspect="equal")
    ax.axis("off")
    return fig, ax, res


def _center_x(panel, ax):
    box = panel.get_window_extent()
    return ax.transData.inverted().transform([0.5 * (box.x0 + box.x1), 0.0])[0]


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_data_units_hang_panels_under_every_station(mode):
    hwostyle.use(mode)
    fig, ax, rail = _data_rail()
    res = ep.rail_panels(rail, width=2.4, bottom=-5.5, height=2.4)
    assert res.ax is ax
    assert res.artists == {}
    assert res.update is None
    assert len(res.insets) == 4
    assert all(panel in ax.child_axes for panel in res.insets)
    assert not any(panel.get_in_layout() for panel in res.insets)
    for figsize in ((8.0, 4.0), (5.0, 5.0)):
        fig.set_size_inches(figsize)
        fig.canvas.draw()
        for panel, x in zip(res.insets, POSITIONS, strict=True):
            box = panel.get_window_extent()
            (x0, y0), (x1, y1) = ax.transData.inverted().transform(
                [(box.x0, box.y0), (box.x1, box.y1)]
            )
            assert 0.5 * (x0 + x1) == pytest.approx(x, abs=1e-6)
            assert (x1 - x0, y0, y1 - y0) == pytest.approx((2.4, -5.5, 2.4), abs=1e-6)
    fig.savefig(io.BytesIO(), format="png", dpi=72)
    plt.close(fig)


def test_labels_choose_and_order_the_panels_case_insensitively():
    fig, ax, rail = _data_rail()
    res = ep.rail_panels(rail, ["lyot", "Pupil"], width=2.0, bottom=-5.0, height=2.0)
    fig.canvas.draw()
    centers = [_center_x(panel, ax) for panel in res.insets]
    np.testing.assert_allclose(centers, [7.0, 1.0], atol=1e-6)
    one = ep.rail_panels(rail, "FPM", width=2.0, bottom=-5.0, height=2.0)
    fig.canvas.draw()
    assert _center_x(one.insets[0], ax) == pytest.approx(4.0, abs=1e-6)
    plt.close(fig)


def test_a_rail_in_its_own_axes_takes_axes_fraction_data_units():
    fig = plt.figure(figsize=(8.0, 4.0))
    rail_ax = fig.add_axes([0.05, 0.6, 0.9, 0.35])
    rail = ep.rail(TRAIN, ax=rail_ax, positions=(0.1, 0.37, 0.63, 0.9))
    res = ep.rail_panels(rail, width=0.18, bottom=-1.5, height=1.2)
    fig.canvas.draw()
    for panel, x in zip(res.insets, (0.1, 0.37, 0.63, 0.9), strict=True):
        assert _center_x(panel, rail_ax) == pytest.approx(x, abs=1e-6)
        assert panel.get_window_extent().y1 < rail_ax.get_window_extent().y0
    plt.close(fig)


@pytest.mark.parametrize("units", ["figure", "axes"])
def test_figure_and_axes_units_center_on_the_station(units):
    fig, ax, rail = _data_rail()
    res = ep.rail_panels(rail, width=0.15, bottom=0.05, height=0.3, units=units)
    for figsize in ((8.0, 4.0), (6.0, 6.0)):
        fig.set_size_inches(figsize)
        fig.canvas.draw()
        frame = fig.bbox if units == "figure" else ax.get_window_extent()
        for panel, x in zip(res.insets, POSITIONS, strict=True):
            assert _center_x(panel, ax) == pytest.approx(x, abs=1e-6)
            box = panel.get_window_extent()
            assert box.width == pytest.approx(0.15 * frame.width, rel=1e-6)
            assert box.height == pytest.approx(0.3 * frame.height, rel=1e-6)
            assert box.y0 == pytest.approx(frame.y0 + 0.05 * frame.height, abs=1e-6)
    plt.close(fig)


def test_shared_labels_are_named_by_occurrence():
    fig, ax = plt.subplots()
    rail = ep.rail(
        [("Pupil", "pupil"), ("Focal", "focal"), ("Pupil", "lyot"), ("Focal", "focal")],
        ax=ax,
    )
    res = ep.rail_panels(
        rail, ["pupil#1", "Focal#0"], width=0.1, bottom=-0.5, height=0.3
    )
    fig.canvas.draw()
    centers = [_center_x(panel, ax) for panel in res.insets]
    stations = np.linspace(0.10, 0.90, 4)
    np.testing.assert_allclose(centers, [stations[2], stations[1]], atol=1e-6)
    with pytest.raises(ValueError, match="several planes"):
        ep.rail_panels(rail, ["Pupil"], width=0.1, bottom=-0.5, height=0.3)
    plt.close(fig)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"labels": ["Nope"]}, "unknown rail_panels label"),
        ({"units": "inches"}, "units"),
        ({"width": 0.0}, "positive"),
        ({"height": -1.0}, "positive"),
    ],
)
def test_bad_arguments_raise(kwargs, match):
    fig, _, rail = _data_rail()
    call = {"width": 2.0, "bottom": -5.0, "height": 2.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        ep.rail_panels(rail, **call)
    plt.close(fig)


def test_a_result_that_is_not_a_rail_raises():
    fig, ax = plt.subplots()
    res = ep.convergence(np.ones(5), ax=ax, refs=[1.0])
    fake = ep.PlotResult(ax=ax, artists={"lines": res.artists["lines"]})
    with pytest.raises(ValueError, match="rail result"):
        ep.rail_panels(fake, width=1.0, bottom=0.0, height=1.0)
    plt.close(fig)
