"""A relay note is fixed neutral furniture in the panel corner."""

import hwostyle
import matplotlib as mpl
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


@pytest.mark.parametrize("mode", ["light", "dark"])
@pytest.mark.parametrize("corner", ["upper left", "upper right"])
def test_carried_tag_placement_and_ink(mode, corner):
    hwostyle.use(mode)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
    tag = ep.carried_tag(ax, "(from clip 1)", corner=corner)
    text = tag.artists["text"]
    fig.canvas.draw()
    box = text.get_window_extent()
    gap = 6 * fig.dpi / 72
    edge = box.x0 - ax.bbox.x0 if corner == "upper left" else ax.bbox.x1 - box.x1
    assert edge == pytest.approx(gap)
    assert ax.bbox.y1 - box.y1 == pytest.approx(gap)
    assert text.get_fontstyle() == "italic"
    assert text.get_fontsize() < mpl.rcParams["font.size"]
    assert to_rgba(text.get_color()) == to_rgba(ep.neutral(0.55))
    assert text in tag.contract.rest


def test_tag_fades_with_panel_but_stays_fixed_when_labels_lift():
    fig, ax = plt.subplots()
    text = ep.carried_tag(ax, "(from clip 1)").artists["text"]
    fig.canvas.draw()
    before = text.get_window_extent().get_points().copy()
    stage = ep.Stage({"tag": text})
    stage.weigh({"tag": 0.5}, ease="linear", lift_pt=3)
    fig.canvas.draw()
    np.testing.assert_array_equal(text.get_window_extent().get_points(), before)
    assert to_rgba(text.get_color())[3] * (text.get_alpha() or 1) == pytest.approx(0.5)
    assert text in stage.contract.rest and text not in stage.contract.moves
    panel = ep.Stage({"panel": ax})
    panel.set(shown=[])
    assert not text.get_visible() or to_rgba(text.get_color())[3] == 0


def test_tag_is_not_data_and_is_exempt_only_from_the_main_text_size_floor():
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_gid("curve")
    text = ep.carried_tag(ax, "(from clip 1)", fontsize=4).artists["text"]
    assert ept.readability_faults(fig) == []
    assert ept.meaning_faults(fig, keys={"curve": "measured"}) == []
    ax.text(0, 1, "overlap", transform=text.get_transform(), va="top")
    assert "overlap" in [f.kind for f in ept.readability_faults(fig)]
