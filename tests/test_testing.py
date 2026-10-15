"""Readability checks: a row collision, a clean figure, a control per fault."""

import matplotlib.pyplot as plt
import pytest

import eyepiece as ep
import eyepiece.testing as ept

DPI = 100
PITCH = 0.06  # figure fraction between two stacked rows


def _figure():
    fig = plt.figure(figsize=(6, 3), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _collision(fig, overlap_px):
    """Third label whose box overlaps the second row's by `overlap_px` px."""
    fig.text(0.1, 0.8, "1  Steer the beam", fontsize=14)
    second = fig.text(0.1, 0.8 - PITCH, "2  Hold the star on the mask", fontsize=14)
    fig.canvas.draw()
    box = second.get_window_extent()
    third = fig.text(0.1, 0.5, "axis label", fontsize=14, va="top")
    # The label's top edge sits `overlap_px` px above the second row's bottom.
    third.set_position((0.1, (box.y0 + overlap_px) / fig.bbox.height))
    return second, third


def test_catches_the_step_collision():
    fig = _figure()
    second, third = _collision(fig, overlap_px=4)
    found = ept.readability_faults(fig, venue="video")
    overlaps = [f for f in found if f.kind == "overlap"]
    assert len(overlaps) == 1
    assert {overlaps[0].artist, overlaps[0].other} == {second, third}
    _, y0, _, y1 = overlaps[0].box
    assert y1 - y0 == pytest.approx(4, abs=0.6)
    with pytest.raises(ept.ReadabilityError, match="overlap"):
        ept.assert_readable(fig)


def test_collision_below_tolerance_passes():
    fig = _figure()
    _collision(fig, overlap_px=1)
    assert not [f for f in ept.readability_faults(fig) if f.kind == "overlap"]


def test_clean_figure_passes():
    fig = _figure()
    fig.text(0.1, 0.8, "1  Steer the beam", fontsize=14)
    fig.text(0.1, 0.6, "2  Hold the star on the mask", fontsize=14)
    ax = fig.add_axes([0.5, 0.2, 0.4, 0.5])
    ax.plot([0, 1], [0, 1], color="C0")
    ax.set_xlabel("time")
    ax.set_ylabel("flux")
    ept.assert_readable(fig, min_text_px=10)
    assert ept.readability_faults(fig, min_text_px=10) == []


def test_text_and_a_backing_box_overlap():
    fig = _figure()
    fig.text(0.1, 0.5, "a boxed label", fontsize=14, bbox=ep.backing())
    fig.text(0.1, 0.5, "another", fontsize=14, va="top")
    fig.canvas.draw()
    box = fig.texts[0].get_bbox_patch().get_window_extent()
    fig.texts[1].set_position((0.1, (box.y0 + 8) / fig.bbox.height))
    found = ept.readability_faults(fig, kinds=("overlap",))
    assert len(found) == 1


def test_outside_frame_text():
    fig = _figure()
    label = fig.text(0.98, 0.5, "clipped by the edge", fontsize=14)
    found = ept.readability_faults(fig, kinds=("outside",))
    assert [f.artist for f in found] == [label]
    assert "right" in found[0].detail


def test_outside_frame_mark():
    fig = _figure()
    ax = fig.add_axes([0.5, 0.2, 0.6, 0.5])  # spills past the right edge
    line = ax.plot([0, 1], [0, 1])[0]
    found = ept.readability_faults(fig, kinds=("outside",))
    assert line in [f.artist for f in found]


def test_small_text_by_venue():
    fig = _figure()
    fig.text(0.1, 0.5, "tiny", fontsize=6)
    small = ept.readability_faults(fig, kinds=("small",), venue="video")
    assert len(small) == 1
    assert ept.readability_faults(fig, kinds=("small",), min_text_px=8) == []
    # The same figure rendered larger has larger text in pixels.
    assert ept.readability_faults(fig, kinds=("small",), venue="video", dpi=400) == []
    assert fig.dpi == DPI
    with pytest.raises(ValueError, match="venue"):
        ept.readability_faults(fig, venue="billboard")


def test_text_on_marks_same_color():
    fig = _figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0, 1], [0.5, 0.5], color="C0", lw=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "label", color="C0", ha="center", va="center", fontsize=16)
    found = ept.readability_faults(fig, kinds=("on-marks",))
    assert len(found) == 1
    assert found[0].other.get_color() == "C0"


def test_text_on_marks_other_color_or_backed_passes():
    fig = _figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0, 1], [0.5, 0.5], color="C0", lw=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.2, 0.5, "other", color="k", ha="center", va="center", fontsize=16)
    ax.text(
        0.7,
        0.5,
        "backed",
        color="C0",
        ha="center",
        va="center",
        fontsize=16,
        bbox=ep.backing(alpha=1.0),
    )
    assert ept.readability_faults(fig, kinds=("on-marks",)) == []


def test_text_on_filled_mark():
    fig = _figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.fill_between([0, 1], 0.3, 0.7, color="C1", alpha=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, "band", color="C1", ha="center", va="center", fontsize=16)
    assert len(ept.readability_faults(fig, kinds=("on-marks",))) == 1


def test_leaders_are_left_to_audit():
    fig = _figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0, 1], [0, 1], color="C0")
    note = ax.text(0.2, 0.8, "note", color="C0", fontsize=14)
    ep.leader(note, ep.Anchor.from_data(ax, (0.8, 0.8)), strict=False)
    assert ept.readability_faults(fig, kinds=("on-marks",)) == []


def test_clip_sampled_at_times_and_hidden_text_skipped():
    fig = _figure()
    first = fig.text(0.1, 0.8, "first row of the list", fontsize=14)
    second = fig.text(0.1, 0.78, "second row of the list", fontsize=14)

    def pose(t):
        second.set_alpha(0.0 if t < 1.0 else 1.0)

    found = ept.readability_faults(fig, at=pose, times=[0.5, 1.5], kinds=("overlap",))
    assert [f.time for f in found] == [1.5]
    assert {found[0].artist, found[0].other} == {first, second}
    assert "t=1.5" in str(found[0])
    with pytest.raises(ValueError, match="together"):
        ept.readability_faults(fig, times=[1.0])


def test_invisible_text_and_out_of_view_ticks_are_not_checked():
    fig = _figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0, 1], [0, 1], color="C0")
    ax.set_xlim(0.2, 0.8)
    ax.text(0.5, 0.5, "hidden", visible=False)
    assert ept.readability_faults(fig, min_text_px=5) == []


def test_deterministic():
    def build():
        fig = _figure()
        _collision(fig, overlap_px=5)
        fig.text(0.97, 0.3, "off the edge", fontsize=5)
        return fig

    one = ept.readability_faults(build(), at=lambda t: None, times=[0.0, 1.0])
    two = ept.readability_faults(build(), at=lambda t: None, times=[0.0, 1.0])
    assert [(f.kind, f.time, f.box, f.detail) for f in one] == [
        (f.kind, f.time, f.box, f.detail) for f in two
    ]
    assert len(one) > 4


def _band(fig, line, height=0.14):
    """Figure-fraction (low, high) of the caption band, measured from the top."""
    return line / 100.0, line / 100.0 + height


def _text_spans(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    spans = []
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        spans.append((1 - box.y1 / fig.bbox.height, 1 - box.y0 / fig.bbox.height))
    return spans


def test_caption_line_is_the_bottom_when_the_bottom_is_empty():
    fig = _figure()
    fig.text(0.1, 0.8, "a heading", fontsize=14)
    assert ept.caption_line(fig) == pytest.approx(100 * (1 - 0.14 - 0.01))


def test_caption_line_clears_text_at_the_bottom():
    fig = _figure()
    fig.text(0.5, 0.04, "what is left: area 0.24 and 0.07", fontsize=14, ha="center")
    fig.text(0.1, 0.9, "heading", fontsize=14)
    line = ept.caption_line(fig)
    low, high = _band(fig, line)
    assert line < 100 * (1 - 0.14 - 0.01)
    for top, bottom in _text_spans(fig):
        assert bottom <= low or top >= high


def test_caption_line_considers_every_sampled_time():
    fig = _figure()
    late = fig.text(0.5, 0.04, "late label", fontsize=14, ha="center")
    late.set_visible(False)

    def pose(t):
        late.set_visible(t > 1.0)

    early = ept.caption_line(fig, at=pose, times=[0.0])
    both = ept.caption_line(fig, at=pose, times=[0.0, 2.0])
    assert early > both


def test_caption_line_ignores_text_outside_the_band_width():
    fig = _figure()
    fig.text(0.0, 0.04, "x", fontsize=14)
    assert ept.caption_line(fig) == pytest.approx(100 * (1 - 0.14 - 0.01))


def test_caption_line_refuses_a_full_figure_and_unpaired_arguments():
    fig = _figure()
    for y in range(0, 100, 5):
        fig.text(0.5, y / 100, "row of text", fontsize=14, ha="center")
    with pytest.raises(ValueError, match="caption band"):
        ept.caption_line(fig)
    with pytest.raises(ValueError, match="together"):
        ept.caption_line(fig, times=[0.0])
