"""Readability checks: a row collision, a clean figure, a control per fault."""

import matplotlib
import matplotlib.pyplot as plt
import pytest

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import narration as nr

DPI = 100
PITCH = 0.06  # figure fraction between two stacked rows


def _figure():
    fig = plt.figure(figsize=(6, 3), dpi=DPI)
    fig.patch.set_facecolor("white")
    return fig


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


@pytest.mark.parametrize("overlap_px", [1, 4])
def test_catches_the_step_collision(overlap_px):
    fig = _figure()
    second, third = _collision(fig, overlap_px=overlap_px)
    found = ept.readability_faults(fig, venue="video")
    overlaps = [f for f in found if f.kind == "overlap"]
    if overlap_px == 1:
        assert overlaps == []
        return
    assert len(overlaps) == 1
    assert {overlaps[0].artist, overlaps[0].other} == {second, third}
    _, y0, _, y1 = overlaps[0].box
    assert y1 - y0 == pytest.approx(4, abs=0.6)
    with pytest.raises(ept.ReadabilityError, match="overlap"):
        ept.assert_readable(fig)


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


@pytest.mark.parametrize(
    ("bbox", "expected_faults"),
    [
        pytest.param(dict(facecolor="white", alpha=0.85), 0, id="alpha-0.85"),
        pytest.param(dict(facecolor="white", alpha=0.7), 1, id="alpha-0.7"),
        pytest.param(dict(facecolor=(1.0, 1.0, 1.0, 0.85)), 0, id="facecolor-alpha"),
    ],
)
def test_text_on_marks_backing_opacity(bbox, expected_faults):
    fig = _figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot([0, 1], [0.5, 0.5], color="C0", lw=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.5,
        "label",
        color="C0",
        ha="center",
        va="center",
        fontsize=16,
        bbox=bbox,
    )
    found = ept.readability_faults(fig, kinds=("on-marks",))
    assert [fault.kind for fault in found] == ["on-marks"] * expected_faults


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
    fig.text(0.0, 0.04, "x", fontsize=14)
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


def test_caption_line_refuses_a_full_figure_and_unpaired_arguments():
    fig = _figure()
    for y in range(0, 100, 5):
        fig.text(0.5, y / 100, "row of text", fontsize=14, ha="center")
    with pytest.raises(ValueError, match="caption band"):
        ept.caption_line(fig)
    with pytest.raises(ValueError, match="together"):
        ept.caption_line(fig, times=[0.0])


# Speech rate


def _narration(rates, n_words=20, pad_s=1.5):
    """A narration with one line per rate (words/min), words evenly spaced."""
    lines, spoken = [], {}
    for i, wpm in enumerate(rates):
        key = f"l{i}"
        lines.append(
            nr.Line(key, " ".join(f"w{j}" for j in range(n_words)), pad_before_s=pad_s)
        )
        step = 60.0 / wpm
        words = tuple(
            nr.Word(f"w{j}", j * step, (j + 1) * step) for j in range(n_words)
        )
        spoken[key] = nr.Spoken(key, n_words * step, words)
    return nr.Narration(nr.Script(tuple(lines)), spoken)


def test_speech_rate_recovers_a_known_rate_ignoring_pauses():
    # 1.5 s of padding before each line and 0.5 s after must not matter.
    result = ept.speech_rate(_narration([120.0, 90.0]))
    assert [k for k, _ in result.rates] == ["l0", "l1"]
    assert [w for _, w in result.rates] == pytest.approx([120.0, 90.0])
    assert result.fast == ()
    rates, fast, _ = result  # unpacks
    assert rates == result.rates and fast == ()


def test_speech_rate_reports_just_above_the_ceiling_not_just_below():
    result = ept.speech_rate(_narration([159.0, 161.0]), ceiling=160)
    assert [k for k, _ in result.fast] == ["l1"]
    assert result.ceiling == 160.0


def test_speech_rate_finds_the_fast_line_of_a_d51_like_spread():
    spread = [124.0, 138.0, 151.0, 187.0, 143.0]
    result = ept.speech_rate(_narration(spread))
    assert [(k, round(w)) for k, w in result.fast] == [("l3", 187)]
    with pytest.raises(ept.SpeechRateError, match="l3: 187 words/min") as err:
        ept.assert_speech_rate(_narration(spread))
    assert err.value.result.fast == result.fast
    assert ept.assert_speech_rate(_narration(spread), ceiling=190).fast == ()


def test_speech_rate_counts_a_pause_inside_a_line_and_skips_bare_dashes():
    line = nr.Line("a", "one two - three four")
    words = (
        nr.Word("one", 0.0, 0.5),
        nr.Word("two", 0.5, 1.0),
        nr.Word("-", 1.0, 1.0),
        nr.Word("three", 2.0, 2.5),
        nr.Word("four", 2.5, 3.0),
    )
    narration = nr.Narration(nr.Script((line,)), {"a": nr.Spoken("a", 3.0, words)})
    ((key, wpm),) = ept.speech_rate(narration).rates
    assert key == "a" and wpm == pytest.approx(60.0 * 4 / 3.0)


def test_halo_lint_names_the_text_and_ignores_backing_boxes():
    from matplotlib import patheffects

    fig = _figure()
    halo = fig.text(
        0.1,
        0.7,
        "haloed label",
        gid="subject-label",
        path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
    )
    fig.text(0.1, 0.4, "boxed label", bbox=ep.backing())
    faults = ept.halo_faults(fig)
    assert len(faults) == 1
    assert faults[0].artist is halo
    assert faults[0].text == "haloed label"
    assert all(
        word in str(faults[0]) for word in ("subject-label", "haloed label", "backing")
    )
    with pytest.raises(ept.HaloError, match="haloed label") as error:
        ept.assert_no_halos(fig)
    assert error.value.faults[0].artist is halo
    halo.set_path_effects([])
    assert ept.assert_no_halos(fig) == []


def test_halo_lint_skips_labels_that_are_not_drawn():
    from matplotlib import patheffects

    fig = _figure()
    fig.text(
        0.1,
        0.7,
        "hidden",
        visible=False,
        path_effects=[patheffects.withStroke(linewidth=3)],
    )
    fig.text(
        0.1,
        0.4,
        "transparent",
        alpha=0,
        path_effects=[patheffects.withStroke(linewidth=3)],
    )
    assert ept.halo_faults(fig) == []


def test_halo_lint_catches_an_opaque_stroke_with_transparent_fill():
    from matplotlib import patheffects

    fig = _figure()
    label = fig.text(
        0.1,
        0.7,
        "outline label",
        color="none",
        path_effects=[patheffects.withStroke(linewidth=3, foreground="black")],
    )
    faults = ept.halo_faults(fig)
    assert [fault.artist for fault in faults] == [label]
    label.set_alpha(0)
    assert ept.halo_faults(fig) == []


@pytest.mark.parametrize(
    "stroke",
    [
        dict(linewidth=3, foreground="none"),
        dict(linewidth=3, foreground="black", alpha=0),
        dict(linewidth=0, foreground="black"),
    ],
)
def test_halo_lint_skips_transparent_fill_without_a_visible_stroke(stroke):
    from matplotlib import patheffects

    fig = _figure()
    fig.text(
        0.1,
        0.7,
        "invisible outline",
        color="none",
        path_effects=[patheffects.withStroke(**stroke)],
    )
    assert ept.halo_faults(fig) == []


# Visual gates: data occlusion and panels that draw too little to see.
@pytest.mark.parametrize("mesh", [False, True])
@pytest.mark.parametrize("backed", [False, True])
def test_text_hiding_image_data_is_a_fault(mesh, backed):
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    data = np.arange(100).reshape(10, 10)
    image = ax.pcolormesh(data) if mesh else ax.imshow(data, origin="lower")
    label = ax.text(
        0.5,
        0.5,
        "data",
        transform=ax.transAxes,
        ha="center",
        va="center",
        bbox={"facecolor": "white", "alpha": 1} if backed else None,
    )
    found = ept.readability_faults(fig, kinds=("on-image",))
    assert len(found) == 1
    assert found[0].artist is label and found[0].other is image
    assert found[0].time is None
    with pytest.raises(ept.ReadabilityError, match="on-image"):
        ept.assert_readable(fig, kinds=("on-image",))
    assert (
        ept.readability_faults(fig, kinds=("on-image",), may_overlap=[(image, label)])
        == []
    )
    result = ep.PlotResult(
        ax=ax, artists={}, contract=ep.MotionContract(may_overlap=((label, image),))
    )
    assert ept.readability_faults(fig, kinds=("on-image",), moves=[result]) == []


@pytest.mark.parametrize("mesh", [False, True])
@pytest.mark.parametrize("invalid", ["mask", "nan"])
def test_text_over_invalid_image_region_is_clean(mesh, invalid):
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    data = np.ma.array(np.arange(100, dtype=float).reshape(10, 10), mask=False)
    if invalid == "mask":
        data.mask[2:8, 2:8] = True
    else:
        data[2:8, 2:8] = np.nan
    # Even an opaque bad-sample color must not count as valid data.
    cmap = plt.get_cmap("viridis").with_extremes(bad="red")
    if mesh:
        ax.pcolormesh(data, cmap=cmap)
    else:
        ax.imshow(data, origin="lower", cmap=cmap)
    ax.text(0.5, 0.5, "x", transform=ax.transAxes, ha="center", va="center")
    assert ept.readability_faults(fig, kinds=("on-image",)) == []


def test_image_overlap_respects_clipping_visibility_and_drawn_extent():
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    image = ax.imshow(np.arange(4).reshape(2, 2), extent=(0, 1, 0, 1))
    ax.set(xlim=(0, 2), ylim=(0, 2))
    label = ax.text(1.5, 1.5, "x", ha="center", va="center")
    assert ept.readability_faults(fig, kinds=("on-image",)) == []
    label.set_position((0.5, 0.5))
    image.set_alpha(0)
    assert ept.readability_faults(fig, kinds=("on-image",)) == []
    image.set_alpha(1)
    label.set_alpha(0)
    assert ept.readability_faults(fig, kinds=("on-image",)) == []
    label.set_alpha(1)
    label.set_clip_on(True)
    label.set_clip_box(matplotlib.transforms.Bbox.from_bounds(0, 0, 1, 1))
    assert ept.readability_faults(fig, kinds=("on-image",)) == []


@pytest.mark.parametrize(
    "content", ["tiny", "faint", "line", "blank", "uniform", "image"]
)
def test_visibly_empty_panel_and_clean_controls(content):
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    if content == "tiny":
        ax.plot([0.5], [0.5], "o", ms=0.1, mew=0)
        ax.grid(True)
        ax.set_title("Decoration is not data")
    elif content == "faint":
        ax.plot([0, 1], [0, 1], alpha=0.01)
    elif content == "line":
        ax.plot([0, 1], [0, 1])
    else:
        data = np.zeros((10, 10))
        if content == "uniform":
            data[5:, :] = 0.005
        elif content == "image":
            data[5:, :] = 1
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
    found = ept.readability_faults(fig, kinds=("empty-panel",))
    if content in ("tiny", "faint", "blank", "uniform"):
        assert len(found) == 1 and found[0].artist is ax
        assert found[0].other is None
    else:
        assert found == []


def test_empty_panel_thresholds_and_waiver():
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    ax.plot([0.5], [0.5], "o", ms=0.1, mew=0)
    assert ept.readability_faults(fig, kinds=("empty-panel",), min_ink_fraction=0) == []
    ax.clear()
    data = np.zeros((10, 10))
    data[5:, :] = 0.005
    ax.imshow(data, cmap="gray", vmin=0, vmax=1)
    assert ept.readability_faults(fig, kinds=("empty-panel",), min_image_range=0) == []
    waiver = ept.Waive("empty-panel", artist=ax, reason="intentional blank control")
    ept.assert_readable(fig, kinds=("empty-panel",), waive=[waiver])
    assert (
        len(ept.readability_faults(fig, kinds=("empty-panel",), waive=[waiver]).waived)
        == 1
    )


@pytest.mark.parametrize(
    "parameter", ["min_ink_fraction", "min_image_range", "ink_tol"]
)
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_visual_thresholds_raise(parameter, value):
    fig = _figure()
    with pytest.raises(ValueError, match=parameter):
        ept.readability_faults(fig, **{parameter: value})


def test_unused_axes_and_hidden_content_are_not_empty_panels():
    fig, axes = plt.subplots(1, 3, figsize=(3, 1), dpi=80)
    axes[1].plot([0, 1], [0, 1], visible=False)
    axes[2].plot([0, 1], [0, 1], alpha=0)
    assert ept.readability_faults(fig, kinds=("empty-panel",)) == []


@pytest.mark.parametrize("mesh", [False, True])
def test_image_overlap_uses_rendered_transforms_and_view(mesh):
    import numpy as np
    from matplotlib.transforms import Affine2D

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    data = np.arange(16).reshape(4, 4)
    transform = Affine2D().translate(10, 20) + ax.transData
    if mesh:
        ax.pcolormesh(data, transform=transform)
    else:
        ax.imshow(data, origin="upper", extent=(0, 4, 0, 4), transform=transform)
    ax.set(xlim=(14, 10), ylim=(20, 24))
    label = ax.text(12, 22, "x", ha="center", va="center")
    found = ept.readability_faults(fig, kinds=("on-image",))
    assert len(found) == 1 and found[0].artist is label
    ax.set_xlim(0, 4)
    assert ept.readability_faults(fig, kinds=("on-image",)) == []


@pytest.mark.parametrize("rgba", [False, True])
def test_image_overlap_respects_array_opacity(rgba):
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    data = np.arange(100).reshape(10, 10)
    if rgba:
        data = plt.get_cmap("viridis")(data / 100)
    opacity = np.ones((10, 10))
    opacity[2:8, 2:8] = 0
    ax.imshow(data, origin="lower", alpha=opacity)
    ax.text(4.5, 4.5, "x", ha="center", va="center")
    assert ept.readability_faults(fig, kinds=("on-image",)) == []


def test_overlap_contract_span_is_applied_when_reusing_frames():
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    image = ax.imshow(np.arange(100).reshape(10, 10))
    label = ax.text(4, 4, "x")
    declared = {"span": (0.0, 0.2), "may_overlap": [(label, image)]}
    found = ept.readability_faults(
        fig,
        at=lambda t: None,
        times=[0, 0.2, 0.3, 0.4],
        frame_key=lambda t: 1,
        kinds=("on-image",),
        moves=[declared],
    )
    assert [f.time for f in found] == [0.4]


def test_visual_checks_preserve_image_data_canvas_and_pixels():
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    image = ax.imshow(np.arange(100).reshape(10, 10), alpha=0.5)
    ax.text(4, 4, "x")
    fig.canvas.draw()
    canvas = fig.canvas
    before = np.asarray(canvas.buffer_rgba()).copy()
    data = image.get_array().copy()
    ept.readability_faults(fig)
    assert fig.canvas is canvas
    np.testing.assert_array_equal(image.get_array(), data)
    np.testing.assert_array_equal(canvas.buffer_rgba(), before)
    assert image.get_alpha() == 0.5


def test_image_panel_range_uses_the_composited_display():
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    ax.imshow(np.arange(100).reshape(10, 10))
    ax.imshow(np.zeros((10, 10)))
    found = ept.readability_faults(fig, kinds=("empty-panel",))
    assert len(found) == 1 and found[0].artist is ax


def test_visual_checks_use_agg_and_restore_a_vector_canvas():
    import numpy as np
    from matplotlib.backends.backend_svg import FigureCanvasSVG

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    ax.imshow(np.arange(100).reshape(10, 10))
    label = ax.text(4, 4, "x")
    canvas = FigureCanvasSVG(fig)
    found = ept.readability_faults(fig, kinds=("on-image",))
    assert len(found) == 1 and found[0].artist is label
    assert fig.canvas is canvas and fig.dpi == 80


def test_image_range_threshold_uses_displayed_opacity():
    import numpy as np

    fig, ax = plt.subplots(figsize=(2, 2), dpi=80)
    data = np.zeros((10, 10))
    data[5:, :] = 1
    ax.imshow(data, cmap="gray", vmin=0, vmax=1, alpha=0.04)
    # Four percent displayed contrast clears a one percent range threshold.
    assert (
        ept.readability_faults(fig, kinds=("empty-panel",), min_image_range=0.01) == []
    )
