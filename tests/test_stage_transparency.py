"""Pixel and visibility contracts for absent Stage artists."""

import io

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import eyepiece as ep


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def test_stage_hides_and_restores_only_its_own_visibility():
    fig, ax = plt.subplots()
    text = ax.text(0.2, 0.5, "absent", in_layout=False)
    caller = ax.text(0.6, 0.5, "caller", visible=False, in_layout=False)
    stage = ep.motion.Stage({"both": [text, caller]})
    plain = pixels(fig)
    for level in (0, 0.3, 0, 1, 0, 1):
        stage.weigh({"both": level})
        assert text.get_visible() == (level > 0)
        assert not caller.get_visible()
    np.testing.assert_array_equal(pixels(fig), plain)
    stage.weigh({"both": 0})
    text.set_visible(False)  # An explicit caller hide, even while already absent.
    stage.rest()
    assert not text.get_visible()
    text.set_visible(True)
    stage.weigh({"both": 0})
    stage.rest()
    assert text.get_visible()


@pytest.mark.parametrize("layout", [None, "constrained", "tight"])
def test_transparent_layout_text_keeps_layout_and_tight_bbox(layout):
    results = []
    for optimized in (False, True):
        fig, ax = plt.subplots(figsize=(3, 2), layout=layout)
        text = ax.text(-0.2, 1.2, "layout participant", transform=ax.transAxes)
        if optimized:
            stage = ep.motion.Stage({"text": text})
            stage.weigh({"text": 0})
        else:
            text.set_color((0, 0, 0, 0))
        actual = pixels(fig)
        extent = fig.get_tightbbox(fig.canvas.get_renderer()).bounds
        stream = io.BytesIO()
        fig.savefig(stream, format="png", bbox_inches="tight")
        assert text.get_visible()
        results.append((actual, extent, stream.getvalue()))
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_allclose(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]


@pytest.mark.parametrize("kind", ["bbox", "effect", "hatch", "marker", "gap"])
def test_naive_transparency_is_not_no_ink(kind):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    if kind in ("bbox", "effect"):
        artist = ax.text(0.2, 0.5, "ink", color=(0, 0, 0, 0), fontsize=25)
        if kind == "bbox":
            artist.set_bbox({"facecolor": "red"})
        else:
            artist.set_path_effects([pe.withStroke(linewidth=3, foreground="red")])
    elif kind == "hatch":
        artist = Rectangle(
            (0.2, 0.2), 0.6, 0.6, facecolor="none", edgecolor="red", hatch="//"
        )
        ax.add_patch(artist)
    else:
        (artist,) = ax.plot([0.2, 0.8], [0.5, 0.5], color=(0, 0, 0, 0), lw=8)
        if kind == "marker":
            artist.set_marker("o")
            artist.set_markersize(16)
            artist.set_markerfacecolor("red")
        else:
            artist.set_linestyle("--")
            artist.set_gapcolor("red")
    drawn = pixels(fig)
    artist.set_visible(False)
    assert not np.array_equal(pixels(fig), drawn)


@pytest.mark.parametrize("kind", ["bbox", "effect", "hatch", "gap", "subclass"])
def test_stage_keeps_unsafe_draw_paths(kind):
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    if kind in ("bbox", "effect"):
        artist = ax.text(0.2, 0.5, "ink", in_layout=False)
        if kind == "bbox":
            artist.set_bbox({"facecolor": "red"})
        else:
            artist.set_path_effects([pe.withStroke(linewidth=3, foreground="red")])
    elif kind == "hatch":
        artist = Rectangle((0.2, 0.2), 0.6, 0.6, hatch="//")
        ax.add_patch(artist)
    elif kind == "subclass":

        class CustomLine(Line2D):
            pass

        artist = CustomLine([0, 1], [0.5, 0.5])
        ax.add_line(artist)
    else:
        (artist,) = ax.plot([0, 1], [0.5, 0.5], ls="--", gapcolor="red")
    stage = ep.motion.Stage({"ink": artist})
    stage.weigh({"ink": 0})
    assert artist.get_visible()
    drawn = pixels(fig)
    artist.draw = type(artist).draw.__get__(artist)
    np.testing.assert_array_equal(pixels(fig), drawn)


def test_hidden_line_keeps_legacy_on_marks_and_outside_findings():
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.4, 0.5, "label", color="black")
    (line,) = ax.plot(
        [-2, 2], [0.5, 0.5], color="black", alpha=1, clip_on=False, in_layout=False
    )
    stage = ep.motion.Stage({"line": line})
    stage.weigh({"line": 0})
    assert not line.get_visible()
    hidden = [str(f) for f in ep.testing.readability_faults(fig)]
    assert any("on-marks" in f for f in hidden)
    assert any("outside" in f for f in hidden)
    line.set_visible(True)
    assert [str(f) for f in ep.testing.readability_faults(fig)] == hidden
    line.set_visible(False)
    assert not any("on-marks" in str(f) for f in ep.testing.readability_faults(fig))


def test_zero_only_and_current_marker_colors():
    fig, ax = plt.subplots()
    (line,) = ax.plot(
        [0.2, 0.8], [0.5, 0.5], marker="o", fillstyle="left", markerfacecoloralt="red"
    )
    text = ax.text(0.2, 0.5, "text", in_layout=False)
    stage = ep.motion.Stage({"all": [line, text]})
    for level in (1e-12, 0.01, 1):
        stage.weigh({"all": level})
        assert line.get_visible() and text.get_visible()
    stage.weigh({"all": 0})
    assert not line.get_visible() and not text.get_visible()
    # The render guard must examine current properties after caller re-showing.
    text.set_visible(True)
    text.set_color("red")
    current = pixels(fig)
    text.draw = type(text).draw.__get__(text)
    np.testing.assert_array_equal(pixels(fig), current)


def test_large_stage_matches_unoptimized_frames_and_findings(monkeypatch):
    from stage_bench import findings, scene

    from eyepiece import attention

    reference, ref_stage, ref_draw = scene()
    optimized, opt_stage, opt_draw = scene()
    frames = [0, 9, 3, 1, 8, 2, 7, 4, 6, 5, 0, 4, 2, 9, 1, 0]
    for k in frames:
        with monkeypatch.context() as baseline:
            baseline.setattr(attention, "_stage_no_ink", lambda artist: False)
            ref_draw(k)
            expected = pixels(reference)
        opt_draw(k)
        np.testing.assert_array_equal(pixels(optimized), expected)
    with monkeypatch.context() as baseline:
        baseline.setattr(attention, "_stage_no_ink", lambda artist: False)
        expected = findings(reference, ref_stage, ref_draw)
    assert findings(optimized, opt_stage, opt_draw) == expected
    # Exact settled geometry after nonsequential updates.
    with monkeypatch.context() as baseline:
        baseline.setattr(attention, "_stage_no_ink", lambda artist: False)
        ref_stage.rest()
        expected = pixels(reference)
    opt_stage.rest()
    np.testing.assert_array_equal(pixels(optimized), expected)


def test_transparent_hatch_still_paints_after_stage():
    fig, ax = plt.subplots(figsize=(3, 2), dpi=80)
    ax.set_axis_off()
    patch = Rectangle(
        (0.2, 0.2), 0.6, 0.6, facecolor="blue", edgecolor="black", hatch="///"
    )
    ax.add_patch(patch)
    stage = ep.motion.Stage({"patch": patch})
    stage.weigh({"patch": 0})
    assert patch.get_facecolor()[3] == patch.get_edgecolor()[3] == 0
    assert patch.get_hatchcolor()[3] > 0
    expected = pixels(fig)
    assert patch.get_visible()
    patch.set_visible(False)
    assert not np.array_equal(pixels(fig), expected)


def test_custom_instance_draw_is_not_skipped():
    fig, ax = plt.subplots()
    text = ax.text(0.2, 0.5, "custom", in_layout=False)
    original = text.draw
    calls = []

    def custom(renderer):
        calls.append(True)
        original(renderer)

    text.draw = custom
    stage = ep.motion.Stage({"text": text})
    stage.weigh({"text": 0})
    pixels(fig)
    assert text.get_visible()
    assert calls


@pytest.mark.parametrize("secondary", ["bbox", "effect", "filter", "scalar-alpha"])
def test_guard_rechecks_secondary_ink_after_caller_show(secondary):
    fig, ax = plt.subplots()
    text = ax.text(0.2, 0.5, "current", in_layout=False)
    stage = ep.motion.Stage({"text": text})
    stage.weigh({"text": 0})
    text.set_visible(True)
    if secondary == "bbox":
        text.set_bbox({"facecolor": "red"})
    elif secondary == "effect":
        text.set_path_effects([pe.withStroke(linewidth=3, foreground="red")])
    elif secondary == "filter":
        text.set_agg_filter(lambda image, dpi: (image, 0, 0))
    else:
        text.set_alpha(1.0)
    guarded = pixels(fig)
    text.draw = type(text).draw.__get__(text)
    np.testing.assert_array_equal(pixels(fig), guarded)


def test_staged_figure_pickle_retains_visibility_ownership():
    import pickle

    fig, ax = plt.subplots()
    text = ax.text(0.2, 0.5, "saved", in_layout=False)
    stage = ep.motion.Stage({"text": text})
    stage.weigh({"text": 0})
    restored_fig, restored_stage = pickle.loads(pickle.dumps((fig, stage)))
    restored_text = restored_fig.axes[0].texts[0]
    assert not restored_text.get_visible()
    restored_stage.rest()
    assert restored_text.get_visible()
    stage.rest()
    np.testing.assert_array_equal(pixels(restored_fig), pixels(fig))


@pytest.mark.parametrize("decoration", ["title", "xlabel", "ylabel", "tick"])
@pytest.mark.parametrize("layout", ["constrained", "tight"])
def test_axis_decorations_keep_layout_even_with_in_layout_false(decoration, layout):
    frames = []
    for optimized in (False, True):
        fig, ax = plt.subplots(figsize=(3, 2), layout=layout)
        if decoration == "tick":
            ax.set_xticks([0.5], ["large tick label"])
            text = ax.get_xticklabels()[0]
        else:
            text = getattr(ax, "set_" + decoration)("large label")
        text.set_in_layout(False)
        if optimized:
            ep.motion.Stage({"text": text}).weigh({"text": 0})
        else:
            text.set_color((0, 0, 0, 0))
        frames.append(pixels(fig))
        assert text.get_visible()
    np.testing.assert_array_equal(*frames)
