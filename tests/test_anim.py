"""record/animate: multi-sink, generator safety, no clearing, ffmpeg error.

The second half of this file guards the mechanics the facade exists to own
-- scoped rcParams, per-suffix dpi, facecolor at grab time, writer teardown
-- so that a later "simplification" of any of them fails a test rather than
silently producing broken output.
"""

import logging
import shutil

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.layout_engine import TightLayoutEngine
from PIL import Image

import eyepiece
from eyepiece.anim import PRESETS, _make_writer, _sink_dpi, animate, record


def _fig_line():
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [])
    ax.set_xlim(0, 9)
    ax.set_ylim(-1, 1)
    return fig, line


def test_record_multi_sink_one_pass(tmp_path):
    fig, line = _fig_line()
    x = np.arange(10)
    with record(fig, tmp_path / "a.gif", tmp_path / "a.html", fps=5) as rec:
        for k in range(3):
            line.set_data(x[: k + 1], np.sin(x[: k + 1]))
            rec.frame()
    assert (tmp_path / "a.gif").stat().st_size > 0
    assert (tmp_path / "a.html").stat().st_size > 0
    plt.close(fig)


def test_record_creates_missing_parent_directories(tmp_path):
    fig, _ = _fig_line()
    out = tmp_path / "runs" / "today" / "a.gif"
    with record(fig, out, fps=5) as rec:
        rec.hold(2)
    assert out.stat().st_size > 0
    plt.close(fig)


def test_record_hold_repeats_frames(tmp_path):
    fig, line = _fig_line()
    with record(fig, tmp_path / "h.html", fps=5) as rec:
        line.set_data([0, 1], [0, 0])
        rec.hold(4)
    html = (tmp_path / "h.html").read_text()
    assert html.count("data:image/png") == 4
    plt.close(fig)


def test_animate_does_not_clear(tmp_path):
    fig, line = _fig_line()
    n_axes_before = len(fig.axes)

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    animate(fig, draw, 3, fps=5).save(tmp_path / "n.gif")
    assert len(fig.axes) == n_axes_before
    assert len(fig.axes[0].lines) == 1
    plt.close(fig)


def test_animate_returns_public_animation_type(tmp_path):
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    anim = animate(fig, draw, 3, fps=5)
    assert isinstance(anim, eyepiece.Animation)
    plt.close(fig)


def test_generator_second_pass_raises(tmp_path):
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    gen = (k for k in range(3))
    anim = animate(fig, draw, gen, fps=5)
    anim.save(tmp_path / "g1.gif")
    with pytest.raises(RuntimeError, match="generator"):
        anim.save(tmp_path / "g2.gif")
    assert not (tmp_path / "g2.gif").exists()
    plt.close(fig)


def test_callable_frames_render_twice(tmp_path):
    fig, line = _fig_line()
    seen = []

    def draw(fig_, k):
        seen.append(k)
        line.set_data([0, k], [0, 0])

    anim = animate(fig, draw, lambda: (k for k in range(3)), fps=5)
    anim.save(tmp_path / "c1.gif")
    anim.save(tmp_path / "c2.gif")
    assert seen == [0, 1, 2, 0, 1, 2]
    for name in ("c1.gif", "c2.gif"):
        with Image.open(tmp_path / name) as im:
            assert im.n_frames == 3
    plt.close(fig)


def test_jshtml_returns_html_and_closes_nothing_extra():
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    html = animate(fig, draw, 3, fps=5).jshtml(dpi=60)
    assert "<script" in html or "animation" in html.lower()
    assert not plt.get_fignums()


def test_presets_are_the_measured_values():
    assert PRESETS == {
        "talk": {"fps": 10, "dpi": 120},
        "gif": {"fps": 8, "dpi": 85},
        "jshtml": {"fps": 14, "dpi": 100},
        "docs": {"fps": 10, "dpi": 100},
    }


def test_a_preset_splats_into_every_renderer(tmp_path):
    # the whole point of PRESETS being fps/dpi pairs is that they splat, so
    # every renderer has to accept both keys -- save and video took only dpi
    # until 0.1.1, which made `anim.save(p, **PRESETS["talk"])` a TypeError
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    anim = animate(fig, draw, 3, fps=2)
    anim.save(tmp_path / "a.gif", **PRESETS["gif"])
    anim.video(tmp_path / "b.gif", **PRESETS["gif"])
    assert "animation" in anim.jshtml(**PRESETS["jshtml"]).lower()


def test_fps_override_reaches_the_written_frames(tmp_path):
    fig, line = _fig_line()

    def draw(fig_, k):
        line.set_data([0, k], [0, 0])

    anim = animate(fig, draw, 3, fps=2)
    anim.save(tmp_path / "own.gif")
    anim.save(tmp_path / "override.gif", fps=10)
    # a gif stores its delay in centiseconds, so read these as 2 and 10 fps
    with Image.open(tmp_path / "own.gif") as im:
        assert im.info["duration"] == 500  # the animation's own 2 fps
    with Image.open(tmp_path / "override.gif") as im:
        assert im.info["duration"] == 100  # 10 fps, not 2
    plt.close(fig)


def test_default_dpi_is_per_suffix(tmp_path):
    assert _sink_dpi(tmp_path / "a.gif", None) == 85
    assert _sink_dpi(tmp_path / "a.html", None) == 100
    assert _sink_dpi(tmp_path / "a.mp4", None) == 120
    assert _sink_dpi(tmp_path / "a.gif", 42) == 42


def test_default_dpi_reaches_the_written_frames(tmp_path):
    fig, _ = plt.subplots(figsize=(2, 2))
    with record(fig, tmp_path / "default.gif", fps=5) as rec:
        rec.hold(1)
    with record(fig, tmp_path / "explicit.gif", fps=5, dpi=50) as rec:
        rec.hold(1)
    with Image.open(tmp_path / "default.gif") as im:
        assert im.size == (170, 170)  # 2 in at the .gif default of 85 dpi
    with Image.open(tmp_path / "explicit.gif") as im:
        assert im.size == (100, 100)
    plt.close(fig)


def test_mp4_writer_leads_with_the_even_dimension_crop(tmp_path):
    writer = _make_writer(tmp_path / "a.mp4", 10, ["-preset", "slow"])
    # h264 rejects an odd width or height, which a figsize times a dpi hits
    assert writer.extra_args[:2] == ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"]
    assert writer.extra_args[2:] == ["-preset", "slow"]


def test_unsupported_suffix_raises_naming_it(tmp_path):
    fig, _ = _fig_line()
    with pytest.raises(ValueError, match=r"\.webm"):
        with record(fig, tmp_path / "x.webm", fps=5):
            pass
    plt.close(fig)


def test_record_scopes_its_rcparam_overrides(tmp_path, caplog):
    fig, _ = _fig_line()
    ffmpeg_before = matplotlib.rcParams["animation.ffmpeg_path"]
    caplog.set_level(logging.INFO, logger="matplotlib.animation")
    with matplotlib.rc_context({"savefig.bbox": "tight"}):
        with record(fig, tmp_path / "s.gif", fps=5) as rec:
            assert matplotlib.rcParams["savefig.bbox"] is None
            rec.hold(2)
        assert matplotlib.rcParams["savefig.bbox"] == "tight"
    assert matplotlib.rcParams["animation.ffmpeg_path"] == ffmpeg_before
    # matplotlib logs this only when a writer opens while the bbox is still
    # tight, so its absence is what proves the override was already in force
    # -- matplotlib scopes the bbox itself once saving() is under way, which
    # makes every later observable identical with or without the override.
    assert not [r for r in caplog.records if "savefig.bbox" in r.getMessage()]
    plt.close(fig)


def test_record_freezes_layout_engine_after_first_frame(tmp_path):
    fig, ax = plt.subplots(layout="constrained")
    ax.plot([], [])
    original_engine = fig.get_layout_engine()
    with record(fig, tmp_path / "a.gif", fps=5) as rec:
        rec.frame()
        frozen_engine = fig.get_layout_engine()
        # frame 1 forced a real solve, then locked it: the engine object
        # itself changes identity, so a re-solve on frame 2 cannot occur
        assert frozen_engine is not original_engine
        rec.frame()
        assert fig.get_layout_engine() is frozen_engine
    plt.close(fig)


def test_record_restores_layout_engine_after_normal_exit(tmp_path):
    fig, ax = plt.subplots(layout="constrained")
    ax.plot([], [])
    original_engine = fig.get_layout_engine()
    with record(fig, tmp_path / "a.gif", fps=5) as rec:
        rec.hold(2)
        assert fig.get_layout_engine() is not original_engine
    assert fig.get_layout_engine() is original_engine
    plt.close(fig)


def test_record_restores_layout_engine_after_exception(tmp_path):
    fig, ax = plt.subplots(layout="constrained")
    ax.plot([], [])
    original_engine = fig.get_layout_engine()
    with pytest.raises(ValueError, match="boom"):
        with record(fig, tmp_path / "a.gif", fps=5) as rec:
            rec.frame()
            raise ValueError("boom")
    assert fig.get_layout_engine() is original_engine
    plt.close(fig)


def test_record_with_no_layout_engine_records_without_error(tmp_path):
    fig, _ = _fig_line()
    assert fig.get_layout_engine() is None
    with record(fig, tmp_path / "a.gif", fps=5) as rec:
        rec.hold(2)
        assert fig.get_layout_engine() is None
    assert fig.get_layout_engine() is None
    assert (tmp_path / "a.gif").stat().st_size > 0
    plt.close(fig)


def test_record_restore_installs_no_engine_under_autolayout(tmp_path):
    # set_layout_engine(None) is not "no engine": matplotlib consults
    # figure.autolayout (and figure.constrained_layout.use) and installs
    # whichever they name. A figure that had no engine must therefore have
    # its restore SKIPPED, not replayed with None, or the exit hands back a
    # tight-layout engine the figure never had.
    fig, _ = _fig_line()
    assert fig.get_layout_engine() is None
    with matplotlib.rc_context({"figure.autolayout": True}):
        with record(fig, tmp_path / "a.gif", fps=5) as rec:
            rec.hold(2)
            engine_during = fig.get_layout_engine()
        assert fig.get_layout_engine() is engine_during
        assert not isinstance(fig.get_layout_engine(), TightLayoutEngine)
    plt.close(fig)


class _SnappingManager:
    """A figure manager that rounds the figure down to whole pixels.

    Stands in for an interactive backend (macosx is the one that bit us):
    `set_size_inches(forward=True)` asks the manager to resize, and the
    manager hands back a size truncated to whole device pixels. The test
    suite runs headless, so the behavior is injected rather than staged.
    """

    def __init__(self, fig):
        self.fig = fig

    def resize(self, w, h):
        self.fig.set_size_inches(w / self.fig.dpi, h / self.fig.dpi, forward=False)


def test_record_ignores_a_backend_that_snaps_the_figure_size(tmp_path):
    # 11.5in at 130dpi is 1495px, odd, so the mp4 writer shrinks the figure
    # to 1494px and restores that size before every grab -- while the frame
    # size it declares to ffmpeg is read back off the figure. A snapping
    # manager rounds that read-back down to 1493, ffmpeg then takes each
    # 1494px row as 1493px, and the movie shears sideways a pixel per row
    # with nothing raising. Recording through an Agg canvas, which has no
    # manager to snap anything, keeps the two numbers equal.
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot([], [])
    fig.canvas.manager = _SnappingManager(fig)
    with record(fig, tmp_path / "s.mp4", fps=5, dpi=130) as rec:
        rec.hold(2)
        assert rec.writers[0].frame_size == (1494, 624)
    assert (tmp_path / "s.mp4").stat().st_size > 0
    plt.close(fig)


def test_record_restores_the_original_canvas(tmp_path):
    fig, _ = _fig_line()
    original = fig.canvas
    with record(fig, tmp_path / "a.gif", fps=5) as rec:
        rec.hold(2)
        assert fig.canvas is not original
    assert fig.canvas is original
    plt.close(fig)


def test_record_restores_the_original_canvas_after_an_exception(tmp_path):
    fig, _ = _fig_line()
    original = fig.canvas
    with pytest.raises(ValueError, match="boom"):
        with record(fig, tmp_path / "a.gif", fps=5) as rec:
            rec.frame()
            raise ValueError("boom")
    assert fig.canvas is original
    plt.close(fig)


def test_exception_in_body_still_finalizes_every_writer(tmp_path):
    fig, line = _fig_line()
    with pytest.raises(ValueError, match="boom"):
        with record(fig, tmp_path / "e.gif", tmp_path / "e.html", fps=5) as rec:
            for k in range(2):  # distinct frames: gif drops a repeated one
                line.set_data([0, k + 1], [0, 0])
                rec.frame()
            raise ValueError("boom")
    with Image.open(tmp_path / "e.gif") as im:
        assert im.n_frames == 2  # a writer never finished writes no file at all
    assert (tmp_path / "e.html").read_text().count("data:image/png") == 2
    assert matplotlib.rcParams["savefig.bbox"] != "tight"
    plt.close(fig)


def test_body_error_survives_a_writer_that_never_got_a_frame(tmp_path):
    # The teardown must never outrank the body. A writer that received no
    # frame cannot finalize -- PillowWriter indexes self._frames[0] -- and
    # matplotlib finalizes in a bare `finally`, so without a guard that
    # IndexError REPLACES the real error on its way out and the caller is
    # sent to debug the wrong library.
    fig, _ = _fig_line()
    with pytest.raises(ValueError, match="boom"):
        with record(fig, tmp_path / "z.gif", fps=5):
            raise ValueError("boom")
    plt.close(fig)


def test_body_error_survives_across_every_sink(tmp_path):
    # Each sink unwinds separately, so one sink that cannot finalize must
    # not clobber the error either -- the guard has to hold per writer.
    fig, _ = _fig_line()
    with pytest.raises(ValueError, match="boom"):
        with record(fig, tmp_path / "z.gif", tmp_path / "z.html", fps=5):
            raise ValueError("boom")
    plt.close(fig)


def test_raising_draw_reports_itself_not_a_teardown_error(tmp_path):
    # The animate() path is where this bites in practice: draw raises on
    # frame 0, no frame is ever grabbed, and the caller used to see only
    # `IndexError: list index out of range` from inside matplotlib.
    fig, _ = _fig_line()

    def draw(_fig, _ctx):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        animate(fig, draw, 3, fps=5).save(tmp_path / "d.gif")
    plt.close(fig)


def test_draw_with_the_wrong_arity_reports_the_type_error(tmp_path):
    # draw is called as draw(fig, ctx); a one-argument draw is the easiest
    # mistake to make against that contract and must say so.
    fig, _ = _fig_line()

    def draw(_ctx):
        pass

    with pytest.raises(TypeError):
        animate(fig, draw, 3, fps=5).save(tmp_path / "d.gif")
    plt.close(fig)


def test_recording_no_frames_at_all_raises_actionable(tmp_path):
    # A clean exit having grabbed nothing is a caller bug, not a crash
    # site: say what was not written and what to call.
    fig, _ = _fig_line()
    with pytest.raises(RuntimeError, match=r"no frames|frame\(\)"):
        with record(fig, tmp_path / "n.gif", fps=5):
            pass
    plt.close(fig)


def test_dark_figure_frame_is_not_saved_on_white(tmp_path):
    fig, ax = plt.subplots(facecolor="black")
    ax.set_facecolor("black")
    # a style that pins savefig.facecolor is the case the mechanic exists for:
    # the default "auto" already follows the figure, so it hides the bug
    with matplotlib.rc_context({"savefig.facecolor": "white"}):
        with record(fig, tmp_path / "dark.gif", fps=5) as rec:
            rec.hold(2)
    with Image.open(tmp_path / "dark.gif") as im:
        corner = im.convert("RGB").getpixel((0, 0))
    assert max(corner) < 128
    plt.close(fig)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_stale_ffmpeg_path_falls_through_and_is_restored(tmp_path):
    fig, _ = _fig_line()
    stale = str(tmp_path / "no-such-ffmpeg")
    with matplotlib.rc_context({"animation.ffmpeg_path": stale}):
        with record(fig, tmp_path / "f.mp4", fps=5) as rec:
            assert matplotlib.rcParams["animation.ffmpeg_path"] != stale
            rec.hold(2)
        assert matplotlib.rcParams["animation.ffmpeg_path"] == stale
    assert (tmp_path / "f.mp4").stat().st_size > 0
    plt.close(fig)


@pytest.mark.skipif(shutil.which("ffmpeg") is not None, reason="ffmpeg present")
def test_mp4_without_ffmpeg_raises_actionable(tmp_path):
    fig, _ = _fig_line()
    with pytest.raises(RuntimeError, match=r"imageio-ffmpeg|apt_packages|ffmpeg"):
        with record(fig, tmp_path / "x.mp4", fps=5) as rec:
            rec.frame()
    plt.close(fig)


def test_recording_no_frames_names_every_sink(tmp_path):
    # A zero-frame gif is never written, but a zero-frame html player IS left
    # on disk as an empty stub, so the message has to name every sink and
    # must not claim they all went unwritten. Deliberately no mp4 sink: an
    # ffmpeg-less machine would raise the ffmpeg error before the recording
    # ever starts, which is a different code path and not what this pins.
    fig, _ = _fig_line()
    paths = [tmp_path / "n.gif", tmp_path / "n.html"]
    with pytest.raises(RuntimeError) as excinfo:
        with record(fig, *paths, fps=5):
            pass
    message = str(excinfo.value)
    for p in paths:
        assert str(p) in message
    plt.close(fig)
