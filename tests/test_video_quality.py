"""Video quality defaults: the encoder profile, frame rates, unsnapped moves, seams."""

import itertools
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

import eyepiece as ep
from eyepiece import _style
from eyepiece._ffmpeg import ENCODER_PROFILE, encoder_args, resolve_ffmpeg
from eyepiece._phasor import _ring
from eyepiece.anim import PRESETS

try:
    FFMPEG = resolve_ffmpeg()
    subprocess.run([FFMPEG, "-version"], capture_output=True, check=True)
except (RuntimeError, OSError, subprocess.CalledProcessError):
    FFMPEG = None

needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="needs ffmpeg")

BRAND = (201, 42, 98)


def _info(path):
    """The stream line ffmpeg prints for a video, with its color tags."""
    done = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    return next(line for line in done.stderr.splitlines() if "Video:" in line)


def _decode(path, size, matrix="bt709"):
    """Decoded RGB frames, converting with an explicit YUV matrix."""
    out = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-vf",
         f"scale=in_color_matrix={matrix}:in_range=limited,format=rgb24",
         "-f", "rawvideo", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    w, h = size
    return np.frombuffer(out.stdout, np.uint8).reshape(-1, h, w, 3)


def _flat_frames(n, size):
    frame = np.zeros((size[1], size[0], 4), np.uint8)
    frame[..., :3] = BRAND
    frame[..., 3] = 255
    return [frame] * n


def test_talk_runs_at_60_fps_and_docs_at_30_with_a_still_resolution_doc():
    assert PRESETS["talk"] == {"fps": 60, "dpi": 120}
    assert PRESETS["docs"]["fps"] == 30
    assert PRESETS["talk_60"] == {"fps": 60, "dpi": 120}
    assert PRESETS["doc_hd"] == {"fps": 30, "dpi": 200}
    assert PRESETS["gif"] == {"fps": 8, "dpi": 85}


def test_profile_arguments_follow_the_profile_and_its_overrides():
    args = encoder_args(30)
    assert args[args.index("-crf") + 1] == "18"
    assert args[args.index("-tune") + 1] == "animation"
    assert args[args.index("-g") + 1] == "60"
    assert args[args.index("-colorspace") + 1] == "bt709"
    assert "+faststart" in args
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert ep.ENCODER_PROFILE is ENCODER_PROFILE
    mine = encoder_args(60, {"crf": 23, "tune": None, "colorspace": None})
    assert mine[mine.index("-crf") + 1] == "23"
    assert mine[mine.index("-g") + 1] == "120"
    assert "-tune" not in mine and "-colorspace" not in mine
    with pytest.raises(ValueError, match="unknown encoder keys"):
        encoder_args(30, {"crf_value": 1})


@needs_ffmpeg
def test_yuv444p_is_an_opt_in_pixel_format_for_both_writers(tmp_path):
    size = (64, 48)
    for name, encoder, want in (
        ("default", None, "yuv420p"),
        ("full", {"pix_fmt": "yuv444p"}, "yuv444p"),
    ):
        raw = tmp_path / f"raw-{name}.mp4"
        with ep.RawSink(raw, size, fps=30, encoder=encoder) as sink:
            for frame in _flat_frames(3, size):
                sink.write(frame)
        fig, _ax = plt.subplots(figsize=(0.64, 0.48))
        recorded = tmp_path / f"rec-{name}.mp4"
        with ep.record(fig, recorded, fps=30, dpi=100, encoder=encoder) as rec:
            rec.hold(3)
        plt.close(fig)
        for path in (raw, recorded):
            line = _info(path)
            assert want in line and "bt709" in line, (name, line)


@needs_ffmpeg
def test_raw_sink_tags_bt709_and_a_flat_patch_decodes_to_its_rgb(tmp_path):
    size = (64, 48)
    path = tmp_path / "flat.mp4"
    with ep.RawSink(path, size, fps=30) as sink:
        for frame in _flat_frames(6, size):
            sink.write(frame)
    line = _info(path)
    assert "bt709" in line and "tv" in line
    patch = _decode(path, size)[3, 8:40, 8:56].reshape(-1, 3).mean(axis=0)
    assert np.abs(patch - BRAND).max() <= 2
    # decoded with the wrong matrix the same pixels are visibly off
    wrong = _decode(path, size, matrix="bt601")[3, 8:40, 8:56].reshape(-1, 3).mean(0)
    assert np.abs(wrong - BRAND).max() > 8


@needs_ffmpeg
def test_record_uses_the_profile_and_overrides_reach_it(tmp_path):
    fig, ax = plt.subplots(figsize=(1.6, 1.2))
    fig.patch.set_facecolor(tuple(v / 255 for v in BRAND))
    ax.set_axis_off()
    paths = {"profile": tmp_path / "a.mp4", "plain": tmp_path / "b.mp4"}
    with ep.record(fig, paths["profile"], fps=30, dpi=40) as rec:
        rec.hold(30)
    with ep.record(
        fig, paths["plain"], fps=30, dpi=40, encoder={"colorspace": None}
    ) as rec:
        rec.hold(3)
    assert "bt709" in _info(paths["profile"])
    assert "bt709" not in _info(paths["plain"])
    frames = _decode(paths["profile"], (64, 48))
    assert len(frames) == 30
    assert np.abs(frames[10][12:36, 12:52].reshape(-1, 3).mean(0) - BRAND).max() <= 2
    plt.close(fig)


@needs_ffmpeg
def test_a_short_talk_clip_has_the_frames_of_60_fps(tmp_path):
    fig, ax = plt.subplots(figsize=(1.6, 1.2))
    (dot,) = ax.plot([0.5], [0.5], "o")
    path = tmp_path / "t.mp4"
    with ep.record(fig, path, **{**PRESETS["talk"], "dpi": 40}) as rec:
        for k in range(90):
            dot.set_data([k / 90], [0.5])
            rec.frame()
    out = subprocess.run(
        [FFMPEG, "-i", str(path), "-f", "null", "-"], capture_output=True, text=True
    )
    count = int(re.findall(r"frame=\s*(\d+)", out.stderr)[-1])
    assert count == 90
    assert "60 fps" in _info(path)
    plt.close(fig)


# Unsnapping


_CORNERS = np.array([(40, 30), (90, 30), (90, 70), (40, 70)])


def _box_scene():
    """A rectangle polygon on a one-pixel-per-0.01-unit axes, drawn crisp."""
    fig = plt.figure(figsize=(2.0, 1.0), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set(xlim=(0, 200), ylim=(0, 100))
    box = Polygon(_CORNERS, closed=True, fc="k", ec="none")
    ax.add_patch(box)
    return fig, ax, box


def _gray(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba(), dtype=float)[..., 0].copy()


def test_a_warped_rectangle_moves_a_tenth_of_a_pixel_per_frame_without_snapping():
    fig, _ax, box = _box_scene()
    shift = 3.0  # data units are pixels here
    move = ep.warp(box, lambda p: p + np.array([shift, 0.0]))
    steps = 30
    frames = []
    for k in range(1, steps):
        move.update(k / steps)
        frames.append(_gray(fig))
    changes = sum(not np.array_equal(a, b) for a, b in itertools.pairwise(frames))
    assert changes == len(frames) - 1
    # the same sub-pixel positions drawn with the default snapping jump
    fig2, _ax2, box2 = _box_scene()
    snapped = []
    for k in range(1, steps):
        box2.set_xy(np.add(_CORNERS, (shift * k / steps, 0)))
        snapped.append(_gray(fig2))
    jumps = sum(not np.array_equal(a, b) for a, b in itertools.pairwise(snapped))
    assert jumps < steps // 2
    plt.close(fig)
    plt.close(fig2)


def _vline_scene():
    """A vertical line and a square marker on a one-pixel-per-unit axes."""
    fig = plt.figure(figsize=(1.0, 0.5), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set(xlim=(0, 100), ylim=(0, 50))
    line = ax.axvline(40.0, color="k", lw=3)
    (dot,) = ax.plot([60.0], [25.0], "s", color="k", ms=6)
    return fig, line, dot


def _slow_stills(tmp_path, steps, **kwargs):
    """Stills of the scene moved a tenth of a pixel per frame inside `record`."""
    fig, line, dot = _vline_scene()
    stills = {tmp_path / f"s{k:02d}.png": k / 10 for k in range(steps)}
    with ep.record(fig, fps=10, dpi=100, stills=stills, **kwargs) as rec:
        for k in range(steps):
            line.set_xdata([40.0 + k / 10] * 2)
            dot.set_xdata([60.0 + k / 10])
            rec.frame()
    plt.close(fig)
    return [plt.imread(path) for path in sorted(stills)]


def test_record_draws_frames_unsnapped_by_default(tmp_path):
    steps = 12
    glide = _slow_stills(tmp_path / "glide", steps)
    assert all(not np.array_equal(a, b) for a, b in itertools.pairwise(glide))
    crisp = _slow_stills(tmp_path / "crisp", steps, snap=True)
    stalls = sum(np.array_equal(a, b) for a, b in itertools.pairwise(crisp))
    assert stalls >= steps // 2


def test_record_leaves_the_callers_snapping_as_it_was(tmp_path):
    before = plt.rcParams["path.snap"]
    fig, line, _dot = _vline_scene()
    with ep.record(fig, fps=10, stills={tmp_path / "s.png": 0}) as rec:
        rec.frame()
    assert plt.rcParams["path.snap"] == before
    assert line.get_snap() is None
    plt.close(fig)


def test_video_rc_gives_lines_butt_caps_and_round_joins(tmp_path):
    assert ep.VIDEO_RC["lines.solid_capstyle"] == "butt"
    assert ep.VIDEO_RC["lines.solid_joinstyle"] == "round"
    fig, ax = plt.subplots(figsize=(1, 1))
    before = plt.rcParams["lines.solid_capstyle"]
    with ep.record(fig, fps=10, stills={tmp_path / "s.png": 0}) as rec:
        (made,) = ax.plot([0, 1], [0, 1])
        rec.frame()
    assert made.get_solid_capstyle() == "butt"
    assert made.get_solid_joinstyle() == "round"
    assert plt.rcParams["lines.solid_capstyle"] == before
    plt.close(fig)


def test_talk_rc_sizes_type_and_stroke_from_the_frame_height():
    import matplotlib

    rc = ep.talk_rc()  # 1080 px at the talk preset's 120 dpi
    assert rc["font.size"] == pytest.approx(35.0, abs=0.1)
    assert rc["axes.titlesize"] == pytest.approx(37.6, abs=0.1)
    assert rc["axes.labelsize"] == pytest.approx(25.3, abs=0.1)
    assert rc["xtick.labelsize"] == pytest.approx(22.0, abs=0.1)
    assert rc["lines.linewidth"] == pytest.approx(3.24, abs=0.01)
    assert rc["axes.linewidth"] == pytest.approx(1.94, abs=0.01)
    assert rc["xtick.major.size"] == 0.0
    assert rc["lines.solid_capstyle"] == "butt"
    small = ep.talk_rc(720, dpi=100)
    assert small["font.size"] == pytest.approx(28.0, abs=0.1)
    assert small["lines.linewidth"] == pytest.approx(2.59, abs=0.01)
    # one em of body text is 5.4 percent of the frame, at any frame size
    for height, dpi in ((1080, 120), (720, 100)):
        with matplotlib.rc_context(ep.talk_rc(height, dpi=dpi)):
            fig = plt.figure(figsize=(height * 16 / 9 / dpi, height / dpi), dpi=dpi)
            text = fig.text(0.5, 0.5, "x")
            em_px = text.get_fontsize() * dpi / 72.0
            plt.close(fig)
        assert em_px / height == pytest.approx(0.054)
    # the documentation sizes are untouched
    assert plt.rcParams["font.size"] != rc["font.size"]


def test_unsnapped_lets_a_slow_mark_glide_outside_a_recording():
    fig, line, dot = _vline_scene()
    assert ep.unsnapped([line, dot]) == [line, dot]
    assert ep.unsnapped(line) is line
    frames = []
    for k in range(12):
        line.set_xdata([40.0 + k / 10] * 2)
        dot.set_xdata([60.0 + k / 10])
        frames.append(_gray(fig))
    assert all(not np.array_equal(a, b) for a, b in itertools.pairwise(frames))
    plt.close(fig)


def test_the_rest_state_after_a_warp_is_pixel_exact_against_the_snapped_still():
    fig, _ax, box = _box_scene()
    move = ep.warp(box, lambda p: p + np.array([3.0, 0.0]))
    assert box.get_snap() is None
    move.update(0.4)
    assert box.get_snap() is False
    move.update(1.0)
    assert box.get_snap() is None
    fig2, _ax2, box2 = _box_scene()
    box2.set_xy(np.add(_CORNERS, (3.0, 0.0)))
    np.testing.assert_array_equal(_gray(fig), _gray(fig2))
    move.update(0.0)
    fig3, _ax3, _box3 = _box_scene()
    np.testing.assert_array_equal(_gray(fig), _gray(fig3))
    for f in (fig, fig2, fig3):
        plt.close(f)


def test_a_snap_setting_the_caller_chose_comes_back_at_rest():
    fig, _ax, box = _box_scene()
    box.set_snap(True)
    move = ep.warp(box, np.array([[1.0, 0.2], [0.0, 1.0]]))
    move.update(0.5)
    assert box.get_snap() is False
    move.update(1.0)
    assert box.get_snap() is True
    plt.close(fig)


@pytest.mark.parametrize("name", ["arrow", "angle_arc"])
def test_arrow_moves_unsnap_only_between_their_ends(name):
    fig, ax = plt.subplots()
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    if name == "arrow":
        result = ep.arrow((-1.0, 0.0), (1.0, 0.5), ax=ax)
    else:
        result = ep.angle_arc((0.0, 0.0), 0.0, 90.0, 1.0, ax=ax)
    patch = result.artists["arrow"]
    for u, expected in ((0.0, None), (0.5, False), (1.0, None)):
        result.update(u)
        assert patch.get_snap() is expected
    plt.close(fig)


def test_compass_moves_unsnap_both_arms():
    fig, ax = plt.subplots()
    ax.set(xlim=(-2, 2), ylim=(-2, 2))
    result = ep.compass((0.0, 0.0), 1.0, ax=ax, east="right")
    arms = result.artists["arrow"]
    result.update(0.5)
    assert [a.get_snap() for a in arms] == [False, False]
    result.update(1.0)
    assert [a.get_snap() for a in arms] == [None, None]
    plt.close(fig)


def test_ghosts_of_the_travelling_moves_are_never_snapped():
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 2), ylim=(0, 2))
    (line,) = ax.plot([0.2, 0.8], [0.5, 0.5], lw=3)
    other = ax.plot([1.5], [1.5], "o")[0]
    fly = ep.flight(line, ep.Anchor.from_artist(other), fig=fig)
    assert fly.artists["line"].get_snap() is False
    cells = np.array([[[0.2, 1.0], [0.6, 1.0]], [[0.7, 1.0], [0.9, 1.0]]])
    gather = ep.collapse(
        ax, cells, [1.0, 2.0],
        ep.Anchor.from_data(ax, (0.2, 1.8)), ep.Anchor.from_data(ax, (1.0, 1.8)),
        fig=fig, colors=["C0", "C1"],
    )  # fmt: skip
    assert gather.artists["line"].get_snap() is False
    theta = np.linspace(0, 2 * np.pi, 40)
    ring = np.column_stack([np.cos(theta), np.cos(theta) * 0 + np.sin(theta)]) * 0.3
    morph = ep.morph(ring + 1.0, ring + 0.5, ax=ax, closed=True)
    assert morph.artists["fill"].get_snap() is False
    plt.close(fig)


# Seams


def _ring_luminance(make, joints):
    """Luminance of the ring's centerline at each segment joint and midpoint."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
    ax.set_position((0, 0, 1, 1))
    ax.add_collection(make())
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=float)[..., :3]
    lum = img @ np.array([0.2126, 0.7152, 0.0722])
    plt.close(fig)
    px = 4 * 200 / 2.6

    def at(phi):
        x = (np.cos(phi) + 1.3) * px
        y = (1.3 - np.sin(phi)) * px
        return lum[round(y), round(x)]

    n = len(joints) - 1
    mids = 0.5 * (joints[:-1] + joints[1:])
    return np.array([at(p) for p in joints[1:-1]]), np.array([at(p) for p in mids]), n


def test_the_phase_ring_has_no_dark_stripe_at_its_segment_joints():
    phi = np.linspace(-np.pi, np.pi, 181)

    def seamless():
        return _ring(1.0, _style.cmap("phase", None), 4.5)

    ring = seamless()
    segments, colors = ring.get_segments(), ring.get_colors()
    joints, mids, _ = _ring_luminance(seamless, phi)
    neighbours = 0.5 * (mids[:-1] + mids[1:])
    dip = np.max(neighbours - joints)
    assert dip < 6.0, dip

    # positive control: the same ring in butt-capped segments shows the stripes
    def butt():
        return LineCollection(
            segments, colors=colors, linewidths=ring.get_linewidths(), capstyle="butt"
        )

    joints_b, mids_b, _ = _ring_luminance(butt, phi)
    dip_b = np.max(0.5 * (mids_b[:-1] + mids_b[1:]) - joints_b)
    assert dip_b > 3.0 * max(dip, 1.0), (dip, dip_b)


def test_the_phasor_ring_uses_the_same_seamless_stroke():
    ring = ep.phasor([1.0], ring=True).artists["collection"]
    assert ring.get_capstyle() == "round" and ring.get_joinstyle() == "round"
    assert len(ring.get_segments()) == 180


@needs_ffmpeg
def test_a_caller_owned_pixel_format_turns_the_color_conversion_off(tmp_path):
    size = (32, 32)
    path = tmp_path / "rgb.mp4"
    lossless = ["-vcodec", "libx264rgb", "-crf", "0", "-pix_fmt", "rgb24"]
    with ep.RawSink(path, size, fps=5, extra_ffmpeg_args=lossless) as sink:
        for frame in _flat_frames(3, size):
            sink.write(frame)
    assert "bt709" not in _info(path)


@needs_ffmpeg
def test_mux_keeps_the_picture_tags_and_the_audio_arguments_together(tmp_path):
    import wave

    from eyepiece.narration import mux

    size = (64, 48)
    video = tmp_path / "v.mp4"
    with ep.RawSink(video, size, fps=30, encoder={"colorspace": None}) as sink:
        for frame in _flat_frames(30, size):
            sink.write(frame)
    assert "bt709" not in _info(video)
    wav = tmp_path / "a.wav"
    t = np.arange(24000) / 24000
    tone = (3000 * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(tone.tobytes())
    out = mux(video, wav, tmp_path / "m.mp4")
    done = subprocess.run([FFMPEG, "-i", str(out)], capture_output=True, text=True)
    assert "bt709" in _info(out)
    assert "48000 Hz" in done.stderr
