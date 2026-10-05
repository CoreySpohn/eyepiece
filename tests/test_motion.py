"""Presentation timing and camera geometry: timeline, easing, zoom path, quads."""

import math
import subprocess
import sys

import numpy as np
import pytest

import eyepiece as ep

# ---------------------------------------------------------------------------
# frame_count, ease, stagger


def test_frame_count_rounds_half_up_after_snapping():
    assert ep.frame_count(0.15, 10) == 2  # 1.5000000000000002 snaps to 1.5
    assert ep.frame_count(0.25, 10) == 3  # half rounds up, not to even
    assert ep.frame_count(2.0, 24) == 48
    assert ep.frame_count(0.0, 30) == 0


def test_frame_count_rounds_exact_halves_up_despite_float_noise():
    assert ep.frame_count(0.29, 50) == 15  # 14.499999999999998 is 14.5
    assert ep.frame_count(0.9, 25) == 23


def test_beat_boundaries_follow_the_exact_cumulative_time():
    tl = ep.Timeline()
    for _ in range(3):
        tl.hold(0.3)
    assert tl.at_fps(25).n_frames == 23
    rng = np.random.default_rng(3)
    for _ in range(500):
        tl = ep.Timeline()
        for seconds in rng.choice([0.1, 0.2, 0.3, 0.7, 1.1, 1.5], size=8):
            tl.hold(float(seconds))
        for fps in (24, 25, 30, 50):
            assert tl.at_fps(fps).n_frames == ep.frame_count(tl.seconds, fps)


@pytest.mark.parametrize("seconds", [-1.0, math.inf, math.nan])
def test_frame_count_rejects_bad_durations(seconds):
    with pytest.raises(ValueError, match="seconds"):
        ep.frame_count(seconds, 30)


@pytest.mark.parametrize("kind", ["linear", "smoothstep", "smootherstep", "cosine"])
def test_every_easing_fixes_both_ends_and_clips(kind):
    assert ep.ease(0.0, kind) == pytest.approx(0.0)
    assert ep.ease(1.0, kind) == pytest.approx(1.0)
    assert ep.ease(-3.0, kind) == pytest.approx(0.0)
    assert ep.ease(7.0, kind) == pytest.approx(1.0)
    assert ep.ease(0.5, kind) == pytest.approx(0.5)
    values = ep.ease(np.linspace(0.0, 1.0, 11), kind)
    assert isinstance(values, np.ndarray)
    assert np.all(np.diff(values) >= 0.0)


def test_ease_returns_a_float_for_a_scalar_and_names_bad_kinds():
    assert isinstance(ep.ease(0.3), float)
    with pytest.raises(ValueError, match="bounce"):
        ep.ease(0.3, "bounce")


def test_stagger_starts_low_ranks_first():
    rank = np.array([0.0, 0.5, 1.0])
    assert np.allclose(ep.stagger(0.0, rank), 0.0)
    assert np.allclose(ep.stagger(1.0, rank), 1.0)
    mid = ep.stagger(0.5, rank)
    assert mid[0] > mid[1] > mid[2]
    assert mid[0] == pytest.approx(1.0)


@pytest.mark.parametrize("span", [0.0, -0.1, 1.5])
def test_stagger_rejects_spans_outside_the_beat(span):
    with pytest.raises(ValueError, match="span"):
        ep.stagger(0.5, [0.0, 1.0], span=span)


# ---------------------------------------------------------------------------
# Timeline and Plan


def _simple():
    tl = ep.Timeline(knobs={"text": "intro", "theta": 0.0})
    tl.hold(1.0, name="intro")
    tl.hold(0.5, name="cue", set={"text": "move"})
    tl.ramp(1.0, name="move", to={"theta": 2.0}, ease="linear")
    tl.hold(1.0, name="end")
    return tl


def test_beats_resolve_from_cumulative_seconds_at_any_fps():
    tl = _simple()
    assert tl.seconds == pytest.approx(3.5)
    for fps in (24, 30):
        plan = tl.at_fps(fps)
        assert plan.n_frames == ep.frame_count(3.5, fps)
        stops = [b.stop for b in plan.beats]
        assert stops == [ep.frame_count(t, fps) for t in (1.0, 1.5, 2.5, 3.5)]
        assert plan.beats[0].start == 0
        assert all(
            a.stop == b.start for a, b in zip(plan.beats, plan.beats[1:], strict=False)
        )


def test_ramp_reaches_its_target_on_its_last_frame():
    plan = _simple().at_fps(10)
    move = next(b for b in plan.beats if b.name == "move")
    assert (move.start, move.stop) == (15, 25)
    first = plan.frames[move.start]
    assert first["theta"] == pytest.approx(0.2)  # (i + 1) / n: already moving
    assert plan.frames[move.stop - 1]["theta"] == pytest.approx(2.0)
    assert plan.frames[move.start - 1]["theta"] == 0.0
    assert plan.frames[-1]["theta"] == 2.0  # carried through the next hold


def test_set_applies_at_the_holds_start():
    plan = _simple().at_fps(10)
    assert plan.frames[9]["text"] == "intro"
    assert plan.frames[10]["text"] == "move"
    assert plan.frames[-1]["text"] == "move"


def test_frames_carry_reserved_fields_and_are_independent_dicts():
    plan = _simple().at_fps(10)
    f = plan.frames[17]
    assert f["beat"] == "move"
    assert f["index"] == 17
    assert f["t"] == pytest.approx(1.7)
    assert f["u"] == pytest.approx(0.3)
    plan.frames[0]["text"] = "changed"
    assert plan.frames[1]["text"] == "intro"


def test_progress_reads_one_clock_per_beat():
    plan = _simple().at_fps(10)
    assert plan.progress(0, "move") == 0.0
    assert plan.progress(14, "move") == 0.0
    assert plan.progress(15, "move") == pytest.approx(0.1)
    assert plan.progress(24, "move") == 1.0
    assert plan.progress(30, "move") == 1.0


def test_progress_eased_and_raw_differ_for_a_curved_ramp():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0).ramp(1.0, name="go", to={"x": 1.0}, ease="smoothstep")
    plan = tl.at_fps(10)
    k = plan.start("go") + 2
    raw = plan.progress(k, "go", eased=False)
    assert raw == pytest.approx(0.3)
    assert plan.progress(k, "go") == pytest.approx(ep.ease(0.3, "smoothstep"))
    assert plan.frames[k]["x"] == pytest.approx(ep.ease(0.3, "smoothstep"))


def test_ramp_accepts_a_callable_easing():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0).ramp(1.0, name="go", to={"x": 1.0}, ease=lambda u: u * u)
    plan = tl.at_fps(10)
    assert plan.frames[plan.start("go") + 4]["x"] == pytest.approx(0.25)


def test_log_space_ramp_interpolates_geometrically():
    tl = ep.Timeline(knobs={"w": 64.0})
    tl.hold(1.0).ramp(1.0, name="zoom", to={"w": 4.0}, ease="linear", space="log")
    plan = tl.at_fps(4)
    widths = [plan.frames[k]["w"] for k in range(plan.start("zoom"), plan.n_frames)]
    assert widths == pytest.approx([32.0, 16.0, 8.0, 4.0])


def test_read_holds_for_reading_time_with_a_floor():
    tl = ep.Timeline(knobs={"text": ""})
    tl.hold(1.0)
    tl.read("one two three", name="short", set={"text": "a"})
    tl.read(" ".join(["w"] * 20), name="long")
    plan = tl.at_fps(10)
    short = next(b for b in plan.beats if b.name == "short")
    long_ = next(b for b in plan.beats if b.name == "long")
    assert short.seconds == pytest.approx(2.5)
    assert long_.seconds == pytest.approx(6.0)


def test_beat_lookups():
    plan = _simple().at_fps(10)
    assert plan.beat_at(16).name == "move"
    assert plan.start("move") == 15
    assert plan.last("move") == 24
    assert plan.mid("move") == 19  # progress 0.5 of a 10-frame ramp
    assert plan.find({"text": "move"}) == 10
    assert plan.find({"text": "move"}, last=True) == plan.n_frames - 1
    assert plan.nearest("theta", 1.0) == 19
    with pytest.raises(KeyError, match="nope"):
        plan.start("nope")
    with pytest.raises(ValueError, match="no frame"):
        plan.find({"text": "absent"})
    with pytest.raises(IndexError):
        plan.beat_at(plan.n_frames)


def test_a_beat_that_rounds_to_no_frames_raises_naming_it():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0).hold(0.01, name="blink").hold(1.0)
    tl.at_fps(200)  # two frames there
    with pytest.raises(ValueError, match="blink"):
        tl.at_fps(10)


def test_undeclared_and_reserved_knob_names_raise():
    with pytest.raises(ValueError, match="reserved"):
        ep.Timeline(knobs={"beat": 1})
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0)
    with pytest.raises(ValueError, match="radus"):
        tl.ramp(1.0, to={"radus": 3.0})
    with pytest.raises(ValueError, match="radus"):
        tl.hold(1.0, set={"radus": 3.0})


def test_ramps_on_non_numeric_knobs_or_across_zero_in_log_raise():
    tl = ep.Timeline(knobs={"text": "a", "x": 0.0, "w": 1.0})
    tl.hold(1.0)
    with pytest.raises(ValueError, match="numeric"):
        tl.ramp(1.0, to={"text": "b"})
    with pytest.raises(ValueError, match="positive"):
        tl.ramp(1.0, to={"x": 2.0}, space="log")
    with pytest.raises(ValueError, match="positive"):
        tl.ramp(1.0, to={"w": -2.0}, space="log")


def test_a_timeline_cannot_open_with_a_ramp_and_names_are_unique():
    tl = ep.Timeline(knobs={"x": 0.0})
    with pytest.raises(ValueError, match="hold"):
        tl.ramp(1.0, to={"x": 1.0})
    tl.hold(1.0, name="a")
    with pytest.raises(ValueError, match="already"):
        tl.hold(1.0, name="a")
    with pytest.raises(ValueError, match="seconds"):
        tl.hold(0.0)


def test_unnamed_beats_get_distinct_names():
    tl = ep.Timeline()
    tl.hold(1.0).hold(1.0)
    names = [b.name for b in tl.at_fps(10).beats]
    assert len(set(names)) == 2


# ---------------------------------------------------------------------------
# zoom_path, view_limits


def test_zoom_path_hits_both_views():
    path, length = ep.zoom_path((0.0, 0.0, 64.0), (8.0, 0.0, 3.0))
    assert np.allclose(path(0.0), (0.0, 0.0, 64.0))
    assert np.allclose(path(1.0), (8.0, 0.0, 3.0))
    assert length > 0


def test_pure_zoom_is_geometric_in_width():
    path, length = ep.zoom_path((1.0, 2.0, 10.0), (1.0, 2.0, 0.1))
    assert length == pytest.approx(math.log(100.0) / math.sqrt(2.0))
    assert path(0.5)[2] == pytest.approx(1.0)


def test_a_pan_longer_than_the_view_zooms_out_in_transit():
    path, _ = ep.zoom_path((-8.0, 4.0, 4.0), (0.0, 0.0, 4.0))
    widths = [path(s)[2] for s in np.linspace(0.0, 1.0, 101)]
    assert max(widths) > 8.0


def test_identical_views_give_a_constant_path():
    path, length = ep.zoom_path((1.0, 2.0, 5.0), (1.0, 2.0, 5.0))
    assert length == 0.0
    assert np.allclose(path(0.5), (1.0, 2.0, 5.0))


def test_zoom_path_accepts_arrays_and_rejects_bad_widths():
    path, _ = ep.zoom_path((0.0, 0.0, 10.0), (5.0, 0.0, 1.0))
    _, _, w = path(np.array([0.0, 1.0]))
    assert np.allclose(w, [10.0, 1.0])
    with pytest.raises(ValueError, match="width"):
        ep.zoom_path((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


def test_zoom_out_with_a_tiny_pan_keeps_exact_endpoints():
    for pan in (1e-9, 1e-5, 1e-3):
        path, _ = ep.zoom_path((0.0, 0.0, 1.0), (pan, 0.0, 10.0))
        assert path(1.0)[2] == pytest.approx(10.0, rel=1e-9)
        assert path(1.0)[0] == pytest.approx(pan, abs=1e-12)


def test_random_view_pairs_land_on_both_views():
    rng = np.random.default_rng(7)
    for _ in range(2000):
        a = (*rng.normal(0.0, 10.0, 2), 10.0 ** rng.uniform(-3, 3))
        pan = rng.normal(0.0, 1.0, 2) * 10.0 ** rng.uniform(-9, 2)
        b = (*(np.asarray(a[:2]) + pan), 10.0 ** rng.uniform(-3, 3))
        path, length = ep.zoom_path(a, b)
        assert np.isfinite(length)
        assert np.allclose(path(0.0), a, rtol=1e-7, atol=1e-9)
        assert np.allclose(path(1.0), b, rtol=1e-7, atol=1e-9)


def test_view_limits():
    xlim, ylim = ep.view_limits((2.0, -1.0, 4.0), aspect=0.5)
    assert xlim == (0.0, 4.0)
    assert ylim == (-2.0, 0.0)
    with pytest.raises(ValueError, match="width"):
        ep.view_limits((0.0, 0.0, -1.0))


# ---------------------------------------------------------------------------
# pixel_quads


def test_pixel_quads_wind_counter_clockwise_and_honor_keep():
    y, x = np.mgrid[0:3, 0:4].astype(float)
    corners = np.stack([x, y], axis=-1)
    quads = ep.pixel_quads(corners)
    assert quads.shape == (6, 4, 2)
    q = quads[0]
    area = 0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])
    assert area == pytest.approx(1.0)
    keep = np.zeros((2, 3), bool)
    keep[1, 2] = True
    kept = ep.pixel_quads(corners, keep)
    assert kept.shape == (1, 4, 2)
    assert np.allclose(kept[0, 0], (2.0, 1.0))


def test_pixel_quads_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="corners"):
        ep.pixel_quads(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="keep"):
        ep.pixel_quads(np.zeros((3, 4, 2)), np.ones((3, 4), bool))


def test_motion_module_imports_without_matplotlib():
    code = (
        "import sys, eyepiece; eyepiece.ease(0.5); eyepiece.Timeline(); "
        "eyepiece.zoom_path((0, 0, 1), (1, 1, 1)); eyepiece.pixel_quads; "
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
