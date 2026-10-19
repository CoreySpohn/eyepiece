"""Presentation timing and camera geometry: timeline, easing, zoom path, quads."""

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import eyepiece as ep

DATA = Path(__file__).parent / "data"

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


def test_span_boundaries_follow_the_exact_cumulative_time():
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


@pytest.mark.parametrize(
    "kind", ["linear", "smooth", "smoothstep", "smootherstep", "cosine"]
)
def test_every_easing_fixes_both_ends_and_clips(kind):
    assert ep.ease(0.0, kind) == pytest.approx(0.0)
    assert ep.ease(1.0, kind) == pytest.approx(1.0)
    assert ep.ease(-3.0, kind) == pytest.approx(0.0)
    assert ep.ease(7.0, kind) == pytest.approx(1.0)
    assert ep.ease(0.5, kind) == pytest.approx(0.5)
    values = ep.ease(np.linspace(0.0, 1.0, 11), kind)
    assert isinstance(values, np.ndarray)
    assert np.all(np.diff(values) >= 0.0)


# Manim's ``smooth`` (manim/utils/rate_functions.py, inflection 10) at nine
# points, from its formula (sigmoid(10 (t - 0.5)) - e) / (1 - 2 e), e = sigmoid(-5).
MANIM_SMOOTH = {
    0.1: 0.011446579538697649,
    0.2: 0.041285659785796076,
    0.25: 0.07010371654510815,
    0.3: 0.11403653008945559,
    0.4: 0.26580657804886093,
    0.5: 0.5,
    0.6: 0.7341934219511391,
    0.75: 0.929896283454892,
    0.9: 0.9885534204613023,
}


def test_smooth_is_manims_sigmoid():
    for t, want in MANIM_SMOOTH.items():
        assert ep.ease(t, "smooth") == pytest.approx(want, rel=1e-12, abs=1e-15)
    u = np.array(list(MANIM_SMOOTH))
    assert np.allclose(ep.ease(u, "smooth"), list(MANIM_SMOOTH.values()), rtol=1e-12)
    # longer dwell than smoothstep near the ends, steeper through the middle
    assert ep.ease(0.1, "smooth") < ep.ease(0.1, "smoothstep")
    assert ep.ease(0.4, "smooth") < ep.ease(0.4, "smoothstep")


def test_smoothstep_stays_available_by_name():
    assert ep.ease(0.25, "smoothstep") == pytest.approx(0.15625)


def test_ease_default_is_the_module_default():
    from eyepiece._motion import DEFAULT_EASE

    assert ep.ease(0.3) == ep.ease(0.3, DEFAULT_EASE)


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
# Timeline and Plan: the sequential subset of the clock


def _simple():
    tl = ep.Timeline(knobs={"text": "intro", "theta": 0.0})
    tl.hold(1.0, name="intro")
    tl.hold(0.5, name="cue", to={"text": "move"})
    tl.move("move", 1.0, to={"theta": 2.0}, ease="linear")
    tl.hold(1.0, name="end")
    return tl


def test_spans_resolve_from_cumulative_seconds_at_any_fps():
    tl = _simple()
    assert tl.seconds == pytest.approx(3.5)
    for fps in (24, 30):
        plan = tl.at_fps(fps)
        assert plan.n_frames == ep.frame_count(3.5, fps)
        spans = list(plan.spans.values())
        assert [s.k1 for s in spans] == [
            ep.frame_count(t, fps) for t in (1.0, 1.5, 2.5, 3.5)
        ]
        assert spans[0].k0 == 0
        assert all(a.k1 == b.k0 for a, b in itertools.pairwise(spans))


def test_a_move_reaches_its_target_on_its_last_frame():
    plan = _simple().at_fps(10)
    move = plan.spans["move"]
    assert (move.k0, move.k1) == (15, 25)
    assert plan.value(move.k0, "theta") == pytest.approx(0.2)  # (i + 1) / n
    assert plan.value(move.k1 - 1, "theta") == pytest.approx(2.0)
    assert plan.value(move.k0 - 1, "theta") == 0.0
    assert plan.value(plan.n_frames - 1, "theta") == 2.0  # carried through the hold


def test_a_hold_steps_its_knobs_when_it_starts():
    plan = _simple().at_fps(10)
    assert plan.value(9, "text") == "intro"
    assert plan.value(10, "text") == "move"
    assert plan.value(plan.n_frames - 1, "text") == "move"


def test_frames_carry_index_time_and_section_and_are_independent_dicts():
    plan = _simple().at_fps(10)
    f = plan.frames[17]
    assert set(f) == {"text", "theta", "index", "t", "section"}
    assert f["index"] == 17
    assert f["t"] == pytest.approx(1.7)
    assert f["theta"] == pytest.approx(0.6)
    plan.frames[0]["text"] = "changed"
    assert plan.frames[1]["text"] == "intro"
    assert len(plan.frames) == plan.n_frames
    assert plan.frames[-1]["index"] == plan.n_frames - 1


def test_progress_reads_one_clock_per_span():
    plan = _simple().at_fps(10)
    assert plan.progress(0, "move") == 0.0
    assert plan.progress(14, "move") == 0.0
    assert plan.progress(15, "move") == pytest.approx(0.1)
    assert plan.progress(24, "move") == 1.0
    assert plan.progress(30, "move") == 1.0


def test_progress_eased_and_raw_differ_for_a_curved_move():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0)
    tl.move("go", 1.0, to={"x": 1.0}, ease="smoothstep")
    plan = tl.at_fps(10)
    k = plan.spans["go"].k0 + 2
    assert plan.progress(k, "go", eased=False) == pytest.approx(0.3)
    assert plan.progress(k, "go") == pytest.approx(ep.ease(0.3, "smoothstep"))
    assert plan.value(k, "x") == pytest.approx(ep.ease(0.3, "smoothstep"))


def test_a_move_accepts_a_callable_easing():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0)
    tl.move("go", 1.0, to={"x": 1.0}, ease=lambda u: u * u)
    plan = tl.at_fps(10)
    assert plan.value(plan.spans["go"].k0 + 4, "x") == pytest.approx(0.25)


def test_log_space_move_interpolates_geometrically():
    tl = ep.Timeline(knobs={"w": 64.0})
    tl.hold(1.0)
    tl.move("zoom", 1.0, to={"w": 4.0}, ease="linear", space="log")
    plan = tl.at_fps(4)
    ks = range(plan.spans["zoom"].k0, plan.n_frames)
    assert [plan.value(k, "w") for k in ks] == pytest.approx([32.0, 16.0, 8.0, 4.0])


def test_read_holds_for_the_tempo_reading_time():
    tl = ep.Timeline(knobs={"text": ""})
    tl.hold(1.0)
    tl.read("one two three", name="short", to={"text": "a"})
    tl.read(" ".join(["w"] * 20), name="long", tempo=ep.Tempo(factor=2.0))
    spans = tl.at_fps(100).spans
    assert spans["short"].end_s - spans["short"].start_s == pytest.approx(
        ep.Tempo().read_s("one two three")
    )
    assert spans["long"].end_s - spans["long"].start_s == pytest.approx(
        2.0 * (0.75 + 0.3 * 20)
    )


def test_span_lookups_name_the_frame_of_a_span():
    plan = _simple().at_fps(10)
    assert plan.frame_of("move", "start") == 15
    assert plan.frame_of("move", "last") == 24
    assert plan.frame_of("move", "mid") == 19  # progress 0.5 of a 10-frame move
    with pytest.raises(KeyError, match="nope"):
        plan.frame_of("nope", "start")
    with pytest.raises(ValueError, match="start, mid or last"):
        plan.frame_of("move", "end")
    with pytest.raises(IndexError):
        plan.value(plan.n_frames, "theta")


def test_a_span_that_rounds_to_no_frames_raises_naming_it():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0)
    tl.hold(0.01, name="blink")
    tl.hold(1.0)
    tl.at_fps(200)  # two frames there
    with pytest.raises(ValueError, match="blink"):
        tl.at_fps(10)


def test_undeclared_and_reserved_knob_names_raise():
    for name in ("index", "t", "section"):
        with pytest.raises(ValueError, match="reserved"):
            ep.Timeline(knobs={name: 1})
    tl = ep.Timeline(knobs={"x": 0.0, "radius": 1.0})
    tl.hold(1.0)
    with pytest.raises(ValueError, match=r"radus.*radius"):
        tl.move("m", 1.0, to={"radus": 3.0})
    with pytest.raises(ValueError, match="radus"):
        tl.hold(1.0, to={"radus": 3.0})


def test_moves_of_non_numeric_knobs_step_and_log_needs_positive_ends():
    tl = ep.Timeline(knobs={"text": "a", "x": 0.0, "w": 1.0})
    tl.hold(1.0)
    tl.move("step", 1.0, to={"text": "b"})
    plan = tl.at_fps(10)
    assert plan.value(plan.spans["step"].k0, "text") == "b"
    tl2 = ep.Timeline(knobs={"x": 0.0, "w": 1.0})
    tl2.hold(1.0)
    tl2.move("a", 1.0, to={"x": 2.0}, space="log")
    with pytest.raises(ValueError, match=r"'x'.*positive"):
        tl2.at_fps(10)
    tl3 = ep.Timeline(knobs={"w": 1.0})
    tl3.move("a", 1.0, to={"w": -2.0}, space="log")
    with pytest.raises(ValueError, match=r"'w'.*positive"):
        tl3.at_fps(10)


def test_names_are_unique_and_seconds_positive():
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(1.0, name="a")
    with pytest.raises(ValueError, match="already"):
        tl.hold(1.0, name="a")
    with pytest.raises(ValueError, match="seconds"):
        tl.hold(0.0)


def _golden_timeline():
    tl = ep.Timeline(knobs={"title": "intro", "theta": 0.0, "w": 64.0, "n": 3})
    tl.hold(1.0, name="a")
    tl.hold(0.5, name="b", to={"title": "turn"})
    tl.move("turn", 1.3, to={"theta": 2.0}, ease="smoothstep")
    tl.hold(0.37, name="c", to={"n": 4})
    tl.move("zoom", 0.9, to={"w": 4.0}, ease="smootherstep", space="log")
    tl.move("back", 0.61, to={"theta": 0.5}, ease="cosine")
    tl.hold(0.8, name="end", to={"title": "end"})
    return tl


@pytest.mark.parametrize("fps", [24, 30])
def test_a_sequential_timeline_reproduces_the_frames_the_old_beats_gave(fps):
    """The golden file holds the frames the holds-and-ramps Timeline produced."""
    golden = json.loads((DATA / "sequential_frames.json").read_text())[str(fps)]
    plan = _golden_timeline().at_fps(fps)
    frames = [{k: v for k, v in f.items() if k != "section"} for f in plan.frames]
    assert frames == golden


def test_unnamed_spans_get_distinct_names():
    tl = ep.Timeline()
    tl.hold(1.0)
    tl.hold(1.0)
    assert len(set(tl.at_fps(10).spans)) == 2


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


# Overshoot and anticipation eases

OVERSHOOT_KINDS = ["back_out", "running_start"]


@pytest.mark.parametrize("kind", OVERSHOOT_KINDS)
def test_overshoot_eases_end_exactly_at_zero_and_one(kind):
    for strength in (None, 0.05, 0.1, 0.5):
        assert ep.ease(0.0, kind, strength) == 0.0
        assert ep.ease(1.0, kind, strength) == 1.0
    # Only the input is clipped, so the ends hold outside [0, 1].
    assert ep.ease(-2.0, kind) == 0.0
    assert ep.ease(3.0, kind) == 1.0
    values = ep.ease(np.array([-1.0, 0.0, 1.0, 2.0]), kind)
    assert values.tolist() == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("strength", [0.02, 0.1, 0.3, 1.0])
def test_back_out_peaks_at_one_plus_strength_then_settles(strength):
    s = np.linspace(0.0, 1.0, 20001)
    values = ep.ease(s, "back_out", strength)
    assert values.max() == pytest.approx(1.0 + strength, abs=1e-6)
    peak = int(np.argmax(values))
    assert np.all(np.diff(values[: peak + 1]) >= 0.0)
    assert np.all(np.diff(values[peak:]) <= 1e-12)
    assert values[-1] == 1.0


@pytest.mark.parametrize("strength", [0.02, 0.1, 0.3])
def test_running_start_dips_to_minus_strength_then_goes(strength):
    s = np.linspace(0.0, 1.0, 20001)
    values = ep.ease(s, "running_start", strength)
    assert values.min() == pytest.approx(-strength, abs=1e-4)
    assert values.max() == 1.0
    low = int(np.argmin(values))
    assert 0 < low < len(s) // 2
    assert np.all(np.diff(values[low:]) >= -1e-12)


def test_the_default_back_out_is_the_classic_ten_percent_overshoot():
    s = np.linspace(0.0, 1.0, 20001)
    assert ep.ease(s, "back_out").max() == pytest.approx(1.1, abs=1e-6)
    assert ep.ease(s, "running_start").min() == pytest.approx(-0.1, abs=1e-4)


@pytest.mark.parametrize("kind", ["linear", "smoothstep", "smootherstep", "cosine"])
def test_other_kinds_take_no_strength_and_stay_inside_the_unit_interval(kind):
    with pytest.raises(ValueError, match="strength"):
        ep.ease(0.5, kind, 0.1)
    values = ep.ease(np.linspace(0.0, 1.0, 101), kind)
    assert values.min() >= 0.0 and values.max() <= 1.0


@pytest.mark.parametrize("strength", [0.0, -0.1, math.inf, math.nan])
def test_a_strength_must_be_positive_and_finite(strength):
    with pytest.raises(ValueError, match="strength"):
        ep.ease(0.5, "back_out", strength)


@pytest.mark.parametrize("kind", OVERSHOOT_KINDS)
def test_a_move_accepts_an_overshoot_ease_and_still_lands_on_its_target(kind):
    tl = ep.Timeline(knobs={"x": 0.0})
    tl.hold(0.1)
    tl.move("go", 1.0, to={"x": 2.0}, ease=kind)
    plan = tl.at_fps(24)
    xs = [f["x"] for f in plan.frames]
    assert xs[-1] == 2.0
    assert (max(xs) > 2.0) if kind == "back_out" else (min(xs) < 0.0)
