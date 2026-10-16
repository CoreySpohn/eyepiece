"""The clock: spans, knobs, the build-time checks, events and the cursor."""

import itertools
import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import eyepiece as ep
from eyepiece import anim
from eyepiece import narration as nr

SMALL = nr.Script(
    (
        nr.Line("open", "The coronagraph blocks the {indicate:star}starlight."),
        nr.Line("leak", "What leaks through is {indicate:speckle}speckle."),
    )
)
TEMPO = ep.Tempo(factor=1.0)
FPS = (24, 25, 30, 50, 60)


def _codes(plan):
    return [f.code for f in plan.faults]


# ---------------------------------------------------------------------------
# Frames and progress


def test_a_span_covers_the_frames_the_rounded_times_say():
    tl = ep.Timeline({"x": 0.0})
    tl.hold(0.25)
    tl.move("go", 0.75, to={"x": 1.0})
    s = tl.at_fps(10).spans["go"]
    assert (s.k0, s.k1, s.n) == (3, 10, 7)  # 2.5 frames rounds half up to 3


def test_progress_is_zero_before_eased_inside_and_exactly_one_from_the_end():
    tl = ep.Timeline({"x": 0.0})
    tl.hold(1.0)
    tl.move("go", 1.0, to={"x": 2.0}, ease="smoothstep")
    tl.hold(1.0)
    plan = tl.at_fps(10)
    s = plan.spans["go"]
    assert plan.progress(s.k0 - 1, "go") == 0.0
    assert plan.progress(s.k0, "go") == pytest.approx(ep.ease(0.1, "smoothstep"))
    assert plan.progress(s.k0, "go", eased=False) == pytest.approx(0.1)
    assert plan.progress(s.k1 - 1, "go") == 1.0
    assert plan.progress(s.k1, "go") == 1.0
    assert plan.value(s.k1 - 1, "x") == 2.0


def test_a_span_rounded_to_no_frames_is_a_fault_that_a_strict_timeline_raises():
    def build(strict):
        tl = ep.Timeline(strict=strict)
        tl.hold(1.0)
        tl.hold(0.01, name="blink")
        tl.hold(1.0)
        return tl

    plan = build(False).at_fps(10)
    assert plan.spans["blink"].collapsed and plan.spans["blink"].n == 1
    assert _codes(plan) == ["collapsed"]
    assert "blink" in plan.faults[0].message
    with pytest.raises(ValueError, match="blink"):
        build(True).at_fps(10)


def test_the_frames_of_a_plan_are_fresh_dicts_with_the_frame_fields():
    tl = ep.Timeline({"x": 1.0})
    tl.section("only")
    tl.hold(0.5)
    plan = tl.at_fps(10)
    assert len(plan.frames) == plan.n_frames == 5
    frame = plan.frames[2]
    assert frame == {"x": 1.0, "index": 2, "t": 0.2, "section": "only"}
    frame["x"] = 9.0
    assert plan.frames[2]["x"] == 1.0
    assert plan.frames[-1]["index"] == 4
    with pytest.raises(IndexError):
        plan.frames[5]
    assert plan.seconds == 0.5


# ---------------------------------------------------------------------------
# One frame rule for the stills


@pytest.mark.parametrize("fps", [4, 8, 10])
def test_a_still_in_seconds_rounds_half_up_not_to_even(fps):
    for k in range(6):
        t = (k + 0.5) / fps
        assert anim._still_frame(t, None, fps) == ep.frame_count(t, fps) == k + 1


def test_a_still_by_span_name_is_the_frame_of_the_matching_event():
    tl = ep.Timeline({"x": 0.0})
    tl.hold(0.5)
    tl.move("go", 1.0, to={"x": 1.0})
    plan = tl.at_fps(8)
    at = {e.name: e.frame for e in plan.events(kind="move")}
    assert anim._still_frame(("go", "start"), plan, 8) == at["go.start"] == 4
    assert anim._still_frame(("go", "mid"), plan, 8) == at["go.mid"] == 7
    assert anim._still_frame(("go", "last"), plan, 8) == at["go.end"] == 11
    assert anim._still_frame("go", plan, 8) == 11
    with pytest.raises(ValueError, match="no span named"):
        anim._still_frame("nope", plan, 8)
    with pytest.raises(ValueError, match="needs plan="):
        anim._still_frame("go", None, 8)


# ---------------------------------------------------------------------------
# Events


def test_event_frames_follow_the_rules_and_before_is_clamped_at_zero():
    tl = ep.Timeline({"x": 0.0})
    tl.move("go", 1.0, to={"x": 1.0})
    plan = tl.at_fps(10)
    events = plan.events(kind="move")
    assert [e.name for e in events] == ["go.start", "go.mid", "go.end"]
    assert [e.what for e in events] == ["start", "mid", "end"]
    assert [e.frame for e in events] == [0, 4, 9]
    assert [e.before for e in events] == [0, 3, 8]
    assert [e.t_s for e in events] == [0.0, 0.5, 1.0]


def test_events_filter_by_what_kind_and_section_and_come_out_in_time_order():
    tl = ep.Timeline({"x": 0.0})
    tl.section("a")
    tl.hold(1.0, name="h")
    tl.section("b")
    tl.move("go", 1.0, to={"x": 1.0})
    plan = tl.at_fps(10)
    assert {e.section for e in plan.events(section="b")} == {"b"}
    assert {e.kind for e in plan.events(kind="hold")} == {"hold"}
    assert [e.name for e in plan.events(what="section-end")] == [
        "section:a.end",
        "section:b.end",
    ]
    times = [e.t_s for e in plan.events()]
    assert times == sorted(times)


def test_review_frames_are_every_middle_end_and_section_end_once_sorted():
    tl = ep.Timeline({"x": 0.0})
    tl.section("a")
    tl.move("go", 1.0, to={"x": 1.0})
    tl.section("b")
    tl.hold(1.0, name="h")
    plan = tl.at_fps(10)
    assert plan.review_frames() == (4, 9, 14, 19)


def test_spans_json_is_the_manifest_of_rounded_start_and_end_times():
    tl = ep.Timeline()
    tl.hold(0.3333333, name="a")
    tl.hold(1.0, name="b")
    assert tl.at_fps(30).spans_json() == {"a": [0.0, 0.333], "b": [0.333, 1.333]}


# ---------------------------------------------------------------------------
# Times


def test_times_are_exact_sums_not_accumulated_floats():
    tl = ep.Timeline()
    for _ in range(10):
        tl.hold(0.1)
    assert tl.now == 1.0
    t = ep.Time(0.1) + 0.2 + 0.3
    assert t == math.fsum([0.1, 0.2, 0.3])
    assert isinstance(t, ep.Time) and isinstance(0.5 + t, ep.Time)
    assert isinstance(t - 0.1, ep.Time) and isinstance(1.0 - t, ep.Time)
    assert ep.Time(1.0) - 0.25 == 0.75 and 1.0 - ep.Time(0.25) == 0.75


# ---------------------------------------------------------------------------
# Knob values


def test_a_knob_holds_its_declared_value_until_its_writer_and_the_last_after():
    tl = ep.Timeline({"x": 3.0, "y": 7.0})
    tl.hold(1.0)
    tl.move("go", 1.0, to={"x": 5.0}, ease="linear")
    tl.hold(1.0)
    plan = tl.at_fps(10)
    assert plan.value(0, "x") == plan.value(9, "x") == 3.0
    assert plan.value(10, "x") == pytest.approx(3.2)
    assert plan.value(19, "x") == 5.0 == plan.value(29, "x")
    assert all(plan.value(k, "y") == 7.0 for k in range(plan.n_frames))


def test_a_subject_knob_starts_at_zero_and_any_other_knob_must_be_declared():
    tl = ep.Timeline()
    tl.hold(1.0)
    tl.enter("thing", 1.0)
    plan = tl.at_fps(10)
    assert plan.value(0, "thing") == 0.0 and plan.value(19, "thing") == 1.0
    with pytest.raises(ValueError, match="never_declared"):
        tl.move("m", 1.0, to={"never_declared": 1.0})


def test_a_move_starts_from_the_value_on_the_frame_before_and_never_relative():
    tl = ep.Timeline({"x": 0.0})
    tl.move("a", 1.0, to={"x": 1.0}, ease="linear")
    tl.move("b", 1.0, to={"x": 3.0}, ease="linear")
    plan = tl.at_fps(10)
    assert plan.value(9, "x") == 1.0
    assert plan.value(10, "x") == pytest.approx(1.2)
    with pytest.raises(ValueError, match="relative"):
        tl.move("c", 1.0, to={"x": "+=1"})


def test_a_leave_returns_the_subject_knobs_to_zero_and_is_named_for_it():
    tl = ep.Timeline()
    tl.enter("s", 1.0, also=("u",))
    tl.leave("s", 1.0, also=("u",))
    tl.enter("s2", 1.0)
    plan = tl.at_fps(10)
    assert "s.leave" in plan.spans
    assert plan.value(9, "s") == plan.value(9, "u") == 1.0
    assert plan.value(19, "s") == plan.value(19, "u") == 0.0


def test_a_hold_steps_its_knobs_on_its_first_frame():
    tl = ep.Timeline({"mode": "a"})
    tl.hold(1.0)
    tl.hold(1.0, to={"mode": "b"})
    plan = tl.at_fps(10)
    assert plan.value(9, "mode") == "a" and plan.value(10, "mode") == "b"


def test_an_unknown_knob_in_the_plan_names_the_near_matches():
    tl = ep.Timeline()
    tl.enter("water", 1.0)
    plan = tl.at_fps(10)
    with pytest.raises(KeyError, match="did you mean"):
        plan.value(0, "watr")
    with pytest.raises(IndexError):
        plan.value(plan.n_frames, "water")


def test_the_back_out_ease_overshoots_the_value_but_not_the_progress():
    tl = ep.Timeline({"x": 0.0})
    tl.move("go", 1.0, to={"x": 1.0}, ease="back_out")
    plan = tl.at_fps(30)
    ps = [plan.progress(k, "go") for k in range(plan.n_frames)]
    assert all(0.0 <= p <= 1.0 for p in ps)
    assert max(plan.value(k, "x") for k in range(plan.n_frames)) > 1.0
    assert plan.value(plan.n_frames - 1, "x") == 1.0


def test_a_progress_only_move_writes_no_knob():
    tl = ep.Timeline()
    tl.hold(1.0)
    tl.span("slide", 1.0, kind="move")
    plan = tl.at_fps(10)
    assert plan.knobs == ()
    assert plan.progress(plan.spans["slide"].k1 - 1, "slide") == 1.0


# ---------------------------------------------------------------------------
# Overlaps and references


def test_two_writers_on_one_frame_raise_naming_both_and_the_knob():
    tl = ep.Timeline({"x": 0.0})
    tl.move("a", 1.0, to={"x": 1.0})
    tl.move("b", 1.0, to={"x": 2.0}, at=0.5)
    with pytest.raises(ValueError, match="knob 'x' is written by 'a' and 'b'"):
        tl.at_fps(10)


def test_ranges_that_only_touch_are_fine_and_a_step_owns_just_its_frame():
    touch = ep.Timeline({"x": 0.0})
    touch.move("a", 1.0, to={"x": 1.0})
    touch.move("b", 1.0, to={"x": 2.0}, at=1.0)
    touch.at_fps(10)

    clash = ep.Timeline({"x": 0.0})
    clash.hold(1.0, name="h", to={"x": 5.0})
    clash.move("m", 1.0, to={"x": 2.0}, at=0.0)
    with pytest.raises(ValueError, match="'x'"):
        clash.at_fps(10)

    after = ep.Timeline({"x": 0.0})
    after.hold(1.0, name="h", to={"x": 5.0})
    after.move("m", 1.0, to={"x": 2.0}, at=0.1)
    plan = after.at_fps(10)
    assert plan.value(0, "x") == 5.0 and plan.value(10, "x") == 2.0


def test_a_log_move_names_the_knob_whose_end_is_not_positive():
    tl = ep.Timeline({"R": 0.0, "w": 2.0})
    tl.move("m", 1.0, to={"R": 3.0, "w": 4.0}, space="log")
    with pytest.raises(ValueError, match=r"'R'.*positive"):
        tl.at_fps(10)


def test_a_log_move_is_geometric_between_its_ends():
    tl = ep.Timeline({"w": 1.0})
    tl.move("m", 1.0, to={"w": 100.0}, space="log", ease="linear")
    plan = tl.at_fps(10)
    assert plan.value(4, "w") == pytest.approx(10.0)
    assert plan.value(9, "w") == 100.0


def test_a_with_name_must_be_an_earlier_span():
    tl = ep.Timeline()
    tl.enter("a")
    with pytest.raises(ValueError, match=r"'later'.*\['a'\]"):
        tl.enter("b", with_=("later",))


def test_at_may_not_precede_the_start_of_its_section():
    tl = ep.Timeline()
    tl.section("one")
    tl.hold(2.0)
    tl.section("two")
    with pytest.raises(ValueError, match="before the start of section 'two'"):
        tl.enter("x", at=1.0)
    tl.enter("x", at=tl.now + 0.5)


def test_a_section_end_is_unknown_while_it_is_open():
    tl = ep.Timeline()
    sec = tl.section("one")
    tl.hold(1.0)
    with pytest.raises(ValueError, match="still open"):
        _ = sec.end
    tl.section("two")
    assert sec.end == 1.0


def test_a_span_and_a_section_may_share_a_name_but_two_sections_may_not():
    tl = ep.Timeline()
    tl.section("same")
    tl.enter("same")
    with pytest.raises(ValueError, match="already used"):
        tl.section("same")
    tl.enter("other")
    with pytest.raises(ValueError, match="already used"):
        tl.hold(1.0, name="other")
    assert "same" in tl.at_fps(10).spans


def test_a_line_and_a_span_may_share_a_name_but_two_lines_may_not():
    small = nr.Narration.estimated(SMALL)
    tl = ep.Timeline(narration=small, strict=False)
    tl.line("open")
    tl.hold(0.5, name="open")
    tl.line("leak")
    with pytest.raises(ValueError, match="already used"):
        tl.section("leak")
    plan = tl.at_fps(30)
    assert plan.events(what="line-start") and "open" in plan.spans


def test_a_colon_in_any_user_given_name_is_refused():
    with pytest.raises(ValueError, match="':'"):
        ep.Timeline({"a:b": 0.0})
    tl = ep.Timeline({"x": 0.0})
    with pytest.raises(ValueError, match="':'"):
        tl.section("a:b")
    with pytest.raises(ValueError, match="':'"):
        tl.enter("a:b")
    with pytest.raises(ValueError, match="':'"):
        tl.hold(1.0, name="a:b")


def test_bad_arguments_are_refused_with_the_valid_choices():
    tl = ep.Timeline()
    with pytest.raises(ValueError, match="no spans"):
        tl.at_fps(10)
    with pytest.raises(ValueError, match="unknown span kind"):
        tl.span("x", 1.0, kind="bogus")
    with pytest.raises(ValueError, match="unknown easing"):
        tl.move("m", 1.0, ease="bounce")
    with pytest.raises(ValueError, match="unknown space"):
        tl.move("m", 1.0, space="sqrt")
    with pytest.raises(ValueError, match="read, next or never"):
        tl.caption("c", "x", leaves="soon")
    with pytest.raises(ValueError, match="positive"):
        tl.hold(0.0)
    with pytest.raises(ValueError, match="reserved"):
        ep.Timeline({"t": 1.0})


# ---------------------------------------------------------------------------
# Clip length


def test_a_pulse_past_the_end_is_truncated_at_the_clip_and_recorded():
    tl = ep.Timeline()
    tl.hold(1.0)
    tl.pulse("p", at=0.5)
    plan = tl.at_fps(10)
    assert plan.n_frames == 10
    span = plan.spans["pulse.p"]
    assert (span.k0, span.n) == (5, 9)  # the speed is kept: 0.9 s is 9 frames
    assert _codes(plan) == ["clamped-to-clip"]
    assert plan.faults[0].severity == "info"
    assert "truncated by 4 frames" in plan.faults[0].message
    assert plan.value(9, "pulse.p") == pytest.approx(plan.progress(9, "pulse.p"))
    assert plan.progress(9, "pulse.p") == pytest.approx(5 / 9)


def test_a_truncated_pulse_runs_at_full_speed_not_compressed():
    tl = ep.Timeline()
    tl.hold(1.0)
    tl.pulse("p", at=0.1)
    full = tl.at_fps(30).spans["pulse.p"].n
    tl2 = ep.Timeline()
    tl2.hold(1.0)
    tl2.pulse("p", at=0.5)
    cut = tl2.at_fps(30).spans["pulse.p"]
    assert cut.n == full == 27


# ---------------------------------------------------------------------------
# The kind checks


def test_a_caption_arriving_over_a_move_raises_unless_the_pair_is_declared():
    tl = ep.Timeline({"x": 0.0})
    m = tl.move("m", 2.0, to={"x": 1.0})
    tl.caption("c", "words", at=m.start + 0.5)
    with pytest.raises(ValueError, match="caption 'c' arrives while move 'm'"):
        tl.at_fps(10)
    quiet = ep.Timeline({"x": 0.0}, strict=False)
    m = quiet.move("m", 2.0, to={"x": 1.0})
    quiet.caption("c", "words", at=m.start + 0.5)
    plan = quiet.at_fps(10)
    assert _codes(plan) == ["text-over-motion"]
    assert plan.faults[0].severity == "error"
    assert set(plan.faults[0].spans) == {"c", "m"}


def test_a_caption_declared_with_its_subject_arrives_before_it_in_a_silent_clip():
    tl = ep.Timeline(tempo=ep.Tempo())
    tl.enter("subject", 0.5, caption="two words")
    plan = tl.at_fps(30)
    cap, subject = plan.spans["cap_subject"], plan.spans["subject"]
    assert cap.k1 <= subject.k0
    assert cap.name in subject.with_ and subject.name in cap.with_


def test_pulses_and_leaves_may_overlap_motion():
    tl = ep.Timeline({"x": 0.0})
    m = tl.move("m", 2.0, to={"x": 1.0})
    tl.pulse("p", at=m.start + 0.2)
    tl.leave("gone", 0.5, at=m.start + 0.5)
    tl.at_fps(10)


def test_a_leave_may_not_overlap_the_arrival_of_its_own_caption():
    tl = ep.Timeline(strict=False)
    tl.enter("s", 0.5, caption="hi there")
    tl.leave("s", 0.3, at=0.1)
    assert "leave-over-arrival" in _codes(tl.at_fps(10))


def test_two_enters_on_the_same_frames_are_a_fault_unless_declared():
    tl = ep.Timeline()
    a = tl.enter("a", 1.0)
    tl.enter("b", 1.0, at=a.start + 0.2)
    with pytest.raises(ValueError, match="one subject enters at a time"):
        tl.at_fps(10)
    ok = ep.Timeline()
    a = ok.enter("a", 1.0)
    ok.enter("b", 1.0, at=a.start + 0.2, with_=("a",))
    ok.at_fps(10)


def test_a_caption_leaving_before_it_is_read_is_a_fault():
    def build(strict):
        tl = ep.Timeline(tempo=TEMPO, strict=strict)
        tl.caption("c", "one two three four five six", leaves="next")
        tl.caption("d", "next one", at=1.0)
        return tl

    with pytest.raises(ValueError, match="before it has been read"):
        build(True)
    assert "short-read" in _codes(build(False).at_fps(10))


# ---------------------------------------------------------------------------
# The cursor


def test_default_placement_advances_now_and_at_does_not():
    tl = ep.Timeline()
    a = tl.hold(1.0)
    assert tl.now == 1.0 and a.end == 1.0
    b = tl.enter("b", 1.0, at=a.start)
    assert tl.now == 1.0 and b.start == 0.0
    c = tl.enter("c", 0.5, with_=("b",))
    assert c.start == 1.0 and tl.now == 1.5


def test_together_starts_children_at_now_and_leaves_now_at_the_latest_end():
    tl = ep.Timeline()
    tl.hold(1.0)
    with tl.together():
        a = tl.enter("a", 0.5)
        b = tl.enter("b", 2.0, with_=("a",))
        assert tl.now == 1.0
    assert a.start == b.start == 1.0
    assert tl.now == 3.0


def test_stagger_chains_extending_verbs_and_leaves_now_after_the_group():
    tl = ep.Timeline()
    tl.hold(1.0)
    spans = tl.stagger("enter", ["a", "b", "c"], each_s=0.5, seconds=0.2)
    assert [float(s.start) for s in spans] == [1.0, 1.5, 2.0]
    assert tl.now == pytest.approx(2.2)
    after = tl.enter("d", 0.2)
    assert float(after.start) == pytest.approx(2.2)
    tl.at_fps(30)


def test_stagger_of_a_pulse_does_not_move_now():
    tl = ep.Timeline()
    tl.hold(1.0)
    spans = tl.stagger("pulse", ["a", "b"], each_s=0.3)
    assert [float(s.start) for s in spans] == [1.0, 1.3]
    assert tl.now == 1.0


def test_staggered_enters_that_overlap_raise_and_point_to_sequential_placement():
    tl = ep.Timeline()
    tl.hold(1.0)
    with pytest.raises(ValueError, match="one subject enters at a time") as err:
        tl.stagger("enter", ["a", "b"], each_s=0.1, seconds=0.5)
    assert "each_s of at least 0.500" in str(err.value)


def test_staggered_leaves_may_overlap():
    tl = ep.Timeline({"a": 1.0, "b": 1.0})
    tl.hold(1.0)
    spans = tl.stagger("leave", ["a", "b"], each_s=0.1, seconds=0.5)
    assert tl.now == pytest.approx(1.6)
    assert len(spans) == 2
    tl.at_fps(30)


def test_a_section_ends_at_its_last_extending_span_not_at_a_pulse():
    tl = ep.Timeline()
    one = tl.section("one")
    e = tl.enter("e", 1.0)
    tl.pulse("p", at=e.start + 0.5)
    two = tl.section("two")
    assert one.end == 1.0 and two.start == 1.0


def test_settle_rests_and_read_holds_for_the_tempo():
    tl = ep.Timeline(tempo=ep.Tempo())
    s = tl.settle()
    assert s.end - s.start == pytest.approx(ep.Tempo().rest_s)
    r = tl.read("one two three")
    assert r.end - r.start == pytest.approx(ep.Tempo().read_s("one two three"))


def test_a_tempo_paces_holds_enters_and_moves_but_not_runs_or_attention():
    tl = ep.Timeline({"p": 0.0}, tempo=ep.Tempo(factor=2.0))
    h = tl.hold(1.0)
    e = tl.enter("e", 1.0)
    r = tl.run("r", 1.0, to={"p": 1.0})
    p = tl.pulse("t", at=0.0)
    assert h.end - h.start == 2.0 and e.end - e.start == 2.0
    assert r.end - r.start == 1.0
    assert p.end - p.start == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Captions in a silent clip


def test_a_silent_caption_leaves_once_read_and_its_section_waits_for_it():
    tl = ep.Timeline(tempo=TEMPO)
    one = tl.section("one")
    cap = tl.caption("c", "four short words here")
    tl.enter("e", 0.5)
    tl.section("two")
    read = TEMPO.read_s("four short words here")
    assert one.end == pytest.approx(cap.end + read)
    plan = tl.at_fps(30)
    leave = plan.spans["c.leave"]
    assert leave.start_s == pytest.approx(cap.end + read)
    assert plan.spans["c"].text == "four short words here"
    assert plan.spans["c"].read_s == pytest.approx(read)
    assert plan.value(plan.spans["c"].k1 - 1, "caption.c") == 1.0
    assert plan.value(plan.n_frames - 1, "caption.c") == 0.0
    assert plan.n_frames == ep.frame_count(leave.end_s, 30)


def test_a_caption_waits_for_the_motion_of_its_section_before_it_leaves():
    tl = ep.Timeline(tempo=TEMPO)
    cap = tl.caption("c", "hi")
    m = tl.move("m", 10.0)
    plan = tl.at_fps(10)
    assert plan.spans["c.leave"].start_s == pytest.approx(m.end)
    assert cap.end < m.end


def test_clear_caption_takes_the_caption_down_at_now_and_needs_one_on_screen():
    tl = ep.Timeline(tempo=TEMPO)
    tl.caption("c", "hi", leaves="next")
    tl.hold(2.0)
    gone = tl.clear_caption()
    assert gone.name == "c.leave" and gone.start == pytest.approx(2.4)
    assert tl.now == pytest.approx(2.7)
    with pytest.raises(ValueError, match="no caption"):
        tl.clear_caption()


def test_a_never_caption_stays_and_a_next_caption_leaves_when_the_next_arrives():
    tl = ep.Timeline(tempo=TEMPO)
    tl.caption("a", "x", leaves="next")
    tl.hold(5.0)
    b = tl.caption("b", "y", leaves="never")
    plan = tl.at_fps(10)
    assert plan.spans["a.leave"].start_s == pytest.approx(b.start)
    assert "b.leave" not in plan.spans
    assert plan.value(plan.n_frames - 1, "caption.b") == 1.0
    assert plan.value(plan.n_frames - 1, "caption.a") == 0.0


# ---------------------------------------------------------------------------
# A silent clip


def test_an_attention_span_placed_only_by_a_cue_is_skipped_and_recorded():
    tl = ep.Timeline()
    tl.enter("a", 1.0)
    skipped = tl.pulse("a")
    placed = tl.pulse("a", at=0.0)
    plan = tl.at_fps(10)
    assert skipped.extends is False and skipped.start == skipped.end
    assert plan.spans["pulse.a"].start_s == 0.0
    infos = [f for f in plan.faults if f.code == "skipped"]
    assert len(infos) == 1 and infos[0].severity == "info"
    assert placed.name == "pulse.a"

    only = ep.Timeline()
    only.hold(1.0)
    only.pulse("z")
    assert only.at_fps(10).value(5, "pulse.z") == 0.0


def test_a_cue_is_ignored_without_narration_and_line_needs_one():
    tl = ep.Timeline()
    e = tl.enter("a", 1.0, cue="show:nothing")
    assert e.start == 0.0
    assert tl.cue("show:a") == tl.now
    with pytest.raises(ValueError, match="narration"):
        tl.line("x")


# ---------------------------------------------------------------------------
# Properties

_seconds = st.integers(min_value=1, max_value=400).map(lambda n: n / 100.0)
_fps = st.sampled_from(FPS)


@given(st.lists(_seconds, min_size=1, max_size=8), _fps)
@settings(max_examples=100, deadline=None)
def test_property_sequential_spans_tile_and_total_the_rounded_sum(durations, fps):
    assume(all(d * fps >= 1.0 for d in durations))
    tl = ep.Timeline({"x": 0.0})
    for i, d in enumerate(durations):
        if i % 2 == 0:
            tl.hold(d)
        else:
            tl.move(f"m{i}", d, to={"x": float(i)})
    plan = tl.at_fps(fps)
    spans = list(plan.spans.values())
    assert spans[0].k0 == 0
    for a, b in itertools.pairwise(spans):
        assert b.k0 == a.k1
    assert spans[-1].k1 == plan.n_frames == ep.frame_count(math.fsum(durations), fps)


@given(st.lists(_seconds, min_size=1, max_size=6), _fps)
@settings(max_examples=100, deadline=None)
def test_property_every_span_has_at_least_one_frame_at_every_fps(durations, fps):
    tl = ep.Timeline(strict=False)
    for d in durations:
        tl.hold(d)
    plan = tl.at_fps(fps)
    assert all(s.n >= 1 for s in plan.spans.values())


@given(
    st.sampled_from(["linear", "smoothstep", "smootherstep", "cosine"]),
    _seconds,
    _seconds,
    _fps,
)
@settings(max_examples=100, deadline=None)
def test_property_progress_is_monotone_and_lands_on_one(kind, lead, dur, fps):
    assume(dur * fps >= 1.0 and lead * fps >= 1.0)
    tl = ep.Timeline({"x": 0.0})
    tl.hold(lead)
    tl.move("go", dur, to={"x": 1.0}, ease=kind)
    plan = tl.at_fps(fps)
    s = plan.spans["go"]
    ps = [plan.progress(k, "go") for k in range(plan.n_frames)]
    assert ps == sorted(ps)
    assert plan.progress(s.k1 - 1, "go") == 1.0
    if s.k0 > 0:
        assert plan.progress(s.k0 - 1, "go") == 0.0


@given(st.integers(0, 200), _seconds, st.integers(0, 200), _seconds, _fps)
@settings(max_examples=200, deadline=None)
def test_property_two_writers_raise_if_and_only_if_their_frames_intersect(
    s1, d1, s2, d2, fps
):
    a0, b0 = s1 / 100.0, s2 / 100.0
    tl = ep.Timeline({"x": 0.0}, strict=False)
    tl.move("a", d1, to={"x": 1.0}, at=a0)
    tl.move("b", d2, to={"x": 2.0}, at=b0)

    def frames(start, dur):
        k0 = ep.frame_count(start, fps)
        return k0, max(k0 + 1, ep.frame_count(start + dur, fps))

    (a_0, a_1), (b_0, b_1) = frames(a0, d1), frames(b0, d2)
    if a_0 < b_1 and b_0 < a_1:
        with pytest.raises(ValueError, match="'x'"):
            tl.at_fps(fps)
    else:
        tl.at_fps(fps)


@given(st.lists(st.tuples(st.integers(0, 2), _seconds), min_size=1, max_size=12))
@settings(max_examples=100, deadline=None)
def test_property_now_never_decreases(calls):
    tl = ep.Timeline()
    last = tl.now
    for i, (how, d) in enumerate(calls):
        if how == 0:
            tl.enter(f"e{i}", d, at=tl.now)
        elif how == 1:
            tl.hold(d)
        else:
            with tl.together():
                tl.enter(f"e{i}", d)
        assert tl.now >= last
        last = tl.now


@given(
    st.lists(_seconds, min_size=1, max_size=6), st.sampled_from([12, 15, 24, 25, 30])
)
@settings(max_examples=100, deadline=None)
def test_property_doubling_the_fps_moves_each_event_by_at_most_one_coarse_frame(
    durations, fps
):
    assume(all(d * fps >= 1.0 for d in durations))
    tl = ep.Timeline({"x": 0.0}, strict=False)
    for i, d in enumerate(durations):
        tl.move(f"m{i}", d, to={"x": float(i + 1)})
    coarse, fine = tl.at_fps(fps), tl.at_fps(2 * fps)
    for a, b in zip(coarse.events(), fine.events(), strict=True):
        assert a.name == b.name
        assert abs(b.frame / 2.0 - a.frame) <= 1.0


@given(st.lists(_seconds, min_size=2, max_size=6), _fps)
@settings(max_examples=100, deadline=None)
def test_property_a_still_at_a_start_time_lands_on_the_start_event_frame(
    durations, fps
):
    tl = ep.Timeline(strict=False)
    for d in durations:
        tl.hold(d)
    plan = tl.at_fps(fps)
    for e in plan.events(what="start"):
        assert anim._still_frame(e.t_s, plan, fps) == e.frame


def test_unused_knobs_counts_value_frames_and_progress_reads_and_skips_pulse_knobs():
    tl = ep.Timeline({"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0})
    tl.move("go", 1.0, to={"a": 1.0, "b": 2.0})
    tl.move("go2", 1.0, to={"c": 1.0})
    tl.pulse("t", at=0.1)
    plan = tl.at_fps(10)
    unread, constant = plan.unused_knobs()
    assert unread == ("a", "b", "c", "d") and constant == ("d",)
    plan.value(0, "a")
    _ = plan.frames[0]["b"]
    plan.progress(0, "go2")
    unread, constant = plan.unused_knobs()
    assert unread == ("d",) and constant == ("d",)
    assert not any(k.startswith("pulse.") for k in (*unread, *constant))


def _field_plan():
    tl = ep.Timeline({"a": 0.0})
    tl.section("intro")
    tl.move("go", 1.0, to={"a": 1.0})
    return tl.at_fps(10)


def test_value_returns_the_frame_fields():
    plan = _field_plan()
    for k in (0, 3, plan.n_frames - 1):
        assert plan.value(k, "t") == k / plan.fps == plan.frames[k]["t"]
        assert plan.value(k, "index") == k
        assert plan.value(k, "section") == plan.frames[k]["section"] == "intro"
    with pytest.raises(IndexError):
        plan.value(plan.n_frames, "t")


def test_reading_frame_fields_leaves_unused_knobs_unchanged():
    plan = _field_plan()
    before = plan.unused_knobs()
    for name in ("t", "index", "section"):
        plan.value(0, name)
    assert plan.unused_knobs() == before == (("a",), ())


def test_value_still_rejects_unknown_names():
    plan = _field_plan()
    with pytest.raises(KeyError, match="no knob named 'tt'"):
        plan.value(0, "tt")
