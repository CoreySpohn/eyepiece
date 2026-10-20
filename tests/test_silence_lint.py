"""The silence lint: line cue coverage, undeclared silence, the hero beat."""

import pytest

import eyepiece as ep
from eyepiece import narration as nr

LONG = "word " * 24  # about 24 * 5 / 12 = 10 s of speech


def _codes(faults):
    return [f.code for f in faults]


def _nar(*lines, **kwargs):
    return nr.Narration.estimated(nr.Script(tuple(lines)), **kwargs)


def test_a_clean_clip_has_no_findings():
    tl = _nar(
        nr.Line("a", "The {show:x}star is bright.", pad_after_s=0.5),
        nr.Pause("beat", 3.0, "the result stays on screen"),
        nr.Line("b", "Then the {show:y}planet.", pad_after_s=0.5),
    )
    assert nr.check_silence(tl) == []


def test_a_first_cue_far_after_the_line_start_is_flagged():
    tl = _nar(nr.Line("a", f"{LONG}{{show:x}}end."))
    faults = nr.check_silence(tl)
    assert _codes(faults) == ["late-first-cue"]
    assert faults[0].key == "a"


def test_a_last_cue_far_before_the_line_end_is_flagged():
    tl = _nar(nr.Line("a", f"{{show:x}}start {LONG}"))
    assert _codes(nr.check_silence(tl)) == ["early-last-cue"]


def test_the_cue_distance_is_a_parameter():
    tl = _nar(nr.Line("a", f"{LONG}{{show:x}}end."))
    assert nr.check_silence(tl, max_cue_gap_s=60.0) == []


def test_a_line_without_cues_is_not_a_cue_finding():
    assert nr.check_silence(_nar(nr.Line("a", LONG, pad_after_s=0.0))) == []


def test_a_silent_stretch_over_two_seconds_is_a_finding_unless_declared():
    padded = _nar(
        nr.Line("a", "One {show:x}line.", pad_after_s=3.0),
        nr.Line("b", "Two {show:y}line.", pad_after_s=0.0),
    )
    faults = nr.check_silence(padded)
    assert _codes(faults) == ["undeclared-silence"]
    assert faults[0].start_s == pytest.approx(padded.beat("a").speech_end_s)
    assert faults[0].end_s == pytest.approx(padded.beat("b").speech_start_s)
    declared = _nar(
        nr.Line("a", "One {show:x}line.", pad_after_s=0.0),
        nr.Pause("beat", 3.0, "result on screen"),
        nr.Line("b", "Two {show:y}line.", pad_after_s=0.0),
    )
    assert nr.check_silence(declared) == []


def test_silence_before_the_first_line_and_after_the_last_counts():
    tl = _nar(nr.Line("a", "One {show:x}line.", pad_before_s=3.0, pad_after_s=3.0))
    faults = nr.check_silence(tl)
    assert _codes(faults) == ["undeclared-silence"] * 2


def test_a_hero_window_declares_the_silence_inside_it():
    tl = _nar(
        nr.Line("a", "One {show:x}line.", pad_after_s=4.0),
        nr.Line("b", "Two {show:y}line."),
    )
    start = tl.beat("a").speech_end_s
    assert nr.check_silence(tl, hero=[(start, start + 4.0)]) == []
    partly = nr.check_silence(tl, hero=[(start, start + 1.0)])
    assert _codes(partly) == ["undeclared-silence"]
    assert partly[0].start_s == pytest.approx(start + 1.0)


def test_more_than_one_hero_span_is_a_finding():
    tl = _nar(
        nr.Line("a", "One {show:x}line.", pad_after_s=0.0),
        nr.Pause("beat", 6.0, "still"),
    )
    faults = nr.check_silence(tl, hero=[(3.0, 4.0), (5.0, 6.0)])
    assert "multiple-hero" in _codes(faults)


def test_speech_inside_a_hero_span_is_a_finding():
    tl = _nar(nr.Line("a", "One {show:x}line, then more."))
    faults = nr.check_silence(tl, hero=[(0.2, 0.6)])
    assert _codes(faults) == ["speech-in-hero"]
    assert faults[0].key == "a"


def test_the_plan_declares_the_hero_span():
    tl = nr.Narration.estimated(
        nr.Script(
            (
                nr.Line("a", "One {show:x}line.", pad_after_s=0.0),
                nr.Pause("beat", 4.0, "the result"),
                nr.Line("b", "Two {show:y}line.", pad_after_s=0.0),
            )
        )
    )
    clock = ep.Timeline({"k": 0.0}, narration=tl)
    clock.line("a")
    clock.enter("x")
    clock.line("beat")
    move = clock.move("slide", 4.0, to={"k": 1.0}, at=clock.now)
    clock.hero(move)
    clock.line("b")
    clock.enter("y")
    plan = clock.at_fps(10)
    assert plan.heroes == ("slide",)
    assert nr.check_silence(plan.narration, plan) == []
    # speech over the hero window is caught through the plan too
    assert plan.hero_windows() == [(move.start_s, move.end_s)]


def test_hero_names_a_span_that_exists():
    clock = ep.Timeline({"k": 0.0})
    with pytest.raises(KeyError, match="nothing"):
        clock.hero("nothing")
