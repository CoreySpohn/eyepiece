"""The clock with a narration: cues place the spans, lines solve in one pass.

The gate is a sample clip, an explainer of how a deep band changes the chance
of water, whose cached speech timing is a fixture. It is laid out here by the
clock, and by a port of the two-pass layout the clock replaced; the two must
agree on every line start and end. Nothing needs a speech model.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import eyepiece as ep
from eyepiece import narration as nr

DATA = Path(__file__).parent / "data" / "narration"
CACHE = DATA / "bayes_area"
SHOW_LEAD_S = 0.1
PULSE_S = 0.9
RING_S = 1.3
TAIL_S = 0.6

# The two-line script whose timing is the fixture of test_narration.
SMALL = nr.Script(
    (
        nr.Line(
            "open",
            "The coronagraph blocks the {indicate:star}starlight.",
            pad_after_s=0.4,
        ),
        nr.Line(
            "leak",
            "What leaks through is {indicate:speckle}speckle, and the "
            "{circumscribe:planet}planet must stand out.",
        ),
    )
)

BAYES = nr.Script(
    (
        nr.Line(
            "question",
            "How does seeing a deep band change the chance that a planet has water?",
            pad_after_s=0.6,
        ),
        nr.Line(
            "square",
            "This {show:square}square holds every possible world, and its area is one.",
        ),
        nr.Line(
            "water",
            "{show:water}Water worlds fill the left {indicate:water_label}0.3 "
            "of the square.",
        ),
        nr.Line(
            "band_h",
            "A deep {show:band_h}band appears in {indicate:band_h_label}0.8 "
            "of the water worlds.",
        ),
        nr.Line(
            "band_n",
            "It also appears in the {show:band_n}dry worlds, in "
            "{indicate:band_n_label}0.1 of them.",
        ),
        nr.Line(
            "rule",
            "We see the band, so every world without it is {show:rule}ruled out.",
        ),
        nr.Line(
            "areas",
            "Two {show:areas}areas are left. Water worlds with the band cover "
            "{circumscribe:area_h}0.24, and dry worlds with the band cover "
            "{circumscribe:area_n}0.07.",
        ),
        nr.Line(
            "stretch",
            "{move:dim}Stack those two areas, and {move:stretch}stretch "
            "them until their total height is one.",
        ),
        nr.Line(
            "bar",
            "In the finished bar, the {show:bar}water part has a height of 0.77.",
        ),
        nr.Line(
            "result",
            "{show:eq}That is the water area, {indicate:num}0.24, divided by "
            "the {indicate:den}total that is left. Seeing the band raises the "
            "chance of water from {indicate:prior}0.3 to {indicate:post}0.77.",
            pad_after_s=1.5,
        ),
    )
)

CAPS = {
    "water": "water worlds fill 0.3 of it",
    "band_h": "deep band in 0.8 of water worlds",
    "rule": "deep band seen: other worlds ruled out",
    "areas": "what is left: area 0.24 and 0.07",
    "stretch": "stack what is left, stretch to height 1",
}
END_S = 2.25


def _codes(plan):
    return [f.code for f in plan.faults]


def bayes_clock(narration=None, strict=True):
    """The sample clip's moves; narrated, each fires on the word that names it."""
    tl = ep.Timeline(
        {"stretch": 0.0}, tempo=ep.Tempo(), narration=narration, strict=strict
    )

    def line(key):
        if narration is not None:
            return tl.line(key)
        return tl.section(key)

    line("question")
    tl.hold(0.5)
    line("square")
    tl.enter("square")
    line("water")
    tl.enter("water", caption=CAPS["water"])
    tl.pulse("water_label")
    line("band_h")
    tl.enter("band_h", caption=CAPS["band_h"])
    tl.pulse("band_h_label")
    line("band_n")
    tl.enter("band_n")
    tl.pulse("band_n_label")
    line("rule")
    tl.enter("rule", 0.75, caption=CAPS["rule"])
    line("areas")
    tl.enter("areas", 0.5, caption=CAPS["areas"])
    tl.ring("area_h")
    tl.ring("area_n")
    line("stretch")
    tl.enter("dim", 0.5, also=("rect_in",), caption=CAPS["stretch"], cue="move:dim")
    tl.move("stretch", 1.75, to=1.0)
    line("bar")
    tl.enter("bar", 0.5)
    line("result")
    eq = tl.enter("eq", 0.5)
    tl.leave("dim", 0.5, at=eq.start)
    if narration is not None:
        tl.clear_caption(at=eq.start)
    for name in ("num", "den", "prior", "post"):
        tl.pulse(name)
    if narration is None:
        tl.hold(END_S)
    return tl


@pytest.fixture(scope="module")
def bayes():
    return nr.Narration.load(BAYES, CACHE)


@pytest.mark.parametrize("suffix", [".py", ".json"])
def test_estimate_cli_needs_no_kokoro_or_cache_and_tracks_cached_speech(
    tmp_path, bayes, suffix
):
    record = asdict(BAYES)
    for line in record["lines"]:
        del line["spoken"], line["cues"]
    payload = json.dumps(record)
    path = tmp_path / f"script{suffix}"
    if suffix == ".json":
        path.write_text(payload)
    else:
        path.write_text(
            "import json\nfrom eyepiece.narration import Line, Script\n"
            f"record = json.loads({payload!r})\n"
            "record['lines'] = tuple(Line(**line) for line in record['lines'])\n"
            "SCRIPT = Script(**record)\n"
        )
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['kokoro'] = None; "
            "from eyepiece.narration.synth import main; main(sys.argv[1:])",
            str(path),
            "--estimate",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    rows = [row.split() for row in done.stdout.splitlines()]
    assert [row[0] for row in rows] == [line.key for line in BAYES.lines] + ["total"]
    durations = [float(row[1]) for row in rows]
    assert all(row[2] == "s" for row in rows)
    assert durations[-1] == pytest.approx(sum(durations[:-1]), abs=0.01)
    for line, seconds in zip(BAYES.lines, durations[:-1], strict=True):
        assert seconds == pytest.approx(nr.estimate(line).duration_s, abs=0.001)
    # Total means speech audio only: sum the cached line durations, excluding
    # padding and motion holds, which are not part of the speech estimate.
    cached_speech_s = sum(timing.duration_s for timing in bayes.spoken.values())
    assert durations[-1] == pytest.approx(cached_speech_s, rel=0.1)


# ---------------------------------------------------------------------------
# The reference: the two-pass layout the clock replaced


def two_pass(narration, motion):
    """Line start and end per key, laid out as the hand-written passes did.

    The first pass reads each cue's offset from the start of its line on the
    speech alone; the second holds the line until the motion fired on that cue
    has finished, plus a tail.

    Args:
        narration: A `Narration` laid out on speech alone.
        motion: Rows of (line key, cue kind, cue target, seconds, lead).
    """
    need = {}
    for key, kind, target, seconds, lead in motion:
        beat = narration.beat(key)
        cue = next(c for c in beat.cues if (c.kind, c.target) == (kind, target))
        offset = max(0.0, cue.at_s - lead - beat.start_s)
        need[key] = max(need.get(key, 0.0), offset + seconds + TAIL_S)
    clock, out = 0.0, {}
    for line in narration.script.lines:
        timing = narration.spoken[line.key]
        on = clock + line.pad_before_s
        end = on + timing.duration_s + line.pad_after_s
        end = max(end, clock + need.get(line.key, 0.0))
        out[line.key] = (clock, end)
        clock = end
    return out


def bayes_motion(tempo):
    return (
        ("square", "show", "square", tempo.seconds(0.6), SHOW_LEAD_S),
        ("water", "show", "water", tempo.seconds(0.6), SHOW_LEAD_S),
        ("band_h", "show", "band_h", tempo.seconds(0.6), SHOW_LEAD_S),
        ("band_n", "show", "band_n", tempo.seconds(0.6), SHOW_LEAD_S),
        ("rule", "show", "rule", tempo.seconds(0.75), SHOW_LEAD_S),
        ("areas", "show", "areas", tempo.seconds(0.5), SHOW_LEAD_S),
        ("stretch", "move", "dim", tempo.seconds(0.5), SHOW_LEAD_S),
        ("stretch", "move", "stretch", tempo.seconds(1.75), 0.0),
        ("bar", "show", "bar", tempo.seconds(0.5), SHOW_LEAD_S),
        ("result", "show", "eq", tempo.seconds(0.5), SHOW_LEAD_S),
    )


# ---------------------------------------------------------------------------
# Parity with the two-pass layout


def test_line_starts_and_ends_match_the_two_pass_layout_within_a_millisecond(bayes):
    plan = bayes_clock(bayes).at_fps(30)
    old = two_pass(bayes, bayes_motion(ep.Tempo()))
    assert set(old) == {b.key for b in plan.narration.beats}
    for beat in plan.narration.beats:
        start, end = old[beat.key]
        assert beat.start_s == pytest.approx(start, abs=1e-3), beat.key
        assert beat.end_s == pytest.approx(end, abs=1e-3), beat.key


def test_span_starts_match_the_hand_placed_ones_within_a_millisecond(bayes):
    plan = bayes_clock(bayes).at_fps(30)
    narration = plan.narration

    def word(key, kind, target):
        return next(
            c.at_s
            for c in narration.beat(key).cues
            if (c.kind, c.target) == (kind, target)
        )

    def show(key, target):
        return max(narration.beat(key).start_s, word(key, "show", target) - SHOW_LEAD_S)

    def pulse(key, target):
        start = narration.beat(key).start_s
        return max(start, word(key, "indicate", target) - 0.5 * PULSE_S)

    want = {
        "square": show("square", "square"),
        "water": show("water", "water"),
        "band_h": show("band_h", "band_h"),
        "band_n": show("band_n", "band_n"),
        "rule": show("rule", "rule"),
        "areas": show("areas", "areas"),
        "bar": show("bar", "bar"),
        "eq": show("result", "eq"),
        "dim": max(
            narration.beat("stretch").start_s,
            word("stretch", "move", "dim") - SHOW_LEAD_S,
        ),
        "stretch": word("stretch", "move", "stretch"),
        "ring.area_h": word("areas", "circumscribe", "area_h"),
        "ring.area_n": word("areas", "circumscribe", "area_n"),
        "pulse.water_label": pulse("water", "water_label"),
        "pulse.band_h_label": pulse("band_h", "band_h_label"),
        "pulse.band_n_label": pulse("band_n", "band_n_label"),
        "pulse.num": pulse("result", "num"),
        "pulse.post": pulse("result", "post"),
    }
    for name, start in want.items():
        assert plan.spans[name].start_s == pytest.approx(start, abs=1e-3), name
    leaves = {
        "cap_water.leave": "band_h",
        "cap_band_h.leave": "rule",
        "cap_rule.leave": "areas",
        "cap_areas.leave": "dim",
        "cap_dim.leave": "eq",
    }
    for name, nxt in leaves.items():
        assert plan.spans[name].start_s == pytest.approx(want[nxt], abs=1e-3), name


@pytest.mark.parametrize("fps", [24, 30])
def test_cue_frames_are_the_rounded_cue_times_and_within_a_frame_of_the_floor(
    bayes, fps
):
    plan = bayes_clock(bayes).at_fps(fps)
    cues = plan.events(what="cue")
    assert len(cues) == len(bayes.cues()) == 19
    for event, cue in zip(
        cues, sorted(bayes.cues(), key=lambda c: c.at_s), strict=True
    ):
        assert event.frame == ep.frame_count(cue.at_s, fps)
        floor = int(np.floor(cue.at_s * fps + 1e-9))
        assert event.frame - floor in (0, 1)
        assert event.before == max(event.frame - 1, 0)


@pytest.mark.parametrize("fps", [24, 30])
def test_every_value_a_draw_function_reads_is_within_a_frame_of_the_continuous_one(
    bayes, fps
):
    """Draw functions read only knobs, so equal knobs are equal pictures."""
    plan = bayes_clock(bayes).at_fps(fps)
    for name, s in plan.spans.items():
        if not s.writes or s.end_s - s.start_s < 2.0 / fps:
            continue
        for k in range(plan.n_frames):
            lo = np.clip((k / fps - s.start_s) / (s.end_s - s.start_s), 0.0, 1.0)
            hi = np.clip(((k + 2) / fps - s.start_s) / (s.end_s - s.start_s), 0.0, 1.0)
            lo, hi = ep.ease(lo, s.ease), ep.ease(hi, s.ease)
            got = plan.progress(k, name)
            assert lo - 0.05 <= got <= hi + 0.05, (name, k)


def test_the_narrated_clip_builds_strict_with_only_infos(bayes):
    plan = bayes_clock(bayes, strict=True).at_fps(30)
    assert {f.severity for f in plan.faults} <= {"info"}
    assert "short-read" not in _codes(plan)


def test_one_function_builds_the_silent_clip_too():
    plan = bayes_clock().at_fps(30)
    assert plan.narration is None
    assert plan.n_frames == ep.frame_count(
        max(s.end_s for s in plan.spans.values() if s.extends), 30
    )
    skipped = [f for f in plan.faults if f.code == "skipped"]
    assert len(skipped) == 9 and {f.severity for f in skipped} == {"info"}
    assert not any(n.startswith(("pulse.", "ring.")) for n in plan.spans)
    assert plan.value(plan.n_frames - 1, "stretch") == 1.0
    assert plan.value(plan.n_frames - 1, "caption.cap_water") == 0.0
    assert plan.value(plan.n_frames - 1, "pulse.num") == 0.0


# ---------------------------------------------------------------------------
# The binding, on the two-line fixture


def test_each_span_starts_on_its_word_and_the_cue_events_follow_the_words():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    tl.line("open")
    star = tl.pulse("star")
    tl.line("leak")
    speckle = tl.pulse("speckle")
    planet = tl.ring("planet")
    assert star.start == pytest.approx(1.3 - 0.45)
    assert speckle.start == pytest.approx(3.4 - 0.45)
    assert planet.start == pytest.approx(4.3)
    assert tl.cue("circumscribe:planet") == pytest.approx(4.3)
    plan = tl.at_fps(30)
    events = plan.events(what="cue")
    assert [e.name for e in events] == [
        "cue:open:indicate:star",
        "cue:leak:indicate:speckle",
        "cue:leak:circumscribe:planet",
    ]
    assert [e.frame for e in events] == [39, 102, 129]
    assert plan.events(what="line-start")[1].name == "line:leak.start"
    assert plan.n_frames == 183 and plan.narration.total_s == pytest.approx(6.1)


def test_lines_open_in_script_order_and_every_line_must_open():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    with pytest.raises(ValueError, match=r"script order.*'open'"):
        tl.line("leak")
    with pytest.raises(KeyError, match="no line named"):
        tl.line("nope")
    tl.line("open")
    tl.pulse("star")
    with pytest.raises(ValueError, match=r"never opened.*\['leak'\]"):
        tl.at_fps(30)


def test_a_mark_no_verb_fires_is_a_fault():
    small = nr.Narration.load(SMALL, DATA)

    def build(strict):
        tl = ep.Timeline(narration=small, strict=strict)
        tl.line("open")
        tl.pulse("star")
        tl.line("leak")
        tl.pulse("speckle")
        return tl

    with pytest.raises(ValueError, match="circumscribe:planet is fired by no verb"):
        build(True).at_fps(30)
    assert _codes(build(False).at_fps(30)) == ["unused-mark"]


def test_a_verb_with_no_inferred_mark_names_the_marks_the_line_has():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    tl.line("open")
    with pytest.raises(ValueError, match=r"no mark 'indicate:moon'.*indicate:star"):
        tl.pulse("moon")
    quiet = ep.Timeline(narration=small, strict=False)
    quiet.line("open")
    quiet.pulse("moon")
    quiet.pulse("star")
    quiet.line("leak")
    quiet.pulse("speckle")
    quiet.ring("planet")
    assert _codes(quiet.at_fps(30)) == ["missing-mark"]


def test_a_cue_in_a_later_line_cannot_be_waited_on():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    tl.line("open")
    with pytest.raises(ValueError, match="comes after the current line"):
        tl.pulse("speckle", cue="indicate:speckle")
    with pytest.raises(ValueError, match="comes after the current line"):
        tl.cue("indicate:speckle")


def _late_clock(over_s, **kwargs):
    """A move on the word `star`, with the cursor `over_s` seconds past it."""
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small, **kwargs)
    tl.line("open")
    tl.hold(small.cue("open", "indicate", "star") + over_s)
    tl.move("star", 0.3, cue="indicate:star")
    tl.line("leak")
    tl.pulse("speckle")
    tl.ring("planet")
    return tl


def test_a_span_that_waits_for_the_cursor_is_info_under_max_late_s():
    plan = _late_clock(0.3).at_fps(30)
    late = [f for f in plan.faults if f.code == "late"]
    assert len(late) == 1 and late[0].severity == "info"


def test_a_span_that_waits_past_max_late_s_is_an_error_naming_span_word_and_seconds():
    small = nr.Narration.load(SMALL, DATA)
    word = small.beat("open").cues[0].word.text
    with pytest.raises(
        ValueError, match=r"move 'star' starts 1\.700 s after its cue word"
    ) as err:
        _late_clock(1.7).at_fps(30)
    assert repr(word) in str(err.value) and "max_late_s" in str(err.value)
    plan = _late_clock(1.7, strict=False).at_fps(30)
    late = [f for f in plan.faults if f.code == "late"]
    assert [f.severity for f in late] == ["error"]


def test_max_late_s_is_a_knob_of_the_timeline():
    plan = _late_clock(1.7, max_late_s=5.0).at_fps(30)
    assert {f.severity for f in plan.faults if f.code == "late"} == {"info"}
    with pytest.raises(ValueError, match="max_late_s"):
        ep.Timeline(max_late_s=-1.0)


LONG_CAPTION = " ".join(["words"] * 40)


def test_a_caption_the_next_caption_replaces_needs_no_read_time_when_narrated():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small, tempo=ep.Tempo())
    tl.line("open")
    tl.caption("first", LONG_CAPTION, cue="indicate:star")
    tl.line("leak")
    tl.caption("second", "the next one", leaves="never", cue="indicate:speckle")
    tl.ring("planet")
    plan = tl.at_fps(30)
    assert "short-read" not in _codes(plan)
    assert plan.spans["first.leave"].start_s < plan.spans["first"].start_s + 60


def test_a_silent_caption_the_next_one_replaces_still_needs_its_read_time():
    silent = ep.Timeline(tempo=ep.Tempo(), strict=False)
    silent.caption("first", LONG_CAPTION, leaves="next")
    silent.hold(0.1)
    silent.caption("second", "the next one", leaves="never")
    assert "short-read" in _codes(silent.at_fps(30))


def test_a_cue_in_an_earlier_line_is_clamped_to_the_line_and_flagged():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    tl.line("open")
    tl.pulse("star")
    tl.line("leak")
    tl.pulse("speckle")
    tl.ring("planet")
    again = tl.pulse("star", cue="indicate:star")
    assert again.start == 2.1
    assert tl.cue("indicate:star") == pytest.approx(1.3)
    plan = tl.at_fps(30)
    assert plan.spans[again.name].clamped is True
    assert plan.spans["pulse.speckle"].clamped is False


def test_a_motion_longer_than_its_line_pushes_the_next_line_and_its_cues():
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    tl.line("open")
    tl.pulse("star")
    tl.move("long", 4.0, at=0.0)
    tl.line("leak")
    tl.pulse("speckle")
    tl.ring("planet")
    assert tl.cue("circumscribe:planet") == pytest.approx(4.3 + 4.6 - 2.1)
    plan = tl.at_fps(30)
    opened, leak = plan.narration.beat("open"), plan.narration.beat("leak")
    assert opened.end_s == pytest.approx(4.0 + 0.6)
    assert leak.start_s == pytest.approx(4.6)
    shift = 4.6 - 2.1
    assert plan.spans["ring.planet"].start_s == pytest.approx(4.3 + shift)
    assert plan.n_frames == ep.frame_count(leak.end_s, 30)


@pytest.mark.parametrize("offset", [-1.0, -0.25, 2.0])
def test_a_line_wait_has_no_tail_and_preserves_speech_and_padding(bayes, offset):
    tl = ep.Timeline(narration=bayes)
    sec = tl.line("question")
    tl.hold(0.25)
    padded_end = bayes.beat("question").end_s
    waited = padded_end + offset
    tl.wait_until(waited)
    want = max(padded_end, waited)
    assert tl.narration.beat("question").end_s == want
    following = tl.line("square")
    assert sec.end == following.start == want
    assert tl.narration.beat("question").end_s == want


@pytest.mark.parametrize("offset", [None, 1.5])
def test_a_line_with_no_spans_ends_at_its_speech_or_its_wait(bayes, offset):
    tl = ep.Timeline(narration=bayes)
    sec = tl.line("question")
    padded_end = bayes.beat("question").end_s
    if offset is not None:
        tl.wait_until(padded_end + offset)
    want = padded_end + (offset or 0.0)
    assert tl.narration.beat("question").end_s == want
    following = tl.line("square")
    assert sec.end == following.start == want


def test_a_line_wait_keeps_the_tail_of_a_later_extending_span(bayes):
    tl = ep.Timeline(narration=bayes)
    sec = tl.line("question")
    padded_end = bayes.beat("question").end_s
    tl.wait_until(padded_end + 1.0)
    held = tl.hold(1.0)
    want = held.end + TAIL_S
    assert tl.narration.beat("question").end_s == want
    tl.line("square")
    assert sec.end == want


def test_the_rebuilt_narration_feeds_the_audio_track_with_the_solved_starts(tmp_path):
    small = nr.Narration.load(SMALL, DATA)
    tl = ep.Timeline(narration=small)
    tl.line("open")
    tl.pulse("star")
    tl.move("long", 4.0, at=0.0)
    tl.line("leak")
    tl.pulse("speckle")
    tl.ring("planet")
    solved = tl.narration
    assert solved is not small and small.beat("leak").start_s == 2.1
    for line in SMALL.lines:
        beat = solved.beat(line.key)
        data = np.full(round((beat.speech_end_s - beat.speech_start_s) * 24000), 1000)
        with wave.open(str(tmp_path / f"{SMALL.line_key(line)}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(data.astype("<i2").tobytes())
    path = nr.audio_track(
        solved, tmp_path, tmp_path / "track.wav", rate=None, fade_s=0.0
    )
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert len(pcm) == round(solved.total_s * rate)
    assert pcm[int(1.0 * rate)] == 1000
    assert not pcm[int(2.0 * rate) : int(4.55 * rate)].any()
    assert pcm[int(4.7 * rate)] == 1000


def test_place_is_pure_and_the_clock_never_changes_the_narration_it_was_given():
    small = nr.Narration.load(SMALL, DATA)
    before = [(b.start_s, b.end_s) for b in small.beats]
    one, two = small.place("leak", 3.0), small.place("leak", 3.0)
    assert one == two and one.start_s == 3.0
    tl = ep.Timeline(narration=small)
    tl.line("open")
    tl.pulse("star")
    tl.move("long", 4.0, at=0.0)
    tl.line("leak")
    tl.pulse("speckle")
    tl.ring("planet")
    tl.at_fps(30)
    assert [(b.start_s, b.end_s) for b in small.beats] == before


# ---------------------------------------------------------------------------
# Properties


def _scripts():
    line = st.tuples(
        st.integers(3, 12),  # words
        st.floats(0.0, 1.0),  # where the mark sits, as a fraction of the words
        st.integers(0, 8).map(lambda n: n / 10),  # pad_before_s
        st.integers(0, 8).map(lambda n: n / 10),  # pad_after_s
        st.integers(1, 80).map(lambda n: n / 10),  # seconds of the motion
    )
    return st.lists(line, min_size=1, max_size=5)


@given(_scripts())
@settings(max_examples=100, deadline=None)
def test_property_one_pass_placement_equals_the_two_pass_layout(rows):
    lines, motion = [], []
    for i, (n, where, before, after, seconds) in enumerate(rows):
        words = [f"word{j}x" for j in range(n)]
        at = min(n - 1, int(where * n))
        words[at] = f"{{show:s{i}}}{words[at]}"
        text = " ".join(words)
        lines.append(nr.Line(f"l{i}", text, pad_before_s=before, pad_after_s=after))
        motion.append((f"l{i}", "show", f"s{i}", seconds, SHOW_LEAD_S))
    script = nr.Script(tuple(lines))
    base = nr.Narration.estimated(script)
    tl = ep.Timeline(narration=base, strict=False)
    for (key, _kind, target, seconds, _lead), _row in zip(motion, rows, strict=True):
        tl.line(key)
        tl.enter(target, seconds)
    solved = tl.narration
    old = two_pass(base, motion)
    for beat in solved.beats:
        assert beat.start_s == pytest.approx(old[beat.key][0], abs=1e-9)
        assert beat.end_s == pytest.approx(old[beat.key][1], abs=1e-9)
    assert tl.narration.total_s == pytest.approx(
        tl.at_fps(30).narration.total_s, abs=1e-9
    )


@given(st.integers(0, 1), st.sampled_from(["missing", "key", "text"]))
@settings(max_examples=12, deadline=None)
def test_property_a_cache_with_a_missing_wrong_key_or_wrong_text_file_is_refused(
    which, how
):
    work = Path(tempfile.mkdtemp())
    try:
        shutil.copytree(DATA, work, dirs_exist_ok=True)
        path = work / f"{SMALL.line_key(SMALL.lines[which])}.json"
        record = json.loads(path.read_text())
        if how == "missing":
            path.unlink()
            expect = nr.MissingSynthesisError
        else:
            record[how] = "other" if how == "key" else "some other words"
            path.write_text(json.dumps(record))
            expect = ValueError
        with pytest.raises(expect):
            nr.Narration.load(SMALL, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
