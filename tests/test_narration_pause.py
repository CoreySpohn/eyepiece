"""A pause is a first-class silent beat of a narration script."""

import wave
from pathlib import Path

import numpy as np
import pytest

from eyepiece import narration as nr

LINES = (
    nr.Line(
        "open", "The coronagraph blocks the {indicate:star}starlight.", pad_after_s=0.4
    ),
    nr.Pause("beat", 2.5, "the speckle map holds still"),
    nr.Line("close", "Then the planet appears.", pad_after_s=0.2),
)
SCRIPT = nr.Script(LINES)


def test_a_pause_has_a_key_a_duration_and_a_description():
    pause = SCRIPT.line("beat")
    assert (pause.key, pause.duration_s, pause.description) == (
        "beat",
        2.5,
        "the speckle map holds still",
    )
    assert pause.spoken == "" and pause.words == () and pause.cues == ()


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan")])
def test_a_pause_needs_a_positive_duration(duration):
    with pytest.raises(ValueError, match="duration"):
        nr.Pause("p", duration, "on screen")


def test_a_pause_needs_a_description_and_shares_the_key_namespace():
    with pytest.raises(ValueError, match="description"):
        nr.Pause("p", 1.0, "  ")
    with pytest.raises(ValueError, match="unique"):
        nr.Script((nr.Line("a", "Words."), nr.Pause("a", 1.0, "still")))


def test_a_pause_enters_the_timing_like_a_line_with_no_audio():
    tl = nr.Narration.estimated(SCRIPT)
    open_, beat, close = tl.beats
    assert beat.start_s == open_.end_s
    assert beat.end_s == pytest.approx(beat.start_s + 2.5)
    assert beat.speech_start_s == beat.speech_end_s == beat.start_s
    assert beat.words == () and beat.cues == ()
    assert close.start_s == beat.end_s


def test_a_pause_can_be_held_longer_by_the_motion_inside_it():
    tl = nr.Narration.estimated(SCRIPT, ends={"beat": 9.0})
    assert tl.end("beat") == pytest.approx(9.0)


def test_a_pause_has_no_cache_key_and_is_never_synthesized(tmp_path):
    assert len(SCRIPT.cache_keys()) == 2
    with pytest.raises(ValueError, match="no speech"):
        SCRIPT.line_key(SCRIPT.line("beat"))
    spoken = []

    def speak(line):
        spoken.append(line.key)
        return (
            np.zeros(800, dtype="<i2"),
            8000,
            tuple(nr.Word(w, 0.0, 0.1) for w in line.words),
        )

    from eyepiece.narration import synth

    done = synth.synthesize(
        SCRIPT,
        tmp_path,
        speak=speak,
        write_audio=lambda d, k, a, r: (
            (Path(d) / f"{k}.wav").write_bytes(b"x") and f"{k}.wav"
        ),
    )
    assert done == spoken == ["open", "close"]
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [
        ".json",
        ".json",
        ".wav",
        ".wav",
    ]
    loaded = nr.Narration.load(SCRIPT, tmp_path)
    assert loaded.beat("beat").end_s - loaded.beat("beat").start_s == pytest.approx(2.5)


def test_a_pause_is_nothing_in_the_captions_and_a_silent_row_in_the_audit():
    tl = nr.Narration.estimated(SCRIPT)
    assert [t for _, _, t in nr.sentences(tl)] == [
        "The coronagraph blocks the starlight.",
        "Then the planet appears.",
    ]
    assert all("speckle" not in c[2] for c in nr.captions(tl))
    rows = tl.audit()
    assert [r["kind"] for r in rows] == ["speech", "silent", "speech"]
    silent = rows[1]
    assert silent["key"] == "beat"
    assert silent["start_s"] == tl.start("beat") and silent["end_s"] == tl.end("beat")
    assert silent["text"] == "the speckle map holds still"


def test_the_audio_track_leaves_a_pause_silent(tmp_path):
    tl = nr.Narration.estimated(SCRIPT)
    for line in SCRIPT.lines:
        if isinstance(line, nr.Pause):
            continue
        beat = tl.beat(line.key)
        n = round((beat.speech_end_s - beat.speech_start_s) * 24000)
        with wave.open(str(tmp_path / f"{SCRIPT.line_key(line)}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(np.full(n, 1000, dtype="<i2").tobytes())
    out = nr.audio_track(tl, tmp_path, tmp_path / "t.wav", rate=None, fade_s=0.0)
    with wave.open(str(out), "rb") as handle:
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    gap = slice(round(tl.start("beat") * rate) + 10, round(tl.end("beat") * rate) - 10)
    assert not data[gap].any()
    assert data[round(tl.beat("close").speech_start_s * rate) + 10] == 1000


def test_a_script_with_a_pause_round_trips_through_json(tmp_path):
    path = tmp_path / "script.json"
    SCRIPT.to_json(path)
    assert nr.Script.from_json(path) == SCRIPT
    assert nr.Script.from_json(path).cache_keys() == SCRIPT.cache_keys()


def test_the_speech_checks_pass_over_a_pause():
    nr.check_script_speech(SCRIPT)
