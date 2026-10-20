"""A line spoken one sentence at a time, joined with a declared gap."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from eyepiece import narration as nr
from eyepiece.narration import synth
from eyepiece.narration._script import line_hash, split_sentences

RATE = 8000
TEXT = "Sixteen times {indicate:six}6, 96. Why 6? Because of the rate."


def _spans(text):
    words = text.split()
    return [" ".join(words[a:b]) for a, b in split_sentences(words)]


def test_sentences_split_at_terminal_punctuation_outside_numbers_and_abbreviations():
    assert _spans("16 times 6, 96. Why 6?") == ["16 times 6, 96.", "Why 6?"]
    assert _spans("It is 3.5 mm wide. Done!") == ["It is 3.5 mm wide.", "Done!"]
    assert _spans("Dr. Smith saw e.g. a star. Then left.") == [
        "Dr. Smith saw e.g. a star.",
        "Then left.",
    ]
    assert _spans("J. K. Rowling wrote.") == ["J. K. Rowling wrote."]
    assert _spans('He said "go." Then stopped') == ['He said "go."', "Then stopped"]
    assert _spans("no terminal punctuation") == ["no terminal punctuation"]


def test_the_default_changes_nothing():
    plain = nr.Line("a", "One. Two.")
    script = nr.Script((plain,))
    assert script.line_key(plain) == line_hash(
        "One. Two.",
        script.voice,
        script.speed,
        script.lang,
        engine=script.engine,
        engine_version=script.engine_version,
        model=script.model,
    )
    assert script.cache_keys() == (script.line_key(plain),)
    assert script.sentence_keys(plain) == ()


def test_the_gap_enters_the_line_key_and_each_sentence_has_its_own():
    plain = nr.Line("a", "One. Two.")
    gapped = nr.Line("a", "One. Two.", sentence_gap_s=0.4)
    other = nr.Line("a", "One. Two.", sentence_gap_s=0.8)
    script = nr.Script((gapped,))
    assert script.line_key(gapped) != nr.Script((plain,)).line_key(plain)
    assert script.line_key(gapped) != nr.Script((other,)).line_key(other)
    keys = script.sentence_keys(gapped)
    assert len(keys) == 2 and len(set(keys)) == 2
    assert script.cache_keys() == (script.line_key(gapped), *keys)
    lone = nr.Script((nr.Line("b", "One."),))
    assert keys[0] == lone.line_key(lone.line("b"))


def test_a_gap_survives_a_json_round_trip(tmp_path):
    script = nr.Script(
        (nr.Line("a", "One. Two.", sentence_gap_s=0.4), nr.Line("b", "Three."))
    )
    path = script.to_json(tmp_path / "s.json")
    record = json.loads(path.read_text())
    assert record["lines"][0]["sentence_gap_s"] == 0.4
    assert "sentence_gap_s" not in record["lines"][1]
    assert nr.Script.from_json(path) == script


def test_a_negative_gap_is_refused():
    with pytest.raises(ValueError, match="sentence_gap_s"):
        nr.Line("a", "One. Two.", sentence_gap_s=-0.1)


def _speak(calls):
    def speak(line):
        calls.append(line.spoken)
        n = len(line.words)
        audio = np.full(RATE * n // 4, 0.1, dtype=np.float32)
        words = tuple(nr.Word(w, i / 4, (i + 1) / 4) for i, w in enumerate(line.words))
        return audio, RATE, words

    return speak


def _write(directory, key, audio, rate):
    name = f"{key}.wav"
    data = np.round(np.asarray(audio) * 32767).astype("<i2")
    with wave.open(str(Path(directory) / name), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return name


def _read(directory, name):
    with wave.open(str(Path(directory) / name), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        return data.astype(np.float32) / 32767, handle.getframerate()


def _run(script, cache, calls, **kwargs):
    return synth.synthesize(
        script,
        cache,
        speak=_speak(calls),
        write_audio=_write,
        read_audio=_read,
        **kwargs,
    )


def test_a_gapped_line_is_spoken_a_sentence_at_a_time_and_joined(tmp_path):
    script = nr.Script((nr.Line("a", TEXT, sentence_gap_s=0.5),))
    calls = []
    assert _run(script, tmp_path, calls) == ["a"]
    assert calls == ["Sixteen times 6, 96.", "Why 6?", "Because of the rate."]
    line = script.line("a")
    record = json.loads((tmp_path / f"{script.line_key(line)}.json").read_text())
    # Sentences last 1.0, 0.5 and 1.0 s; 0.5 s of silence joins them.
    assert record["duration_s"] == pytest.approx(1.0 + 0.5 + 0.5 + 0.5 + 1.0)
    assert record["samples"] == round(RATE * 3.5)
    starts = [w[1] for w in record["words"]]
    assert starts[:3] == pytest.approx([0.0, 0.25, 0.5])
    assert starts[4] == pytest.approx(1.0 + 0.5)  # "Why"
    assert starts[6] == pytest.approx(1.0 + 0.5 + 0.5 + 0.5)  # "Because"
    assert record["sentences"] == list(script.sentence_keys(line))
    audio, _ = _read(tmp_path, record["audio"])
    gap = audio[round(1.0 * RATE) : round(1.5 * RATE)]
    assert not gap.any()


def test_cue_marks_stay_bound_to_their_words_on_the_joined_clock(tmp_path):
    script = nr.Script((nr.Line("a", TEXT, sentence_gap_s=0.5),))
    _run(script, tmp_path, [])
    tl = nr.Narration.load(script, tmp_path)
    assert script.line("a").cue("indicate", "six").word == 2
    assert tl.cue("a", "indicate", "six") == pytest.approx(tl.start("a") + 0.5)


def test_sentences_are_cached_and_reused_across_lines_and_gaps(tmp_path):
    first = nr.Script((nr.Line("a", "Why 6? Because.", sentence_gap_s=0.5),))
    calls = []
    _run(first, tmp_path, calls)
    assert calls == ["Why 6?", "Because."]
    # A longer gap re-joins the cached sentences without speaking.
    again = nr.Script((nr.Line("a", "Why 6? Because.", sentence_gap_s=0.9),))
    assert _run(again, tmp_path, calls) == ["a"]
    assert calls == ["Why 6?", "Because."]
    # Another line holding one of the sentences speaks only the new one.
    other = nr.Script((nr.Line("b", "Why 6? Never.", sentence_gap_s=0.5),))
    _run(other, tmp_path, calls)
    assert calls[2:] == ["Never."]
    # Pruning keeps the sentences of a script that uses them.
    assert synth.prune_cache(other, tmp_path) != []
    assert all((tmp_path / f"{k}.json").exists() for k in other.cache_keys())


def test_a_line_without_a_gap_is_one_call(tmp_path):
    script = nr.Script((nr.Line("a", "Why 6? Because."),))
    calls = []
    _run(script, tmp_path, calls)
    assert calls == ["Why 6? Because."]
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_the_estimate_holds_the_gap_after_each_sentence():
    plain = nr.estimate(nr.Line("a", "One two. Three four."))
    gapped = nr.estimate(nr.Line("a", "One two. Three four.", sentence_gap_s=1.0))
    # The gap replaces the usual quarter second after a sentence.
    assert gapped.words[2].start_s - plain.words[2].start_s == pytest.approx(0.75)
    assert gapped.duration_s > plain.duration_s
