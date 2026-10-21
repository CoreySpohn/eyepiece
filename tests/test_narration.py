"""Narration: the cue parser, the cache key, the layout, the captions, the mux.

Nothing here needs Kokoro: the timing is a fixture written beside the tests,
and the synthesis runs through a fake speaker.
"""

import itertools
import json
import subprocess
import sys
import warnings
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from eyepiece import _ffmpeg
from eyepiece import narration as nr
from eyepiece.narration import _align, synth

DATA = Path(__file__).parent / "data" / "narration"

SCRIPT = nr.Script(
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


# The cue parser


def test_marks_are_removed_and_sit_on_the_next_word():
    spoken, cues = nr.parse_cues(
        "What leaks is {indicate:speckle}speckle, not {link:planet}light."
    )
    assert spoken == "What leaks is speckle, not light."
    assert [(c.kind, c.target, spoken.split()[c.word]) for c in cues] == [
        ("indicate", "speckle", "speckle,"),
        ("link", "planet", "light."),
    ]


def test_a_mark_may_open_the_text_and_two_marks_may_share_a_word():
    spoken, cues = nr.parse_cues("{caption:a}{indicate:b} Water worlds.")
    assert spoken == "Water worlds."
    assert [c.word for c in cues] == [0, 0]


def test_text_without_marks_has_no_cues():
    assert nr.parse_cues("Plain words.") == ("Plain words.", ())


@pytest.mark.parametrize(
    "text",
    [
        "A stray { brace.",
        "Unclosed {indicate:x word.",
        "No word after it {indicate:x}",
        "Glued to a word up{indicate:x}down.",
        "Missing target {indicate}word.",
    ],
)
def test_malformed_marks_are_refused(text):
    with pytest.raises(ValueError):
        nr.parse_cues(text)


def test_a_malformed_line_names_its_key():
    with pytest.raises(ValueError, match="line 'bad'"):
        nr.Line("bad", "A stray { brace.")


def test_a_line_finds_its_cue_and_refuses_an_unknown_one():
    line = SCRIPT.line("leak")
    assert line.words[line.cue("indicate", "speckle").word] == "speckle,"
    with pytest.raises(KeyError):
        line.cue("indicate", "planet")


def test_duplicate_keys_and_empty_lines_are_refused():
    with pytest.raises(ValueError, match="unique"):
        nr.Script((nr.Line("a", "x."), nr.Line("a", "y.")))
    with pytest.raises(ValueError, match="no words"):
        nr.Line("a", "   ")


# The cache key


def test_the_default_voice_is_british_george():
    assert (SCRIPT.voice, SCRIPT.lang) == ("bm_george", "b")


def test_the_cache_key_is_pinned_and_ignores_cue_marks():
    assert SCRIPT.line_key(SCRIPT.line("open")) == "491f7023363ff02d"
    moved = nr.Line("open", "The {indicate:star}coronagraph blocks the starlight.")
    assert SCRIPT.line_key(moved) == SCRIPT.line_key(SCRIPT.line("open"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("voice", "am_adam"),
        ("speed", 0.9),
        ("lang", "a"),
        ("engine", "piper"),
        ("engine_version", "0.9.5"),
        ("model", "Kokoro-82M-v2"),
    ],
)
def test_the_cache_key_changes_with_everything_that_voices_the_line(field, value):
    other = nr.Script(SCRIPT.lines, **{field: value})
    line = SCRIPT.line("open")
    assert other.line_key(line) != SCRIPT.line_key(line)


def test_the_cache_key_changes_with_the_words():
    line = SCRIPT.line("open")
    assert nr.line_hash("Other words.", "bm_george", 1.0, "b") != SCRIPT.line_key(line)


def test_the_script_lists_the_keys_of_its_lines():
    assert SCRIPT.cache_keys() == ("491f7023363ff02d", "ce02f99ce8f3585a")


def test_script_json_round_trip_preserves_all_fields_and_cache_keys(tmp_path):
    script = nr.Script(
        (
            nr.Line(
                "post",
                "  The {show:eq}{indicate:post}posterior is 0.943.  ",
                pad_before_s=0.2,
                pad_after_s=1.1,
                display="The posterior is Pr(H|E).",
            ),
            nr.Line("end", "A plain line."),
        ),
        voice="af_heart",
        speed=0.9,
        lang="a",
        engine="another_engine",
        engine_version="2.0",
        model="another_model",
    )
    path = tmp_path / "script.json"
    script.to_json(path)
    restored = nr.Script.from_json(str(path))
    assert asdict(restored) == asdict(script)
    assert isinstance(restored.lines, tuple)
    assert restored.cache_keys() == script.cache_keys()


@pytest.mark.parametrize(
    "digits,spoken",
    [
        ("0.943", "zero point nine four three"),
        ("12", "twelve"),
        ("42", "forty two"),
        ("105", "one hundred and five"),
        ("1200", "one thousand two hundred"),
        ("-12", "minus twelve"),
        ("+12", "plus twelve"),
    ],
)
def test_estimate_times_numbers_as_their_spoken_words(digits, spoken):
    numeric = nr.Line("a", f"It is {digits}.")
    expanded = nr.Line("a", f"It is {spoken}.")
    timing = nr.estimate(numeric)
    assert timing.duration_s == pytest.approx(nr.estimate(expanded).duration_s)
    assert tuple(w.text for w in timing.words) == numeric.words
    assert timing.words[-1].end_s == pytest.approx(
        nr.estimate(expanded).words[-1].end_s
    )


# The layout


def test_beats_run_back_to_back_and_last_speech_plus_padding():
    tl = nr.Narration.load(SCRIPT, DATA)
    a, b = tl.beats
    assert a.start_s == 0.0
    assert a.speech_end_s == pytest.approx(1.7)
    assert a.end_s == pytest.approx(1.7 + 0.4)
    assert b.start_s == a.end_s
    assert b.end_s == pytest.approx(a.end_s + 3.5 + 0.5)
    assert tl.total_s == b.end_s


def test_the_narration_replaces_a_hold_with_the_speech():
    short = nr.Narration.load(SCRIPT, DATA)
    padded = nr.Narration.load(SCRIPT, DATA, starts={"open": 1.0})
    assert padded.total_s == pytest.approx(short.total_s + 1.0)
    assert padded.start("open") == 1.0


def test_motion_longer_than_the_speech_stretches_the_beat():
    tl = nr.Narration.load(SCRIPT, DATA, ends={"open": 5.0})
    assert tl.end("open") == pytest.approx(5.0)
    assert tl.start("leak") == pytest.approx(5.0)


def test_a_cue_is_the_start_of_its_word_with_no_lead():
    tl = nr.Narration.load(SCRIPT, DATA)
    word = tl.beat("leak").words[4]
    assert word.text == "speckle,"
    assert tl.cue("leak", "indicate", "speckle") == word.start_s


def test_place_is_pure_and_puts_the_line_where_it_is_told():
    tl = nr.Narration.load(SCRIPT, DATA)
    one = tl.place("leak", 10.0)
    assert one == tl.place("leak", 10.0)
    base = tl.beat("leak")
    shift = 10.0 - base.start_s
    assert one.end_s - one.start_s == pytest.approx(base.end_s - base.start_s)
    assert one.cues[0].at_s == pytest.approx(base.cues[0].at_s + shift)
    held = tl.place("leak", 10.0, end_s=40.0)
    assert held.end_s == 40.0
    assert held.speech_end_s == one.speech_end_s
    assert tl.place("leak", 10.0, end_s=1.0).end_s == one.end_s  # a floor, not a cut


def test_a_line_cannot_start_before_the_line_before_it_ends():
    tl = nr.Narration.load(SCRIPT, DATA)
    with pytest.raises(ValueError, match="before the line before"):
        nr.Narration(SCRIPT, tl.spoken, starts={"leak": 0.5})


def test_cue_times_are_in_reading_order():
    tl = nr.Narration.load(SCRIPT, DATA)
    times = [tl.cue("leak", c.kind, c.target) for c in SCRIPT.line("leak").cues]
    assert times == sorted(times)
    assert times[0] > tl.start("leak")


def test_beat_at_names_the_running_beat():
    tl = nr.Narration.load(SCRIPT, DATA)
    assert tl.beat_at(-1.0) == "open"
    assert tl.beat_at(tl.start("leak")) == "leak"
    assert tl.beat_at(tl.total_s + 3.0) == "leak"


def test_a_missing_line_is_named():
    tl = nr.Narration.load(SCRIPT, DATA)
    with pytest.raises(KeyError, match="nope"):
        tl.beat("nope")
    with pytest.raises(KeyError, match="leak"):
        nr.Narration(SCRIPT, {"open": tl.spoken["open"]})


def test_a_missing_cache_names_the_command_that_fills_it(tmp_path):
    with pytest.raises(nr.MissingSynthesisError, match=r"narration\.synth"):
        nr.Narration.load(SCRIPT, tmp_path)


def _copy_cache(tmp_path, edit=None):
    for line in SCRIPT.lines:
        key = SCRIPT.line_key(line)
        record = json.loads((DATA / f"{key}.json").read_text())
        if edit is not None and line.key == "open":
            edit(record)
        (tmp_path / f"{key}.json").write_text(json.dumps(record))


def test_timing_that_disagrees_with_the_text_is_refused(tmp_path):
    _copy_cache(tmp_path, lambda r: r.update(words=r["words"][:-1]))
    with pytest.raises(ValueError, match="synthesize it again"):
        nr.Narration.load(SCRIPT, tmp_path)


def test_a_file_whose_key_or_text_is_not_the_lines_is_refused(tmp_path):
    _copy_cache(tmp_path, lambda r: r.update(key="0123456789abcdef"))
    with pytest.raises(ValueError, match="holds the timing of"):
        nr.Narration.load(SCRIPT, tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _copy_cache(other, lambda r: r.update(text="Something else."))
    with pytest.raises(ValueError, match="Something else"):
        nr.Narration.load(SCRIPT, other)


def test_starts_and_ends_name_only_known_lines():
    with pytest.raises(KeyError, match="nope"):
        nr.Narration.estimated(SCRIPT, ends={"nope": 1.0})
    with pytest.raises(KeyError, match="nope"):
        nr.Narration.estimated(SCRIPT, starts={"nope": 1.0})


def test_an_estimated_narration_lays_out_without_a_cache():
    tl = nr.Narration.estimated(SCRIPT)
    assert tl.total_s > 3.0
    assert tl.cue("open", "indicate", "star") > 0.0


# Cue vocabulary


def test_cues_lists_every_cue_in_firing_order():
    tl = nr.Narration.load(SCRIPT, DATA)
    cues = tl.cues()
    assert [(c.line, c.kind, c.target) for c in cues] == [
        ("open", "indicate", "star"),
        ("leak", "indicate", "speckle"),
        ("leak", "circumscribe", "planet"),
    ]


def test_check_refuses_a_kind_or_target_the_clip_does_not_handle():
    tl = nr.Narration.load(SCRIPT, DATA)
    tl.check(kinds={"indicate", "circumscribe"}, targets={"star", "speckle", "planet"})
    with pytest.raises(ValueError, match="circumscribe:planet has an unknown kind"):
        tl.check(kinds={"indicate"})
    with pytest.raises(ValueError, match="indicate:star has an unknown target"):
        tl.check(targets={"speckle", "planet"})


def test_check_reports_a_cue_the_clip_never_asked_for():
    tl = nr.Narration.load(SCRIPT, DATA)
    tl.cue("open", "indicate", "star")
    tl.cue("leak", "indicate", "speckle")
    with pytest.raises(ValueError, match="circumscribe:planet is never used"):
        tl.check(used=True)
    tl.cue("leak", "circumscribe", "planet")
    tl.check(used=True)


# Aligning the words to a speech model's tokens


def test_align_joins_split_tokens_and_skips_punctuation():
    tokens = [
        ("The", 0.0, 0.2),
        ("speck", 0.3, 0.5),
        ("le", 0.5, 0.7),
        (",", 0.7, 0.7),
        ("0.24", 0.8, 1.2),
    ]
    assert _align.align(["The", "speckle,", "0.24"], tokens) == [
        (0.0, 0.2),
        (0.3, 0.7),
        (0.8, 1.2),
    ]


def test_align_fills_a_missing_time_from_its_neighbor():
    tokens = [("one", 0.0, 0.4), ("two", None, None)]
    assert _align.align(["one", "two"], tokens) == [(0.0, 0.4), (0.4, 0.4)]


def test_align_refuses_tokens_that_do_not_spell_the_words():
    with pytest.raises(ValueError):
        _align.align(["alpha"], [("beta", 0.0, 1.0)])
    with pytest.raises(ValueError):
        _align.align(["alpha", "beta"], [("alpha", 0.0, 1.0)])


@pytest.mark.parametrize("dash", ["--", "&", "+", "%"])
def test_align_gives_a_word_without_letters_zero_width_after_its_neighbor(dash):
    tokens = [("a", 0.0, 0.2), ("b", 0.4, 0.6)]
    assert _align.align(["a", dash, "b"], tokens) == [
        (0.0, 0.2),
        (0.2, 0.2),
        (0.4, 0.6),
    ]


def test_align_gives_a_leading_word_without_letters_the_next_start():
    tokens = [("b", 0.4, 0.6)]
    assert _align.align(["&", "b"], tokens) == [(0.4, 0.4), (0.4, 0.6)]
    assert _align.align(["&"], []) == [(0.0, 0.0)]


def test_align_keeps_accented_letters():
    word = "Str\u00f6mgren"
    assert _align.align([word], [("Str\u00f6m", 0.0, 0.3), ("gren", 0.3, 0.5)]) == [
        (0.0, 0.5)
    ]
    with pytest.raises(ValueError):
        _align.align([word], [("Str", 0.0, 0.2), ("omgren", 0.2, 0.5)])


# Synthesis, through a fake speaker


def _fake_speak(calls, rate=8000):
    def speak(line):
        calls.append(line.key)
        n = len(line.words)
        audio = np.full(rate * n // 4, 0.1, dtype=np.float32)
        words = tuple(nr.Word(w, i / 4, (i + 1) / 4) for i, w in enumerate(line.words))
        return audio, rate, words

    return speak


def _write_wav(directory, key, audio, rate):
    name = f"{key}.wav"
    data = np.round(audio * 32767).astype("<i2")
    with wave.open(str(Path(directory) / name), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return name


def _synthesize(tmp_path, calls, **kwargs):
    return synth.synthesize(
        SCRIPT, tmp_path, speak=_fake_speak(calls), write_audio=_write_wav, **kwargs
    )


def test_synthesis_cli_loads_json_and_speaks_the_right_lines(tmp_path, monkeypatch):
    path = tmp_path / "script.json"
    path.write_text(
        json.dumps(
            {
                "lines": [
                    {"key": "first", "text": "The {indicate:star}star."},
                    {"key": "last", "text": "The planet.", "display": "A planet."},
                ],
                "voice": "af_heart",
                "lang": "a",
            }
        )
    )
    calls, scripts = [], []

    def engine(script):
        scripts.append(script)
        return _fake_speak(calls)

    monkeypatch.setitem(synth.ENGINES, "kokoro", engine)
    monkeypatch.setattr(synth, "_write_compressed", _write_wav)
    cache = tmp_path / "cache"
    synth.main([str(path), str(cache)])
    assert calls == ["first", "last"]
    assert len(scripts) == 1
    script = scripts[0]
    assert (script.voice, script.lang) == ("af_heart", "a")
    assert script.line("first").cues == (nr.Cue("indicate", "star", 1),)
    assert script.line("last").display == "A planet."
    assert {p.stem for p in cache.glob("*.json")} == set(script.cache_keys())


def test_synthesis_speaks_each_line_once_and_records_the_pair(tmp_path):
    calls = []
    assert _synthesize(tmp_path, calls) == ["open", "leak"]
    assert _synthesize(tmp_path, calls) == []
    assert calls == ["open", "leak"]
    key = SCRIPT.line_key(SCRIPT.line("open"))
    record = json.loads((tmp_path / f"{key}.json").read_text())
    assert record["audio"] == f"{key}.wav"
    assert record["samples"] == 8000 * 5 // 4
    assert record["rate"] == 8000
    assert record["duration_s"] == pytest.approx(1.25)
    assert (record["engine"], record["engine_version"]) == ("kokoro", "0.9.4")
    assert record["text"] == "The coronagraph blocks the starlight."
    assert nr.Narration.load(SCRIPT, tmp_path).total_s > 0.0


def test_json_without_its_audio_is_an_error_unless_respeak_is_given(tmp_path):
    _synthesize(tmp_path, [])
    key = SCRIPT.line_key(SCRIPT.line("open"))
    (tmp_path / f"{key}.wav").unlink()
    calls = []
    with pytest.raises(nr.MissingAudioError, match="respeak"):
        _synthesize(tmp_path, calls)
    assert calls == []
    assert _synthesize(tmp_path, calls, respeak=True) == ["open"]
    assert (tmp_path / f"{key}.wav").exists()


def test_audio_without_json_is_spoken_again(tmp_path):
    _synthesize(tmp_path, [])
    key = SCRIPT.line_key(SCRIPT.line("leak"))
    (tmp_path / f"{key}.json").unlink()
    assert _synthesize(tmp_path, []) == ["leak"]


def test_force_speaks_every_line_again(tmp_path):
    _synthesize(tmp_path, [])
    assert _synthesize(tmp_path, [], force=True) == ["open", "leak"]


def test_an_interrupted_line_leaves_no_json(tmp_path):
    def broken(directory, key, audio, rate):
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        synth.synthesize(SCRIPT, tmp_path, speak=_fake_speak([]), write_audio=broken)
    assert not list(tmp_path.glob("*.json"))


def test_the_json_is_written_last_and_atomically(tmp_path):
    order = []

    def write(directory, key, audio, rate):
        order.append("audio")
        return _write_wav(directory, key, audio, rate)

    real = synth._write_json

    def write_json(path, record):
        assert (tmp_path / record["audio"]).exists()
        order.append("json")
        real(path, record)

    synth_json, synth._write_json = synth._write_json, write_json
    try:
        synth.synthesize(SCRIPT, tmp_path, speak=_fake_speak([]), write_audio=write)
    finally:
        synth._write_json = synth_json
    assert order == ["audio", "json", "audio", "json"]
    assert not list(tmp_path.glob(".*"))


def test_prune_leaves_exactly_the_keys_of_the_script(tmp_path):
    _synthesize(tmp_path, [])
    for stale in ("0123456789abcdef.json", "0123456789abcdef.opus"):
        (tmp_path / stale).write_text("{}")
    (tmp_path / "notes.txt").write_text("keep")
    removed = synth.prune_cache(SCRIPT, tmp_path)
    assert sorted(p.name for p in removed) == [
        "0123456789abcdef.json",
        "0123456789abcdef.opus",
    ]
    stems = {p.stem for p in tmp_path.iterdir() if p.suffix in (".json", ".wav")}
    assert stems == set(SCRIPT.cache_keys())
    assert (tmp_path / "notes.txt").exists()


def test_synthesize_can_prune(tmp_path):
    (tmp_path / "0123456789abcdef.json").write_text("{}")
    _synthesize(tmp_path, [], prune=True)
    assert not (tmp_path / "0123456789abcdef.json").exists()


def test_list_cache_names_the_line_of_each_entry(tmp_path):
    _synthesize(tmp_path, [])
    (tmp_path / "0123456789abcdef.json").write_text('{"text": "Old words."}')
    rows = synth.list_cache(SCRIPT, tmp_path)
    by_key = {key: (line, has_audio) for key, line, _, has_audio in rows}
    assert by_key[SCRIPT.line_key(SCRIPT.line("leak"))] == ("leak", True)
    assert by_key["0123456789abcdef"] == (None, False)


def test_the_engine_is_chosen_from_a_dict_by_name(tmp_path):
    script = nr.Script(SCRIPT.lines, engine="fake")
    made = []

    def factory(received):
        made.append(received)
        return _fake_speak([])

    synth.ENGINES["fake"] = factory
    try:
        synth.synthesize(script, tmp_path, write_audio=_write_wav)
    finally:
        del synth.ENGINES["fake"]
    assert made == [script]


def test_all_cached_lines_never_build_the_engine(tmp_path, monkeypatch):
    _synthesize(tmp_path, [])

    def boom(script):
        raise AssertionError("the engine was built for a cached script")

    monkeypatch.setitem(synth.ENGINES, "kokoro", boom)
    assert synth.synthesize(SCRIPT, tmp_path, write_audio=_write_wav) == []


def test_a_missing_kokoro_names_the_extra_and_sets_the_download_switch(monkeypatch):
    monkeypatch.setitem(sys.modules, "kokoro", None)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    with pytest.raises(ImportError, match=r"eyepiece\[narration\]") as err:
        synth._kokoro(SCRIPT)
    assert "kokoro==0.9.4" in str(err.value)
    assert synth.os.environ["HF_HUB_DISABLE_XET"] == "1"
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    with pytest.raises(ImportError):
        synth._kokoro(SCRIPT)
    assert synth.os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_importing_the_package_loads_neither_matplotlib_nor_kokoro():
    code = (
        "import sys, eyepiece.narration, eyepiece.narration.synth; "
        "heavy = ('matplotlib', 'kokoro', 'soundfile', 'torch'); "
        "bad = [m for m in heavy if m in sys.modules]; "
        "assert not bad, bad"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


# Captions


def test_sentences_split_at_sentence_punctuation_not_commas():
    tl = nr.Narration.load(SCRIPT, DATA)
    texts = [t for _, _, t in nr.sentences(tl)]
    assert texts == [
        "The coronagraph blocks the starlight.",
        "What leaks through is speckle, and the planet must stand out.",
    ]


def test_srt_has_numbered_cues_with_increasing_times(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    path = nr.write_srt(tl, tmp_path / "clip.srt")
    blocks = path.read_text().strip().split("\n\n")
    assert len(blocks) == 2
    first = blocks[0].splitlines()
    assert first[0] == "1"
    assert first[1].startswith("00:00:00,100 --> 00:00:01,")
    assert first[2] == "The coronagraph blocks the starlight."


def _stamps(text, sep):
    out = []
    for line in text.splitlines():
        if "-->" in line:
            a, b = (x.strip().replace(sep, ".") for x in line.split("-->"))
            out.append((float(a.split(":")[-1]), float(b.split(":")[-1])))
    return out


def test_captions_never_overlap_when_sentences_are_close(tmp_path):
    close = nr.Script(
        (nr.Line("a", "One.", pad_after_s=0.05), nr.Line("b", "Two.", pad_after_s=0.0))
    )
    spoken = {
        "a": nr.Spoken("x", 1.0, (nr.Word("One.", 0.0, 1.0),)),
        "b": nr.Spoken("y", 1.0, (nr.Word("Two.", 0.0, 1.0),)),
    }
    tl = nr.Narration(close, spoken)
    for write, sep in ((nr.write_srt, ","), (nr.write_vtt, ".")):
        path = write(tl, tmp_path / f"clip{sep}x")
        (first, second) = _stamps(path.read_text(), sep)
        assert first[0] < first[1] <= second[0]


def test_vtt_is_webvtt_with_dot_timestamps(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    text = nr.write_vtt(tl, tmp_path / "clip.vtt").read_text()
    lines = text.splitlines()
    assert lines[0] == "WEBVTT"
    assert lines[2] == "00:00:00.100 --> 00:00:01.950"
    assert lines[3] == "The coronagraph blocks the starlight."
    assert "," not in lines[2]


# The audio track and the mux


def _write_clip(path, seconds, rate=24000, level=1000):
    data = np.full(int(seconds * rate), level, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def _fill_audio(tl, directory, scale=1.0):
    for line in SCRIPT.lines:
        beat = tl.beat(line.key)
        _write_clip(
            Path(directory) / f"{SCRIPT.line_key(line)}.wav",
            (beat.speech_end_s - beat.speech_start_s) * scale,
        )


def test_the_audio_track_places_each_line_at_its_speech_start(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA, starts={"open": 1.0})
    _fill_audio(tl, tmp_path)
    path = nr.audio_track(
        tl, tmp_path, tmp_path / "track.wav", rate=None, room_tone_db=None, fade_s=0.0
    )
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert len(data) == round(tl.total_s * rate)
    assert not data[: int(0.99 * rate)].any()
    assert data[int(1.05 * rate)] == 1000
    gap = slice(
        round(tl.beat("open").speech_end_s * rate) + 10,
        round(tl.start("leak") * rate) - 10,
    )
    assert not data[gap].any()


def test_the_audio_track_names_a_missing_line(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    with pytest.raises(nr.MissingAudioError, match="open"):
        nr.audio_track(tl, tmp_path, tmp_path / "track.wav")


def test_audio_of_the_wrong_length_is_refused_with_its_line(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    _fill_audio(tl, tmp_path, scale=1.2)
    with pytest.raises(ValueError, match=r"line 'open'.*different syntheses"):
        nr.audio_track(tl, tmp_path, tmp_path / "track.wav")


def test_a_line_past_the_end_of_the_track_raises_instead_of_vanishing(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    _fill_audio(tl, tmp_path, scale=2.0)
    with pytest.raises(ValueError, match="past the end"):
        nr.audio_track(tl, tmp_path, tmp_path / "track.wav", tol_s=100.0)


def _ffmpeg_or_skip():
    try:
        return _ffmpeg.resolve_ffmpeg()
    except RuntimeError as err:
        pytest.skip(str(err))


def test_the_audio_track_reads_compressed_audio(tmp_path):
    ffmpeg = _ffmpeg_or_skip()
    tl = nr.Narration.load(SCRIPT, DATA)
    _fill_audio(tl, tmp_path)
    for wav in list(tmp_path.glob("*.wav")):
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav),
             str(wav.with_suffix(".flac"))],
            check=True,
        )  # fmt: skip
        wav.unlink()
    path = nr.audio_track(
        tl,
        tmp_path,
        tmp_path / "out" / "track.wav",
        rate=None,
        room_tone_db=None,
        fade_s=0.0,
    )
    with wave.open(str(path), "rb") as handle:
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    assert data[int(0.5 * 24000)] == 1000


def test_mux_adds_an_audio_stream_and_copies_the_picture(tmp_path):
    ffmpeg = _ffmpeg_or_skip()
    video = tmp_path / "silent.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=white:s=64x64:r=10:d=2", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    wav = tmp_path / "a.wav"
    _write_clip(wav, 2.0)
    out = nr.mux(video, wav, tmp_path / "out.mp4", loudness=None)
    probe = subprocess.run(
        [ffmpeg, "-i", str(out)], capture_output=True, text=True, check=False
    ).stderr
    assert "Video: h264" in probe
    assert "Audio: aac" in probe


def test_mux_reports_ffmpegs_own_message(tmp_path):
    _ffmpeg_or_skip()
    (tmp_path / "bad.mp4").write_text("not a video")
    wav = tmp_path / "a.wav"
    _write_clip(wav, 1.0)
    with pytest.raises(RuntimeError, match=r"bad\.mp4"):
        nr.mux(tmp_path / "bad.mp4", wav, tmp_path / "out.mp4")


@pytest.mark.parametrize("audio_s", [1.0, 3.0])
@pytest.mark.parametrize("loudness", [None, -16.0])
def test_mux_pads_or_trims_audio_to_the_video_length(tmp_path, audio_s, loudness):
    ffmpeg = _ffmpeg_or_skip()
    video = tmp_path / "v.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         "color=c=white:s=64x64:r=30:d=2.1", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    wav = tmp_path / "a.wav"
    _write_clip(wav, audio_s)
    out = nr.mux(video, wav, tmp_path / "out.mp4", loudness=loudness)
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(out), "-f", "s16le", "-ac", "1",
         "-ar", "48000", "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    assert len(decoded.stdout) / (2 * 48000) == pytest.approx(2.1, abs=1 / 30)


# Audio finishing


def _read_track(path):
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return rate, data.astype(float)


def _dbfs(x):
    return 20.0 * np.log10(np.sqrt(np.mean(x**2)) / 32767.0)


def test_the_track_is_48_khz_with_a_finite_room_tone_below_the_floor(tmp_path):
    _ffmpeg_or_skip()
    tl = nr.Narration.load(SCRIPT, DATA, starts={"open": 1.0})
    _fill_audio(tl, tmp_path)
    path = nr.audio_track(tl, tmp_path, tmp_path / "track.wav", room_tone_db=-60.0)
    rate, data = _read_track(path)
    assert rate == 48000
    assert len(data) == round(tl.total_s * rate)
    gap = data[int(0.1 * rate) : int(0.9 * rate)]
    level = _dbfs(gap)
    assert np.isfinite(level)
    assert level < -55.0
    assert level > -65.0


def test_the_track_has_no_bed_unless_asked(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA, starts={"open": 1.0})
    _fill_audio(tl, tmp_path)
    path = nr.audio_track(
        tl, tmp_path, tmp_path / "t.wav", rate=None, room_tone_db=None
    )
    assert not _read_track(path)[1][:20000].any()


def test_lines_start_and_stop_without_a_click(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA, starts={"open": 1.0})
    _fill_audio(tl, tmp_path)
    kwargs = {"rate": None, "room_tone_db": None}
    bare = nr.audio_track(tl, tmp_path, tmp_path / "a.wav", fade_s=0.0, **kwargs)
    faded = nr.audio_track(tl, tmp_path, tmp_path / "b.wav", fade_s=0.008, **kwargs)
    rate, hard = _read_track(bare)
    _, soft = _read_track(faded)
    assert np.abs(np.diff(hard)).max() >= 1000
    assert np.abs(np.diff(soft)).max() < 20
    first = round(tl.beat("open").speech_start_s * rate)
    assert soft[first] == 0.0
    assert soft[first + int(0.01 * rate)] == 1000


def test_the_track_is_reproducible_and_the_seed_changes_it(tmp_path):
    _ffmpeg_or_skip()
    tl = nr.Narration.load(SCRIPT, DATA, starts={"open": 1.0})
    _fill_audio(tl, tmp_path)
    kw = {"room_tone_db": -60.0}
    one = nr.audio_track(tl, tmp_path, tmp_path / "one.wav", **kw)
    two = nr.audio_track(tl, tmp_path, tmp_path / "two.wav", **kw)
    other = nr.audio_track(tl, tmp_path, tmp_path / "three.wav", seed=1, **kw)
    assert one.read_bytes() == two.read_bytes()
    assert one.read_bytes() != other.read_bytes()


def test_a_negative_fade_is_refused(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    with pytest.raises(ValueError, match="fade_s"):
        nr.audio_track(tl, tmp_path, tmp_path / "t.wav", fade_s=-1.0)


def _tone(path, seconds, level, rate=24000, hz=440.0):
    t = np.arange(int(seconds * rate)) / rate
    data = np.round(level * np.sin(2 * np.pi * hz * t)).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())


def _integrated_lufs(ffmpeg, path):
    done = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128",
         "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    summary = done.stderr.rsplit("Integrated loudness:", 1)[1]
    return float(summary.split("I:")[1].split("LUFS")[0])


def _silent_video(ffmpeg, path, seconds):
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
         f"color=c=white:s=64x64:r=10:d={seconds}", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )  # fmt: skip


@pytest.mark.parametrize("level", [800, 6000])
def test_mux_normalizes_to_the_target_within_one_lu(tmp_path, level):
    ffmpeg = _ffmpeg_or_skip()
    _silent_video(ffmpeg, tmp_path / "v.mp4", 6)
    _tone(tmp_path / "a.wav", 6.0, level)
    out = nr.mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")
    assert abs(_integrated_lufs(ffmpeg, out) - (-16.0)) < 1.0


def test_the_loudness_target_is_a_parameter(tmp_path):
    ffmpeg = _ffmpeg_or_skip()
    _silent_video(ffmpeg, tmp_path / "v.mp4", 6)
    _tone(tmp_path / "a.wav", 6.0, 3000)
    out = nr.mux(
        tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4", loudness=-23.0
    )
    assert abs(_integrated_lufs(ffmpeg, out) - (-23.0)) < 1.0


def test_mux_keeps_silence_silent_instead_of_failing(tmp_path):
    ffmpeg = _ffmpeg_or_skip()
    _silent_video(ffmpeg, tmp_path / "v.mp4", 2)
    _tone(tmp_path / "a.wav", 2.0, 0)
    out = nr.mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")
    assert out.exists()


# Caption limits


def _one_line_narration(text, spans, display=None):
    """A one-line narration whose words last as `spans` says."""
    line = nr.Line("a", text, pad_after_s=0.0, display=display)
    words = tuple(nr.Word(w, a, b) for w, (a, b) in zip(line.words, spans, strict=True))
    return nr.Narration(nr.Script((line,)), {"a": nr.Spoken("x", spans[-1][1], words)})


D51 = "Water worlds with the band cover 0.24, and dry worlds with the band cover 0.07."


def _even(n, total):
    return [(i * total / n, (i + 1) * total / n) for i in range(n)]


def test_caption_at_includes_starts_and_excludes_ends_and_gaps():
    tl = _one_line_narration("First. Second.", [(1.0, 1.5), (3.0, 3.5)])
    cues = nr.captions(tl, min_s=0.0)
    first, second = cues
    assert "caption_at" in nr.__all__
    assert nr.caption_at(cues, first[0]) == first[2]
    assert nr.caption_at(cues, first[0] - 1e-6) == ""
    assert nr.caption_at(cues, first[1] - 1e-6) == first[2]
    assert nr.caption_at(cues, first[1]) == ""
    assert nr.caption_at(cues, (first[1] + second[0]) / 2) == ""
    assert nr.caption_at(cues, second[0] - 1e-6) == ""
    assert nr.caption_at(cues, second[0]) == second[2]
    assert nr.caption_at(cues, second[1]) == ""
    assert nr.caption_at(cues, 0.0) == ""
    assert nr.caption_at(cues, second[1] + 1.0) == ""
    assert nr.caption_at([], 0.0) == ""


@pytest.mark.parametrize("max_lines", [1, 2])
def test_caption_breaks_keep_operators_with_their_operands(max_lines):
    text = "Compute a / b - 1; cap +/-1% applies."
    tl = _one_line_narration(text, _even(len(text.split()), 6.0))
    cues = nr.captions(tl, max_chars=14, max_lines=max_lines)
    assert " ".join(t.replace("\n", " ") for _, _, t in cues) == text
    for _, _, cue in cues:
        for line in cue.splitlines():
            assert not line.startswith(("/", "-", "+/-"))
            assert any(c.isalnum() for c in line.split()[0])
            assert any(c.isalnum() for c in line.split()[-1])


@pytest.mark.parametrize("max_lines", [1, 2])
def test_caption_breaks_keep_parenthetical_text_together(max_lines):
    text = "The measured curve (secant d^6.1; shown at 4.4) applies."
    tl = _one_line_narration(text, _even(len(text.split()), 6.0))
    cues = nr.captions(tl, max_chars=32, max_lines=max_lines)
    assert " ".join(t.replace("\n", " ") for _, _, t in cues) == text
    for _, _, cue in cues:
        assert cue.count("(") == cue.count(")")
        for line in cue.splitlines():
            assert line.count("(") == line.count(")")


def test_the_79_character_line_becomes_two_lines_within_the_limits():
    words = D51.split()
    tl = _one_line_narration(D51, _even(len(words), 5.0))
    (cue,) = nr.captions(tl)
    start, end, text = cue
    assert len(D51) == 79
    lines = text.split("\n")
    assert lines == [
        "Water worlds with the band cover 0.24,",
        "and dry worlds with the band cover 0.07.",
    ]
    assert all(len(x) <= 42 for x in lines)
    assert start == 0.0
    assert 5.0 <= end <= 5.0 + 0.15 + 1e-9


def test_captions_without_a_clause_break_split_evenly():
    text = "one two three four five six seven eight nine ten eleven twelve " * 2
    tl = _one_line_narration(text.strip(), _even(24, 6.0))
    cues = nr.captions(tl)
    assert len(cues) >= 2
    for _, _, t in cues:
        assert len(t.split("\n")) <= 2
        assert all(len(x) <= 42 for x in t.split("\n"))
    joined = " ".join(t.replace("\n", " ") for _, _, t in cues)
    assert joined == text.strip()


def test_a_long_sentence_splits_at_a_clause_boundary_with_timing_kept():
    words = (
        "Light from the star reaches the mask, and the mask removes most of it, "
        "so the planet stays."
    ).split()
    tl = _one_line_narration(" ".join(words), _even(len(words), 12.0))
    cues = nr.captions(tl)
    assert len(cues) >= 2
    assert cues[0][0] == 0.0
    assert all(b - a <= 7.0 + 0.15 + 1e-9 for a, b, _ in cues)
    assert cues[0][2].replace("\n", " ").endswith(",")
    for (_, end, _), (start, _, _) in itertools.pairwise(cues):
        assert end <= start


def test_a_cue_lasts_at_least_the_minimum_and_never_into_the_next():
    tl = _one_line_narration("Yes. No.", [(0.0, 0.2), (0.3, 0.5)])
    first, second = nr.captions(tl, min_s=1.0)
    assert first[1] == second[0]
    assert second[1] - second[0] >= 1.0


def test_reading_speed_holds_a_fast_cue_longer():
    words = ["alpha"] * 6
    tl = _one_line_narration(" ".join(words) + ".", _even(6, 1.0))
    ((a, b, text),) = nr.captions(tl, min_s=0.0, max_cps=10.0)
    assert b - a >= len(text) / 10.0 - 1e-9


def test_caption_limits_are_validated():
    tl = _one_line_narration("Yes.", [(0.0, 0.5)])
    with pytest.raises(ValueError, match="max_lines"):
        nr.captions(tl, max_lines=3)
    with pytest.raises(ValueError, match="positive"):
        nr.captions(tl, max_cps=0.0)


def test_srt_and_vtt_carry_the_wrapped_cue(tmp_path):
    words = D51.split()
    tl = _one_line_narration(D51, _even(len(words), 5.0))
    vtt, srt = nr.write_captions(tl, tmp_path / "clip")
    assert vtt.suffix == ".vtt" and srt.suffix == ".srt"
    assert "0.24,\nand dry worlds" in vtt.read_text()
    assert "0.24,\nand dry worlds" in srt.read_text()


def test_vtt_line_setting_is_written_on_every_cue(tmp_path):
    tl = nr.Narration.load(SCRIPT, DATA)
    text = nr.write_vtt(tl, tmp_path / "c.vtt", line=62.5).read_text()
    stamps = [x for x in text.splitlines() if "-->" in x]
    assert len(stamps) == 2
    assert all(x.endswith(" line:62.5%") for x in stamps)
    plain = nr.write_vtt(tl, tmp_path / "d.vtt").read_text()
    assert "line:" not in plain


# Spoken versus displayed text


def test_a_caption_shows_the_display_text_and_the_times_do_not_move():
    spoken = "The posterior is P of H given E."
    spans = _even(8, 4.0)
    plain = _one_line_narration(spoken, spans)
    shown = _one_line_narration(spoken, spans, display="The posterior is Pr(H|E).")
    (a0, b0, t0), (a1, b1, t1) = nr.captions(plain)[0], nr.captions(shown)[0]
    assert t0 == "The posterior is P of H given E."
    assert t1 == "The posterior is Pr(H|E)."
    assert (a0, b0) == (a1, b1)


def test_display_text_leaves_the_cue_marks_and_cache_key_alone():
    marked = "The {indicate:post}posterior is P of H given E."
    a = nr.Line("a", marked)
    b = nr.Line("a", marked, display="The posterior is Pr(H|E).")
    assert b.cues == a.cues
    assert b.spoken == a.spoken
    script_a, script_b = nr.Script((a,)), nr.Script((b,))
    assert script_a.cache_keys() == script_b.cache_keys()
    spoken = nr.estimate(a)
    tl_a = nr.Narration(script_a, {"a": spoken})
    tl_b = nr.Narration(script_b, {"a": spoken})
    assert tl_a.cue("a", "indicate", "post") == tl_b.cue("a", "indicate", "post")
    assert nr.sentences(tl_a) == nr.sentences(tl_b)


def test_display_text_with_the_same_sentence_count_keeps_sentence_spans():
    line = nr.Line("a", "One two. Three four.", pad_after_s=0.0, display="1 2. 3 4.")
    words = tuple(nr.Word(w, i * 1.0, i * 1.0 + 0.9) for i, w in enumerate(line.words))
    tl = nr.Narration(nr.Script((line,)), {"a": nr.Spoken("x", 4.0, words)})
    cues = nr.captions(tl, min_s=0.0)
    assert [c[2] for c in cues] == ["1 2.", "3 4."]
    assert cues[0][0] == 0.0 and cues[1][0] == 2.0


def test_captions_warn_once_per_mismatched_line_and_never_for_matched_lines():
    script = nr.Script(
        (
            nr.Line("mismatch", "One. Two.", display="One and two together."),
            nr.Line("matched", "Three. Four.", display="3. 4."),
            nr.Line("plain", "Five. Six."),
            nr.Line("also_bad", "Seven and eight.", display="Seven. Eight."),
        )
    )
    narration = nr.Narration.estimated(script)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cues = nr.captions(narration, max_chars=8, max_lines=1)
    assert len(cues) > len(script.lines)
    assert len(caught) == 2
    assert all(w.category is UserWarning for w in caught)
    assert "mismatch" in str(caught[0].message)
    assert "also_bad" in str(caught[1].message)
    assert all("sentence" in str(w.message) for w in caught)


def test_display_text_may_not_be_empty_or_carry_marks():
    with pytest.raises(ValueError, match="display"):
        nr.Line("a", "Hello.", display="  ")
    with pytest.raises(ValueError, match="cue marks"):
        nr.Line("a", "Hello.", display="{indicate:x}Hello.")


def _gap_clip(tmp_path, level):
    """A muxed clip: tone, a 2 s gap of digital silence, tone."""
    ffmpeg = _ffmpeg_or_skip()
    rate = 24000
    t = np.arange(2 * rate) / rate
    tone = np.round(level * np.sin(2 * np.pi * 440.0 * t))
    data = np.concatenate([tone, np.zeros(2 * rate), tone]).astype("<i2")
    with wave.open(str(tmp_path / "a.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    _silent_video(ffmpeg, tmp_path / "v.mp4", 6)
    return ffmpeg


def _gap_rms_db(ffmpeg, path):
    done = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-ss", "2.3", "-t", "1.4", "-i", str(path),
         "-af", "astats", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    overall = done.stderr.split("Overall")[1]
    return float(overall.split("RMS level dB:")[1].split()[0])


@pytest.mark.parametrize("level", [300, 9000])
def test_the_muxed_gap_floor_is_the_requested_level_whatever_the_gain(tmp_path, level):
    ffmpeg = _gap_clip(tmp_path, level)
    out = nr.mux(tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "o.mp4")
    assert abs(_gap_rms_db(ffmpeg, out) - (-60.0)) < 2.0
    assert abs(_integrated_lufs(ffmpeg, out) - (-16.0)) < 1.0


def test_the_muxed_bed_level_and_seed_are_parameters(tmp_path):
    ffmpeg = _gap_clip(tmp_path, 3000)
    a = nr.mux(
        tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "a.mp4", room_tone_db=-50.0
    )
    b = nr.mux(
        tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "b.mp4", room_tone_db=-50.0
    )
    c = nr.mux(
        tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "c.mp4", room_tone_db=None
    )
    assert abs(_gap_rms_db(ffmpeg, a) - (-50.0)) < 2.0
    assert a.read_bytes() == b.read_bytes()
    assert _gap_rms_db(ffmpeg, c) < -90.0


@pytest.mark.parametrize("cached", [False, True])
def test_synthesis_refuses_faults_before_building_an_engine(
    tmp_path, monkeypatch, cached
):
    script = nr.Script(
        (nr.Line("clean", "A clean line."), nr.Line("bad", "It is 1e-15."))
    )
    if cached:
        key = script.line_key(script.line("bad"))
        (tmp_path / f"{key}.json").write_text(json.dumps({"audio": f"{key}.wav"}))
        (tmp_path / f"{key}.wav").write_bytes(b"cached audio")
    made = []

    def engine(received):
        made.append(received)
        raise AssertionError("the engine must not be built for faulty speech")

    monkeypatch.setitem(synth.ENGINES, "kokoro", engine)
    with pytest.raises(getattr(nr, "SpeechTextError", ValueError), match=r"bad.*1e-15"):
        synth.synthesize(script, tmp_path, write_audio=_write_wav)
    assert made == []
    assert not (tmp_path / f"{script.line_key(script.line('clean'))}.json").exists()


def test_synthesis_accepts_faults_in_display_text(tmp_path):
    script = nr.Script(
        (nr.Line("contrast", "One times ten to the minus 15.", display="1e-15"),)
    )
    nr.check_script_speech(script)
    calls = []
    assert synth.synthesize(
        script, tmp_path, speak=_fake_speak(calls), write_audio=_write_wav
    ) == ["contrast"]
    assert calls == ["contrast"]
    key = script.cache_keys()[0]
    assert (
        json.loads((tmp_path / f"{key}.json").read_text())["text"]
        == script.lines[0].spoken
    )


def test_synthesis_checks_speech_with_an_explicit_speaker(tmp_path):
    script = nr.Script((nr.Line("bad", "It is 1e-15."),))
    calls = []
    with pytest.raises(getattr(nr, "SpeechTextError", ValueError), match=r"bad.*1e-15"):
        synth.synthesize(
            script, tmp_path, speak=_fake_speak(calls), write_audio=_write_wav
        )
    assert calls == []


def test_synthesis_cli_reports_bad_speech_without_a_traceback(tmp_path):
    script_path = tmp_path / "script.py"
    script_path.write_text(
        "from eyepiece import narration as nr\n"
        "SCRIPT = nr.Script((nr.Line('contrast', 'It is 1e-15.'),))\n"
    )
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "eyepiece.narration.synth",
            str(script_path),
            str(tmp_path / "cache"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode != 0
    assert "contrast" in done.stderr and "1e-15" in done.stderr
    assert "Traceback" not in done.stderr


def test_estimate_cli_rejects_speech_faults_before_printing_or_synthesis(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "script.py"
    path.write_text(
        "from eyepiece import narration as nr\n"
        "SCRIPT = nr.Script((nr.Line('clean', 'A clean line.'), "
        "nr.Line('bad', 'It is 1e-15.')))\n"
    )
    script = synth._load_script(path)
    with pytest.raises(nr.SpeechTextError) as expected:
        nr.check_script_speech(script)
    monkeypatch.setitem(sys.modules, "kokoro", None)
    with pytest.raises(SystemExit) as stopped:
        synth.main([str(path), "--estimate"])
    assert stopped.value.code == 2
    captured = capsys.readouterr()
    assert str(expected.value) in captured.err
    assert captured.out == ""
