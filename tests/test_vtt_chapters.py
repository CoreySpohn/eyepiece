"""A WebVTT chapters file: one cue per declared beat."""

import pytest

from eyepiece import narration as nr

SCRIPT = nr.Script(
    (
        nr.Line("open", "The coronagraph blocks the starlight.", pad_after_s=0.4),
        nr.Pause("hold", 2.5, "the speckle map holds still"),
        nr.Line("close", "Then the planet appears.", pad_after_s=0.2),
    )
)


def _cues(text):
    blocks = [b for b in text.strip().split("\n\n")][1:]
    return [b.split("\n") for b in blocks]


def test_one_cue_per_beat_on_the_beat_times(tmp_path):
    tl = nr.Narration.estimated(SCRIPT)
    path = nr.write_chapters(tl, tmp_path / "clip.chapters.vtt")
    text = path.read_text()
    assert text.startswith("WEBVTT\n")
    cues = _cues(text)
    assert [c[0] for c in cues] == ["open", "hold", "close"]
    for (key, stamp, title), beat in zip(cues, tl.beats, strict=True):
        a, b = stamp.split(" --> ")
        assert a == nr._mux._stamp(beat.start_s, ".")
        assert b == nr._mux._stamp(beat.end_s, ".")
        assert title == key


def test_titles_override_the_key(tmp_path):
    tl = nr.Narration.estimated(SCRIPT)
    path = nr.write_chapters(
        tl,
        tmp_path / "c.vtt",
        titles={"open": "Blocking the star", "close": "The planet"},
    )
    titles = [c[2] for c in _cues(path.read_text())]
    assert titles == ["Blocking the star", "hold", "The planet"]


def test_a_title_for_no_beat_is_refused(tmp_path):
    tl = nr.Narration.estimated(SCRIPT)
    with pytest.raises(ValueError, match="nope"):
        nr.write_chapters(tl, tmp_path / "c.vtt", titles={"nope": "x"})


def test_write_captions_adds_chapters_only_when_asked(tmp_path):
    tl = nr.Narration.estimated(SCRIPT)
    plain = nr.write_captions(tl, tmp_path / "clip")
    assert len(plain) == 2
    assert not (tmp_path / "clip.chapters.vtt").exists()
    vtt, _srt, chapters = nr.write_captions(tl, tmp_path / "clip", chapters=True)
    assert chapters.name == "clip.chapters.vtt" and chapters.exists()
    assert vtt.read_text() == plain[0].read_text()
