"""Cue sounds mixed into the speech track before the loudness pass."""

import hashlib
import wave

import numpy as np
import pytest

from eyepiece import narration as nr

SCRIPT = nr.Script((nr.Line("a", "One short line about a count.", pad_after_s=3.0),))
RATE = 24000
VOICE_HZ = 300.0
# The track written with cues=None, taken on the code before cues existed.
GOLDEN = "6d20c2cd86fc55975503f3b2ef51f9b13a2e8ab9c6268942b637f33f4e877798"
# The line is spoken from 0 to 2.17 s of a 5.17 s track.
EVENTS = nr.CueEvents(
    count_times=(2.5, 4.0),
    count_values=(4.0, 16.0),
    reference=4.0,
    tick_times=(1.0, 1.25, 1.5),
    sting_s=4.8,
)


def _track(tmp_path, name, **kwargs):
    """The line as a 300 Hz tone at 0.3 of full scale, written by audio_track."""
    tl = nr.Narration.estimated(SCRIPT)
    beat = tl.beat("a")
    n = round((beat.speech_end_s - beat.speech_start_s) * RATE)
    voice = 0.3 * 32767 * np.sin(2 * np.pi * VOICE_HZ * np.arange(n) / RATE)
    with wave.open(
        str(tmp_path / f"{SCRIPT.line_key(SCRIPT.line('a'))}.wav"), "wb"
    ) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(voice.astype("<i2").tobytes())
    out = nr.audio_track(
        tl, tmp_path, tmp_path / name, rate=None, room_tone_db=-50.0, **kwargs
    )
    with wave.open(str(out), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
    return out, data


def _magnitude(x, t, hz, width_s=0.05):
    """Spectral magnitude at `hz` of the window of `width_s` centered at `t`."""
    n = round(width_s * RATE)
    i = round(t * RATE) - n // 2
    window = x[i : i + n] * np.hanning(n)
    return np.abs(np.sum(window * np.exp(-2j * np.pi * hz * np.arange(n) / RATE)))


def test_no_cues_writes_the_same_bytes_as_before(tmp_path):
    out, _ = _track(tmp_path, "plain.wav", cues=None)
    assert hashlib.sha256(out.read_bytes()).hexdigest() == GOLDEN
    again, _ = _track(tmp_path, "default.wav")
    assert again.read_bytes() == out.read_bytes()


def test_cues_keep_the_length_and_the_pitch(tmp_path):
    _, plain = _track(tmp_path, "plain.wav")
    _, cued = _track(tmp_path, "cued.wav", cues=EVENTS)
    assert len(cued) == len(plain)
    assert np.max(np.abs(cued)) < 32767
    # The count is 10 at 3.25 s, so the tone is at 220 x 10 / 4 = 550 Hz.
    n = round(0.05 * RATE)
    i = round(3.25 * RATE) - n // 2
    spectrum = np.abs(np.fft.rfft(cued[i : i + n] * np.hanning(n)))
    peak = np.fft.rfftfreq(n, 1.0 / RATE)[np.argmax(spectrum)]
    assert peak == pytest.approx(550.0, abs=20.0)


def test_the_voice_drops_out_under_the_ticks_and_the_bed_stays(tmp_path):
    _, plain = _track(tmp_path, "plain.wav")
    _, cued = _track(tmp_path, "cued.wav", cues=EVENTS)
    # Between ticks inside the tick window, the voice is gone.
    inside = _magnitude(cued, 1.125, VOICE_HZ)
    assert inside < 0.01 * _magnitude(plain, 1.125, VOICE_HZ)
    # Away from every cue, the voice and the bed are as they were.
    np.testing.assert_array_equal(cued[: round(0.8 * RATE)], plain[: round(0.8 * RATE)])
    bed = slice(round(2.2 * RATE), round(2.45 * RATE))
    assert np.std(cued[bed]) == pytest.approx(np.std(plain[bed]), rel=0.02)


def test_cues_past_the_end_of_the_clip_are_refused(tmp_path):
    late = nr.CueEvents(tick_times=(1.0, 9.0))
    with pytest.raises(ValueError, match="times"):
        _track(tmp_path, "late.wav", cues=late)
    with pytest.raises(TypeError, match="CueEvents"):
        _track(tmp_path, "array.wav", cues=np.zeros(10))
