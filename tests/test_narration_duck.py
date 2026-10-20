"""Duck windows drop the room-tone bed to its floor; default output is unchanged."""

import wave

import numpy as np
import pytest

from eyepiece import narration as nr

SCRIPT = nr.Script((nr.Line("a", "One short line.", pad_after_s=3.0),))


def _track(tmp_path, name, **kwargs):
    tl = nr.Narration.estimated(SCRIPT)
    beat = tl.beat("a")
    n = round((beat.speech_end_s - beat.speech_start_s) * 24000)
    with wave.open(
        str(tmp_path / f"{SCRIPT.line_key(SCRIPT.line('a'))}.wav"), "wb"
    ) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(np.zeros(n, dtype="<i2").tobytes())
    out = nr.audio_track(
        tl, tmp_path, tmp_path / name, rate=None, room_tone_db=-40.0, **kwargs
    )
    with wave.open(str(out), "rb") as w:
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float)
    return rate, data, tl


def _rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)) / 32767.0)


def test_no_windows_leaves_the_track_byte_identical(tmp_path):
    _track(tmp_path, "plain.wav")
    _track(tmp_path, "none.wav", duck=None)
    _track(tmp_path, "empty.wav", duck=[])
    plain = (tmp_path / "plain.wav").read_bytes()
    assert (tmp_path / "none.wav").read_bytes() == plain
    assert (tmp_path / "empty.wav").read_bytes() == plain


def test_a_window_drops_the_bed_to_the_floor_and_leaves_the_rest(tmp_path):
    rate, plain, _ = _track(tmp_path, "plain.wav")
    _, ducked, _ = _track(
        tmp_path, "ducked.wav", duck=[(1.0, 2.0)], duck_floor_db=-80.0
    )
    inside = slice(round(1.2 * rate), round(1.8 * rate))
    assert _rms_db(plain[inside]) == pytest.approx(-40.0, abs=0.5)
    assert _rms_db(ducked[inside]) < -70.0
    outside = np.r_[0 : round(0.9 * rate), round(2.1 * rate) : len(plain)]
    np.testing.assert_array_equal(ducked[outside], plain[outside])


def test_the_window_edges_are_faded_not_cut():
    from eyepiece.narration._mux import _duck_gain

    rate = 1000
    gain = _duck_gain(4000, rate, [(1.0, 3.0)], 0.01, 0.1)
    assert gain[:1000].min() == 1.0 and gain[3000:].min() == 1.0
    assert gain[1500:2500].max() == pytest.approx(0.01)
    down, up = gain[1000:1100], gain[2900:3000]
    assert np.all(np.diff(down) <= 0) and np.all(np.diff(up) >= 0)
    assert np.max(np.abs(np.diff(gain))) < 0.05  # no step: the fade is 100 samples


@pytest.mark.parametrize(
    "windows", [[(2.0, 1.0)], [(-1.0, 1.0)], [(0.0, 2.0), (1.0, 3.0)], [(1.0, 1.0)]]
)
def test_bad_windows_are_refused(tmp_path, windows):
    with pytest.raises(ValueError, match="duck"):
        _track(tmp_path, "bad.wav", duck=windows)
