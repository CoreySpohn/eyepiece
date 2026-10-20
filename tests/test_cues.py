"""Cue synthesizers: a count tone, pixel-read ticks and a click sting."""

import numpy as np
import pytest

from eyepiece import narration as nr
from eyepiece.narration import _cues

RATE = 48000
WINDOW_S = 0.05
BIN_HZ = 1.0 / WINDOW_S  # one FFT bin of a 50 ms window


def peak_hz(x, t, rate=RATE):
    """The FFT peak of the 50 ms window centered at `t`."""
    n = round(WINDOW_S * rate)
    i = round(t * rate) - n // 2
    window = x[i : i + n] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(window))
    return np.fft.rfftfreq(n, 1.0 / rate)[np.argmax(spectrum)]


def onsets(x):
    """Sample index of the first nonzero sample of each run of nonzero samples."""
    nonzero = x != 0.0
    return np.flatnonzero(nonzero & ~np.r_[False, nonzero[:-1]])


def db(x):
    return 20.0 * np.log10(x)


def test_pitch_follows_the_count():
    times, values = [0.0, 1.0, 2.0, 3.0], [4.0, 6.0, 12.0, 16.0]
    x = nr.count_tone(times, values, reference=4.0, duration_s=3.0, rate=RATE)
    for t in (0.3, 0.9, 1.5, 2.1, 2.5):
        expected = 220.0 * np.interp(t, times, values) / 4.0
        assert peak_hz(x, t) == pytest.approx(expected, abs=BIN_HZ)


def test_four_to_sixteen_is_two_octaves():
    x = nr.count_tone(
        [0.0, 0.6, 1.4, 2.0], [4.0, 4.0, 16.0, 16.0], reference=4.0, duration_s=2.0
    )
    low, high = peak_hz(x, 0.3), peak_hz(x, 1.7)
    assert low == pytest.approx(220.0, abs=BIN_HZ)
    assert high == pytest.approx(880.0, abs=BIN_HZ)
    assert high / low == pytest.approx(4.0, abs=4.0 * 2 * BIN_HZ / 220.0)


def test_the_tone_has_no_phase_jump():
    x = nr.count_tone([0.0, 1.0], [4.0, 16.0], reference=4.0, duration_s=1.0)
    # A continuous sine at most 880 Hz moves at most 2 pi f / rate per sample.
    step = 2 * np.pi * 880.0 / RATE * np.max(np.abs(x))
    assert np.max(np.abs(np.diff(x))) <= step * 1.01


def test_tone_level_attack_and_decay():
    x = nr.count_tone([0.0, 2.0], [4.0, 4.0], reference=4.0, duration_s=2.0)
    middle = x[round(0.5 * RATE) : round(1.5 * RATE)]
    assert db(np.sqrt(np.mean(middle**2))) == pytest.approx(-30.0, abs=1.0)
    full = np.max(np.abs(middle))
    # Attack 40 ms: rising over the first 40 ms, at full level after it.
    assert np.max(np.abs(x[: round(0.01 * RATE)])) < 0.5 * full
    assert np.max(np.abs(x[round(0.04 * RATE) : round(0.06 * RATE)])) > 0.99 * full
    # Decay 250 ms at the end: still sounding 250 ms before it, silent at it.
    assert np.max(np.abs(x[round(1.70 * RATE) : round(1.74 * RATE)])) > 0.9 * full
    assert np.max(np.abs(x[round(1.99 * RATE) :])) < 0.05 * full


def test_octave_bell_at_the_stated_count():
    plain = nr.count_tone([0.0, 2.0], [4.0, 16.0], reference=4.0, duration_s=2.0)
    belled = nr.count_tone(
        [0.0, 2.0], [4.0, 16.0], reference=4.0, duration_s=2.0, bell_at=8.0
    )
    extra = np.flatnonzero(belled - plain)
    # The count reaches 8 at t = 2/3 s; the bell is 60 ms there, at 440 Hz.
    assert extra[0] / RATE == pytest.approx(2.0 / 3.0, abs=1.0 / RATE + 1e-9)
    assert (extra[-1] - extra[0] + 1) / RATE == pytest.approx(0.06, abs=2.0 / RATE)
    assert peak_hz(belled - plain, 2.0 / 3.0 + 0.03) == pytest.approx(440.0, abs=BIN_HZ)


@pytest.mark.parametrize("n", [4, 96])
def test_one_tick_per_time_within_a_millisecond(n):
    times = 0.5 + np.arange(n) / n  # n ticks inside one second
    x = nr.pixel_ticks(times, duration_s=2.0, rate=RATE, seed=3)
    found = onsets(x)
    assert len(found) == n
    assert np.max(np.abs(found - times * RATE)) <= 0.001 * RATE + 1


def test_ticks_are_jittered_by_the_seed():
    times = 0.5 + np.arange(96) / 96
    offsets = [
        onsets(nr.pixel_ticks(times, duration_s=2.0, seed=s)) - times * RATE
        for s in (0, 1)
    ]
    assert not np.array_equal(offsets[0], offsets[1])
    assert np.std(np.diff(offsets[0])) > 0.1 * 0.001 * RATE


def test_tick_shape_and_level():
    x = nr.pixel_ticks([0.5], duration_s=1.0, rate=RATE)
    hit = np.flatnonzero(x)
    assert (hit[-1] - hit[0] + 2) / RATE == pytest.approx(0.004, abs=2.0 / RATE)
    assert db(np.max(np.abs(x))) == pytest.approx(-26.0, abs=1.0)
    padded = np.zeros(round(0.05 * RATE))
    padded[: len(hit) + 2] = x[hit[0] - 1 : hit[-1] + 2]
    spectrum = np.abs(np.fft.rfft(padded))
    assert np.fft.rfftfreq(len(padded), 1.0 / RATE)[np.argmax(spectrum)] == (
        pytest.approx(2400.0, abs=BIN_HZ)
    )


def test_sting_ends_at_its_time_with_two_partials():
    t_end = 0.8
    x = nr.click_sting(t_end, duration_s=1.0, rate=RATE)
    hit = np.flatnonzero(x)
    assert hit[0] / RATE >= t_end - 0.09 - 1.0 / RATE
    assert hit[-1] / RATE < t_end
    full = np.max(np.abs(x))
    assert np.max(np.abs(x[round((t_end - 0.005) * RATE) : round(t_end * RATE)])) < (
        0.2 * full
    )
    assert db(full) == pytest.approx(-26.0, abs=1.0)
    n = round(0.09 * RATE)
    segment = np.zeros(4 * n)
    segment[:n] = x[round(t_end * RATE) - n : round(t_end * RATE)]
    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / RATE)
    top = sorted(freqs[np.argsort(spectrum)[-2:]])
    resolution = RATE / len(segment)
    assert top[0] == pytest.approx(880.0, abs=resolution)
    assert top[1] == pytest.approx(1318.5, abs=resolution)


@pytest.mark.parametrize("t_end", [0.0, 1.0 / RATE, 0.01])
def test_early_sting_keeps_the_complete_stings_ending_envelope(t_end):
    x = nr.click_sting(t_end, duration_s=1.0, rate=RATE)
    assert np.all(np.isfinite(x))
    end = round(t_end * RATE)
    np.testing.assert_array_equal(x[end:], 0.0)
    if end:
        complete = nr.click_sting(0.09, duration_s=0.09, rate=RATE)
        np.testing.assert_allclose(x[:end], complete[-end:], atol=1e-15)
        assert x[end - 1] == 0.0
        tail = x[max(0, end - round(0.005 * RATE)) : end]
        assert np.max(np.abs(tail)) < 0.02 * 10 ** (-26.0 / 20.0)


@pytest.mark.parametrize("sample", [np.nan, np.inf, -np.inf])
def test_composite_refuses_nonfinite_samples(monkeypatch, sample):
    monkeypatch.setattr(
        _cues, "pixel_ticks", lambda *args, **kwargs: np.full(RATE, sample)
    )
    with pytest.raises(ValueError, match=r"finite|full scale"):
        nr.cue_track(nr.CueEvents(tick_times=(0.5,)), 1.0, rate=RATE)


def events(**kwargs):
    base = dict(
        count_times=(1.0, 3.0),
        count_values=(4.0, 16.0),
        reference=4.0,
        tick_times=tuple(3.5 + np.arange(96) / 96),
        sting_s=4.8,
        bell_at=8.0,
    )
    base.update(kwargs)
    return nr.CueEvents(**base)


def test_cue_track_is_deterministic_and_of_exact_length():
    a = nr.cue_track(events(), 5.0, rate=RATE, seed=0)
    b = nr.cue_track(events(), 5.0, rate=RATE, seed=0)
    assert a.tobytes() == b.tobytes()
    assert len(a) == round(5.0 * RATE)
    assert 0.0 < np.max(np.abs(a)) < 1.0
    odd = nr.cue_track(events(), 4.99999, rate=44100)
    assert len(odd) == round(4.99999 * 44100)


def test_a_different_seed_moves_only_the_ticks():
    a = nr.cue_track(events(), 5.0, seed=0)
    b = nr.cue_track(events(), 5.0, seed=1)
    differ = np.flatnonzero(a != b) / RATE
    assert len(differ)
    assert differ.min() >= 3.5 - 0.0011 and differ.max() <= 4.5 + 0.0051


def test_the_track_is_the_sum_of_its_voices():
    e = events()
    track = nr.cue_track(e, 5.0, seed=2)
    parts = (
        nr.count_tone(
            e.count_times, e.count_values, reference=4.0, duration_s=5.0, bell_at=8.0
        )
        + nr.pixel_ticks(e.tick_times, duration_s=5.0, seed=2)
        + nr.click_sting(4.8, duration_s=5.0)
    )
    np.testing.assert_allclose(track, parts, atol=1e-12)


def tone(times, values, *, reference=4.0, duration_s=1.0, **kwargs):
    return nr.count_tone(
        times, values, reference=reference, duration_s=duration_s, **kwargs
    )


@pytest.mark.parametrize(
    ("call", "name"),
    [
        (lambda: tone([], []), "times"),
        (lambda: tone([0.5, 0.5], [4, 5]), "times"),
        (lambda: tone([0.0, 1.0], [4, 5], reference=0.0), "reference"),
        (lambda: tone([0.0, 1.0], [4, -1]), "values"),
        (lambda: tone([0.0, 2.0], [4, 5]), "times"),
        (lambda: tone([0.0, 1.0], [4, 5], rate=4000), "rate"),
        (lambda: nr.pixel_ticks([], duration_s=1.0), "times"),
        (lambda: nr.pixel_ticks([0.3, 0.2], duration_s=1.0), "times"),
        (lambda: nr.pixel_ticks([1.5], duration_s=1.0), "times"),
        (lambda: nr.pixel_ticks([0.5], duration_s=1.0, rate=7999), "rate"),
        (lambda: nr.click_sting(1.5, duration_s=1.0), "t_end"),
        (lambda: nr.click_sting(-0.1, duration_s=1.0), "t_end"),
        (lambda: nr.click_sting(0.5, duration_s=1.0, rate=100), "rate"),
        (lambda: nr.cue_track(events(sting_s=9.0), 5.0), "sting_s"),
        (lambda: nr.cue_track(events(tick_times=(6.0,)), 5.0), "times"),
    ],
)
def test_bad_arguments_are_refused_by_name(call, name):
    with pytest.raises(ValueError, match=name):
        call()
