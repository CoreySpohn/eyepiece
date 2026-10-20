"""Cue sounds: a count heard as pitch, one tick per pixel read, a click sting.

Three voices, each computed from numbers and nothing else (no sample files,
no network), returned as a mono float array in full-scale units (1.0 is
0 dBFS) of ``round(duration_s * rate)`` samples:

- `count_tone`: a sine whose frequency is ``220 Hz x value / reference``, so
  pitch is proportional to the count and equal ratios are equal musical
  intervals (4 to 16 is exactly two octaves). Its phase is the running
  integral of the frequency, so the pitch glides without a phase jump. It
  sounds from the first to the last knot, with a 40 ms attack, a 250 ms decay
  ending at the last knot, and a level of -30 dBFS RMS. An optional octave
  marker bell (440 Hz, 60 ms) rings once where the count first reaches a
  stated value.
- `pixel_ticks`: one tick per time, each a 4 ms raised-cosine burst at
  2.4 kHz (the raised cosine band-limits it), -26 dBFS peak, with its onset
  moved by at most 1 ms by a seeded generator, so a fast run of ticks does
  not sound like a buzz at a fixed rate.
- `click_sting`: a 90 ms two-partial tone, A5 (880 Hz) and E6 (1318.5 Hz),
  a fifth (3:2) apart, with a soft attack, ending exactly at its time.

`cue_track` sums the three on one array. The bell and the sting both default
to -26 dBFS peak.
"""

import math
from dataclasses import dataclass

import numpy as np

TONE_BASE_HZ = 220.0
TONE_DB = -30.0
TONE_ATTACK_S = 0.040
TONE_DECAY_S = 0.250
BELL_HZ = 440.0
BELL_S = 0.060
TICK_HZ = 2400.0
TICK_S = 0.004
TICK_DB = -26.0
TICK_JITTER_S = 0.001
STING_HZ = (880.0, 1318.5)
STING_S = 0.090
STING_ATTACK_S = 0.015
STING_DB = -26.0
MIN_RATE = 8000


@dataclass(frozen=True)
class CueEvents:
    """What the cue voices play, in seconds on the clip's clock.

    Attributes:
        count_times: Knot times of the count, strictly increasing.
        count_values: The count at each knot, positive.
        reference: The count heard at 220 Hz, positive.
        tick_times: One time per pixel read, strictly increasing; empty for
            no ticks.
        sting_s: When the click's swell ends, or None for no sting.
        bell_at: The count at which the octave bell rings, or None.
    """

    count_times: tuple = ()
    count_values: tuple = ()
    reference: float = 1.0
    tick_times: tuple = ()
    sting_s: float | None = None
    bell_at: float | None = None


def _length(duration_s, rate):
    if not (isinstance(rate, int) and rate >= MIN_RATE):
        raise ValueError(f"rate must be an int of at least {MIN_RATE} Hz, got {rate!r}")
    if not (math.isfinite(duration_s) and duration_s > 0.0):
        raise ValueError(f"duration_s must be positive, got {duration_s!r}")
    return round(duration_s * rate)


def _times(times, duration_s, name="times"):
    """The times as floats, refused when empty, unordered or off the clip."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(f"{name} is empty; give at least one time")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    if times[0] < 0.0 or times[-1] > duration_s:
        raise ValueError(f"{name} must lie in [0, duration_s={duration_s}]")
    return times


def _raised_cosine(n):
    """``n`` samples rising from 0 toward 1 along half a cosine."""
    return 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / max(n, 1))


def _amplitude(db):
    return 10.0 ** (db / 20.0)


def count_tone(times, values, *, reference, duration_s, rate=48000, bell_at=None):
    """A sine whose pitch is proportional to a count.

    The frequency at time ``t`` is ``220 Hz x value(t) / reference``, with
    ``value(t)`` interpolated linearly between the knots; the phase is the
    running integral of that frequency, so the tone glides without a jump.
    It sounds from the first knot to the last: a 40 ms raised-cosine attack,
    a level of -30 dBFS RMS (amplitude ``sqrt(2) x 10^(-30/20)``), and a
    250 ms raised-cosine decay that ends at the last knot. With `bell_at`,
    a 60 ms bell at 440 Hz (the pitch of a count twice the reference), at
    -26 dBFS peak under a raised-cosine window, starts where the count first
    reaches `bell_at`, and none rings if it never does.

    Args:
        times: Knot times in seconds, strictly increasing, inside the clip.
        values: The count at each knot, positive.
        reference: The count heard at 220 Hz, positive.
        duration_s: Length of the returned track in seconds.
        rate: Sample rate in Hz, at least 8000.
        bell_at: The count that rings the octave bell, or None.

    Returns:
        A float array of ``round(duration_s * rate)`` samples.

    Raises:
        ValueError: If `times` is empty, not strictly increasing or off the
            clip; `values` does not match it or is not positive;
            `reference` is not positive; or `rate` is below 8000.
    """
    size = _length(duration_s, rate)
    times = _times(times, duration_s)
    values = np.asarray(values, dtype=float)
    if values.shape != times.shape:
        raise ValueError("values must have one count per time")
    if not (math.isfinite(reference) and reference > 0.0):
        raise ValueError(f"reference must be positive, got {reference!r}")
    if not np.all(np.isfinite(values) & (values > 0.0)):
        raise ValueError("values must be positive counts")
    out = np.zeros(size)
    i0, i1 = round(times[0] * rate), min(round(times[-1] * rate), size)
    if i1 <= i0:
        return out
    t = np.arange(i0, i1) / rate
    freq = TONE_BASE_HZ * np.interp(t, times, values) / reference
    phase = 2.0 * np.pi * np.cumsum(freq) / rate
    envelope = np.ones(i1 - i0)
    attack = min(round(TONE_ATTACK_S * rate), len(envelope))
    envelope[:attack] = _raised_cosine(attack)
    decay = min(round(TONE_DECAY_S * rate), len(envelope))
    envelope[len(envelope) - decay :] *= _raised_cosine(decay)[::-1]
    out[i0:i1] = math.sqrt(2.0) * _amplitude(TONE_DB) * envelope * np.sin(phase)
    if bell_at is not None:
        reached = np.flatnonzero(np.interp(t, times, values) >= bell_at)
        if reached.size:
            j = i0 + reached[0]
            n = min(round(BELL_S * rate), size - j)
            window = np.hanning(round(BELL_S * rate) + 2)[1:-1][:n]
            ring = np.sin(2.0 * np.pi * BELL_HZ * np.arange(n) / rate)
            out[j : j + n] += _amplitude(TICK_DB) * window * ring
    return out


def pixel_ticks(times, *, duration_s, rate=48000, seed=0):
    """One short click per pixel read.

    Each tick is a 2.4 kHz sine under a 4 ms raised-cosine (Hann) window,
    which band-limits it, at -26 dBFS peak. Its onset is the given time
    moved by a jitter drawn uniformly from -1 ms to +1 ms by
    ``numpy.random.default_rng(seed)``, so a fast run does not buzz at a
    fixed rate, and the same seed gives the same ticks. Pass the frame times
    of the flashes the picture shows, so picture and sound agree to within a
    video frame. A tick that would start before the clip starts at 0, and
    one that would run past its end is cut there.

    Args:
        times: Tick times in seconds, strictly increasing, inside the clip.
        duration_s: Length of the returned track in seconds.
        rate: Sample rate in Hz, at least 8000.
        seed: Seed of the onset jitter.

    Returns:
        A float array of ``round(duration_s * rate)`` samples.

    Raises:
        ValueError: If `times` is empty, not strictly increasing or off the
            clip, or `rate` is below 8000.
    """
    size = _length(duration_s, rate)
    times = _times(times, duration_s)
    jitter = np.random.default_rng(seed).uniform(
        -TICK_JITTER_S, TICK_JITTER_S, times.size
    )
    n = round(TICK_S * rate)
    tick = np.hanning(n) * np.cos(
        2.0 * np.pi * TICK_HZ * (np.arange(n) - (n - 1) / 2) / rate
    )
    tick *= _amplitude(TICK_DB) / np.max(np.abs(tick))
    out = np.zeros(size)
    for t in np.clip(times + jitter, 0.0, None):
        i = round(t * rate)
        m = min(n, size - i)
        if m > 0:
            out[i : i + m] += tick[:m]
    return out


def click_sting(t_end, *, duration_s, rate=48000):
    """A two-note chime that ends exactly when the click's swell ends.

    Two equal partials, A5 at 880 Hz and E6 at 1318.5 Hz (a fifth, 3:2),
    sound for the 90 ms before `t_end`: a 15 ms raised-cosine attack, then a
    raised-cosine fall that reaches zero at `t_end`. The peak is -26 dBFS.
    If the sting would start before the clip, its beginning is cropped,
    preserving the complete sting's normalization and ending decay.

    Args:
        t_end: When the sting ends, in seconds, inside the clip.
        duration_s: Length of the returned track in seconds.
        rate: Sample rate in Hz, at least 8000.

    Returns:
        A float array of ``round(duration_s * rate)`` samples.

    Raises:
        ValueError: If `t_end` is off the clip or `rate` is below 8000.
    """
    size = _length(duration_s, rate)
    if not (math.isfinite(t_end) and 0.0 <= t_end <= duration_s):
        raise ValueError(f"t_end must lie in [0, duration_s={duration_s}], got {t_end}")
    out = np.zeros(size)
    j = min(round(t_end * rate), size)
    n = round(STING_S * rate)
    i = max(j - n, 0)
    k = np.arange(n)
    attack = round(STING_ATTACK_S * rate)
    envelope = np.empty(n)
    envelope[:attack] = _raised_cosine(attack)
    envelope[attack:] = 0.5 + 0.5 * np.cos(
        np.pi * np.arange(1, n - attack + 1) / (n - attack)
    )
    tone = sum(np.sin(2.0 * np.pi * f * k / rate) for f in STING_HZ)
    sting = envelope * tone
    sting = _amplitude(STING_DB) * sting / np.max(np.abs(sting))
    out[i:j] = sting[n - (j - i) :]
    return out


def cue_track(events, duration_s, *, rate=48000, seed=0):
    """The three cue voices of `events` on one track.

    The count tone (with its bell), the ticks and the sting are each placed
    as their own functions place them and summed. Each voice is absent when
    `events` holds none of it.

    Args:
        events: A `CueEvents`.
        duration_s: Length of the track in seconds.
        rate: Sample rate in Hz, at least 8000.
        seed: Seed of the tick jitter; it changes the ticks and nothing else.

    Returns:
        A float array of ``round(duration_s * rate)`` samples, peak below 1.

    Raises:
        ValueError: If a voice's arguments are refused (named as there, the
            sting time as ``sting_s``), or the sum is non-finite or would
            reach full scale.
    """
    out = np.zeros(_length(duration_s, rate))
    if len(events.count_times):
        out += count_tone(
            events.count_times,
            events.count_values,
            reference=events.reference,
            duration_s=duration_s,
            rate=rate,
            bell_at=events.bell_at,
        )
    if len(events.tick_times):
        out += pixel_ticks(
            events.tick_times, duration_s=duration_s, rate=rate, seed=seed
        )
    if events.sting_s is not None:
        if not 0.0 <= events.sting_s <= duration_s:
            raise ValueError(
                f"sting_s must lie in [0, duration_s={duration_s}], "
                f"got {events.sting_s}"
            )
        out += click_sting(events.sting_s, duration_s=duration_s, rate=rate)
    if not np.all(np.isfinite(out)):
        raise ValueError("the cue voices contain non-finite samples")
    if np.max(np.abs(out), initial=0.0) >= 1.0:
        raise ValueError("the cue voices sum to full scale; lower their density")
    return out
