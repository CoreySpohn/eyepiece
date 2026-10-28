"""Put the speech under a silent render, and write its captions.

`audio_track` places each line's audio at its speech start on the clip's
clock, with a short fade on every line and a low noise bed under the gaps.
`mux` joins that track with the video through ffmpeg without re-encoding the
picture, after a two-pass loudness normalization of the audio. `captions`
turns the script into cues timed from the word times, wrapped and split
within reading limits, and `write_srt` and `write_vtt` write them. WebVTT is
the format an HTML ``<track>`` reads.
"""

import json
import re
import subprocess
import tempfile
import warnings
import wave
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import numpy as np

from eyepiece._ffmpeg import _merged_profile, resolve_ffmpeg

from ._cues import TICK_JITTER_S, TICK_S, CueEvents, cue_track
from ._script import Pause
from ._timeline import MissingAudioError

_SENTENCE_END = re.compile(r"[.?!]$")
_AUDIO_SUFFIXES = (".opus", ".flac", ".wav")
_CAPTION_TAIL_S = 0.15
_DEFAULT_RATE = 24000
_CLAUSE_END = re.compile(r"(?:[,;:]|--|\u2013|\u2014)$")
_MIN_BALANCE = 0.25
_FULL_SCALE = 32767.0
_TICK_RUN_GAP_S = 1.0


def _read_wav(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono")
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return rate, data


def _decode(path, rate, target):
    """Read a 16-bit mono audio file, resampled to ``target`` Hz when given.

    Compressed formats, and WAV files at another rate than the target, go
    through ffmpeg; ``rate`` is the rate assumed for a compressed file when
    there is no target.
    """
    if path.suffix == ".wav":
        native, data = _read_wav(path)
        if target in (None, native):
            return native, data
    rate = target or rate
    cmd = [
        resolve_ffmpeg(), "-v", "error", "-i", str(path),
        "-f", "s16le", "-ac", "1", "-ar", str(rate), "-",
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, check=False)
    if done.returncode:
        raise RuntimeError(f"ffmpeg could not read {path}: {done.stderr.decode()}")
    return rate, np.frombuffer(done.stdout, dtype="<i2")


def _audio_file(directory, key, spoken):
    if spoken.audio is not None:
        names = [spoken.audio]
    else:
        names = [f"{key}{suffix}" for suffix in _AUDIO_SUFFIXES]
    for name in names:
        if (Path(directory) / name).exists():
            return Path(directory) / name
    return None


def _bed(size, level_db, seed):
    """Seeded white noise of ``size`` samples at ``level_db`` dBFS RMS."""
    sigma = _FULL_SCALE * 10.0 ** (level_db / 20.0)
    return np.random.default_rng(seed).normal(0.0, sigma, size)


def _duck_gain(size, rate, windows, floor_gain, fade_s):
    """Gain of the bed: 1 outside the windows, ``floor_gain`` inside.

    Each window fades down over its first ``fade_s`` and back up over its last
    ``fade_s`` (raised cosine), so the bed never steps.
    """
    gain = np.ones(size)
    n = round(fade_s * rate)
    for t0, t1 in windows:
        i, j = round(t0 * rate), min(round(t1 * rate), size)
        if i >= size:
            continue
        n_here = min(n, (j - i) // 2)
        gain[i:j] = floor_gain
        if n_here > 0:
            ramp = 0.5 + 0.5 * np.cos(np.pi * np.arange(n_here) / n_here)
            fall = floor_gain + (1.0 - floor_gain) * ramp
            gain[i : i + n_here] = fall
            gain[j - n_here : j] = fall[::-1]
    return gain


def _check_duck(duck):
    """The sorted, validated duck windows as a list of ``(t0, t1)``."""
    windows = sorted((float(a), float(b)) for a, b in (duck or ()))
    for a, b in windows:
        if not 0.0 <= a < b:
            raise ValueError(f"duck windows need 0 <= t0 < t1, got ({a}, {b})")
    for (_, end), (start, _) in pairwise(windows):
        if start < end:
            raise ValueError("duck windows must not overlap")
    return windows


def _bed_gain(size, rate, level_db, duck, floor_db, fade_s):
    """The bed's per-sample gain for the duck windows, or None for no ducking."""
    windows = _check_duck(duck)
    if not windows:
        return None
    return _duck_gain(
        size, rate, windows, 10.0 ** ((floor_db - level_db) / 20.0), fade_s
    )


def _cue_windows(cues, fade_s, end_s):
    """Where the voice yields to the cues: the tone, and each run of ticks.

    Ticks less than a second apart form one run. Each window is widened by
    the fade on both sides, so the voice is fully out while a cue plays, and
    overlapping windows merge.
    """
    windows = []
    if len(cues.count_times):
        windows.append((cues.count_times[0], cues.count_times[-1]))
    ticks = sorted(cues.tick_times)
    if ticks:
        first = last = ticks[0]
        for t in ticks[1:]:
            if t - last > _TICK_RUN_GAP_S:
                windows.append((first - TICK_JITTER_S, last + TICK_S + TICK_JITTER_S))
                first = t
            last = t
        windows.append((first - TICK_JITTER_S, last + TICK_S + TICK_JITTER_S))
    merged = []
    for a, b in sorted(
        (max(a - fade_s, 0.0), min(b + fade_s, end_s)) for a, b in windows
    ):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def _write_wav(path, data, rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.astype("<i2").tobytes())
    return path


def _fade(data, n):
    """The line with a raised-cosine fade of ``n`` samples at each end."""
    n = min(n, len(data) // 2)
    out = data.astype(float)
    if n > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
        out[:n] *= ramp
        out[len(out) - n :] *= ramp[::-1]
    return out


def audio_track(
    narration,
    audio_dir,
    path,
    *,
    tail_s=0.0,
    tol_s=0.03,
    rate=48000,
    room_tone_db=None,
    fade_s=0.008,
    seed=0,
    duck=None,
    duck_floor_db=-90.0,
    duck_fade_s=0.05,
    cues=None,
):
    """Write the whole clip's speech as one 16-bit mono WAV.

    Each line gets a short fade at both ends, so no line starts or stops on
    a click. `mux` adds the room-tone bed after it normalizes the loudness,
    so the bed keeps its level whatever gain the speech needs. For a track
    that `mux` will not normalize (``loudness=None``), pass ``room_tone_db``
    here instead, or to `mux`. The bed comes from a seeded generator, so the
    same inputs give the same file.

    Args:
        narration: The `Narration`.
        audio_dir: Directory holding each line's audio file, beside its JSON.
            Ogg Opus, FLAC and WAV files are read.
        path: Output WAV path.
        tail_s: Silence added after the clip's end.
        tol_s: Most a file's length may differ from the duration its JSON
            records, in seconds. A larger difference means the audio is not
            the audio the cue times were measured on.
        rate: Output sample rate in Hz; lines are resampled to it with
            ffmpeg. None keeps the rate of the files, which must agree.
        room_tone_db: RMS level of a white-noise bed under the whole track,
            in dB relative to full scale. None, the default, writes exact
            silence between the lines.
        fade_s: Length of the fade on each end of a line, in seconds.
        seed: Seed of the noise bed.
        duck: Windows ``[(t0, t1), ...]`` in seconds inside which the room-tone
            bed drops to ``duck_floor_db``; None or empty ducks nothing and
            leaves the track byte-identical. Windows must not overlap.
        duck_floor_db: RMS level of the bed inside a window, in dBFS.
        duck_fade_s: Fade down at the start of a window and up at its end.
        cues: A `CueEvents`, whose `cue_track` (seeded with `seed`) is summed
            into the track, or None (the default) for no cue sounds and a
            byte-identical track. The voice yields to the cues: it fades
            out, over ``duck_fade_s``, from the first to the last knot of
            the count tone and over each run of ticks (ticks less than 1 s
            apart), through the duck machinery the bed uses; the bed keeps
            its level. Cues are synthesized at fixed levels and mixed before
            normalization. `mux` normalizes the combined speech and cue
            track, with no enforced cue-to-voice margin. For cue-only clips,
            pass ``loudness=None`` to `mux` to retain the synthesis levels.

    Returns:
        The path written.

    Raises:
        TypeError: If `cues` is given and is not a `CueEvents`.
        MissingAudioError: If a line has no audio file.
        ValueError: If the files differ in sample rate (with ``rate=None``),
            a file's length does not match its recorded duration, a line
            runs past the end of the track, ``fade_s`` is negative, or a cue
            lies past the end of the track.
    """
    if fade_s < 0.0:
        raise ValueError(f"fade_s must not be negative, got {fade_s}")
    if cues is not None and not isinstance(cues, CueEvents):
        raise TypeError(f"cues must be a CueEvents or None, got {type(cues).__name__}")
    clips, out_rate = [], rate
    for beat in narration.beats:
        line = narration.script.line(beat.key)
        if isinstance(line, Pause):
            continue
        spoken = narration.spoken[beat.key]
        key = narration.script.line_key(line)
        file = _audio_file(audio_dir, key, spoken)
        if file is None:
            raise MissingAudioError(
                f"no audio for line {beat.key!r} ({key}) in {audio_dir}"
            )
        r, data = _decode(file, spoken.rate or _DEFAULT_RATE, rate)
        if out_rate not in (None, r):
            raise ValueError("lines differ in sample rate")
        out_rate = r
        if abs(len(data) / r - spoken.duration_s) > tol_s:
            raise ValueError(
                f"line {beat.key!r}: {file.name} lasts {len(data) / r:.3f} s but "
                f"its timing says {spoken.duration_s:.3f} s; the audio and the "
                "cue times are from different syntheses"
            )
        clips.append((beat.key, beat.speech_start_s, data))
    if out_rate is None:
        raise ValueError("the script has no lines")
    size = round((narration.total_s + tail_s) * out_rate)
    mix = np.zeros(size)
    if room_tone_db is not None:
        bed = _bed(size, room_tone_db, seed)
        gain = _bed_gain(size, out_rate, room_tone_db, duck, duck_floor_db, duck_fade_s)
        mix += bed if gain is None else bed * gain
    speech = mix if cues is None else np.zeros(size)
    for key, start, data in clips:
        i = round(start * out_rate)
        if i + len(data) > size + 2:
            raise ValueError(f"line {key!r} runs past the end of the track")
        data = data[: size - i]
        speech[i : i + len(data)] += _fade(data, round(fade_s * out_rate))
    if cues is not None:
        sound = cue_track(cues, size / out_rate, rate=out_rate, seed=seed)
        windows = _cue_windows(cues, duck_fade_s, size / out_rate)
        mix += speech * _duck_gain(size, out_rate, windows, 0.0, duck_fade_s)
        mix += _FULL_SCALE * sound
    out = np.clip(np.round(mix), -32768, 32767).astype("<i2")
    return _write_wav(path, out, out_rate)


def _measure(ffmpeg, audio, filt):
    """First pass of loudnorm: the measured statistics, or None for silence."""
    cmd = [
        ffmpeg, "-hide_banner", "-nostats", "-i", str(audio),
        "-af", f"{filt}:print_format=json", "-f", "null", "-",
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    found = re.findall(r"\{[^{}]*\}", done.stderr)
    if done.returncode or not found:
        raise RuntimeError(f"ffmpeg could not measure {audio}: {done.stderr}")
    stats = json.loads(found[-1])
    if not np.isfinite(float(stats["input_i"])):
        return None
    return stats


def _finish_audio(ffmpeg, audio, tmp, loudness, true_peak, lra, rate):
    """The audio after the two-pass loudnorm, as a WAV in ``tmp``; or as given."""
    if loudness is None:
        return audio
    target = f"loudnorm=I={loudness}:TP={true_peak}:LRA={lra}"
    stats = _measure(ffmpeg, audio, target)
    if stats is None:
        return audio
    target += (
        f":measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}"
        f":measured_TP={stats['input_tp']}"
        f":measured_thresh={stats['input_thresh']}"
        f":offset={stats['target_offset']}:linear=true"
    )
    out = tmp / "norm.wav"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-i", str(audio),
        "-af", target, "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", str(out),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"ffmpeg could not normalize {audio}: {done.stderr}")
    return out


def _video_duration(ffmpeg, video):
    """Read the picture's presentation end, without decoding or needing ffprobe.

    The framecrc muxer reports packet PTS and duration in an exact time base.
    Use presentation times: copied H.264's decode clock can end early because
    of B-frame reordering. The human-readable input header rounds to 0.01 s.
    """
    cmd = [
        ffmpeg, "-v", "error", "-i", str(video), "-map", "0:v:0",
        "-c:v", "copy", "-f", "framecrc", "-",
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"ffmpeg could not read {video}: {done.stderr}")
    time_base, end = None, 0
    for line in done.stdout.splitlines():
        if line.startswith("#tb 0:"):
            time_base = Fraction(line.split(":", 1)[1].strip())
        elif line and not line.startswith("#"):
            fields = line.split(",")
            end = max(end, int(fields[2]) + int(fields[3]))
    if time_base is None or end <= 0:
        raise RuntimeError(f"ffmpeg could not determine the video duration of {video}")
    return float(end * time_base)


def mux(
    video,
    audio,
    path,
    *,
    bitrate="160k",
    loudness=-16.0,
    true_peak=-1.0,
    lra=11.0,
    rate=48000,
    room_tone_db=-60.0,
    seed=0,
    encoder=None,
    duck=None,
    duck_floor_db=-90.0,
    duck_fade_s=0.05,
    work_dir=None,
):
    """Join a silent video and an audio track into one MP4.

    The picture is copied, not re-encoded; the audio is AAC, padded with
    silence or trimmed to the video's duration. The audio is normalized in
    two passes of ffmpeg's ``loudnorm``: the first measures the track, the
    second applies a linear gain from that measurement, so the speech keeps
    its dynamics. The default, -16 LUFS with a -1 dBTP ceiling, is the usual
    target for spoken
    web audio; broadcast uses -23 LUFS. A silent track is left as it is. The
    container follows the shared encoder profile (`eyepiece.ENCODER_PROFILE`):
    the BT.709 tags are written into the copied stream and the index moves to
    the front.

    A seeded white-noise bed is added after the normalization, so
    ``room_tone_db`` is its level in the muxed file whatever gain the speech
    needed. Normalizing first also keeps the bed out of the loudness
    measurement.

    Args:
        video: Silent MP4 from the render.
        audio: Audio file from `audio_track`.
        path: Output MP4 path.
        bitrate: AAC bitrate.
        loudness: Integrated loudness target in LUFS. None skips the
            normalization and encodes the track as it is.
        true_peak: True-peak ceiling in dBTP.
        lra: Loudness range target in LU.
        rate: Sample rate of the encoded audio in Hz.
        room_tone_db: RMS level in dBFS of the noise bed added under the
            whole track after normalization. None adds none. It applies with
            ``loudness=None`` too, so a track that already has a bed from
            `audio_track` should pass None here.
        seed: Seed of the noise bed.
        encoder: Keys of `eyepiece.ENCODER_PROFILE` to replace; only the
            color tags and ``faststart`` apply to a copied picture.
        duck: Windows ``[(t0, t1), ...]`` in seconds where the bed drops to
            ``duck_floor_db``, as in `audio_track`; None ducks nothing.
        duck_floor_db: RMS level of the bed inside a window, in dBFS.
        duck_fade_s: Fade at each window edge, in seconds.
        work_dir: Parent folder for temporary audio files; None uses the
            system temporary folder. Created when supplied and missing.

    Returns:
        The path written.

    Raises:
        RuntimeError: If no ffmpeg is found, or ffmpeg fails (its message is
            included).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg()
    duration_s = _video_duration(ffmpeg, video)
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        source = _finish_audio(ffmpeg, audio, Path(tmp), loudness, true_peak, lra, rate)
        if room_tone_db is not None:
            _, data = _decode(Path(source), rate, rate)
            bed = _bed(len(data), room_tone_db, seed)
            gain = _bed_gain(
                len(data), rate, room_tone_db, duck, duck_floor_db, duck_fade_s
            )
            noisy = np.round(data + (bed if gain is None else bed * gain))
            source = _write_wav(
                Path(tmp) / "bed.wav", np.clip(noisy, -32768, 32767), rate
            )
        audio_args = [] if loudness is None else ["-ar", str(rate)]
        profile = _merged_profile(encoder)
        tags = (
            [
                "-bsf:v",
                "h264_metadata=colour_primaries=1:transfer_characteristics=1"
                ":matrix_coefficients=1:video_full_range_flag=0",
            ]
            if profile["colorspace"] == "bt709"
            else []
        )
        faststart = ["-movflags", "+faststart"] if profile["faststart"] else []
        # FFmpeg 4.4's -shortest stops AAC at the copied video's decode end,
        # before its last B-frames are presented. Pad and use the presentation
        # duration instead, allowing the AAC encoder to flush its final frames.
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(video), "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", *tags, "-c:a", "aac", "-b:a", bitrate, *audio_args,
            "-af", "apad", *faststart, "-t", str(duration_s), str(path),
        ]  # fmt: skip
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode:
            raise RuntimeError(
                f"ffmpeg failed to join {video} and {audio}: {done.stderr}"
            )
    return path


def _group(items, text):
    """Split a run of words into sentences, each a list."""
    out, group = [], []
    for item in items:
        group.append(item)
        if _SENTENCE_END.search(text(item)):
            out.append(group)
            group = []
    if group:
        out.append(group)
    return out


def sentences(narration):
    """The script as sentences with their times.

    Returns:
        A list of ``(start_s, end_s, text)``. A sentence ends at a word that
        ends in a period, question mark or exclamation mark, or at the end
        of its line. The text is the spoken text, whatever a line's
        ``display`` says; `captions` shows the display text.
    """
    out = []
    for beat in narration.beats:
        out += _group(beat.words, lambda w: w.text)
    return [(g[0].start_s, g[-1].end_s, " ".join(w.text for w in g)) for g in out]


def _spread(tokens, start, end):
    """Display tokens timed across a span, in proportion to their length."""
    weights = np.array([len(t) + 1 for t in tokens], dtype=float)
    fraction = np.cumsum(np.concatenate([[0.0], weights])) / weights.sum()
    edges = start + (end - start) * fraction
    edges[0] = start
    edges[-1] = end
    return [(t, float(edges[i]), float(edges[i + 1])) for i, t in enumerate(tokens)]


def _units(narration):
    """Sentences as lists of ``(text, start_s, end_s)`` tokens.

    A line with display text shows it in place of the spoken words. When the
    display text has as many sentences as the speech, each takes the span of
    its spoken sentence; otherwise the whole line is one run over the line's
    speech. Either way the span comes from the spoken word times.
    """
    out = []
    for beat in narration.beats:
        spoken = _group(beat.words, lambda w: w.text)
        shown = narration.script.line(beat.key).display
        if shown is None:
            out += [[(w.text, w.start_s, w.end_s) for w in g] for g in spoken]
            continue
        pieces = _group(shown.split(), str)
        if len(pieces) == len(spoken):
            pairs = list(zip(spoken, pieces, strict=True))
        else:
            warnings.warn(
                f"line {beat.key!r}: display text has {len(pieces)} sentences "
                f"but spoken text has {len(spoken)}; caption timing may drift "
                "within the line",
                UserWarning,
                stacklevel=3,
            )
            pairs = [(list(beat.words), shown.split())]
        out += [_spread(tokens, g[0].start_s, g[-1].end_s) for g, tokens in pairs]
    return out


def _breaks(texts):
    """Word breaks outside brackets and away from standalone operators.

    If none are available, return every word break so long text still splits.
    """
    found = set()
    depth = 0
    for i, text in enumerate(texts[:-1], start=1):
        for char in text:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
        if (
            depth == 0
            and any(c.isalnum() for c in text)
            and any(c.isalnum() for c in texts[i])
        ):
            found.add(i)
    return found or set(range(1, len(texts)))


def _best_break(texts, allowed):
    """Index at which to break a token list, preferring a clause boundary.

    Among the breaks ``allowed`` accepts, a break after a comma, colon,
    semicolon or dash wins when it leaves two parts that are not badly
    unequal; otherwise the most equal parts win. Bracketed text and standalone
    operators stay together unless every word break is forbidden. None if no
    break meets ``allowed``.
    """
    sizes = [len(t) for t in texts]
    found = []
    for i in sorted(_breaks(texts)):
        if not allowed(i):
            continue
        left, right = sum(sizes[:i]) + i - 1, sum(sizes[i:]) + len(texts) - i - 1
        balance = min(left, right) / max(left, right, 1)
        found.append(
            (i, abs(left - right), balance, bool(_CLAUSE_END.search(texts[i - 1])))
        )
    if not found:
        return None
    clause = [f for f in found if f[3] and f[2] >= _MIN_BALANCE]
    return min(clause or found, key=lambda f: f[1])[0]


def _wrap(texts, max_chars, max_lines):
    """The text as lines within the limit, or None if it does not fit."""
    full = " ".join(texts)
    if len(full) <= max_chars:
        return [full]
    if max_lines < 2:
        return None
    i = _best_break(
        texts,
        lambda i: (
            len(" ".join(texts[:i])) <= max_chars
            and len(" ".join(texts[i:])) <= max_chars
        ),
    )
    if i is None:
        return None
    return [" ".join(texts[:i]), " ".join(texts[i:])]


def _pieces(tokens, max_chars, max_lines, max_cue_s):
    """Split a sentence's tokens until each piece wraps and is short enough."""
    texts = [t for t, _, _ in tokens]
    fits = _wrap(texts, max_chars, max_lines) is not None
    if (fits and tokens[-1][2] - tokens[0][1] <= max_cue_s) or len(tokens) == 1:
        return [tokens]
    i = _best_break(texts, lambda i: True)
    return _pieces(tokens[:i], max_chars, max_lines, max_cue_s) + _pieces(
        tokens[i:], max_chars, max_lines, max_cue_s
    )


def captions(
    narration,
    *,
    max_chars=42,
    max_lines=2,
    max_cue_s=7.0,
    max_cps=20.0,
    min_s=1.0,
):
    """The script as caption cues, wrapped and split to reading limits.

    These are subtitle cues for a caption track (`write_vtt`, `write_srt`),
    not on-screen text: `eyepiece.Timeline.caption` places a caption in the
    picture, and `eyepiece.callout` labels a subject.

    A sentence becomes one cue. A sentence that does not fit in `max_lines`
    lines of `max_chars` characters, or that lasts longer than `max_cue_s`,
    is split first at a clause boundary (after a comma, colon, semicolon or
    dash), and at the most even word break when it has none. Within a cue the
    lines break at a clause boundary when one leaves two even lines. Breaks
    stay outside brackets and away from tokens with no letters or digits,
    unless every word break is forbidden. Cue start times are the start times
    of the first spoken word; a cue ends a moment after its last word, and is
    held to at least `min_s` and to the
    time its characters need at `max_cps`, but never past the start of the
    next cue. A line with display text shows it; see `Line`. If its sentence
    count differs from the speech, a `UserWarning` names the line once per
    call, since caption timing may drift within that line.

    Args:
        narration: The `Narration`.
        max_chars: Most characters on one caption line.
        max_lines: Most lines in one cue, 1 or 2.
        max_cue_s: Longest speech span of one cue, in seconds.
        max_cps: Reading speed the cue is held for, in characters per second.
        min_s: Shortest cue, in seconds.

    Returns:
        A list of ``(start_s, end_s, text)``, the text holding a newline
        between its lines.

    Raises:
        ValueError: If `max_lines` is not 1 or 2, or a limit is not positive.
    """
    if max_lines not in (1, 2):
        raise ValueError(f"max_lines must be 1 or 2, got {max_lines}")
    if min(max_chars, max_cue_s, max_cps) <= 0.0 or min_s < 0.0:
        raise ValueError("caption limits must be positive")
    cues = []
    for tokens in _units(narration):
        for piece in _pieces(tokens, max_chars, max_lines, max_cue_s):
            texts = [t for t, _, _ in piece]
            lines = _wrap(texts, max_chars, max_lines) or [" ".join(texts)]
            cues.append((piece[0][1], piece[-1][2], "\n".join(lines)))
    out = []
    for i, (a, b, text) in enumerate(cues):
        need = max(min_s, len(text.replace("\n", " ")) / max_cps)
        end = max(b + _CAPTION_TAIL_S, a + need)
        if i + 1 < len(cues):
            end = min(end, cues[i + 1][0])
        out.append((a, max(end, a), text))
    return out


def caption_at(cues, t) -> str:
    """Return the caption active at a time on the clip's clock.

    Args:
        cues: The list of ``(start_s, end_s, text)`` from `captions`.
        t: Time in seconds, with cue starts included and ends excluded.

    Returns:
        The active cue's text, or an empty string between or outside cues.
    """
    for start, end, text in cues:
        if start <= t < end:
            return text
    return ""


def _stamp(seconds, sep):
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def write_srt(narration, path, **limits):
    """Write the script's captions as an SRT file.

    The cues are those of `captions`.

    Args:
        narration: The `Narration`.
        path: Output path.
        **limits: Keyword limits of `captions` (``max_chars``, ``max_lines``,
            ``max_cue_s``, ``max_cps``, ``min_s``).

    Returns:
        The path written.
    """
    lines = []
    for i, (a, b, text) in enumerate(captions(narration, **limits), start=1):
        lines += [str(i), f"{_stamp(a, ',')} --> {_stamp(b, ',')}", text, ""]
    return _write(path, "\n".join(lines))


def write_vtt(narration, path, *, line=None, **limits):
    """Write the script's captions as a WebVTT file.

    The cues are those of `captions`. WebVTT, not SRT, is what an HTML
    ``<track kind="captions">`` reads.

    Args:
        narration: The `Narration`.
        path: Output path.
        line: Where the cues sit, as a percentage of the picture height
            measured to the top of the cue, written as the ``line:`` setting
            of every cue. None leaves the player's default, the bottom.
            `eyepiece.testing.caption_line` computes a value clear of the
            figure's text.
        **limits: Keyword limits of `captions`.

    Returns:
        The path written.
    """
    setting = "" if line is None else f" line:{float(line):g}%"
    lines = ["WEBVTT", ""]
    for a, b, text in captions(narration, **limits):
        lines += [f"{_stamp(a, '.')} --> {_stamp(b, '.')}{setting}", text, ""]
    return _write(path, "\n".join(lines))


def write_chapters(narration, path, *, titles=None):
    """Write a WebVTT chapters file with one cue per beat.

    Each cue spans its beat, from its start to its end, and its identifier is
    the line's key. A pause is a beat like any other. An HTML
    ``<track kind="chapters">`` reads the file.

    Args:
        narration: The `Narration`.
        path: Output path.
        titles: Mapping from a line key to the chapter title shown for it.
            A beat without an entry is titled with its key.

    Returns:
        The path written.

    Raises:
        ValueError: If `titles` names a key that is not a beat, or a title is
            empty once its whitespace is collapsed.
    """
    titles = {} if titles is None else dict(titles)
    unknown = sorted(set(titles) - {b.key for b in narration.beats})
    if unknown:
        raise ValueError(f"chapter titles for lines not in the script: {unknown}")
    lines = ["WEBVTT", ""]
    for beat in narration.beats:
        title = " ".join(str(titles.get(beat.key, beat.key)).split())
        if not title:
            raise ValueError(f"the chapter title for {beat.key!r} is empty")
        lines += [
            beat.key,
            f"{_stamp(beat.start_s, '.')} --> {_stamp(beat.end_s, '.')}",
            title,
            "",
        ]
    return _write(path, "\n".join(lines))


def write_captions(narration, stem, *, line=None, chapters=False, **limits):
    """Write the captions as both WebVTT and SRT.

    Args:
        narration: The `Narration`.
        stem: Output path without a suffix; ``.vtt`` and ``.srt`` are added.
        line: The ``line:`` setting of the WebVTT file; see `write_vtt`.
        chapters: Also write ``<stem>.chapters.vtt``, see `write_chapters`.
        **limits: Keyword limits of `captions`.

    Returns:
        ``(vtt_path, srt_path)``, with the chapters path added when
        `chapters` is true.
    """
    stem = str(stem)
    paths = (
        write_vtt(narration, stem + ".vtt", line=line, **limits),
        write_srt(narration, stem + ".srt", **limits),
    )
    if chapters:
        paths += (write_chapters(narration, stem + ".chapters.vtt"),)
    return paths
