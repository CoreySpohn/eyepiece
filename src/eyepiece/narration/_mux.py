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
import wave
from pathlib import Path

import numpy as np

from eyepiece._ffmpeg import _merged_profile, resolve_ffmpeg

from ._timeline import MissingAudioError

_SENTENCE_END = re.compile(r"[.?!]$")
_AUDIO_SUFFIXES = (".opus", ".flac", ".wav")
_CAPTION_TAIL_S = 0.15
_DEFAULT_RATE = 24000
_CLAUSE_END = re.compile(r"(?:[,;:]|--|\u2013|\u2014)$")
_MIN_BALANCE = 0.25
_FULL_SCALE = 32767.0


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
    room_tone_db=-60.0,
    fade_s=0.008,
    seed=0,
):
    """Write the whole clip's speech as one 16-bit mono WAV.

    Each line gets a short fade at both ends, so no line starts or stops on
    a click, and a low white-noise bed runs under the whole track, so the
    gaps are not digital silence. The bed comes from a seeded generator, so
    the same inputs give the same file.

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
        room_tone_db: RMS level of the noise bed in dB relative to full
            scale. None writes exact silence between the lines.
        fade_s: Length of the fade on each end of a line, in seconds.
        seed: Seed of the noise bed.

    Returns:
        The path written.

    Raises:
        MissingAudioError: If a line has no audio file.
        ValueError: If the files differ in sample rate (with ``rate=None``),
            a file's length does not match its recorded duration, a line
            runs past the end of the track, or ``fade_s`` is negative.
    """
    if fade_s < 0.0:
        raise ValueError(f"fade_s must not be negative, got {fade_s}")
    clips, out_rate = [], rate
    for beat in narration.beats:
        spoken = narration.spoken[beat.key]
        key = narration.script.line_key(narration.script.line(beat.key))
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
        sigma = _FULL_SCALE * 10.0 ** (room_tone_db / 20.0)
        mix += np.random.default_rng(seed).normal(0.0, sigma, size)
    for key, start, data in clips:
        i = round(start * out_rate)
        if i + len(data) > size + 2:
            raise ValueError(f"line {key!r} runs past the end of the track")
        data = data[: size - i]
        mix[i : i + len(data)] += _fade(data, round(fade_s * out_rate))
    out = np.clip(np.round(mix), -32768, 32767).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(out_rate)
        handle.writeframes(out.tobytes())
    return path


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
    encoder=None,
):
    """Join a silent video and an audio track into one MP4.

    The picture is copied, not re-encoded; the audio is AAC. The output ends
    with the shorter of the two. The audio is normalized in two passes of
    ffmpeg's ``loudnorm``: the first measures the track, the second applies a
    linear gain from that measurement, so the speech keeps its dynamics. The
    default, -16 LUFS with a -1 dBTP ceiling, is the usual target for spoken
    web audio; broadcast uses -23 LUFS. A silent track is left as it is. The
    container follows the shared encoder profile (`eyepiece.ENCODER_PROFILE`):
    the BT.709 tags are written into the copied stream and the index moves to
    the front.

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
        encoder: Keys of `eyepiece.ENCODER_PROFILE` to replace; only the
            color tags and ``faststart`` apply to a copied picture.

    Returns:
        The path written.

    Raises:
        RuntimeError: If no ffmpeg is found, or ffmpeg fails (its message is
            included).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_ffmpeg()
    audio_args = []
    if loudness is not None:
        target = f"loudnorm=I={loudness}:TP={true_peak}:LRA={lra}"
        stats = _measure(ffmpeg, audio, target)
        audio_args = ["-ar", str(rate)]
        if stats is not None:
            target += (
                f":measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}"
                f":measured_TP={stats['input_tp']}"
                f":measured_thresh={stats['input_thresh']}"
                f":offset={stats['target_offset']}:linear=true"
            )
            audio_args += ["-af", target]
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
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", *tags, "-c:a", "aac", "-b:a", bitrate, *audio_args,
        *faststart, "-shortest", str(path),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"ffmpeg failed to join {video} and {audio}: {done.stderr}")
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
            pairs = [(list(beat.words), shown.split())]
        out += [_spread(tokens, g[0].start_s, g[-1].end_s) for g, tokens in pairs]
    return out


def _best_break(texts, allowed):
    """Index at which to break a token list, preferring a clause boundary.

    Among the breaks ``allowed`` accepts, a break after a comma, colon,
    semicolon or dash wins when it leaves two parts that are not badly
    unequal; otherwise the most equal parts win. None if no break is allowed.
    """
    sizes = [len(t) for t in texts]
    found = []
    for i in range(1, len(texts)):
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

    A sentence becomes one cue. A sentence that does not fit in `max_lines`
    lines of `max_chars` characters, or that lasts longer than `max_cue_s`,
    is split first at a clause boundary (after a comma, colon, semicolon or
    dash), and at the most even word break when it has none. Within a cue the
    lines break at a clause boundary when one leaves two even lines. Cue
    start times are the start times of the first spoken word; a cue ends a
    moment after its last word, and is held to at least `min_s` and to the
    time its characters need at `max_cps`, but never past the start of the
    next cue. A line with display text shows it; see `Line`.

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


def write_captions(narration, stem, *, line=None, **limits):
    """Write the captions as both WebVTT and SRT.

    Args:
        narration: The `Narration`.
        stem: Output path without a suffix; ``.vtt`` and ``.srt`` are added.
        line: The ``line:`` setting of the WebVTT file; see `write_vtt`.
        **limits: Keyword limits of `captions`.

    Returns:
        ``(vtt_path, srt_path)``.
    """
    stem = str(stem)
    return (
        write_vtt(narration, stem + ".vtt", line=line, **limits),
        write_srt(narration, stem + ".srt", **limits),
    )
