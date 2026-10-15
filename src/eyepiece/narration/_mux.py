"""Put the speech under a silent render, and write its captions.

`audio_track` places each line's audio at its speech start on the clip's
clock. `mux` joins that track with the video through ffmpeg without
re-encoding the picture. `write_srt` and `write_vtt` write the same script
as captions, one cue per sentence, timed from the word times. WebVTT is the
format an HTML ``<track>`` reads.
"""

import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from eyepiece._ffmpeg import resolve_ffmpeg

from ._timeline import MissingAudioError

_SENTENCE_END = re.compile(r"[.?!]$")
_AUDIO_SUFFIXES = (".opus", ".flac", ".wav")
_CAPTION_TAIL_S = 0.15
_DEFAULT_RATE = 24000


def _read_wav(path):
    with wave.open(str(path), "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono")
        rate = handle.getframerate()
        data = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return rate, data


def _decode(path, rate):
    """Read a 16-bit mono audio file; compressed formats go through ffmpeg."""
    if path.suffix == ".wav":
        return _read_wav(path)
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


def audio_track(narration, audio_dir, path, *, tail_s=0.0, tol_s=0.03):
    """Write the whole clip's speech as one 16-bit mono WAV.

    Args:
        narration: The `Narration`.
        audio_dir: Directory holding each line's audio file, beside its JSON.
            Ogg Opus, FLAC and WAV files are read.
        path: Output WAV path.
        tail_s: Silence added after the clip's end.
        tol_s: Most a file's length may differ from the duration its JSON
            records, in seconds. A larger difference means the audio is not
            the audio the cue times were measured on.

    Returns:
        The path written.

    Raises:
        MissingAudioError: If a line has no audio file.
        ValueError: If the files differ in sample rate, a file's length does
            not match its recorded duration, or a line runs past the end of
            the track.
    """
    clips, rate = [], None
    for beat in narration.beats:
        spoken = narration.spoken[beat.key]
        key = narration.script.line_key(narration.script.line(beat.key))
        file = _audio_file(audio_dir, key, spoken)
        if file is None:
            raise MissingAudioError(
                f"no audio for line {beat.key!r} ({key}) in {audio_dir}"
            )
        r, data = _decode(file, spoken.rate or _DEFAULT_RATE)
        if rate not in (None, r):
            raise ValueError("lines differ in sample rate")
        rate = r
        if abs(len(data) / r - spoken.duration_s) > tol_s:
            raise ValueError(
                f"line {beat.key!r}: {file.name} lasts {len(data) / r:.3f} s but "
                f"its timing says {spoken.duration_s:.3f} s; the audio and the "
                "cue times are from different syntheses"
            )
        clips.append((beat.key, beat.speech_start_s, data))
    if rate is None:
        raise ValueError("the script has no lines")
    out = np.zeros(round((narration.total_s + tail_s) * rate), dtype="<i2")
    for key, start, data in clips:
        i = round(start * rate)
        if i + len(data) > len(out) + 2:
            raise ValueError(f"line {key!r} runs past the end of the track")
        data = data[: len(out) - i]
        out[i : i + len(data)] = data
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(out.tobytes())
    return path


def mux(video, audio, path, *, bitrate="160k"):
    """Join a silent video and an audio track into one MP4.

    The picture is copied, not re-encoded; the audio is AAC. The output ends
    with the shorter of the two.

    Args:
        video: Silent MP4 from the render.
        audio: Audio file from `audio_track`.
        path: Output MP4 path.
        bitrate: AAC bitrate.

    Returns:
        The path written.

    Raises:
        RuntimeError: If no ffmpeg is found, or ffmpeg fails (its message is
            included).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        resolve_ffmpeg(), "-y", "-loglevel", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", bitrate,
        "-movflags", "+faststart", "-shortest", str(path),
    ]  # fmt: skip
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"ffmpeg failed to join {video} and {audio}: {done.stderr}")
    return path


def sentences(narration):
    """The script as sentences with their times.

    Returns:
        A list of ``(start_s, end_s, text)``. A sentence ends at a word that
        ends in a period, question mark or exclamation mark, or at the end
        of its line.
    """
    out = []
    for beat in narration.beats:
        group = []
        for word in beat.words:
            group.append(word)
            if _SENTENCE_END.search(word.text):
                out.append(group)
                group = []
        if group:
            out.append(group)
    return [(g[0].start_s, g[-1].end_s, " ".join(w.text for w in g)) for g in out]


def _captions(narration):
    """Sentences with caption end times, held a moment but never overlapping."""
    spans = sentences(narration)
    out = []
    for i, (a, b, text) in enumerate(spans):
        end = b + _CAPTION_TAIL_S
        if i + 1 < len(spans):
            end = min(end, spans[i + 1][0])
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


def write_srt(narration, path):
    """Write the script's captions as an SRT file.

    Each caption ends a moment after its sentence but never after the next
    one starts.

    Args:
        narration: The `Narration`.
        path: Output path.

    Returns:
        The path written.
    """
    lines = []
    for i, (a, b, text) in enumerate(_captions(narration), start=1):
        lines += [str(i), f"{_stamp(a, ',')} --> {_stamp(b, ',')}", text, ""]
    return _write(path, "\n".join(lines))


def write_vtt(narration, path):
    """Write the script's captions as a WebVTT file.

    The cues are those of `write_srt`. WebVTT, not SRT, is what an HTML
    ``<track kind="captions">`` reads.

    Args:
        narration: The `Narration`.
        path: Output path.

    Returns:
        The path written.
    """
    lines = ["WEBVTT", ""]
    for a, b, text in _captions(narration):
        lines += [f"{_stamp(a, '.')} --> {_stamp(b, '.')}", text, ""]
    return _write(path, "\n".join(lines))
