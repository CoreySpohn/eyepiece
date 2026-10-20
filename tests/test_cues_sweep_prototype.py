"""The cue sounds prototype: a short sweep MP4 with its count heard."""

import json
import runpy
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SWEEP = runpy.run_path(str(ROOT / "docs" / "narration" / "cues_sweep" / "sweep.py"))
DURATION_S = 6.0
RATE = 48000

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the prototype writes MP4s through ffmpeg, and ffprobe reads them",
)


def _streams(path):
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(done.stdout)["streams"]


def _samples(path):
    """The decoded audio as 16-bit mono samples at 48 kHz."""
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
         "-ar", str(RATE), "-"],
        capture_output=True, check=True,
    )  # fmt: skip
    return np.frombuffer(done.stdout, dtype="<i2").astype(float)


def _video_md5(path):
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v", "-c", "copy",
         "-f", "md5", "-"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    return done.stdout.strip()


def _ticks(x):
    """Onsets of 2.4 kHz bursts: band-pass 1.8 to 3.2 kHz, then rising edges."""
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / RATE)
    spectrum[(freqs < 1800.0) | (freqs > 3200.0)] = 0.0
    band = np.abs(np.fft.irfft(spectrum, len(x)))
    n = round(0.001 * RATE)
    envelope = np.convolve(band, np.ones(n) / n, mode="same")
    above = envelope > 0.3 * envelope.max()
    edges = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    keep = [e for i, e in enumerate(edges) if i == 0 or e - edges[i - 1] > 0.003 * RATE]
    return np.array(keep)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("sweep")
    paths = {
        "a": SWEEP["build"](out / "a.mp4", duration_s=DURATION_S, dpi=20),
        "b": SWEEP["build"](out / "b.mp4", duration_s=DURATION_S, dpi=20),
        "silent": SWEEP["build"](
            out / "silent.mp4", duration_s=DURATION_S, dpi=20, cues=False
        ),
    }
    return paths


def test_one_audio_and_one_video_stream_of_one_length(built):
    streams = _streams(built["a"])
    kinds = sorted(s["codec_type"] for s in streams)
    assert kinds == ["audio", "video"]
    video = next(s for s in streams if s["codec_type"] == "video")
    frame_s = 1.0 / SWEEP["FPS"]
    assert float(video["duration"]) == pytest.approx(DURATION_S, abs=frame_s)
    audio_s = len(_samples(built["a"])) / RATE
    assert audio_s == pytest.approx(float(video["duration"]), abs=frame_s)


def test_two_builds_have_the_same_audio_samples(built):
    assert _samples(built["a"]).tobytes() == _samples(built["b"]).tobytes()


def test_one_tick_per_pixel_read(built):
    plan = SWEEP["make_plan"](DURATION_S)
    flashes = sum(
        round(max(plan.value(k, f"read.{name}") for k in range(plan.n_frames)))
        for name in SWEEP["PIXELS"]
    )
    assert flashes == 100
    events = SWEEP["cue_events"](plan)
    assert len(events.tick_times) == flashes
    found = _ticks(_samples(built["a"]))
    assert len(found) == flashes
    # Each tick lands within one video frame (plus the 1 ms jitter) of its flash.
    lag = found / RATE - np.array(events.tick_times)
    assert np.max(np.abs(lag)) < 1.0 / SWEEP["FPS"] + 0.001


def test_the_silent_variant_has_the_same_picture_and_no_ticks(built):
    assert _video_md5(built["silent"]) == _video_md5(built["a"])
    silent = _samples(built["silent"])
    assert len(silent) == len(_samples(built["a"]))
    assert np.max(np.abs(silent)) < 32767 * 10 ** (-40 / 20)
