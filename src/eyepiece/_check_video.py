"""Stream timestamped RGB video frames without optional decoding packages."""

import math
import os
import re
import subprocess
from collections import deque
from decimal import Decimal
from fractions import Fraction
from queue import Empty, Full, Queue
from threading import Event, Thread

import numpy as np

from eyepiece._check_samples import _FrameStore
from eyepiece._ffmpeg import resolve_ffmpeg


def _microseconds(time):
    """Parse decimal seconds at ffmpeg's input-seek precision."""
    return int(Decimal(str(max(0.0, time))) * 1_000_000)


def _ticks(time, base):
    """Rescale microseconds to stream ticks with nearest, half-up rounding."""
    numerator = time * base.denominator
    denominator = 1_000_000 * base.numerator
    whole, remainder = divmod(numerator, denominator)
    return whole + (2 * remainder >= denominator)


def _video_frames(source, start=0.0):
    """Yield stream timestamps, time bases and RGB frames from one decoder."""
    args = [
        resolve_ffmpeg(), "-v", "info", "-nostdin", "-ss", f"{start:.6f}",
        "-i", os.fspath(source), "-map", "0:v:0", "-an", "-sn",
        "-vf", "showinfo", "-fps_mode", "passthrough",
        "-c:v", "ppm", "-pix_fmt", "rgb24", "-f", "image2pipe", "pipe:1",
    ]  # fmt: skip
    timestamps = Queue(maxsize=4)
    stopped = Event()
    errors = deque(maxlen=100)
    pattern = re.compile(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s+pts_time:")
    clock = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")

    def put(value):
        while not stopped.is_set():
            try:
                timestamps.put(value, timeout=0.1)
                return
            except Full:
                continue

    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:

        def read_errors():
            base = None
            try:
                for line in proc.stderr:
                    errors.append(line)
                    found = clock.search(line)
                    if found:
                        base = Fraction(int(found[1]), int(found[2]))
                    match = pattern.search(line)
                    if match and base is not None:
                        put((int(match[1]) + _ticks(_microseconds(start), base), base))
            except (OSError, ValueError):
                if not stopped.is_set():
                    raise
            finally:
                put(None)

        thread = Thread(target=read_errors, daemon=True)
        thread.start()
        try:
            while True:
                magic = proc.stdout.readline()
                if not magic:
                    break
                if magic != b"P6\n":
                    raise RuntimeError("ffmpeg returned an invalid RGB frame header")
                width, height = map(int, proc.stdout.readline().split())
                if proc.stdout.readline() != b"255\n":
                    raise RuntimeError("ffmpeg returned non-8-bit RGB pixels")
                data = proc.stdout.read(width * height * 3)
                if len(data) != width * height * 3:
                    raise RuntimeError("ffmpeg returned an incomplete RGB frame")
                while True:
                    try:
                        stamp = timestamps.get(timeout=0.1)
                        break
                    except Empty:
                        if not thread.is_alive():
                            raise RuntimeError(
                                "ffmpeg omitted a frame timestamp"
                            ) from None
                if stamp is None:
                    raise RuntimeError("ffmpeg omitted a frame timestamp")
                yield (
                    *stamp,
                    np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3),
                )
            if proc.wait():
                thread.join(timeout=5)
                raise subprocess.CalledProcessError(
                    proc.returncode, args, stderr=b"".join(errors)
                )
        finally:
            stopped.set()
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            thread.join(timeout=5)


def _source_samples(source, times):
    """Yield requested samples with the same frame selection as input seeks.

    Requests may repeat or be out of order. Selected frames are compressed
    to temporary storage so memory stays bounded, including for long clips.
    """
    from eyepiece.testing import _source_frame

    times = list(times)
    if callable(source) and not isinstance(source, (str, os.PathLike)):
        for t in times:
            yield _source_frame(source, t)
        return
    if not isinstance(source, (str, os.PathLike)):
        raise TypeError("source must be a video path or a callable source(t)")
    if any(not math.isfinite(t) for t in times):
        raise ValueError("sample times must be finite")
    if not times:
        return
    targets = sorted(set(_microseconds(t) for t in times))
    start = targets[0] / 1_000_000
    with _FrameStore() as frames:
        indices = {}
        for seek in [start, 0.0] if start else [0.0]:
            decoded = _video_frames(source, seek)
            try:
                current = next(decoded, None)
                if current is None:
                    continue
                exhausted = False
                for target in targets:
                    while current[0] < _ticks(target, current[1]) and not exhausted:
                        following = next(decoded, None)
                        if following is None:
                            exhausted = True
                        else:
                            current = following
                    indices[target] = len(frames)
                    frames.append(current[2])
                break
            finally:
                decoded.close()
        else:
            raise RuntimeError("video contains no decodable frames")
        for t in times:
            yield frames[indices[_microseconds(t)]]
