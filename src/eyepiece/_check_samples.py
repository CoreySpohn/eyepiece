"""Call-scoped samples shared by the clip checks, with bounded raster reuse."""

import hashlib
import tempfile
import zlib
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.layout_engine import PlaceHolderLayoutEngine

_ACTIVE = ContextVar("eyepiece_check_samples", default=None)


class _FrameStore:
    """Append lossless rasters to disk with a bounded decoded-frame cache.

    Geometry and scalar observations stay in the trace. Pixels remain
    addressable for checks whose regions are known only after sampling.
    Consecutive equal frames share a record, after exact equality checking.
    """

    def __init__(self, max_bytes=32 * 1024 * 1024):
        self.file = None
        self.max_bytes = max_bytes
        self.records = []
        self.cache = OrderedDict()
        self.nbytes = 0
        self.disk_bytes = 0
        self.previous = None
        self.file = tempfile.TemporaryFile()

    @property
    def closed(self):
        return self.file is None or self.file.closed

    def append(self, frame):
        previous = self.previous
        if (
            previous is not None
            and previous.shape == frame.shape
            and previous.dtype == frame.dtype
            and np.array_equal(previous, frame)
        ):
            self.records.append(self.records[-1])
            return
        frame = np.ascontiguousarray(frame).copy()
        frame.flags.writeable = False
        data = zlib.compress(memoryview(frame), level=1)
        offset = self.disk_bytes
        self.file.seek(offset)
        self.file.write(data)
        self.disk_bytes += len(data)
        self.records.append((offset, len(data), frame.shape, frame.dtype.str))
        self.previous = frame

    def __len__(self):
        return len(self.records)

    def repeat(self):
        """Append the preceding record without decoding or copying it."""
        self.records.append(self.records[-1])

    def __getitem__(self, index):
        record = self.records[index]
        if record in self.cache:
            self.cache.move_to_end(record)
            return self.cache[record]
        offset, size, shape, dtype = record
        self.file.seek(offset)
        frame = np.frombuffer(zlib.decompress(self.file.read(size)), dtype=dtype)
        frame = frame.reshape(shape)
        if frame.nbytes <= self.max_bytes:
            self.cache[record] = frame
            self.nbytes += frame.nbytes
        while self.nbytes > self.max_bytes:
            _, removed = self.cache.popitem(last=False)
            self.nbytes -= removed.nbytes
        return frame

    def close(self):
        if self.file is not None:
            self.file.close()
        self.cache.clear()
        self.previous = None
        self.nbytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()


class _FrameView:
    """One stored frame without retaining the decoded raster."""

    def __init__(self, store, index):
        self.store, self.index = store, index

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index not in (0, -1):
            raise IndexError(index)
        return self.store[self.index]


def _reuse_key(fig, key, time, *, inspect_artists=False):
    """Return an explicit key only where draw-time layout cannot converge."""
    if key is None:
        return None
    value = key(time)
    if value is None:
        return None
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError("frame_key must return a hashable value or None") from exc
    engine = fig.get_layout_engine()
    if (
        type(fig.canvas) is not FigureCanvasAgg
        or "draw" in fig.canvas.__dict__
        or (engine is not None and not isinstance(engine, PlaceHolderLayoutEngine))
        or (
            inspect_artists
            and any(artist.get_agg_filter() is not None for artist in fig.findobj())
        )
    ):
        return None
    return value, fig.dpi, tuple(fig.bbox.bounds), id(fig.canvas)


def _draw_geometry(fig):
    """Refresh draw-time state, retaining custom canvas and filter behavior."""
    engine = fig.get_layout_engine()
    if (
        type(fig.canvas) is not FigureCanvasAgg
        or "draw" in fig.canvas.__dict__
        or (engine is not None and not isinstance(engine, PlaceHolderLayoutEngine))
        or any(artist.get_agg_filter() is not None for artist in fig.findobj())
    ):
        fig.canvas.draw()
    else:
        fig.draw_without_rendering()


class _RasterPool:
    """Intern exact rasters; hashes select candidates, equality verifies them."""

    def __init__(self, max_bytes=32 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.frames = OrderedDict()
        self.nbytes = 0

    def intern(self, frame, *, copy=True):
        frame = np.ascontiguousarray(frame)
        key = (frame.shape, frame.dtype.str, hashlib.sha256(frame).digest())
        old = self.frames.pop(key, None)
        if old is not None:
            self.nbytes -= old.nbytes
            if np.array_equal(old, frame):
                frame = old
        if frame is not old:
            if copy:
                frame = frame.copy()
            frame.flags.writeable = False
        if frame.nbytes <= self.max_bytes:
            self.frames[key] = frame
            self.nbytes += frame.nbytes
        while self.nbytes > self.max_bytes:
            _, removed = self.frames.popitem(last=False)
            self.nbytes -= removed.nbytes
        return frame


class _Samples:
    def __init__(self, fig, times, capture):
        self.fig = fig
        self.times = set(times)
        self.capture = capture
        self.time = None
        self.entries = {}
        self.frames = _FrameStore()


@contextmanager
def _share_samples(fig, times, capture):
    """Share only explicitly selected times within one absolute-frame callback."""
    active = _Samples(fig, times, capture)
    token = _ACTIVE.set(active)
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        active.frames.close()


def _select_sample(fig, time):
    active = _ACTIVE.get()
    if active is not None and active.fig is fig:
        active.time = time


def _draw_sample(fig):
    """Draw once and return captured observations when this call shares samples."""
    active = _ACTIVE.get()
    if active is None or active.fig is not fig or active.time not in active.times:
        fig.canvas.draw()
        return None
    engine = fig.get_layout_engine()
    if (
        type(fig.canvas) is not FigureCanvasAgg
        or (engine is not None and not isinstance(engine, PlaceHolderLayoutEngine))
        or any(artist.get_agg_filter() is not None for artist in fig.findobj())
    ):
        # A pose callback may change the canvas or enable live layout.
        # Repeated layout solves can converge differently: keep all passes.
        active.entries.clear()
        fig.canvas.draw()
        return None
    key = (active.time, fig.dpi, tuple(fig.bbox.bounds), id(fig.canvas))
    if key not in active.entries:
        fig.canvas.draw()
        entry = active.capture(active.time)
        trace = entry["trace"]
        active.frames.append(trace.frames[0])
        trace.frames = _FrameView(active.frames, len(active.frames) - 1)
        active.entries[key] = entry
    else:
        # Preserve draw-time artist geometry (text backings, connectors,
        # etc.) as well as the canvas contents, without rasterizing again.
        frame = active.entries[key]["trace"].frames[0]
        # The public method can clear the Agg buffer while obtaining its
        # renderer. Restore the sampled pixels after refreshing geometry.
        fig.draw_without_rendering()
        np.copyto(np.asarray(fig.canvas.buffer_rgba()), frame)
    return active.entries[key]
