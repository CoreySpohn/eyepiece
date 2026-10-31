"""Call-scoped samples shared by the clip checks, with bounded raster reuse."""

import hashlib
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.layout_engine import PlaceHolderLayoutEngine

_ACTIVE = ContextVar("eyepiece_check_samples", default=None)


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


@contextmanager
def _share_samples(fig, times, capture):
    """Share only explicitly selected times within one absolute-frame callback."""
    token = _ACTIVE.set(_Samples(fig, times, capture))
    try:
        yield
    finally:
        _ACTIVE.reset(token)


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
        active.entries[key] = active.capture(active.time)
    else:
        # Preserve draw-time artist geometry (text backings, connectors,
        # etc.) as well as the canvas contents, without rasterizing again.
        frame = active.entries[key]["trace"].frames[0]
        np.copyto(np.asarray(fig.canvas.buffer_rgba()), frame)
        renderer = fig.canvas.get_renderer()
        with renderer._draw_disabled():
            fig.draw(renderer)
    return active.entries[key]
