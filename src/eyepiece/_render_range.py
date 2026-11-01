"""Shared validation and absolute source-frame selection for excerpts."""

import math
from dataclasses import dataclass
from numbers import Real


def _range_bounds(time_range, fps, count=None, *, open_stop=False):
    """Return the half-open source frame interval, preserving full defaults."""
    if time_range is None:
        return 0, None if open_stop else count
    if not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("time_range requires positive finite fps")
    try:
        start, stop = time_range
    except (TypeError, ValueError):
        raise ValueError("time_range must be (start_s, stop_s)") from None
    start = 0.0 if start is None else start
    for value in (start, stop):
        if value is not None and (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("time_range endpoints must be finite nonnegative seconds")
    if stop is not None and stop <= start:
        raise ValueError("time_range stop must be after start")
    if count is not None:
        duration = count / fps
        if start >= duration or (stop is not None and stop > duration + 1e-9):
            raise ValueError("time_range is outside the source duration")
    first = math.ceil(start * fps - 1e-9)
    last = (
        (None if open_stop else count) if stop is None else math.ceil(stop * fps - 1e-9)
    )
    if count is not None and last is not None:
        last = min(count, last)
    if last is not None and last <= first:
        raise ValueError("time_range contains no source frames")
    return first, last


@dataclass(frozen=True)
class _FrameSlice:
    """A picklable sequence retaining the original indices and frame mapping."""

    frames: object
    frame_at: object
    first: int
    stop: int

    def __len__(self):
        return self.stop - self.first

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.source_item(index + self.first)

    def source_item(self, index):
        """Resolve an absolute index, including frame zero for layout setup."""
        if self.frame_at is not None:
            return self.frame_at(index)
        return index if isinstance(self.frames, int) else self.frames[index]
