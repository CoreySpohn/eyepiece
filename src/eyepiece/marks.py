"""Shaded intervals, labeled references, and zero rows on logarithmic axes."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cbook import normalize_kwargs
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, Locator

from eyepiece import _style
from eyepiece._result import PlotResult


def shade_intervals(
    x=None,
    mask=None,
    *,
    intervals=None,
    ax=None,
    edges="midpoint",
    log=False,
    orient="v",
    color=None,
    fill_kw=None,
):
    """Shade masked runs or explicit intervals across a single Axes.

    It was named ``spans`` before 0.16, a name that read like the timeline's
    `Timeline.spans` and `Plan.spans`; ``eyepiece.spans`` still resolves to
    this function with a `DeprecationWarning`.

    Args:
        x: Increasing one-dimensional sample positions. Unused in interval mode.
        mask: boolean, True marks samples to shade; shaped like x.
        intervals: Explicit (N, 2) lower/upper pairs, instead of a mask.
            Infinite ends clip to the axis limits at call time, so set the
            limits first.
        ax: Single Axes, or None to create a figure.
        edges: "midpoint" only. End bins extend by half the adjacent spacing;
            a lone sample gets half a unit (half a decade with log=True).
        log: Use geometric midpoint edges for positive sample positions.
        orient: "v" shades x intervals; "h" shades y intervals.
        color: Fill color; defaults to a faint neutral tone.
        fill_kw: Overrides passed to ax.axvspan or ax.axhspan.

    Returns:
        PlotResult with a "fill" list of span patches, in interval order.

    Example::

        import eyepiece as ep
        result = ep.shade_intervals([0, 1, 2, 3], [False, True, True, False])
    """
    if orient not in ("v", "h"):
        raise ValueError('orient must be "v" or "h"')
    if edges != "midpoint":
        raise ValueError('edges must be "midpoint"')
    if intervals is not None:
        if mask is not None:
            raise ValueError("provide either mask or intervals, not both")
        pairs = np.asarray(intervals, dtype=float)
        if pairs.size == 0:
            pairs = pairs.reshape(0, 2)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError(f"expected intervals shape (N, 2); received {pairs.shape}")
        if np.any(np.isnan(pairs)) or np.any(pairs[:, 0] > pairs[:, 1]):
            raise ValueError("intervals must have ordered, non-NaN endpoints")
    else:
        x = np.asarray(x, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        if x.ndim != 1 or mask.shape != x.shape:
            raise ValueError(
                f"expected x and mask shape (N,); received {x.shape}, {mask.shape}"
            )
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            raise ValueError("x must be finite and strictly increasing")
        if log and np.any(x <= 0):
            raise ValueError("log midpoint edges require positive x")
        positions = np.log10(x) if log else x
        if len(x) == 0:
            pairs = np.empty((0, 2))
        else:
            if len(x) == 1:
                bounds = np.array([positions[0] - 0.5, positions[0] + 0.5])
            else:
                mid = (positions[:-1] + positions[1:]) / 2
                bounds = np.r_[
                    2 * positions[0] - mid[0], mid, 2 * positions[-1] - mid[-1]
                ]
            if log:
                bounds = 10.0**bounds
            transitions = np.diff(np.r_[False, mask, False].astype(int))
            pairs = np.column_stack(
                (
                    bounds[np.flatnonzero(transitions == 1)],
                    bounds[np.flatnonzero(transitions == -1)],
                )
            )
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    limits = sorted(ax.get_xlim() if orient == "v" else ax.get_ylim())
    options = {
        "facecolor": _style.neutral(0.15) if color is None else color,
        "edgecolor": "none",
        "zorder": 0,
    }
    options.update(normalize_kwargs(fill_kw, Patch))
    draw = ax.axvspan if orient == "v" else ax.axhspan
    patches = []
    for low, high in pairs:
        low = np.clip(low, *limits) if np.isinf(low) else low
        high = np.clip(high, *limits) if np.isinf(high) else high
        if low <= high:
            patches.append(draw(low, high, **options))
    return PlotResult(ax, {"fill": patches})


def ref_line(
    ax,
    value,
    label=None,
    *,
    orient="h",
    where="end",
    color=None,
    line_kw=None,
    text_kw=None,
):
    """Draw a reference with its label anchored to an axes edge.

    Args:
        ax: Single Axes, or None to create a figure.
        value: Reference coordinate in data units.
        label: Optional label at the start or end of the reference.
        orient: "h" for a horizontal reference, "v" for vertical.
        where: "start" (axes coordinate 0) or "end" (axes coordinate 1).
        color: Reference color; defaults to a neutral tone.
        line_kw: Overrides passed to ax.axhline or ax.axvline.
        text_kw: Overrides passed to ax.text, including transform and alignment.

    Returns:
        PlotResult with "line" and, when labeled, "text". Text uses the
        y-axis transform for horizontal lines and the x-axis transform for
        vertical lines, so its edge coordinate stays fixed when limits change.

    Example::

        import eyepiece as ep
        result = ep.ref_line(None, 1.0, "reference")
    """
    if orient not in ("h", "v"):
        raise ValueError('orient must be "h" or "v"')
    if where not in ("start", "end"):
        raise ValueError('where must be "start" or "end"')
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    shade = _style.neutral(0.6) if color is None else color
    options = {"color": shade, "linestyle": "--"}
    options.update(normalize_kwargs(line_kw, Line2D))
    draw = ax.axhline if orient == "h" else ax.axvline
    line = draw(value, **options)
    artists = {"line": line}
    if label is not None:
        edge = 1.0 if where == "end" else 0.0
        horizontal = orient == "h"
        coords = (edge, value) if horizontal else (value, edge)
        options = {
            "color": line.get_color(),
            "clip_on": False,
            "bbox": _style.backing(),
            "transform": ax.get_yaxis_transform()
            if horizontal
            else ax.get_xaxis_transform(),
            "ha": ("left" if edge else "right") if horizontal else "center",
            "va": "center" if horizontal else ("bottom" if edge else "top"),
        }
        options = normalize_kwargs(options, Text)
        options.update(normalize_kwargs(text_kw, Text))
        artists["text"] = ax.text(*coords, label, **options)
    return PlotResult(ax, artists)


class _FloorLocator(Locator):
    """Retain logarithmic ticks above a separately labeled display floor."""

    def __init__(self, locator, floor):
        self.locator = locator
        self.floor = floor

    def set_axis(self, axis):
        super().set_axis(axis)
        self.locator.set_axis(axis)

    def __call__(self):
        return self.tick_values(*self.axis.get_view_interval())

    def tick_values(self, vmin, vmax):
        ticks = self.locator.tick_values(vmin, vmax)
        return np.r_[self.floor, ticks[ticks > self.floor]]


def log_zero_floor(ax, floor, *, axis="y", label="0", break_mark=True, mark_kw=None):
    """Give a log axis a zero row at a positive display floor.

    This decorates the axis only; callers map zero-valued data to floor.
    The floor becomes the lower numerical limit, with ordinary log ticks
    above it; the other limit stays free to follow data added later. The
    locator retains the zero label after later limit changes.

    Args:
        ax: Single Axes, or None to create a figure.
        floor: Finite positive display position for zero values.
        axis: "x" or "y".
        label: Tick text at floor.
        break_mark: Draw two short strokes through the bottom or left spine.
        mark_kw: Overrides passed to ax.plot for the spine strokes.

    Returns:
        PlotResult with a "lines" list of break strokes (empty if disabled).

    Example::

        import eyepiece as ep
        result = ep.log_zero_floor(None, 1e-4)
    """
    if axis not in ("x", "y"):
        raise ValueError('axis must be "x" or "y"')
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    target = ax.xaxis if axis == "x" else ax.yaxis
    set_scale = ax.set_xscale if axis == "x" else ax.set_yscale
    set_limits = ax.set_xlim if axis == "x" else ax.set_ylim
    set_scale("log")
    limits = target.get_view_interval()
    upper = max(max(limits), floor * 10)
    # Pin only the lower bound; an invisible sticky line keeps it there when
    # autoscaling later reruns, and leaves the other end free to follow data.
    if limits[0] < limits[1]:
        set_limits(floor, upper, auto=None)
    else:
        set_limits(upper, floor, auto=None)
    anchor = (ax.axvline if axis == "x" else ax.axhline)(floor, visible=False)
    (anchor.sticky_edges.x if axis == "x" else anchor.sticky_edges.y).append(floor)
    locator = target.get_major_locator()
    formatter = target.get_major_formatter()
    target.set_major_locator(_FloorLocator(locator, floor))
    target.set_major_formatter(
        FuncFormatter(
            lambda value, pos: label if value == floor else formatter(value, pos)
        )
    )
    lines = []
    if break_mark:
        # Data coordinate along the scale; axes coordinate across its spine.
        options = {
            "color": _style.neutral(0.6),
            "clip_on": False,
            "linewidth": 1.2,
            "transform": ax.get_xaxis_transform()
            if axis == "x"
            else ax.get_yaxis_transform(),
        }
        options.update(normalize_kwargs(mark_kw, Line2D))
        for factor in (1.15, 1.4):
            coords = ([-0.012, 0.012], [floor * factor, floor * factor * 1.12])
            if axis == "x":
                coords = coords[::-1]
            lines.extend(ax.plot(*coords, **options))
    return PlotResult(ax, {"lines": lines})
