"""Calibration comparisons and reproducible jittered sample strips."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cbook import normalize_kwargs
from matplotlib.lines import Line2D

from eyepiece import _style
from eyepiece._result import PlotResult
from eyepiece.marks import log_zero_floor
from eyepiece.stats import _hollow_points, interval_points


def calibration(
    stated,
    realized,
    *,
    ax=None,
    lo=None,
    hi=None,
    hollow=None,
    floor=None,
    scale="log",
    identity=True,
    tolerance=None,
    lim=None,
    point_kw=None,
    interval_kw=None,
    ref_kw=None,
):
    """Compare stated and realized values on identical axis limits.

    Args:
        stated: Finite one-dimensional stated values (x).
        realized: Finite realized values (y), shaped like stated.
        ax: Single Axes, or None to create a figure.
        lo: Absolute lower endpoints on realized values, scalar or (N,).
        hi: Absolute upper endpoints on realized values, scalar or (N,).
        hollow: Boolean (N,) mask selecting hollow markers; None fills all.
        floor: Positive display coordinate for zeros on both log axes,
            including interval endpoints. None requires strictly positive
            log data. Only zeros are mapped, never positive values.
        scale: "log" (default) or "linear" (useful for probabilities).
        identity: Draw a neutral identity line spanning the shared limits.
        tolerance: Nonnegative absolute half-width of a band around identity.
            The band is clipped to the shared display limits.
        lim: Optional (lo, hi) shared limits, e.g. (0, 1). None uses the
            current limits, widened to show this call's data.
        point_kw: Overrides passed to ax.scatter through interval_points. A
            color array such as {"color": rgba_array} colors points one by
            one and works together with hollow.
        interval_kw: Overrides passed to interval_points' interval lines.
        ref_kw: Overrides passed to ax.plot for the identity line.

    Returns:
        PlotResult with "scatter", "lines" (intervals followed by any
        x/y spine breaks), optionally "line" (identity), and optionally
        "fill" (tolerance band). Limits agree without joining shared axes.

    Example::

        import eyepiece as ep
        result = ep.calibration([0.1, 0.5, 0.9], [0.12, 0.48, 0.88], scale="linear")
    """
    stated, realized = np.asarray(stated, float), np.asarray(realized, float)
    if stated.ndim != 1 or realized.shape != stated.shape:
        raise ValueError(
            "expected stated and realized shape (N,); "
            f"received {stated.shape}, {realized.shape}"
        )
    if scale not in ("log", "linear"):
        raise ValueError('scale must be "log" or "linear"')
    if floor is not None and (scale != "log" or not np.isfinite(floor) or floor <= 0):
        raise ValueError("floor must be finite and positive and requires scale='log'")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    mask = np.zeros(stated.shape, bool) if hollow is None else np.asarray(hollow, bool)
    if mask.shape != stated.shape:
        raise ValueError(f"expected hollow shape {stated.shape}; received {mask.shape}")

    def display(values):
        if values is None:
            return None
        values = np.broadcast_to(np.asarray(values, float), stated.shape)
        if not np.all(np.isfinite(values)):
            raise ValueError("calibration values and endpoints must be finite")
        if scale == "log":
            if np.any(values < 0) or (floor is None and np.any(values == 0)):
                raise ValueError("log values must be positive; provide floor for zeros")
            if floor is not None:
                values = np.where(values == 0, floor, values)
        return values

    x, y, lower, upper = map(display, (stated, realized, lo, hi))
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    result = interval_points(
        x,
        y,
        lower,
        upper,
        ax=ax,
        hollow_zero=False,
        point_kw=point_kw,
        interval_kw=interval_kw,
    )
    if np.any(mask):
        _hollow_points(result.artists["scatter"], mask)
    artists = result.artists.copy()
    limits = (min(*ax.get_xlim(), *ax.get_ylim()), max(*ax.get_xlim(), *ax.get_ylim()))
    # Limits frozen by an earlier call or the caller must still show new data.
    ends = np.concatenate([a.ravel() for a in (x, y, lower, upper) if a is not None])
    ends = ends[np.isfinite(ends)]
    if ends.size and lim is None:
        low, high = ends.min(), ends.max()
        if scale == "log":
            pad = (high / low) ** 0.05 if high > low else 1.0
            low, high = low / pad, high * pad
        else:
            pad = 0.05 * (high - low)
            low, high = low - pad, high + pad
        limits = (
            low if ends.min() < limits[0] else limits[0],
            high if ends.max() > limits[1] else limits[1],
        )
    if lim is not None:
        limits = tuple(float(v) for v in lim)
        if (
            len(limits) != 2
            or not np.all(np.isfinite(limits))
            or limits[0] >= limits[1]
            or (scale == "log" and limits[0] <= 0)
        ):
            raise ValueError("lim must be finite, increasing, and positive on log")
    if floor is not None:
        limits = (floor, max(limits[1], floor * 10))
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    if floor is not None:
        for axis in ("x", "y"):
            marks = log_zero_floor(ax, floor, axis=axis)
            artists["lines"].extend(marks.artists["lines"])
    # Drawing guides must not feed back into later autoscaling.
    data_limits = ax.dataLim.frozen()
    if tolerance is not None:
        # Include every kink of the clipped band, so straight edges are exact.
        grid = np.unique(
            np.clip([*limits, limits[0] + tolerance, limits[1] - tolerance], *limits)
        )
        if scale == "log":
            grid = np.unique(np.r_[grid, np.geomspace(*limits, 200)])
        artists["fill"] = ax.fill_between(
            grid,
            np.clip(grid - tolerance, *limits),
            np.clip(grid + tolerance, *limits),
            color=_style.neutral(0.15),
            zorder=0,
        )
    if identity:
        options = {"color": _style.neutral(0.6), "linestyle": "--", "zorder": 1}
        options.update(normalize_kwargs(ref_kw, Line2D))
        artists["line"] = ax.plot(limits, limits, **options)[0]
    ax.dataLim.set(data_limits)
    return PlotResult(ax, artists)


def strip(
    groups,
    *,
    ax=None,
    positions=None,
    labels=None,
    width=0.28,
    seed=0,
    color=None,
    capped=None,
    point_kw=None,
):
    """Draw reproducibly jittered groups, with triangles for capped values.

    Args:
        groups: Sequence of one-dimensional sample arrays, or an ordered
            mapping from labels to sample arrays. NaN and infinite values
            are omitted, except positive infinity when capped is supplied.
        ax: Single Axes, or None to create a figure.
        positions: Finite (G,) column centers; None uses 0 through G - 1.
        labels: Optional (G,) tick labels; defaults to mapping keys, if any.
        width: Nonnegative total width of uniform horizontal jitter.
        seed: Seed for a local numpy generator, leaving global state alone.
        color: Shared color override; None assigns a palette slot per group.
        capped: Finite cap. Values at or above it appear there as triangles.
        point_kw: Overrides passed to ax.scatter for each group and marker.

    Returns:
        PlotResult with a "scatter" list in group order: one collection per
        group, or two (ordinary then capped) per group when capped is set.
        Empty collections retain their positions in this list.

    Example::

        import eyepiece as ep
        result = ep.strip({"a": [0.1, 0.2], "b": [0.3, 2.0]}, capped=1.0)
    """
    if isinstance(groups, Mapping):
        if labels is None:
            labels = list(groups)
        groups = list(groups.values())
    groups = [np.asarray(group, float) for group in groups]
    if any(group.ndim != 1 for group in groups):
        raise ValueError("expected each group shape (N,); received a non-1D group")
    positions = (
        np.arange(len(groups), dtype=float)
        if positions is None
        else np.asarray(positions, float)
    )
    if positions.shape != (len(groups),):
        raise ValueError(
            f"expected positions shape {(len(groups),)}; received {positions.shape}"
        )
    if labels is not None and len(labels) != len(groups):
        raise ValueError(f"expected {len(groups)} labels; received {len(labels)}")
    if not np.all(np.isfinite(positions)) or not np.isfinite(width) or width < 0:
        raise ValueError("positions must be finite and width finite and nonnegative")
    if capped is not None and not np.isfinite(capped):
        raise ValueError("capped must be finite")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    rng = np.random.default_rng(seed)
    scatters = []
    for i, (position, values) in enumerate(zip(positions, groups, strict=True)):
        jitter = position + rng.uniform(-width / 2, width / 2, len(values))
        is_capped = np.zeros(len(values), bool) if capped is None else values >= capped
        masks = [(np.isfinite(values) & ~is_capped, "o")]
        if capped is not None:
            masks.append((is_capped, "^"))
        for mask, marker in masks:
            options = {"color": _style.color(i, color), "marker": marker}
            options.update(point_kw or {})
            if "c" in options:
                options.pop("color", None)
            y = np.full(mask.sum(), capped) if marker == "^" else values[mask]
            scatters.append(ax.scatter(jitter[mask], y, **options))
    if labels is None:
        ax.set_xticks(positions)
    else:
        ax.set_xticks(positions, labels)
    return PlotResult(ax, {"scatter": scatters})
