"""Weighted draws drawn at their weights, with an optional resampling beat.

Importance sampling, a particle filter or a reweighted ensemble gives draws
that are not equally likely: each carries a weight. A mark that draws them
as equal dots says something false. `weighted_draws` draws each draw with
its area (or opacity) proportional to its weight, on one scale for the whole
clip, and can play the resampling that turns the weighted set into an equal
weight one. The arrays stay in NumPy: this is the I/O edge of a clip, not a
model.
"""

import dataclasses
from collections.abc import Callable

import numpy as np

from eyepiece._result import MotionContract, PlotResult

_ENCODINGS = ("area", "alpha")


@dataclasses.dataclass(frozen=True, eq=False)
class _WeightedResult(PlotResult):
    """A `PlotResult` that also reports the weights drawn."""

    weights: Callable | None = None
    counts: np.ndarray | None = None
    ess: float = 0.0


def _systematic(weights, seed):
    """Offspring counts of each draw under seeded systematic resampling."""
    n = weights.size
    rng = np.random.default_rng(seed)
    points = (rng.random() + np.arange(n)) / n
    edges = np.cumsum(weights)
    edges[-1] = 1.0  # rounding must not leave the last point unowned
    return np.bincount(np.searchsorted(edges, points, side="right"), minlength=n)


def weighted_draws(
    points,
    weights,
    *,
    ax=None,
    encode="area",
    max_size=None,
    resample=None,
    color=None,
    zorder=None,
    scatter_kw=None,
) -> PlotResult:
    """Draw weighted draws with each draw's weight as its area or opacity.

    The weights are drawn at their true values: a draw with twice the weight
    has twice the marker area (``encode="area"``) or twice the opacity
    (``encode="alpha"``), on one fixed scale for the whole clip. No weight is
    floored, capped or equalized for looks, so a draw with a weight near
    zero is drawn near invisible, which is what it is worth.

    With `resample`, the mark also plays the resampling beat: seeded
    systematic resampling draws `n` offspring from the `n` weighted draws,
    each with weight ``1 / n``. The resampled set is drawn on the same
    scale, each parent at the total weight of its offspring (its count over
    `n`), so a parent with three offspring is drawn three times as large as
    one with one, and a parent with none vanishes. ``update(u)`` moves every
    weight linearly from the weighted set (``u = 0``) to the resampled one
    (``u = 1``), and with ``encode="area"`` the area follows the weight, so
    the beat is a continuous change of size with no cut and no change of
    position. The rest state, ``u = 1``, is the resampled set (the weighted
    set without `resample`), and the mark is drawn in it.

    Heavier draws are drawn first, so a light draw is never hidden under a
    heavy neighbor; the scatter's points are in that order, not the input
    order (``weights`` reports in the input order).

    Motion contract: positions never change; only marker sizes (or
    opacities) do. ``contract.moves`` declares the scatter.

    Args:
        points: The draws' positions, an array shaped ``(n, 2)`` in data
            coordinates of `ax`.
        weights: One non-negative finite weight per draw, not all zero;
            normalized to sum to one.
        ax: Axes to draw on, or None for a new figure.
        encode: ``"area"`` or ``"alpha"``.
        max_size: Marker area in points squared of the largest weight drawn
            in the clip (with ``encode="alpha"``, every marker's area). None
            uses four times the default marker area.
        resample: A seed for the resampling beat, or None for no beat.
        color: Marker color; None uses the first palette color.
        zorder: Drawing order of the scatter.
        scatter_kw: Extra kwargs for ``ax.scatter``, applied last.

    Returns:
        A `PlotResult` with ``artists["scatter"]``, ``update(u, level=1.0)``
        (``u`` is the resampling beat, ignored without `resample`; ``level`` dims
        the mark), ``weights(u=None)``, the weight each draw is drawn at (at
        the last `u` set when None), ``counts``, the offspring count of each
        draw (None without `resample`), and ``ess``, the effective sample
        size ``1 / sum(w**2)`` of the normalized weights (for a readout: the
        number of equal-weight draws the weighted set is worth).

    Raises:
        ValueError: If `points` is not ``(n, 2)``, `weights` does not match
            it or is negative, non-finite or all zero, `encode` is unknown,
            or `max_size` is not positive.

    Example::

        mark = ep.weighted_draws(xy, likelihood, ax=ax, resample=3)
        for u in np.linspace(0.0, 1.0, 60):
            mark.update(ep.ease(u))
            rec.frame()
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from eyepiece import _style

    xy = np.asarray(points, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
        raise ValueError(
            f"weighted_draws points must be shaped (n, 2), got {np.shape(points)}"
        )
    n = len(xy)
    raw = np.asarray(weights, dtype=float).ravel()
    if raw.size != n:
        raise ValueError(f"weighted_draws has {raw.size} weights for {n} points")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0.0) or not raw.sum() > 0.0:
        raise ValueError(
            "weighted_draws weights must be finite, non-negative and not all zero"
        )
    if encode not in _ENCODINGS:
        raise ValueError(
            f"unknown weighted_draws encode: {encode!r}; known: {list(_ENCODINGS)}"
        )
    if max_size is None:
        max_size = 4.0 * float(plt.rcParams["lines.markersize"]) ** 2
    max_size = float(max_size)
    if not max_size > 0.0:
        raise ValueError(f"weighted_draws max_size must be positive, got {max_size}")
    start = raw / raw.sum()
    counts = None if resample is None else _systematic(start, resample)
    end = start if counts is None else counts / n
    # One scale for the clip: the largest weight of either state fills
    # max_size, so the beat never rescales the weights it compares.
    top = max(float(start.max()), float(end.max()))
    if ax is None:
        _, ax = plt.subplots()
    rgba = np.array(to_rgba(_style.color(0, color)))
    kw = {"lw": 0, **(scatter_kw or {})}
    if zorder is not None:
        kw["zorder"] = zorder
    # The heaviest draws go down first, so a light draw is never hidden
    # under a heavy neighbor's marker.
    order = np.argsort(-np.maximum(start, end), kind="stable")
    scatter = ax.scatter(xy[order, 0], xy[order, 1], s=max_size, color=rgba, **kw)
    state = {"u": 1.0}

    def drawn(u=None):
        u = state["u"] if u is None else min(max(float(u), 0.0), 1.0)
        return (1.0 - u) * start + u * end

    def update(u, level=1.0):
        u = 0.0 if counts is None else min(max(float(u), 0.0), 1.0)
        level = min(max(float(level), 0.0), 1.0)
        state["u"] = u
        share = drawn(u)[order] / top
        colors = np.tile(rgba, (n, 1))
        if encode == "area":
            scatter.set_sizes(max_size * share)
            colors[:, 3] *= level
        else:
            scatter.set_sizes(np.full(n, max_size))
            colors[:, 3] *= level * share
        scatter.set_facecolor(colors)
        scatter.set_edgecolor(colors)

    update(1.0)
    return _WeightedResult(
        ax=ax,
        artists={"scatter": scatter},
        update=update,
        contract=MotionContract(moves=(scatter,)),
        weights=drawn,
        counts=counts,
        ess=float(1.0 / np.sum(start**2)),
    )
