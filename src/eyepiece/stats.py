"""Sample primitives: corner plots, histograms, covariance ellipses, and traces.

`convergence` draws a sequence of samples beside its running mean or running
sum, converging toward labeled reference values, and reveals the samples one
at a time for an animation. `signed_trace` draws a sequence about a
reference level, shaded one color above it and another below, and
`hist_fill` a histogram that fills as samples arrive on a pinned scale; both
reveal their samples one at a time the same way.

`corner` and `corner_overlay` are a deliberate parity port of the classic
triangle-plot idiom rather than a wrapper around a third-party corner-plot
package: overlaying a second dataset into a caller's existing axes, and
branding through the style library, both go against that kind of package's
API grain. `labels` is a plain dict argument on both -- the parameter-name
table belongs to whichever consumer owns those parameter names, not to this
library.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from eyepiece import _style
from eyepiece._phasor import _is_linestyle, _is_number
from eyepiece._result import MosaicResult, PlotResult


def _corner_axes(n, axes):
    """Create an (n, n) axes grid, or reshape a caller-provided one.

    Returns:
        `(created, fig, axes)` where `created` is whether a new figure was
        made.
    """
    created = axes is None
    if created:
        fig, axes = plt.subplots(n, n, layout="constrained", squeeze=False)
    else:
        axes = np.asarray(axes).reshape(n, n)
        fig = axes[0, 0].figure
    return created, fig, axes


def _hide_upper(axes, n):
    """Hide the (j > i) upper-triangle cells of an (n, n) corner grid."""
    for i in range(n):
        for j in range(i + 1, n):
            axes[i, j].set_visible(False)


def _finish_cell(ax, i, j, n, params, labels):
    """Apply the shared tick/label/grid convention to one corner cell."""
    ax.grid(False)
    if i == n - 1:
        ax.set_xlabel(labels.get(params[j], params[j]))
    else:
        ax.set_xticklabels([])
    if j == 0 and i > 0:
        ax.set_ylabel(labels.get(params[i], params[i]))
    elif i != j:
        ax.set_yticklabels([])


def corner(
    samples,
    params=None,
    *,
    truths=None,
    labels=None,
    color=None,
    bins=30,
    title=None,
    axes=None,
):
    """Draw a triangle plot: 1D histograms on the diagonal, 2D density below.

    Args:
        samples: Dict mapping a parameter name to a 1D array of samples.
        params: Ordered parameter names to show. None uses every key of
            `samples`, in dict order.
        truths: Optional dict of true values, drawn as dashed guide lines.
            A parameter missing from `truths` is drawn without a guide.
        labels: Optional dict mapping a parameter name to its axis label.
            A parameter missing from `labels` falls back to its own name.
        color: Histogram/density color override; None uses `_style.color(0)`.
        bins: Bin count for both the 1D and 2D histograms.
        title: Optional figure suptitle, applied when `corner` creates its
            own figure. Mutually exclusive with `axes`: passing both raises
            `ValueError`, since a caller-supplied grid may share its figure
            with other content that a suptitle would overwrite.
        axes: An `(n, n)` array of Axes to draw into. `corner` hides the
            upper triangle via `set_visible(False)` unconditionally,
            whether it created the axes or was handed them. None creates a
            new figure.

    Returns:
        A `MosaicResult` whose `axes` is the `(n, n)` grid, upper triangle
        hidden via `set_visible(False)`. `artists["hist"]` is the list of
        diagonal step-histogram patch lists (a `histtype="step"` histogram
        returns a list of `Polygon`, not a `BarContainer`) and
        `artists["collection"]` the list of lower-triangle density meshes,
        each a `QuadMesh`, in row-major `(i, j)` order with `j <= i`.
        `artists["line"]` collects the truth guide lines, when `truths` is
        given.

    Note:
        The diagonal histograms are normalized to unit area (`density=True`),
        not drawn as raw counts, so `corner_overlay` can lay a second dataset
        onto the same diagonal at a comparable scale and so that datasets of
        different sample size compare on shape rather than on how many draws
        each happens to carry. The diagonal's y ticks are removed either way:
        the height of a marginal is not a quantity the reader is meant to
        read off.

    Raises:
        ValueError: If both `title` and `axes` are given.
    """
    if title is not None and axes is not None:
        raise ValueError(
            "title is not supported when axes is provided; a caller-supplied "
            "grid may share the figure with other content"
        )
    labels = labels or {}
    if params is None:
        params = list(samples)
    n = len(params)
    data = [np.asarray(samples[p]) for p in params]

    hist_color = _style.color(0, color)
    truth_color = _style.color(1)
    cmap = _style.cmap("intensity")

    _, fig, axes = _corner_axes(n, axes)
    _hide_upper(axes, n)

    hists = []
    density = []
    truth_lines = []
    for i in range(n):
        for j in range(i + 1):
            ax = axes[i, j]
            if i == j:
                # density, not counts: `corner_overlay` normalizes the same
                # way, so an overlay lands on the diagonal at the scale the
                # underlying histogram set. Counts here put the two curves
                # orders of magnitude apart and flatlined the overlay.
                _, _, patches = ax.hist(
                    data[i],
                    bins=bins,
                    color=hist_color,
                    histtype="step",
                    density=True,
                )
                hists.append(patches)
                if truths is not None and params[i] in truths:
                    truth_lines.append(
                        ax.axvline(truths[params[i]], color=truth_color, ls="--")
                    )
                ax.set_yticks([])
            else:
                _, _, _, mesh = ax.hist2d(data[j], data[i], bins=bins, cmap=cmap)
                density.append(mesh)
                if truths is not None:
                    if params[j] in truths:
                        truth_lines.append(
                            ax.axvline(truths[params[j]], color=truth_color, ls="--")
                        )
                    if params[i] in truths:
                        truth_lines.append(
                            ax.axhline(truths[params[i]], color=truth_color, ls="--")
                        )
            _finish_cell(ax, i, j, n, params, labels)

    if title:
        fig.suptitle(title)

    artists = {"hist": hists, "collection": density}
    if truth_lines:
        artists["line"] = truth_lines
    return MosaicResult(axes=axes, artists=artists)


def corner_overlay(
    datasets, params=None, *, axes=None, colors=None, labels=None, names=None, bins=30
):
    """Overlay several sample sets on one corner plot.

    Draws into caller-provided `axes` (e.g. a `corner` result's `axes`) when
    given, so a second dataset can be laid over an existing triangle plot
    without redrawing it.

    Args:
        datasets: List of sample dicts, one per overlay, each mapping a
            parameter name to a 1D array of samples.
        params: Ordered parameter names to show. None uses every key of
            `datasets[0]`, in dict order.
        axes: An `(n, n)` array of Axes to draw into. None creates a new
            figure and hides its upper triangle.
        colors: Optional list of per-dataset color overrides, same length as
            `datasets`. None defaults every dataset to `_style.color(i)`.
        labels: Optional dict mapping a parameter name to its axis label.
        names: Optional list of per-dataset legend labels, same length as
            `datasets`. None labels them "dataset 0", "dataset 1", ...
        bins: Bin count for the diagonal step histograms.

    Returns:
        A `MosaicResult` whose `axes` is the `(n, n)` grid. The top-right
        upper-triangle cell hosts the dataset legend and is made visible
        (with its ticks and spines turned off) for that purpose, even when
        it was hidden by an earlier `corner` call. `artists["scatter"]` is
        the list of lower-triangle `PathCollection` artists and
        `artists["hist"]` the list of diagonal step-histogram patch lists
        (each a list of `Polygon`, as `histtype="step"` draws), in the order
        drawn (dataset-major, then row-major `(i, j)` with `j <= i`).
    """
    labels = labels or {}
    if params is None:
        params = list(datasets[0])
    n = len(params)
    resolved_colors = [
        _style.color(k, colors[k] if colors is not None else None)
        for k in range(len(datasets))
    ]
    resolved_names = (
        names if names is not None else [f"dataset {k}" for k in range(len(datasets))]
    )

    created, _fig, axes = _corner_axes(n, axes)
    if created:
        _hide_upper(axes, n)

    scatters = []
    hists = []
    for k, samp in enumerate(datasets):
        c = resolved_colors[k]
        for i in range(n):
            for j in range(i + 1):
                ax = axes[i, j]
                yi = np.asarray(samp[params[i]])
                if i == j:
                    _, _, patches = ax.hist(
                        yi, bins=bins, color=c, histtype="step", density=True
                    )
                    hists.append(patches)
                else:
                    xj = np.asarray(samp[params[j]])
                    scatters.append(
                        ax.scatter(xj, yi, s=4, color=c, alpha=0.3, edgecolors="none")
                    )
                _finish_cell(ax, i, j, n, params, labels)

    if n > 1:
        legend_ax = axes[0, n - 1]
        legend_ax.set_visible(True)
        legend_ax.axis("off")
        handles = [
            plt.Line2D([], [], color=resolved_colors[k], lw=2, label=resolved_names[k])
            for k in range(len(datasets))
        ]
        legend_ax.legend(handles=handles, loc="center")

    return MosaicResult(axes=axes, artists={"scatter": scatters, "hist": hists})


def hist_vs_pdf(
    samples, pdf, *, ax=None, bins=50, log=False, label=None, line_kw=None, hist_kw=None
):
    """Draw a normalized histogram against an analytic PDF curve.

    Args:
        samples: 1D array-like of samples.
        pdf: Callable evaluated over a linspace spanning the sample range,
            e.g. `lambda x: scipy.stats.norm.pdf(x, mu, sigma)`.
        ax: Axes to draw into. None creates a new figure and axes.
        bins: Histogram bin count.
        log: Whether to set a log y-scale.
        label: Optional histogram legend label.
        line_kw: Extra kwargs passed to `ax.plot` for the PDF curve.
        hist_kw: Extra kwargs passed to `ax.hist`, applied last.

    Returns:
        A `PlotResult` with artists `"hist"` (the `BarContainer` of filled
        bars) and `"line"` (the PDF `Line2D`).
    """
    data = np.asarray(samples)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    hkw = {"color": _style.color(0), "alpha": 0.5, "label": label, **(hist_kw or {})}
    _, _, hist = ax.hist(data, bins=bins, density=True, **hkw)

    x = np.linspace(float(data.min()), float(data.max()), 500)
    lkw = {"color": _style.color(1), **(line_kw or {})}
    (line,) = ax.plot(x, pdf(x), **lkw)

    if log:
        ax.set_yscale("log")

    return PlotResult(ax=ax, artists={"hist": hist, "line": line})


def cov_ellipse(
    mean2,
    cov22,
    *,
    ax=None,
    n_sigma=1,
    color=None,
    label=None,
    ellipse_kw=None,
    text_kw=None,
):
    """Draw an n-sigma covariance ellipse from a 2D mean and covariance.

    Args:
        mean2: Length-2 array-like center, in axis units.
        cov22: `(2, 2)` covariance matrix, in axis units.
        ax: Axes to draw into. None creates a new figure and axes.
        n_sigma: Number of standard deviations the ellipse spans; the
            ellipse's width and height scale linearly with this.
        color: Edge color override; None uses `_style.color(0)`.
        label: Text naming the interval, written on the ellipse itself.
            True writes "<n> sigma" from `n_sigma`; a string is used as
            given; None writes nothing.

            An ellipse is a claim about an interval, and an interval nobody
            names is not one a reader can check: the same outline can be one
            standard deviation, two, or a 95 percent region, and the drawing
            is identical. Nothing on a bare ellipse distinguishes them, so a
            reader either asks or assumes. The label goes on the curve rather
            than into a legend because that is where the reader is looking
            when the question occurs to them.
        ellipse_kw: Extra kwargs passed to `Ellipse`, applied last.
        text_kw: Extra kwargs passed to the label's `ax.annotate`.

    Returns:
        A `PlotResult` with artist `"ellipse"` (the `Ellipse` patch, already
        added to `ax`) and, when `label` is given, `"text"` (the `Text`).
    """
    mean = np.asarray(mean2, dtype=float)
    cov = np.asarray(cov22, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))

    ekw = {"fill": False, "color": _style.color(0, color), **(ellipse_kw or {})}
    ell = Ellipse(
        mean,
        2.0 * n_sigma * np.sqrt(vals[1]),
        2.0 * n_sigma * np.sqrt(vals[0]),
        angle=angle,
        **ekw,
    )
    ax.add_patch(ell)

    artists = {"ellipse": ell}
    if label is not None and label is not False:
        text = f"{n_sigma:g} sigma" if label is True else str(label)
        # the end of the semi-major axis, so the label sits on the curve
        # rather than floating beside it
        major = n_sigma * np.sqrt(vals[1]) * vecs[:, 1]
        tip = mean + (major if major[1] >= 0 else -major)
        tkw = {
            "fontsize": 8,
            "color": ekw.get("color"),
            "ha": "center",
            "va": "bottom",
            "textcoords": "offset points",
            "xytext": (0, 3),
            **(text_kw or {}),
        }
        artists["text"] = ax.annotate(text, xy=tuple(tip), **tkw)

    return PlotResult(ax=ax, artists=artists)


# What `convergence` draws as its line.
_RUNNING = ("mean", "sum", None)

# Where `convergence` writes a reference's label: (vertical alignment, sign of
# the vertical offset).
_LABEL_SIDES = {"above": ("bottom", 1.0), "below": ("top", -1.0)}

# Fraction of the x span added right of the last sample when references are
# labeled, so the labels sit past the data rather than over it.
_LABEL_ROOM = 0.2


def _per_ref(value, n, name, single=_is_number):
    """Broadcast one value (or None) or a length-`n` sequence to a list.

    `single` decides whether `value` is one value for every reference; no
    array is built from `value`, so a sequence may mix kinds of entry.
    """
    if value is None or single(value):
        return [value] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"convergence {name} has {len(values)} entries for {n} refs")
    return values


def _is_label(value):
    return isinstance(value, str)


def _running(values, running):
    if running == "mean":
        return np.cumsum(values) / np.arange(1, values.size + 1)
    if running == "sum":
        return np.cumsum(values)
    return values.copy()


def _padded(lo, hi, frac):
    span = hi - lo
    if span == 0.0:
        span = max(abs(lo), 1.0)
    return lo - frac * span, hi + frac * span


def convergence(
    values,
    *,
    ax=None,
    x=None,
    refs=None,
    ref_labels=None,
    show_samples=True,
    band=None,
    running="mean",
    color=None,
    sample_color=None,
    ref_linestyles=None,
    label_side="above",
    scatter_kw=None,
    line_kw=None,
    ref_kw=None,
    fill_kw=None,
    text_kw=None,
):
    """Draw samples beside their running mean or sum, converging on references.

    Each sample is a dot, and the line is the mean (or sum) of the samples
    up to that point, so the reader watches a scattered quantity settle onto
    the value it is converging to. Each reference is a horizontal line across
    the panel, labeled at its right end on a backing box, and a tolerance
    band can be shaded around it. The samples and the line are data; the
    references, their labels, and the bands are scenery in neutral tones.

    The limits are set from the full data on the first draw -- every sample
    (when shown), the whole running line, the references, and the bands --
    and `update` never changes them, so an animation that reveals the
    samples one at a time keeps one scale throughout. On axes that already
    hold data, the limits grow to cover both, so two calls (two curves
    converging to two references, say) share one panel.

    Args:
        values: 1D array-like of the samples, in order.
        ax: Axes to draw into. None creates a new figure and axes.
        x: Position of each sample along the horizontal axis. None uses
            `1, 2, ..., n`, the number of samples taken so far.
        refs: A reference value or a sequence of them, each drawn as a
            horizontal line. None draws none.
        ref_labels: One label per reference (an entry of None leaves that
            reference unlabeled), written at the panel's right edge, just
            `label_side` of its line. When any reference is labeled the
            right limit leaves a fifth of the x span past the last sample,
            so the labels sit beyond the data.
        show_samples: Draw the samples as dots.
        band: Half-width of a tolerance band shaded around the references:
            one value for every reference, or one per reference (an entry of
            None draws no band there). None draws no band.
        running: What the line shows: `"mean"`, the mean of the first k
            samples at sample k; `"sum"`, their sum; or None, the values
            themselves, for a curve the caller already accumulated (usually
            with `show_samples=False`).
        color: Color of the line. None uses `_style.color(0)`.
        sample_color: Color of the dots. None uses a neutral tone, so the
            line is the data the eye follows.
        ref_linestyles: One line style for every reference or one per
            reference, each a name such as `":"` or a dash pattern such as
            `(0, (3, 2))`, in any mix. None draws them dashed.
        label_side: `"above"` or `"below"` the reference line.
        scatter_kw: Extra kwargs for `ax.scatter`, applied last.
        line_kw: Extra kwargs for the line's `ax.plot` call, applied last.
            A head marker that follows the revealed end of the line is
            `{"marker": "o", "markevery": [-1]}`.
        ref_kw: Extra kwargs for every reference's `ax.axhline`, applied
            last.
        fill_kw: Extra kwargs for every band `Rectangle`, applied last.
        text_kw: Extra kwargs for every reference label's `Text`, applied
            last.

    Returns:
        A `PlotResult` with artists `"line"` (the running `Line2D`),
        `"scatter"` (the samples' `PathCollection`, when shown), `"lines"`
        (the reference `Line2D` list, when any), `"fill"` (the band
        `Rectangle` list, in reference order, when any), and `"text"` (the
        label list, in reference order, when any), and an
        `update(k=None, *, values=None, refs=None)` that reveals the first
        `k` samples and the line up to them, with no new artist. `values`
        replaces the samples (same count) and recomputes the line; `refs`
        moves the references (same count), their labels, and their bands. A
        `k` of None keeps the current count, which starts at every sample.
        The limits never change.

    Raises:
        ValueError: If `values` is empty, `x` does not match it, `running`
            or `label_side` is unknown, a per-reference argument does not
            match `refs`, `band` is given without `refs`, or `update`
            receives a different number of values or references.

    Example::

        rng = np.random.default_rng(0)
        looks = rng.exponential(1.0, 200)
        res = ep.convergence(looks, refs=[1.0], ref_labels=["expected"], band=0.1)
        for k in range(1, 201):
            res.update(k)
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n == 0:
        raise ValueError("convergence needs at least one value")
    if running not in _RUNNING:
        raise ValueError(
            f"unknown convergence running: {running!r}; known: {list(_RUNNING)}"
        )
    if label_side not in _LABEL_SIDES:
        raise ValueError(
            f"unknown convergence label_side: {label_side!r}; "
            f"known: {list(_LABEL_SIDES)}"
        )
    x = np.arange(1.0, n + 1.0) if x is None else np.asarray(x, dtype=float).ravel()
    if x.size != n:
        raise ValueError(f"convergence x has {x.size} entries for {n} values")
    if refs is None:
        ref_values = []
    elif np.ndim(refs) == 0:
        ref_values = [float(refs)]
    else:
        ref_values = [float(r) for r in refs]
    n_refs = len(ref_values)
    if band is not None and n_refs == 0:
        raise ValueError("convergence band= shades around a reference; pass refs")
    labels = _per_ref(ref_labels, n_refs, "ref_labels", single=_is_label)
    styles = _per_ref(ref_linestyles, n_refs, "ref_linestyles", single=_is_linestyle)
    bands = _per_ref(band, n_refs, "band")
    curve = _running(values, running)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    had_data = ax.has_data()
    old_limits = ax.get_xlim(), ax.get_ylim()

    rc = matplotlib.rcParams
    base_width = float(rc["lines.linewidth"])
    artists = {}
    rects = {}
    for i, (ref, half) in enumerate(zip(ref_values, bands, strict=True)):
        if half is None:
            continue
        half = float(half)
        fkw = {
            "facecolor": _style.neutral(0.18),
            "edgecolor": "none",
            "zorder": 0.5,
            **(fill_kw or {}),
        }
        rect = Rectangle(
            (0.0, ref - half),
            1.0,
            2.0 * half,
            transform=ax.get_yaxis_transform(),
            **fkw,
        )
        rects[i] = ax.add_artist(rect)
    ref_lines = []
    for ref, ls in zip(ref_values, styles, strict=True):
        rkw = {
            "color": _style.neutral(0.5),
            "ls": "--" if ls is None else ls,
            "lw": 0.9 * base_width,
            "zorder": 1,
            **(ref_kw or {}),
        }
        ref_lines.append(ax.axhline(ref, **rkw))
    va, sign = _LABEL_SIDES[label_side]
    texts = {}
    for i, (ref, label) in enumerate(zip(ref_values, labels, strict=True)):
        if label is None:
            continue
        tkw = {
            "color": _style.neutral(0.65),
            "fontsize": "small",
            "ha": "right",
            "va": va,
            "zorder": 7,
            "bbox": _style.backing(),
            **(text_kw or {}),
        }
        texts[i] = ax.annotate(
            label,
            (1.0, ref),
            xycoords=ax.get_yaxis_transform(),
            xytext=(-2.0, 2.0 * sign),
            textcoords="offset points",
            **tkw,
        )
    scatter = None
    if show_samples:
        skw = {
            "s": (0.45 * float(rc["lines.markersize"])) ** 2,
            "color": _style.neutral(0.55) if sample_color is None else sample_color,
            "lw": 0,
            "zorder": 5,
            **(scatter_kw or {}),
        }
        scatter = ax.scatter(x, values, **skw)
        artists["scatter"] = scatter
    lkw = {"color": _style.color(0, color), "zorder": 6, **(line_kw or {})}
    (line,) = ax.plot(x, curve, **lkw)
    artists["line"] = line
    if ref_lines:
        artists["lines"] = ref_lines
    if rects:
        artists["fill"] = [rects[i] for i in sorted(rects)]
    if texts:
        artists["text"] = [texts[i] for i in sorted(texts)]

    finite_x = x[np.isfinite(x)]
    x_lo, x_hi = float(finite_x.min()), float(finite_x.max())
    span = x_hi - x_lo
    xlim = _padded(x_lo, x_hi, 0.02)
    if texts:
        xlim = (xlim[0], x_hi + _LABEL_ROOM * (span if span > 0.0 else 1.0))
    ys = [curve]
    if show_samples:
        ys.append(values)
    ys.append(np.asarray(ref_values, dtype=float))
    for ref, half in zip(ref_values, bands, strict=True):
        if half is not None:
            ys.append(np.array([ref - float(half), ref + float(half)]))
    ys = np.concatenate(ys)
    ys = ys[np.isfinite(ys)]
    ylim = _padded(float(ys.min()), float(ys.max()), 0.06)
    if had_data:
        xlim = (min(xlim[0], old_limits[0][0]), max(xlim[1], old_limits[0][1]))
        ylim = (min(ylim[0], old_limits[1][0]), max(ylim[1], old_limits[1][1]))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    state = {"k": n, "values": values, "curve": curve, "refs": ref_values}

    def update(k=None, *, values=None, refs=None):
        if values is not None:
            new = np.asarray(values, dtype=float).ravel()
            if new.size != n:
                raise ValueError(f"convergence update got {new.size} values for {n}")
            state["values"] = new
            state["curve"] = _running(new, running)
        if refs is not None:
            new_refs = [float(refs)] if np.ndim(refs) == 0 else [float(r) for r in refs]
            if len(new_refs) != n_refs:
                raise ValueError(
                    f"convergence update got {len(new_refs)} refs for {n_refs}"
                )
            state["refs"] = new_refs
            for i, ref in enumerate(new_refs):
                ref_lines[i].set_ydata([ref, ref])
                if i in texts:
                    texts[i].xy = (1.0, ref)
                if i in rects:
                    rects[i].set_y(ref - float(bands[i]))
        if k is not None:
            state["k"] = min(max(int(k), 0), n)
        shown = state["k"]
        if scatter is not None:
            offsets = np.column_stack([x[:shown], state["values"][:shown]])
            scatter.set_offsets(offsets.reshape(-1, 2))
        line.set_data(x[:shown], state["curve"][:shown])

    return PlotResult(ax=ax, artists=artists, update=update)


# Where a signed trace's two fills sample the diverging colormap: above the
# level, then below it. Near the ends, short of the extremes, so a fill at
# half opacity stays a clear hue on a light page and a dark one.
_SIGNED_ENDS = (0.85, 0.15)


def _signed_fills(ax, x, values, level, colors, fkw):
    """The fill above `level` and the fill below it, as two new collections."""
    fills = []
    for mask, fill_color in zip((values >= level, values < level), colors, strict=True):
        fills.append(
            ax.fill_between(
                x,
                level,
                values,
                where=mask,
                interpolate=True,
                **{"color": fill_color, **fkw},
            )
        )
    return fills


def signed_trace(
    values,
    *,
    ax=None,
    x=None,
    level=0.0,
    color=None,
    cmap=None,
    fill_colors=None,
    fill_alpha=0.5,
    show_level=False,
    level_label=None,
    line_kw=None,
    fill_kw=None,
    level_kw=None,
    text_kw=None,
):
    """Draw a trace about a reference level, filled one color above, another below.

    The area between the trace and `level` is shaded in two colors, one
    where the trace is above the level and one where it is below, with the
    crossings interpolated, so the reader sees at a glance which way the
    quantity moved and for how long: a pixel brightening and dimming about
    its static value, a residual wandering about zero. The colors come from
    the two ends of the diverging colormap a signed map is drawn with, so
    the fills agree with a signed image beside the trace.

    The limits are set from the full trace and the level on the first draw
    (the x range edge to edge, the y range padded) and `update` never
    changes them, so an animation that reveals the trace sample by sample
    keeps one scale. On axes that already hold data, the limits grow to
    cover both.

    Args:
        values: 1D array-like of the samples, in order.
        ax: Axes to draw into. None creates a new figure and axes.
        x: Position of each sample, such as its time. None uses
            `0, 1, ..., n - 1`.
        level: The reference level the fills are drawn from.
        color: Color of the trace. None uses `_style.color(0)`.
        cmap: Diverging colormap whose ends color the fills (above at 0.85
            of the way along it, below at 0.15). None uses the semantic
            "residual" cmap.
        fill_colors: `(above, below)` colors for the fills, overriding
            `cmap`.
        fill_alpha: Opacity of both fills.
        show_level: Draw the level as a dashed neutral line across the
            panel.
        level_label: Text written at the right end of the level line on a
            backing box; implies `show_level`. None writes nothing.
        line_kw: Extra kwargs for the trace's `ax.plot` call, applied last.
        fill_kw: Extra kwargs for both `ax.fill_between` calls, applied
            last, on every update as on the first draw.
        level_kw: Extra kwargs for the level's `ax.axhline`, applied last.
        text_kw: Extra kwargs for the level label's `Text`, applied last.

    Returns:
        A `PlotResult` with artists `"line"` (the trace `Line2D`), `"fill"`
        (a list holding the fill above the level, then the fill below it,
        while two or more samples are shown, else empty), and, with the
        level drawn, `"lines"` (a list holding the level's `Line2D`) and
        `"text"` (the label, when given). Its
        `update(k=None, *, values=None)` shows the first `k` samples: it
        sets the trace's data and redraws the two fills, removing the old
        pair and putting the new one in the same `"fill"` list, so the axes
        hold the same number of artists after every update. `values`
        replaces the samples (same count). A `k` of None keeps the current
        count, which starts at every sample. The limits never change.

    Raises:
        ValueError: If `values` is empty, `x` does not match it, or
            `update` receives a different number of values.

    Example::

        res = ep.signed_trace(brightness, x=t_hours, level=static,
                              show_level=True, level_label="static")
        for k in range(1, len(t_hours) + 1):
            res.update(k)
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n == 0:
        raise ValueError("signed_trace needs at least one value")
    x = np.arange(float(n)) if x is None else np.asarray(x, dtype=float).ravel()
    if x.size != n:
        raise ValueError(f"signed_trace x has {x.size} entries for {n} values")
    level = float(level)
    if fill_colors is None:
        diverging = _style.cmap("residual", cmap)
        fill_colors = tuple(diverging(end) for end in _SIGNED_ENDS)
    else:
        fill_colors = tuple(fill_colors)

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    had_data = ax.has_data()
    old_limits = ax.get_xlim(), ax.get_ylim()

    rc = matplotlib.rcParams
    artists = {}
    show_level = show_level or level_label is not None
    if show_level:
        rkw = {
            "color": _style.neutral(0.5),
            "ls": "--",
            "lw": 0.9 * float(rc["lines.linewidth"]),
            "zorder": 2,
            **(level_kw or {}),
        }
        artists["lines"] = [ax.axhline(level, **rkw)]
    if level_label is not None:
        tkw = {
            "color": _style.neutral(0.65),
            "fontsize": "small",
            "ha": "right",
            "va": "bottom",
            "zorder": 7,
            "bbox": _style.backing(),
            **(text_kw or {}),
        }
        artists["text"] = ax.annotate(
            level_label,
            (1.0, level),
            xycoords=ax.get_yaxis_transform(),
            xytext=(-2.0, 2.0),
            textcoords="offset points",
            **tkw,
        )
    fkw = {"alpha": fill_alpha, "lw": 0, "zorder": 1, **(fill_kw or {})}
    lkw = {"color": _style.color(0, color), "zorder": 3, **(line_kw or {})}
    (line,) = ax.plot(x, values, **lkw)
    artists["line"] = line
    fills = []
    artists["fill"] = fills

    finite_x = x[np.isfinite(x)]
    xlim = (float(finite_x.min()), float(finite_x.max()))
    if xlim[0] == xlim[1]:
        xlim = _padded(xlim[0], xlim[1], 0.5)
    ys = np.append(values, level)
    ys = ys[np.isfinite(ys)]
    ylim = _padded(float(ys.min()), float(ys.max()), 0.06)
    if had_data:
        xlim = (min(xlim[0], old_limits[0][0]), max(xlim[1], old_limits[0][1]))
        ylim = (min(ylim[0], old_limits[1][0]), max(ylim[1], old_limits[1][1]))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    state = {"k": n, "values": values}

    def update(k=None, *, values=None):
        if values is not None:
            new = np.asarray(values, dtype=float).ravel()
            if new.size != n:
                raise ValueError(f"signed_trace update got {new.size} values for {n}")
            state["values"] = new
        if k is not None:
            state["k"] = min(max(int(k), 0), n)
        shown = state["k"]
        xs, vs = x[:shown], state["values"][:shown]
        line.set_data(xs, vs)
        for fill in fills:
            fill.remove()
        fills.clear()
        if shown >= 2:
            fills.extend(_signed_fills(ax, xs, vs, level, fill_colors, fkw))

    update()
    return PlotResult(ax=ax, artists=artists, update=update)


def _bin_edges(values, bins):
    """Bin edges from an edge array, or `bins` equal bins over the finite data."""
    if np.ndim(bins) == 0:
        finite = values[np.isfinite(values)]
        lo, hi = (0.0, 1.0) if finite.size == 0 else (finite.min(), finite.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        return np.linspace(lo, hi, int(bins) + 1)
    edges = np.asarray(bins, dtype=float).ravel()
    if edges.size < 2 or np.any(np.diff(edges) <= 0.0):
        raise ValueError("hist_fill bins must be a count or increasing edges")
    return edges


def hist_fill(
    values,
    bins=30,
    *,
    ax=None,
    ymax=None,
    law=None,
    law_label=None,
    law_label_x=None,
    rug=0,
    color=None,
    rug_color=None,
    bar_kw=None,
    line_kw=None,
    scatter_kw=None,
    text_kw=None,
):
    """Draw a histogram that fills as samples arrive, on a pinned scale.

    The bars count the samples per bin, and `update(k)` counts only the
    first `k`, so an animation shows the distribution building up sample by
    sample. The y axis is pinned on the first draw to the final counts (and
    to the law's expected counts, when given), so the histogram grows into
    a fixed frame rather than rescaling every frame and hiding how few
    samples the early shape rests on. A law drawn as expected counts shows
    what the bars are converging to.

    On that frame a count of one is a sliver, so the first `rug` samples can
    also be drawn as dots on the baseline, one at each sample's value and
    all at one height: a rug marks where samples fell, not how many, and
    can fade (`rug_alpha`) once the bars take over.

    Args:
        values: 1D array-like of every sample, in arrival order.
        bins: A number of equal bins spanning the finite samples, or the
            bin edges, increasing. Samples outside the edges are not
            counted.
        ax: Axes to draw into. None creates a new figure and axes.
        ymax: Top of the y axis. None takes 1.08 times the largest final
            count or expected count.
        law: Optional density of the samples, a callable `law(x)`, drawn as
            expected counts: the number of samples times the bin width times
            the density. Needs equal bins.
        law_label: Text written just above and right of the law's curve at
            `law_label_x`, on a backing box. None writes nothing.
        law_label_x: Where along x the label sits. None uses 0.35 of the
            way across the bins.
        rug: How many of the first samples the rug can show; 0 draws none.
        color: Color of the bars. None uses a neutral tone, so a law or a
            highlight drawn over them carries the color.
        rug_color: Color of the rug's dots. None uses `_style.color(0)`.
        bar_kw: Extra kwargs for `ax.bar`, applied last.
        line_kw: Extra kwargs for the law's `ax.plot` call, applied last.
        scatter_kw: Extra kwargs for the rug's `ax.scatter`, applied last.
        text_kw: Extra kwargs for the law label's `Text`, applied last.

    Returns:
        A `PlotResult` with artists `"hist"` (the `BarContainer`, one bar per
        bin), `"line"` (the law's expected counts, with `law`), `"text"`
        (the law's label, with `law_label`), and `"scatter"` (the rug's
        `PathCollection`, with `rug`), and an
        `update(k=None, *, rug_alpha=None, show_law=None)` that counts the
        first `k` samples into the same bars and shows the first
        `min(k, rug)` of them on the rug, adding no artist. `rug_alpha` sets
        the rug's opacity and `show_law` shows or hides the law and its
        label; None keeps either as it was. A `k` of None keeps the current
        count, which starts at every sample. The limits never change.

    Raises:
        ValueError: If `values` is empty, `bins` is neither a count nor
            increasing edges, or `law` is given with unequal bins.

    Example::

        looks = np.random.default_rng(0).exponential(1.0, 400)
        res = ep.hist_fill(looks, np.arange(0.0, 6.01, 0.25),
                           law=lambda x: np.exp(-x), law_label="exponential",
                           rug=12)
        for k in range(1, 401):
            res.update(k, rug_alpha=max(0.0, 1.0 - k / 40))
    """
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    if n == 0:
        raise ValueError("hist_fill needs at least one value")
    edges = _bin_edges(values, bins)
    widths = np.diff(edges)
    n_bins = widths.size
    if law is not None and not np.allclose(widths, widths[0], rtol=1e-6, atol=0.0):
        raise ValueError("hist_fill law= draws expected counts and needs equal bins")
    index = np.digitize(values, edges) - 1
    # A sample exactly on the last edge belongs to the last bin, as in hist.
    index[values == edges[-1]] = n_bins - 1

    def counts(k):
        sel = index[:k]
        return np.bincount(sel[(sel >= 0) & (sel < n_bins)], minlength=n_bins)

    final = counts(n)
    grid = expected = None
    if law is not None:
        grid = np.linspace(edges[0], edges[-1], 400)
        expected = n * widths[0] * np.asarray(law(grid), dtype=float)
    if ymax is None:
        top = float(final.max())
        if expected is not None:
            top = max(top, float(np.nanmax(expected)))
        ymax = 1.08 * top if top > 0.0 else 1.0

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    rc = matplotlib.rcParams
    base_width = float(rc["lines.linewidth"])
    bkw = {
        "align": "edge",
        "color": _style.neutral(0.55) if color is None else color,
        "edgecolor": rc["axes.facecolor"],
        "linewidth": 0.4 * base_width,
        "zorder": 2,
        **(bar_kw or {}),
    }
    bars = ax.bar(edges[:-1], final, width=widths, **bkw)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, ymax)
    artists = {"hist": bars}

    rug_dots = None
    if rug:
        skw = {
            "s": (0.55 * float(rc["lines.markersize"])) ** 2,
            "color": _style.color(0, rug_color),
            "edgecolors": rc["text.color"],
            "linewidths": 0.4 * base_width,
            "transform": ax.get_xaxis_transform(),
            "clip_on": False,
            "zorder": 6,
            **(scatter_kw or {}),
        }
        rug_dots = ax.scatter([], [], **skw)
        artists["scatter"] = rug_dots
    curve = label = None
    if law is not None:
        lkw = {"color": rc["text.color"], "lw": 1.3 * base_width, "zorder": 4}
        (curve,) = ax.plot(grid, expected, **{**lkw, **(line_kw or {})})
        artists["line"] = curve
        if law_label is not None:
            if law_label_x is None:
                lx = edges[0] + 0.35 * (edges[-1] - edges[0])
            else:
                lx = float(law_label_x)
            ly = n * widths[0] * float(np.asarray(law(np.array([lx])))[0])
            tkw = {
                "color": rc["text.color"],
                "fontsize": "small",
                "ha": "left",
                "va": "bottom",
                "zorder": 7,
                "bbox": _style.backing(),
                **(text_kw or {}),
            }
            label = ax.text(lx + 0.12 * widths[0], ly + 0.04 * ymax, law_label, **tkw)
            artists["text"] = label

    state = {"k": n}

    def update(k=None, *, rug_alpha=None, show_law=None):
        if k is not None:
            state["k"] = min(max(int(k), 0), n)
        shown = state["k"]
        for bar, count in zip(bars, counts(shown), strict=True):
            bar.set_height(count)
        if rug_dots is not None:
            first = values[: min(shown, int(rug))]
            rug_dots.set_offsets(
                np.column_stack([first, np.full(first.size, 0.025)]).reshape(-1, 2)
            )
            if rug_alpha is not None:
                rug_dots.set_alpha(rug_alpha)
        if show_law is not None:
            for artist in (curve, label):
                if artist is not None:
                    artist.set_visible(bool(show_law))

    update()
    return PlotResult(ax=ax, artists=artists, update=update)
