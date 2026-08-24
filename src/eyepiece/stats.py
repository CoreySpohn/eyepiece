"""Sample-distribution primitives: corner plots, hist-vs-pdf, covariance ellipses.

`corner` and `corner_overlay` are a deliberate parity port of the classic
triangle-plot idiom rather than a wrapper around a third-party corner-plot
package: overlaying a second dataset into a caller's existing axes, and
branding through the style library, both go against that kind of package's
API grain. `labels` is a plain dict argument on both -- the parameter-name
table belongs to whichever consumer owns those parameter names, not to this
library.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from eyepiece import _style
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
