"""Phasors on the complex plane: arrows, chains, resultants, and dials.

`phasor` draws complex numbers as arrows on an equal-aspect complex plane,
either each from a shared origin or chained tip to tail, with an optional
resultant, Re and Im axes, and a ring colored by the phase colormap so an
arrow's direction and a phase map read against one key. Drawn with `at=`,
the plane is a small inset (a dial) centered on a point of another panel.

The module is private because a public `eyepiece.phasor` submodule would
shadow the `phasor` function of the same name once imported.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

from eyepiece import _style
from eyepiece._result import PlotResult

#: Arrowhead shape: length and half-width as fractions of the mutation scale.
HEAD_LENGTH = 0.5
HEAD_WIDTH = 0.28

#: Mutation scale of a full-size head per point of `lines.markersize`, so a
#: head is 0.8 marker sizes long and grows linearly with the marker size.
HEAD_PER_MARKER = 1.6

#: An arrow shorter than this on screen, in points, is not drawn at all.
MIN_LENGTH_PT = 0.05

#: Margin added around the data by the automatic limits, as a fraction of
#: the larger data span.
PAD = 0.08


class _Arrow(FancyArrowPatch):
    """A `FancyArrowPatch` whose head never overshoots a short shaft.

    The head is a fixed size in points, so an arrow shorter on screen than
    its head would draw the head past its own tail. The effective mutation
    scale is therefore capped at the one whose head is exactly as long as
    the arrow, computed from the current transform each time it is read,
    so the cap follows later changes to the limits, the figure size, or
    the save dpi. `set_mutation_scale` still sets the nominal head size.
    """

    def __init__(self, posA, posB, **kwargs):
        super().__init__(posA, posB, **kwargs)
        self._ends = (posA, posB)

    def set_positions(self, posA, posB):
        """Move the arrow, keeping the endpoints the head cap reads."""
        super().set_positions(posA, posB)
        self._ends = (posA, posB)

    def _length_pt(self):
        """On-screen length in points, or None before the arrow is placed."""
        if self.figure is None or self.axes is None:
            return None
        a, b = self.get_transform().transform(np.asarray(self._ends, dtype=float))
        return float(np.hypot(*(b - a))) * 72.0 / self.figure.dpi

    def get_mutation_scale(self):
        """Nominal mutation scale, capped so the head fits the arrow."""
        nominal = super().get_mutation_scale()
        length_pt = self._length_pt()
        if length_pt is None:
            return nominal
        return min(nominal, length_pt / HEAD_LENGTH)

    def _vanishing(self):
        """True when the arrow is too short on screen to draw."""
        length_pt = self._length_pt()
        return length_pt is not None and length_pt < MIN_LENGTH_PT

    def get_path(self):
        """The arrow's path, or a single point for a vanishing arrow.

        Matplotlib builds the head from the shaft's direction, which a
        zero-length arrow does not have (a 0/0 in the head geometry), and
        it asks for the path outside of drawing too: when the patch is
        added, and for window and tight extents. A vanishing arrow is
        therefore its start point, so it never reaches the head geometry.
        """
        if self._vanishing():
            start = np.asarray(self._ends[0], dtype=float)
            return Path(np.array([start, start]))
        return super().get_path()

    def draw(self, renderer):
        """Draw nothing for a vanishing arrow, whose stroke would leave a dot.

        A chain that closes has a zero resultant, and a dot at the origin
        would read as a value. Visibility is left alone, so the arrow draws
        again as soon as an update gives it length.
        """
        if self._vanishing():
            return
        super().draw(renderer)


def _per_arrow(value, n, default, name):
    """Broadcast a scalar or a length-`n` sequence to a list of `n` values."""
    if value is None:
        return [default] * n
    if isinstance(value, str) or np.ndim(value) == 0 or _is_dash_tuple(value):
        return [value] * n
    values = list(value)
    if len(values) != n:
        raise ValueError(f"phasor {name} has {len(values)} entries for {n} vectors")
    return values


def _is_dash_tuple(value):
    """True for a single matplotlib dash pattern such as `(0, (3, 2))`."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and np.ndim(value[0]) == 0
        and isinstance(value[1], tuple)
    )


def _is_color(value):
    """True for one color spec (a name, a hex string, or an RGB/RGBA tuple)."""
    try:
        to_rgba(value)
    except (TypeError, ValueError):
        return False
    return True


def _shaft_fraction(head):
    """Default shaft width, as a fraction of the line width, for a head scale.

    A small head on a full-width shaft disappears into it, so the shaft
    thins with the head, down to 0.4 of the line width.
    """
    return min(1.0, 0.4 + 0.6 * head)


def _segments(vectors, origin, chain, show_sum):
    """Start and end points (complex) of each arrow, the resultant last."""
    vectors = np.asarray(vectors, dtype=complex).ravel()
    origin = complex(origin)
    if chain:
        ends = origin + np.cumsum(vectors)
        starts = np.concatenate([[origin], ends[:-1]])
    else:
        starts = np.full(vectors.shape, origin)
        ends = origin + vectors
    if show_sum:
        starts = np.append(starts, origin)
        ends = np.append(ends, origin + vectors.sum())
    return starts, ends


def _xy(z):
    return (float(z.real), float(z.imag))


def _auto_limits(points):
    """Square limits around `points` (complex), padded on every side."""
    lo_x, hi_x = points.real.min(), points.real.max()
    lo_y, hi_y = points.imag.min(), points.imag.max()
    span = max(hi_x - lo_x, hi_y - lo_y)
    if span == 0.0:
        span = max(abs(points).max(), 1.0)
    half = 0.5 * span * (1.0 + 2.0 * PAD)
    cx, cy = 0.5 * (lo_x + hi_x), 0.5 * (lo_y + hi_y)
    return (cx - half, cx + half, cy - half, cy + half)


def _resolve_limits(lim, points):
    if lim is None:
        return _auto_limits(points)
    if np.ndim(lim) == 0:
        half = float(lim)
        return (-half, half, -half, half)
    lim = tuple(float(v) for v in lim)
    if len(lim) != 4:
        raise ValueError(f"phasor lim must be a number or 4 values, got {lim}")
    return lim


def _dial_axes(ax, at, size):
    """A transparent inset of `ax`, `size` data units square, centered at `at`.

    The inset is placed through the parent's data transform, so it stays
    on its point when the parent's limits change, and it is kept out of
    the layout engine so the parent never gives up room to it.
    """
    size = float(size)
    if not size > 0.0:
        raise ValueError(f"phasor dial size must be positive, got {size}")
    x, y = float(at[0]), float(at[1])
    inset = ax.inset_axes(
        [x - 0.5 * size, y - 0.5 * size, size, size],
        transform=ax.transData,
        zorder=6,
    )
    inset.set_in_layout(False)
    inset.set_anchor("C")
    inset.patch.set_alpha(0.0)
    return inset


def _ring(radius, cmap, width, n=180):
    """The circle of `radius`, each segment colored by the phase at its angle.

    The color at angle phi is the phase colormap's color for phi on the
    same `[-pi, pi]` scale a phase map is drawn on, so the ring is the key.
    """
    phi = np.linspace(-np.pi, np.pi, n + 1)
    pts = radius * np.column_stack([np.cos(phi), np.sin(phi)])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    mid = 0.5 * (phi[:-1] + phi[1:])
    return LineCollection(
        segments,
        colors=cmap((mid + np.pi) / (2.0 * np.pi)),
        linewidths=width,
        capstyle="butt",
        zorder=2,
    )


def phasor(
    vectors,
    *,
    ax=None,
    origin=0j,
    chain=False,
    show_sum=False,
    colors=None,
    linestyles=None,
    widths=None,
    head_scale=1.0,
    sum_color=None,
    at=None,
    size=0.25,
    cross=True,
    axis_labels=None,
    ring=False,
    ring_radius=1.0,
    ring_cmap=None,
    lim=None,
    arrow_kw=None,
    line_kw=None,
    collection_kw=None,
):
    """Draw complex numbers as arrows on the complex plane.

    Each value is an arrow on an equal-aspect, frameless plane with Re on
    the horizontal axis and Im on the vertical. Drawn separately, every
    arrow starts at `origin`; chained, each starts at the previous tip, so
    the chain's last tip is `origin` plus the sum, and a sum of unit
    phasors whose phases wind through a full turn closes on itself.

    Arrowheads are sized in points, not data units, and grow linearly with
    `rcParams["lines.markersize"]`, so a style that enlarges markers for a
    slide enlarges the heads in proportion instead of leaving them as
    specks or, as a head scaled with the square of the marker size would,
    letting them cover the figure. An arrow shorter on screen than its head
    draws a head shrunk to the arrow's length, so the many short links of a
    long chain never overshoot their own tails.

    Args:
        vectors: Complex array-like of the values to draw, flattened to
            one arrow each, in order.
        ax: Axes to draw into. None creates a new figure and axes. With
            `at`, the parent axes that receives the dial.
        origin: Complex start of the first arrow (of every arrow when not
            chained).
        chain: Draw the arrows tip to tail starting at `origin`.
        show_sum: Also draw the resultant, from `origin` to `origin` plus
            the sum of `vectors`, in `sum_color` and 1.5 times the default
            width. It is listed last but layered beneath the other arrows,
            so a chain lying along its own resultant stays visible on top.
        colors: One color for every arrow or one per vector. None uses
            `_style.color(0)`, the first color of the active palette.
        linestyles: One line style for every arrow or one per vector.
            None draws solid arrows.
        widths: One shaft width in points for every arrow or one per
            vector. None uses `rcParams["lines.linewidth"]` for a full-size
            head and thins the shaft with a smaller one, to 0.4 of it at
            the least, so a small head is not lost in its shaft.
        head_scale: Head size relative to a full-size head, one value for
            every arrow or one per vector. The resultant takes the largest.
        sum_color: Color of the resultant. None uses `rcParams["text.color"]`.
        at: `(x, y)` in the data coordinates of `ax`. When given, the plane
            is drawn in a transparent inset (a dial) centered at that point
            and `result.ax` is the inset. The parent's position and limits
            are left unchanged.
        size: Width and height of the dial's box in the parent's data units;
            its equal aspect makes it a square of the smaller side on screen.
            Ignored without `at`.
        cross: Draw the Re and Im axes through 0 as thin neutral lines.
        axis_labels: `(re_label, im_label)` written at the positive ends
            of the axes. None writes "Re" and "Im" on a full plane and
            nothing on a dial; False writes nothing. Ignored without `cross`.
        ring: Draw a circle of `ring_radius` about 0 colored by phase.
        ring_radius: Radius of the ring, the unit circle by default.
        ring_cmap: Colormap override for the ring; None uses the semantic
            "phase" cmap, the one a phase map is drawn with.
        lim: A half-width for symmetric limits about 0, or `(xmin, xmax,
            ymin, ymax)`. None fits a square around every drawn point,
            including 0 when `cross` is set and the ring when drawn.
        arrow_kw: Extra kwargs for every `FancyArrowPatch` (for example
            `zorder` or `gid`), applied last.
        line_kw: Extra kwargs for the axis lines, applied last.
        collection_kw: Extra kwargs for the ring's `LineCollection`,
            applied last.

    Returns:
        A `PlotResult` with artists `"arrow"` (the list of `FancyArrowPatch`
        in input order, the resultant last), `"lines"` (the Re and Im axis
        lines, if drawn), `"collection"` (the ring, if drawn), and `"text"`
        (the axis labels, if drawn), and an `.update(vectors, origin=None)`
        that moves the same arrows to new values, keeping every style. It
        re-chains and recomputes the resultant as on the first draw; an
        `origin` of None keeps the last one. It never rescales the limits
        or the heads' nominal size.

    Raises:
        ValueError: If `at` is given without `ax`, if a per-arrow sequence
            does not match the number of vectors, if `lim` has the wrong
            length, or if `update` receives a different number of vectors.

    Example::

        n = 24
        steps = np.exp(2j * np.pi * (np.arange(n) + 0.5) / n) / n
        result = ep.phasor(steps, chain=True, show_sum=True)

        image = ep.imshow_log(speckles, extent=extent)
        ep.phasor([field], ax=image.ax, at=(x, y), size=1.0, ring=True)
    """
    if at is not None and ax is None:
        raise ValueError("phasor at= draws a dial into an existing ax; pass ax")
    values = np.asarray(vectors, dtype=complex).ravel()
    n = len(values)
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    if at is not None:
        ax = _dial_axes(ax, at, size)

    rc = matplotlib.rcParams
    base_width = float(rc["lines.linewidth"])
    if colors is not None and _is_color(colors):
        colors = [colors] * n
    colors = _per_arrow(colors, n, _style.color(0), "colors")
    linestyles = _per_arrow(linestyles, n, "-", "linestyles")
    heads = [float(h) for h in _per_arrow(head_scale, n, 1.0, "head_scale")]
    if widths is None:
        widths = [base_width * _shaft_fraction(h) for h in heads]
    widths = _per_arrow(widths, n, base_width, "widths")
    zorders = [5] * n
    if show_sum:
        sum_head = max(heads, default=1.0)
        colors.append(rc["text.color"] if sum_color is None else sum_color)
        linestyles.append("-")
        widths.append(1.5 * base_width * _shaft_fraction(sum_head))
        heads.append(sum_head)
        zorders.append(4)

    state = {"origin": complex(origin)}
    starts, ends = _segments(values, state["origin"], chain, show_sum)
    head_pt = HEAD_PER_MARKER * float(rc["lines.markersize"])
    style = f"-|>,head_length={HEAD_LENGTH},head_width={HEAD_WIDTH}"
    arrows = []
    for start, end, color, ls, lw, head, zorder in zip(
        starts, ends, colors, linestyles, widths, heads, zorders, strict=True
    ):
        kw = {
            "arrowstyle": style,
            "mutation_scale": head_pt * head,
            "color": color,
            "linestyle": ls,
            "linewidth": lw,
            "shrinkA": 0.0,
            "shrinkB": 0.0,
            "zorder": zorder,
            **(arrow_kw or {}),
        }
        arrows.append(ax.add_patch(_Arrow(_xy(start), _xy(end), **kw)))

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    artists = {"arrow": arrows}
    if cross:
        lkw = {
            "color": _style.neutral(0.45),
            "lw": 0.7 * base_width,
            "zorder": 1,
            **(line_kw or {}),
        }
        artists["lines"] = [ax.axhline(0.0, **lkw), ax.axvline(0.0, **lkw)]
        labels = axis_labels
        if labels is None:
            labels = ("Re", "Im") if at is None else False
        if labels:
            tkw = {"color": _style.neutral(0.6), "fontsize": "small", "zorder": 1}
            re_text = ax.text(
                1.0,
                0.0,
                f" {labels[0]}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                **tkw,
            )
            im_text = ax.text(
                0.0,
                1.0,
                f" {labels[1]}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                **tkw,
            )
            artists["text"] = [re_text, im_text]

    points = np.concatenate([starts, ends])
    if cross:
        points = np.append(points, 0j)
    if ring:
        radius = float(ring_radius)
        collection = _ring(radius, _style.cmap("phase", ring_cmap), 3.0 * base_width)
        collection.set(**(collection_kw or {}))
        artists["collection"] = ax.add_collection(collection)
        points = np.append(points, [radius + radius * 1j, -radius - radius * 1j])
    xmin, xmax, ymin, ymax = _resolve_limits(lim, points)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def update(new_vectors, origin=None):
        new_values = np.asarray(new_vectors, dtype=complex).ravel()
        if len(new_values) != n:
            raise ValueError(
                f"phasor update got {len(new_values)} vectors for {n} arrows"
            )
        if origin is not None:
            state["origin"] = complex(origin)
        new_starts, new_ends = _segments(new_values, state["origin"], chain, show_sum)
        for patch, start, end in zip(arrows, new_starts, new_ends, strict=True):
            patch.set_positions(_xy(start), _xy(end))

    return PlotResult(ax=ax, artists=artists, update=update)
