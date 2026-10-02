"""Phasors on the complex plane: arrows, chains, resultants, and dials.

`phasor` draws complex numbers as arrows on an equal-aspect complex plane,
either each from a shared origin or chained tip to tail, with an optional
resultant, Re and Im axes, and a ring colored by the phase colormap so an
arrow's direction and a phase map read against one key. Drawn with `at=`,
the plane is a small inset (a dial) centered on a point of another panel.
`phase_ring` draws that ring alone, onto any axes.

The module is private because a public `eyepiece.phasor` submodule would
shadow the `phasor` function of the same name once imported.
"""

from numbers import Number

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import is_color_like
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.transforms import Bbox

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

# Units a dial's `size` can be given in.
_SIZE_UNITS = ("data", "axes")


class _Arrow(FancyArrowPatch):
    """A `FancyArrowPatch` whose head never overshoots a short shaft.

    The head is a fixed size in points, so an arrow shorter on screen than
    its head would draw the head past its own tail. The effective mutation
    scale is therefore capped at the one whose head is exactly as long as
    the arrow, computed from the current transform each time it is read,
    so the cap follows later changes to the limits, the figure size, or
    the save dpi. `set_mutation_scale` still sets the nominal head size.
    `heads` is the number of heads the arrow style draws; a double-headed
    arrow caps each head at half its length, so the two never cross.
    """

    heads = 1

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
        return min(nominal, length_pt / (HEAD_LENGTH * self.heads))

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


def _is_number(value):
    """True for one plain number: a Python or NumPy scalar, or a 0-d array."""
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    return isinstance(value, Number) and not isinstance(value, bool)


def _per_arrow(value, n, default, name, single=_is_number):
    """Broadcast one value or a length-`n` sequence to a list of `n` values.

    `single` decides whether `value` is one value for every arrow. It never
    builds an array from `value`, so a sequence that mixes kinds of entry (a
    color name beside an RGBA tuple, a named line style beside a dash
    pattern) is taken entry by entry.
    """
    if value is None:
        return [default] * n
    if single(value):
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
        and _is_number(value[0])
        and isinstance(value[1], tuple)
        and all(_is_number(v) for v in value[1])
    )


def _is_linestyle(value):
    """True for one line style: a name such as `"--"` or a dash pattern."""
    return isinstance(value, str) or _is_dash_tuple(value)


def _shaft_fraction(head):
    """Default shaft width, as a fraction of the line width, for a head scale.

    A small head on a full-width shaft disappears into it, so the shaft
    thins with the head, down to 0.4 of the line width.
    """
    return min(1.0, 0.4 + 0.6 * head)


def _segments(vectors, origin, chain, show_sum, starts=None):
    """Start and end points (complex) of each arrow, the resultant last.

    `starts`, when given, is the start of each arrow of a plane that is not
    chained; the resultant still starts at `origin`.
    """
    vectors = np.asarray(vectors, dtype=complex).ravel()
    origin = complex(origin)
    if chain:
        ends = origin + np.cumsum(vectors)
        starts = np.concatenate([[origin], ends[:-1]])
    elif starts is not None:
        starts = np.asarray(starts, dtype=complex)
        ends = starts + vectors
    else:
        starts = np.full(vectors.shape, origin)
        ends = origin + vectors
    if show_sum:
        starts = np.append(starts, origin)
        ends = np.append(ends, origin + vectors.sum())
    return starts, ends


def _resolve_starts(starts, n, chain, what):
    """`starts` as a length-`n` complex array, or None when not given.

    Raises:
        ValueError: If `starts` is given on a chained plane, or its length is
            not `n`. The message starts with `what`.
    """
    if starts is None:
        return None
    if chain:
        raise ValueError(
            f"{what} starts= places arrows that are not chained; "
            "a chain starts each arrow at the previous tip"
        )
    starts = np.asarray(starts, dtype=complex).ravel()
    if len(starts) != n:
        raise ValueError(f"{what} got {len(starts)} starts for {n} vectors")
    return starts


def _overlap(a0, a1, b0, b1, tol):
    """True when segment `b` lies on segment `a`'s line and shares a length.

    Segments meeting at one point, as consecutive links of a chain or two
    arrows from one origin do, do not overlap. Lengths, distances from the
    line, and the shared length are compared against `tol`.
    """
    da, db = a1 - a0, b1 - b0
    if abs(da) <= tol or abs(db) <= tol:
        return False
    u = da / abs(da)
    r0 = np.conj(u) * (b0 - a0)
    r1 = np.conj(u) * (b1 - a0)
    if abs(r0.imag) > tol or abs(r1.imag) > tol:
        return False
    shared = min(abs(da), max(r0.real, r1.real)) - max(0.0, min(r0.real, r1.real))
    return shared > tol


def _separate(starts, ends, offset, tol):
    """Shift each arrow that overlaps an earlier one to its right-hand side.

    Arrows are compared in order at their unshifted positions; an arrow that
    lies on the same line as any arrow before it and shares a length with
    it moves by `offset` perpendicular to itself, clockwise from its own
    direction. Nothing moves when `offset` is 0.
    """
    if offset == 0.0:
        return starts, ends
    moved_starts, moved_ends = starts.copy(), ends.copy()
    for k in range(1, len(starts)):
        if any(_overlap(starts[j], ends[j], starts[k], ends[k], tol) for j in range(k)):
            step = ends[k] - starts[k]
            shift = -1j * offset * step / abs(step)
            moved_starts[k] += shift
            moved_ends[k] += shift
    return moved_starts, moved_ends


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


class _FractionLocator:
    """Place a square dial a fraction of its parent's shorter side across.

    Called by the inset at draw time, it reads the parent's box and data
    transform as they are then, so the dial keeps its share of the parent
    and stays on its data point through layout, resizing, and limit
    changes. It returns the box in the parent figure's coordinates.
    """

    def __init__(self, parent, at, size):
        self._parent = parent
        self._at = at
        self._size = size

    def __call__(self, ax, renderer):
        parent = self._parent
        box = parent.bbox
        side = self._size * min(box.width, box.height)
        cx, cy = parent.transData.transform(self._at)
        display = Bbox.from_bounds(cx - 0.5 * side, cy - 0.5 * side, side, side)
        return display.transformed(parent.figure.transSubfigure.inverted())


def _dial_axes(ax, at, size, size_units="data"):
    """A transparent inset of `ax`, `size` across, centered at `at`.

    In `"data"` units the inset is a box `size` data units square, placed
    through the parent's data transform. In `"axes"` units it is a square
    `size` times the parent's shorter side on screen. Either way it stays
    on its point when the parent's limits change, and it is kept out of
    the layout engine so the parent never gives up room to it.
    """
    size = float(size)
    if not size > 0.0:
        raise ValueError(f"phasor dial size must be positive, got {size}")
    x, y = float(at[0]), float(at[1])
    if size_units == "data":
        inset = ax.inset_axes(
            [x - 0.5 * size, y - 0.5 * size, size, size],
            transform=ax.transData,
            zorder=6,
        )
    else:
        locator = _FractionLocator(ax, (x, y), size)
        inset = ax.inset_axes([0.0, 0.0, 1.0, 1.0], zorder=6)
        inset.set_axes_locator(locator)
        inset.set_position(locator(inset, None))
    inset.set_in_layout(False)
    inset.set_anchor("C")
    inset.patch.set_alpha(0.0)
    return inset


def _ring(radius, cmap, width, center=0j, n=180):
    """The circle of `radius`, each segment colored by the phase at its angle.

    The color at angle phi is the phase colormap's color for phi on the
    same `[-pi, pi]` scale a phase map is drawn on, so the ring is the key.
    """
    phi = np.linspace(-np.pi, np.pi, n + 1)
    pts = radius * np.column_stack([np.cos(phi), np.sin(phi)])
    if center != 0j:
        pts = pts + np.array([center.real, center.imag])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    mid = 0.5 * (phi[:-1] + phi[1:])
    return LineCollection(
        segments,
        colors=cmap((mid + np.pi) / (2.0 * np.pi)),
        linewidths=width,
        capstyle="butt",
        zorder=2,
    )


def phase_ring(ax=None, *, radius=1.0, center=0j, cmap=None, width=None, zorder=None):
    """Draw a circle colored by phase: the key a phasor's direction reads against.

    The segment at angle phi about `center` takes the phase colormap's color
    for phi on the `[-pi, pi]` scale a phase map is drawn on, so an arrow
    pointing along phi and a phase-map pixel of value phi share one color.

    The ring is added like any matplotlib collection and touches nothing
    else on a given axes: not its aspect, ticks, spines, or limits, except
    that axes still autoscaling grow to include it, as they would for any
    artist; axes whose limits were set keep them. It looks round only under
    an equal aspect, which the caller sets.

    Args:
        ax: Axes to draw into. None creates a new figure and axes with
            constrained layout and gives that new axes an equal aspect.
        radius: Radius of the ring in data units.
        center: Complex center of the ring, `x + 1j * y` in data units.
        cmap: Colormap override; None uses the semantic "phase" cmap, the
            one a phase map is drawn with.
        width: Stroke width in points. None uses three times
            `rcParams["lines.linewidth"]`.
        zorder: Layering of the ring. None draws it at 2, beneath lines
            and patches.

    Returns:
        A `PlotResult` with the artist `"collection"` (the ring's
        `LineCollection`, 180 segments) and no `update`.

    Example::

        fig, ax = plt.subplots()
        ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal")
        ep.phase_ring(ax, radius=1.0)
    """
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
        ax.set_aspect("equal")
    if width is None:
        width = 3.0 * float(matplotlib.rcParams["lines.linewidth"])
    collection = _ring(
        float(radius), _style.cmap("phase", cmap), width, center=complex(center)
    )
    if zorder is not None:
        collection.set_zorder(zorder)
    ax.add_collection(collection)
    return PlotResult(ax=ax, artists={"collection": collection})


def phasor(
    vectors,
    *,
    ax=None,
    origin=0j,
    starts=None,
    chain=False,
    show_sum=False,
    colors=None,
    linestyles=None,
    widths=None,
    head_scale=1.0,
    sum_color=None,
    sum_head_scale=None,
    sum_width=None,
    at=None,
    size=0.25,
    size_units="data",
    separate=0.0,
    cross=True,
    axis_labels=None,
    text_kw=None,
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
            chained and `starts` is None).
        starts: Complex start of each arrow, one per vector, for a plane
            that is not chained, so arrows from `origin` and arrows placed
            elsewhere (the second of two added fields drawn from the
            first's tip, say) share one call. Arrow i runs from `starts[i]`
            to `starts[i] + vectors[i]`. The resultant still starts at
            `origin`. None starts every arrow at `origin`.
        chain: Draw the arrows tip to tail starting at `origin`. Cannot be
            combined with `starts`.
        show_sum: Also draw the resultant, from `origin` to `origin` plus
            the sum of `vectors`, in `sum_color` and 1.5 times the default
            width. It is listed last but layered beneath the other arrows,
            so a chain lying along its own resultant stays visible on top.
        colors: One color for every arrow or one per vector. Any
            matplotlib color spec works, and a per-vector sequence may mix
            them (names, hex strings, RGB and RGBA tuples); a single RGB or
            RGBA tuple is one color for every arrow. None uses
            `_style.color(0)`, the first color of the active palette.
        linestyles: One line style for every arrow or one per vector, each a
            name such as `"--"` or a dash pattern such as `(0, (3, 2))`, in
            any mix. None draws solid arrows.
        widths: One shaft width in points for every arrow or one per
            vector. None uses `rcParams["lines.linewidth"]` for a full-size
            head and thins the shaft with a smaller one, to 0.4 of it at
            the least, so a small head is not lost in its shaft.
        head_scale: Head size relative to a full-size head, one value for
            every arrow or one per vector. 0 draws a plain shaft with no
            head.
        sum_color: Color of the resultant. None uses `rcParams["text.color"]`.
        sum_head_scale: Head size of the resultant, relative to a full-size
            head. None uses the largest of `head_scale`. Ignored without
            `show_sum`.
        sum_width: Shaft width of the resultant in points. None uses 1.5
            times `rcParams["lines.linewidth"]`, thinned with a smaller
            resultant head as the default shafts are. Ignored without
            `show_sum`.
        at: `(x, y)` in the data coordinates of `ax`. When given, the plane
            is drawn in a transparent inset (a dial) centered at that point
            and `result.ax` is the inset. The parent's position and limits
            are left unchanged.
        size: Width and height of the dial's box, in `size_units`. Ignored
            without `at`.
        size_units: `"data"` (the default) gives `size` in the parent's data
            units, and the dial's equal aspect makes it a square of the
            smaller side on screen. `"axes"` gives it as a fraction of the
            parent axes' shorter side on screen, so `size=0.25` is a square
            a quarter of the parent's height on a wide panel. That square
            is computed from the parent's box at each draw, so it keeps its
            share of the parent through layout and resizing, as the dial
            keeps to its data point.
        separate: Offset, as a fraction of the plane's half-width (half the
            larger span of the limits), that draws apart arrows lying on
            one another. Arrows are taken in the returned order (the
            resultant last), and each one that lies on the same line as an
            arrow before it and shares a length with it, at their unshifted
            positions, moves by this offset perpendicular to itself,
            clockwise from its own direction (below a rightward arrow,
            above a leftward one). So the resultant of a straight chain
            runs just beneath the chain, a link folded back onto the one
            before it runs beside it, and arrows that touch only at a
            point, as consecutive links or arrows from one origin do, never
            move. The automatic limits are fitted to the unshifted arrows.
            0 moves nothing.
        cross: Draw the Re and Im axes through 0 as thin neutral lines.
        axis_labels: `(re_label, im_label)` written at the positive ends
            of the axes. None writes "Re" and "Im" on a full plane and
            nothing on a dial; False writes nothing. Ignored without `cross`.
        text_kw: Extra kwargs for the axis-label `Text` artists (for example
            `fontsize`, `zorder`, `color`, or a backing `bbox`), applied
            last.
        ring: Draw a circle of `ring_radius` about 0 colored by phase, with
            `phase_ring`.
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
        (the axis labels, if drawn), and an
        `.update(vectors, origin=None, starts=None)` that moves the same
        arrows to new values, keeping every style (the resultant's head and
        width among them). It re-chains, recomputes the resultant, and
        separates overlapping arrows as on the first draw, with the offset
        of the first draw; an `origin` or `starts` of None keeps the last
        one. It never rescales the limits or the heads'
        nominal size.

    Raises:
        ValueError: If `at` is given without `ax`, if a per-arrow sequence
            or `starts` does not match the number of vectors, if `starts`
            is given with `chain`, if `size_units` is not `"data"` or
            `"axes"`, if `lim` has the wrong length, or if `update` receives
            a different number of vectors or starts, or starts on a chain.

    Example::

        n = 24
        steps = np.exp(2j * np.pi * (np.arange(n) + 0.5) / n) / n
        result = ep.phasor(steps, chain=True, show_sum=True)

        image = ep.imshow_log(speckles, extent=extent)
        ep.phasor([field], ax=image.ax, at=(x, y), size=1.0, ring=True)

        ep.phasor([1.0, 0.5j], starts=[0j, 1.0], show_sum=True, separate=0.1)
    """
    if at is not None and ax is None:
        raise ValueError("phasor at= draws a dial into an existing ax; pass ax")
    if size_units not in _SIZE_UNITS:
        raise ValueError(
            f"unknown phasor size_units: {size_units!r}; known: {list(_SIZE_UNITS)}"
        )
    values = np.asarray(vectors, dtype=complex).ravel()
    n = len(values)
    starts = _resolve_starts(starts, n, chain, "phasor")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    if at is not None:
        ax = _dial_axes(ax, at, size, size_units)

    rc = matplotlib.rcParams
    base_width = float(rc["lines.linewidth"])
    colors = _per_arrow(colors, n, _style.color(0), "colors", single=is_color_like)
    linestyles = _per_arrow(linestyles, n, "-", "linestyles", single=_is_linestyle)
    heads = [float(h) for h in _per_arrow(head_scale, n, 1.0, "head_scale")]
    if widths is None:
        widths = [base_width * _shaft_fraction(h) for h in heads]
    widths = _per_arrow(widths, n, base_width, "widths")
    zorders = [5] * n
    if show_sum:
        if sum_head_scale is None:
            sum_head = max(heads, default=1.0)
        else:
            sum_head = float(sum_head_scale)
        if sum_width is None:
            sum_width = 1.5 * base_width * _shaft_fraction(sum_head)
        colors.append(rc["text.color"] if sum_color is None else sum_color)
        linestyles.append("-")
        widths.append(float(sum_width))
        heads.append(sum_head)
        zorders.append(4)

    state = {"origin": complex(origin), "starts": starts}
    starts, ends = _segments(values, state["origin"], chain, show_sum, starts)
    points = np.concatenate([starts, ends])
    if cross:
        points = np.append(points, 0j)
    if ring:
        radius = float(ring_radius)
        points = np.append(points, [radius + radius * 1j, -radius - radius * 1j])
    xmin, xmax, ymin, ymax = _resolve_limits(lim, points)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    offset, tol = float(separate) * half, 1e-9 * half
    starts, ends = _separate(starts, ends, offset, tol)
    head_pt = HEAD_PER_MARKER * float(rc["lines.markersize"])
    style = f"-|>,head_length={HEAD_LENGTH},head_width={HEAD_WIDTH}"
    arrows = []
    # A head of zero size is a plain shaft: matplotlib's head geometry
    # divides by the head's size, so a zero head never reaches it.
    for start, end, color, ls, lw, head, zorder in zip(
        starts, ends, colors, linestyles, widths, heads, zorders, strict=True
    ):
        kw = {
            "arrowstyle": style if head > 0.0 else "-",
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
            tkw = {
                "color": _style.neutral(0.6),
                "fontsize": "small",
                "zorder": 1,
                **(text_kw or {}),
            }
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

    if ring:
        collection = phase_ring(
            ax, radius=radius, cmap=ring_cmap, width=3.0 * base_width
        ).artists["collection"]
        collection.set(**(collection_kw or {}))
        artists["collection"] = collection
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def update(new_vectors, origin=None, starts=None):
        new_values = np.asarray(new_vectors, dtype=complex).ravel()
        if len(new_values) != n:
            raise ValueError(
                f"phasor update got {len(new_values)} vectors for {n} arrows"
            )
        if starts is not None:
            state["starts"] = _resolve_starts(starts, n, chain, "phasor update")
        if origin is not None:
            state["origin"] = complex(origin)
        new_starts, new_ends = _segments(
            new_values, state["origin"], chain, show_sum, state["starts"]
        )
        new_starts, new_ends = _separate(new_starts, new_ends, offset, tol)
        for patch, start, end in zip(arrows, new_starts, new_ends, strict=True):
            patch.set_positions(_xy(start), _xy(end))

    return PlotResult(ax=ax, artists=artists, update=update)
