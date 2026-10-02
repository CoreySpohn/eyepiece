"""Small axes standing over chosen points of a curve.

`curve_insets` hangs a square inset above each marked point of a curve the
caller has drawn, joined to its point by a short connector, so a quantity
plotted along the curve can show, at a few places, the picture it was read
from: an image at that separation, or the chain of arrows whose resultant is
the brightness at that angle. The insets are placed at every draw from the
parent's data transform, so they stay on their points through layout,
resizing, and limit changes, as a phasor dial does.
"""

import matplotlib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, IdentityTransform

from eyepiece import _style
from eyepiece._phasor import _auto_limits, phasor
from eyepiece._result import PlotResult

# Units an inset's `size` can be given in.
_SIZE_UNITS = ("axes", "data")


class _StandLocator:
    """Place a square inset standing above a point of its parent's data.

    The inset is centered on `x` and its bottom edge sits `above` (a
    fraction of the parent's height) over the height `base`, both read
    through the parent's data transform when the inset is drawn. `size` is
    a fraction of the parent's shorter side (`"axes"`) or a width in the
    parent's x data units (`"data"`).
    """

    def __init__(self, parent, x, base, above, size, size_units):
        self._parent = parent
        self._x = x
        self._base = base
        self._above = above
        self._size = size
        self._units = size_units

    def box(self):
        """The inset's box in display coordinates, as the parent is now."""
        parent = self._parent
        frame = parent.bbox
        px, py = parent.transData.transform((self._x, self._base))
        if self._units == "axes":
            side = self._size * min(frame.width, frame.height)
        else:
            right = parent.transData.transform((self._x + self._size, self._base))[0]
            side = abs(right - px)
        bottom = py + self._above * frame.height
        return Bbox.from_bounds(px - 0.5 * side, bottom, side, side)

    def __call__(self, ax, renderer):
        parent = self._parent
        return self.box().transformed(parent.figure.transSubfigure.inverted())


class _Connector(Line2D):
    """A line from a point of the parent to the bottom of its inset.

    One end is in the parent's data coordinates and the other on an inset
    whose box is computed at draw time, which no single transform can
    express, so the line recomputes both ends in display coordinates each
    time it is drawn.
    """

    def __init__(self, parent, point, locator, **kwargs):
        self._parent = parent
        self._point = point
        self._locator = locator
        super().__init__(*self._ends(), transform=IdentityTransform(), **kwargs)

    def _ends(self):
        px, py = self._parent.transData.transform(self._point)
        box = self._locator.box()
        return [px, 0.5 * (box.x0 + box.x1)], [py, box.y0]

    def draw(self, renderer):
        """Recompute both ends from the current geometry, then draw."""
        self.set_data(*self._ends())
        super().draw(renderer)


def _per_mark(value, n, name):
    """Broadcast a scalar to `n` floats, or check a length-`n` sequence."""
    if np.ndim(value) == 0:
        return [float(value)] * n
    values = [float(v) for v in value]
    if len(values) != n:
        raise ValueError(f"curve_insets {name} has {len(values)} entries for {n} marks")
    return values


def curve_insets(
    ax,
    x,
    y,
    marks,
    *,
    size=0.18,
    size_units="axes",
    above=0.1,
    heights=None,
    build=None,
    chains=None,
    points=True,
    connectors=True,
    color=None,
    phasor_kw=None,
    scatter_kw=None,
    line_kw=None,
):
    """Stand a square inset above each marked point of a curve.

    The curve itself is the caller's, drawn already or drawn afterwards;
    `x` and `y` only say where it runs, so each mark's height is the curve
    interpolated there. Above each mark, `above` of the parent's height
    higher, stands a square inset, joined to its point by a thin vertical
    connector, and the point itself is marked with a dot. Each inset is
    filled by `build(inset_ax, i)`, or, given complex `chains`, holds that
    mark's chain of arrows and its resultant drawn by `phasor`.

    The insets are placed each time the figure is drawn, from the parent's
    box and data transform as they are then, so they stay on their points
    through layout, resizing, and later limit changes, and they are kept out
    of the layout engine so the parent gives up no room to them. Set the
    parent's limits to leave room above the curve for them.

    Args:
        ax: Parent axes, holding (or about to hold) the curve.
        x: 1D array-like of the curve's x values, strictly increasing.
        y: 1D array-like of the curve's y values, one per `x`.
        marks: x positions of the marked points, in data units.
        size: Side of each square inset, in `size_units`.
        size_units: `"axes"` (the default) gives `size` as a fraction of the
            parent's shorter side on screen, as a phasor dial's `"axes"`
            size is. `"data"` gives it as a width in the parent's x data
            units.
        above: Gap between the point an inset stands on and the inset's
            bottom edge, as a fraction of the parent's height.
        heights: Data height each inset stands on in place of its point,
            one value for a level row of insets or one per mark, for insets
            staggered to keep neighbors apart; the connector still runs from
            the point. None stands each inset on its own point.
        build: Callable `build(inset_ax, i)` called once per inset, in mark
            order, to fill it. None leaves the insets empty for the caller.
        chains: One complex array per mark, the arrows of that mark's chain,
            drawn tip to tail from 0 with their resultant by `phasor(chain=
            True, show_sum=True)`. Every chain is drawn on one square frame
            fitted around all of them, so arrow lengths compare between
            insets. The insets then have no face, like a phasor dial.
            Cannot be combined with `build`.
        points: Mark each point with a dot.
        connectors: Draw the connector from each point to its inset.
        color: Color of the dots. None uses `_style.color(0)`, the color a
            single curve takes by default.
        phasor_kw: Extra kwargs for every `phasor` call (for example
            `head_scale`, `colors`, `sum_color`, or `separate`), applied
            last over `chain=True`, `show_sum=True`, `cross=False`, and the
            shared frame as `lim`. Ignored without `chains`.
        scatter_kw: Extra kwargs for the dots' `ax.scatter`, applied last.
        line_kw: Extra kwargs for every connector `Line2D`, applied last.

    Returns:
        A `PlotResult` on the parent, with artists `"scatter"` (the dots'
        `PathCollection`, when `points`) and `"lines"` (the connector
        `Line2D` list, in mark order, when `connectors`), and the inset axes
        in `result.insets`, a tuple in mark order. With `chains`, each
        inset's arrows are its `patches`, the resultant last. There is no
        `update`.

    Raises:
        ValueError: If `x` and `y` differ in length or `x` is not strictly
            increasing, `size` is not positive, `size_units` is unknown,
            `heights` or `chains` does not match `marks`, or both `build`
            and `chains` are given.

    Example::

        theta = np.linspace(0.0, 3.0, 400)
        ax.plot(theta, np.sinc(theta) ** 2)
        ax.set_ylim(-0.05, 1.9)
        res = ep.curve_insets(ax, theta, np.sinc(theta) ** 2, [0.0, 0.5, 1.0])
        for inset in res.insets:
            inset.set_axis_off()
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"curve_insets x has {x.size} entries and y {y.size}")
    if x.size > 1 and not np.all(np.diff(x) > 0.0):
        raise ValueError("curve_insets x must be strictly increasing")
    if size_units not in _SIZE_UNITS:
        raise ValueError(
            f"unknown curve_insets size_units: {size_units!r}; "
            f"known: {list(_SIZE_UNITS)}"
        )
    size = float(size)
    if not size > 0.0:
        raise ValueError(f"curve_insets size must be positive, got {size}")
    if build is not None and chains is not None:
        raise ValueError("curve_insets takes build or chains, not both")
    marks = [float(m) for m in np.atleast_1d(marks)]
    n = len(marks)
    tops = np.interp(marks, x, y)
    bases = list(tops) if heights is None else _per_mark(heights, n, "heights")
    if chains is not None:
        chains = [np.asarray(c, dtype=complex).ravel() for c in chains]
        if len(chains) != n:
            raise ValueError(
                f"curve_insets chains has {len(chains)} entries for {n} marks"
            )

    rc = matplotlib.rcParams
    insets = []
    connector_lines = []
    lkw = {
        "color": _style.neutral(0.45),
        "lw": 0.7 * float(rc["lines.linewidth"]),
        "zorder": 4,
        **(line_kw or {}),
    }
    for mark, top, base in zip(marks, tops, bases, strict=True):
        locator = _StandLocator(ax, mark, base, float(above), size, size_units)
        inset = ax.inset_axes([0.0, 0.0, 1.0, 1.0], zorder=6)
        inset.set_axes_locator(locator)
        inset.set_position(locator(inset, None))
        inset.set_in_layout(False)
        insets.append(inset)
        if connectors:
            line = _Connector(ax, (mark, float(top)), locator, **lkw)
            line.set_clip_on(False)
            line.set_in_layout(False)
            connector_lines.append(ax.add_artist(line))

    artists = {}
    if points:
        skw = {
            "s": (0.7 * float(rc["lines.markersize"])) ** 2,
            "color": _style.color(0, color),
            "lw": 0,
            "zorder": 5,
            **(scatter_kw or {}),
        }
        artists["scatter"] = ax.scatter(marks, tops, **skw)
    if connector_lines:
        artists["lines"] = connector_lines

    if chains is not None:
        tips = [np.concatenate([[0j], np.cumsum(c)]) for c in chains]
        frame = _auto_limits(np.concatenate(tips))
        for inset, chain in zip(insets, chains, strict=True):
            inset.patch.set_alpha(0.0)
            pkw = {
                "chain": True,
                "show_sum": True,
                "cross": False,
                "lim": frame,
                **(phasor_kw or {}),
            }
            phasor(chain, ax=inset, **pkw)
    elif build is not None:
        for i, inset in enumerate(insets):
            build(inset, i)

    return PlotResult(ax=ax, artists=artists, insets=tuple(insets))
