"""A magnified inset that follows a moving point.

`follow_zoom` hangs a square inset in a corner of a panel. The inset shows
the panel's own artists through a window centered on a target, magnified by
a fixed factor, and a frame box on the panel marks the region the window
covers, joined to the inset by two connector lines. The window, the box and
the connectors are all recomputed each time the figure is drawn, so they
follow the target through layout, resizing and new limits with no per-frame
call.
"""

import contextlib
import math

import matplotlib
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection
from matplotlib.image import _ImageBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.spines import Spine
from matplotlib.transforms import Bbox, IdentityTransform

from eyepiece import _style
from eyepiece._motion import _apply_ease as _ease
from eyepiece._result import PlotResult
from eyepiece.attention import (
    Anchor,
    _as_anchor,
    _check_u,
    _is_reference,
    _Link,
)
from eyepiece.emphasis import _resolve_background

# Where the inset sits in the panel: which sides of the panel it touches.
_LOCS = {
    "upper right": (1, 1),
    "upper left": (-1, 1),
    "lower right": (1, -1),
    "lower left": (-1, -1),
    "right": (1, 0),
    "left": (-1, 0),
    "upper": (0, 1),
    "lower": (0, -1),
}

# The corners `loc="auto"` chooses among, and how much farther (in insets' sides)
# another corner must be before the inset moves to it.
_CORNERS = ("upper right", "upper left", "lower left", "lower right")
_AUTO_MARGIN = 0.25

# Gap between the inset and the panel edge, as a fraction of the inset side.
_MARGIN = 0.08


class _PointAnchor(Anchor):
    """An anchor whose display point is computed by a callable at draw time."""

    def __init__(self, parent, point):
        super().__init__(ax=parent, xy=(0.0, 0.0))
        self._point = point

    def display(self, renderer=None):
        return np.asarray(self._point(), dtype=float)

    def box(self, renderer=None):
        x, y = self.display(renderer)
        return Bbox.from_bounds(float(x), float(y), 0.0, 0.0)


class _Zoom:
    """The geometry of one following zoom, always as the parent is now."""

    def __init__(self, parent, target, zoom, size, loc, time):
        self.parent = parent
        self.zoom = float(zoom)
        self.size = float(size)
        self.loc = loc
        self.time = float(time)
        self.target = target
        self.alpha = 1.0
        self._held = None

    def center(self):
        """The target in the parent's data coordinates, as it is now."""
        t = self.target
        if callable(t):
            x, y = t(self.time)
            return float(x), float(y)
        if isinstance(t, Anchor):
            point = t.display()
            return tuple(
                float(v) for v in self.parent.transData.inverted().transform(point)
            )
        return t

    def corner(self):
        """The place the inset sits now: `loc`, or the corner `"auto"` holds.

        With ``"auto"`` the inset takes the corner farthest from the target
        (largest of the horizontal and vertical gaps between the target and
        the inset's center), and keeps it until another corner is farther by
        more than `_AUTO_MARGIN` of the inset's side. The held corner is
        state, so a target hovering where two corners tie never flips it.
        """
        if self.loc != "auto":
            return self.loc
        frame = self.parent.bbox
        side = self.size * min(frame.width, frame.height)
        tx, ty = self.parent.transData.transform(self.center())
        gaps = {}
        for name in _CORNERS:
            box = self._box(name)
            gaps[name] = max(
                abs(tx - 0.5 * (box.x0 + box.x1)), abs(ty - 0.5 * (box.y0 + box.y1))
            )
        best = max(gaps, key=gaps.get)
        held = self._held
        if held is None or gaps[best] > gaps[held] + _AUTO_MARGIN * side:
            self._held = best
        return self._held

    def inset_box(self):
        """The inset's box in display pixels."""
        return self._box(self.corner())

    def _box(self, loc):
        frame = self.parent.bbox
        side = self.size * min(frame.width, frame.height)
        gap = _MARGIN * side
        sx, sy = _LOCS[loc]
        x0 = {-1: frame.x0 + gap, 0: frame.x0 + 0.5 * (frame.width - side)}.get(
            sx, frame.x1 - gap - side
        )
        y0 = {-1: frame.y0 + gap, 0: frame.y0 + 0.5 * (frame.height - side)}.get(
            sy, frame.y1 - gap - side
        )
        return Bbox.from_bounds(x0, y0, side, side)

    def view(self):
        """The data window ``((x0, x1), (y0, y1))`` the inset shows."""
        parent = self.parent
        box = self.inset_box()
        cx, cy = self.center()
        (xa, xb), (ya, yb) = parent.get_xlim(), parent.get_ylim()
        frame = parent.bbox
        # Data units per pixel on the parent, signed so inverted axes stay so.
        ux = (xb - xa) / frame.width
        uy = (yb - ya) / frame.height
        hx = 0.5 * box.width * ux / self.zoom
        hy = 0.5 * box.height * uy / self.zoom
        return (cx - hx, cx + hx), (cy - hy, cy + hy)

    def corners(self):
        """The frame box on the parent: display corners, counterclockwise."""
        (x0, x1), (y0, y1) = self.view()
        return self.parent.transData.transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def pairs(self):
        """Two ``(box corner, inset corner)`` display pairs for the connectors.

        Rectangles that overlap in one direction join their facing corners;
        diagonal ones join the corners on their silhouette. Overlapping
        rectangles return degenerate pairs, which draw nothing.
        """
        a = self.corners()
        lo, hi = a.min(axis=0), a.max(axis=0)
        ib = self.inset_box()
        ilo, ihi = np.array([ib.x0, ib.y0]), np.array([ib.x1, ib.y1])
        over_x = lo[0] < ihi[0] and ilo[0] < hi[0]
        over_y = lo[1] < ihi[1] and ilo[1] < hi[1]

        def corner(low, high, c):
            return np.array(
                [high[0] if c[0] > 0 else low[0], high[1] if c[1] > 0 else low[1]]
            )

        if over_x and over_y:
            return [(lo, lo), (lo, lo)]
        d = 0.5 * (ilo + ihi) - 0.5 * (lo + hi)
        if over_y:
            s = 1 if d[0] > 0 else -1
            picks = [((s, 1), (-s, 1)), ((s, -1), (-s, -1))]
        elif over_x:
            s = 1 if d[1] > 0 else -1
            picks = [((1, s), (1, -s)), ((-1, s), (-1, -s))]
        else:
            sx = 1 if d[0] > 0 else -1
            sy = 1 if d[1] > 0 else -1
            picks = [((-sx, sy), (-sx, sy)), ((sx, -sy), (sx, -sy))]
        return [(corner(lo, hi, p), corner(ilo, ihi, q)) for p, q in picks]


class _Locator:
    """Place the inset from the zoom geometry when the inset is drawn."""

    def __init__(self, zoom):
        self._zoom = zoom

    def __call__(self, ax, renderer):
        fig = self._zoom.parent.figure
        return self._zoom.inset_box().transformed(fig.transSubfigure.inverted())


class _ZoomBox(Rectangle):
    """The frame box on the parent, recomputed from the target at each draw."""

    def __init__(self, zoom, **kw):
        super().__init__((0, 0), 1, 1, transform=IdentityTransform(), **kw)
        self._zoom = zoom

    def draw(self, renderer):
        """Move onto the target, then draw."""
        if not self.get_visible():
            return
        c = self._zoom.corners()
        self.set_bounds(c[:, 0].min(), c[:, 1].min(), np.ptp(c[:, 0]), np.ptp(c[:, 1]))
        super().draw(renderer)


def _zoom_link(zoom, index, **kw):
    """A connector from the frame box to the inset, audited as a link is.

    The inset is a panel of its own, so the line joins two panels and keeps
    the rules of a `link`: it crosses no data of the parent and no text, and
    takes no data color. The quarter-panel length of a leader does not apply.
    """
    parent = zoom.parent
    source = _PointAnchor(parent, lambda: zoom.pairs()[index][0])
    target = _PointAnchor(parent, lambda: zoom.pairs()[index][1])
    return _Link(source, target, **kw)


class _Mirror(Artist):
    """Draws the parent's own artists through the inset's window.

    Each eligible artist of the parent is drawn once, with its transform,
    clip and opacity borrowed for the draw and returned after it. The
    artists are the parent's own, so data that change between frames show
    in the inset with no copy to refresh.
    """

    def __init__(self, zoom, inset, exclude):
        super().__init__()
        self._zoom = zoom
        self._inset = inset
        self._exclude = list(exclude)
        self.set_zorder(1.5)

    def get_array(self):
        """No data array: a stage weighs this artist by its opacity alone."""
        return None

    def _eligible(self):
        parent = self._zoom.parent
        out = []
        for child in parent.get_children():
            if child is parent.patch or isinstance(child, (Spine, Axes)):
                continue
            if child in self._exclude or not child.get_visible():
                continue
            if not isinstance(child, (Line2D, Patch, Collection, _ImageBase)):
                continue
            if _is_reference(child, parent):
                continue
            out.append(child)
        return sorted(out, key=lambda a: a.get_zorder())

    def sync(self):
        """Set the inset's limits to the window around the target."""
        (x0, x1), (y0, y1) = self._zoom.view()
        self._inset.set_xlim(x0, x1)
        self._inset.set_ylim(y0, y1)

    def draw(self, renderer):
        """Draw the parent's artists through the window."""
        if not self.get_visible():
            return
        zoom, inset = self._zoom, self._inset
        self.sync()
        a = zoom.alpha * (1.0 if self.get_alpha() is None else self.get_alpha())
        old = zoom.parent.transData
        new = inset.transData
        for artist in self._eligible():
            with _borrowed(artist, old, new, inset, a):
                artist.draw(renderer)


def _moved(transform, old, new):
    if transform is old:
        return new
    return transform + old.inverted() + new


@contextlib.contextmanager
def _borrowed(artist, old, new, inset, alpha):
    """Aim `artist` at the inset's window for one draw, then put it back."""
    saved = {
        "transform": Artist.get_transform(artist),
        "clip_box": artist.get_clip_box(),
        "clip_path": artist.get_clip_path(),
        "clip_on": artist.get_clip_on(),
        "alpha": artist.get_alpha(),
    }
    offset = artist.get_offset_transform() if isinstance(artist, Collection) else None
    clip_path = getattr(artist, "_clippath", None)
    try:
        artist.set_transform(_moved(saved["transform"], old, new))
        if offset is not None:
            artist.set_offset_transform(_moved(offset, old, new))
        artist.set_clip_box(inset.bbox)
        artist.set_clip_path(inset.patch)
        artist.set_clip_on(True)
        if alpha < 1.0:
            base = 1.0 if saved["alpha"] is None else saved["alpha"]
            artist.set_alpha(base * alpha)
        yield
    finally:
        artist.set_transform(saved["transform"])
        if offset is not None:
            artist.set_offset_transform(offset)
        artist.set_clip_box(saved["clip_box"])
        artist._clippath = clip_path
        artist.set_clip_on(saved["clip_on"])
        if alpha < 1.0:
            artist.set_alpha(saved["alpha"])


def _target(target, parent):
    """A callable of time, an anchor, or a fixed data point."""
    if callable(target) and not isinstance(target, (Anchor, Artist)):
        return target
    if isinstance(target, (Anchor, Artist)):
        anchor = _as_anchor(target, "target")
        return anchor
    try:
        x, y = (float(v) for v in target)
    except (TypeError, ValueError):
        raise TypeError(
            "follow_zoom target must be an (x, y) callable, an Anchor, an "
            f"artist, or an (x, y) pair, got {type(target).__name__}"
        ) from None
    return (x, y)


def follow_zoom(
    ax,
    target,
    *,
    zoom=4.0,
    loc="upper right",
    size=0.35,
    time=0.0,
    color=None,
    exclude=(),
    ease="smoothstep",
    avoid=None,
    strict=True,
    line_kw=None,
):
    """Hang a magnified inset on a panel that follows a moving point.

    The inset shows the panel's own artists (lines, patches, collections
    and images; they are drawn again, not re-plotted, so a moving mark and
    changing data show in it with nothing to refresh) through a window
    centered on `target` and `zoom` times closer than the panel. A frame
    box on the panel marks the region the window covers, and two connector
    lines join the box to the inset. Texts, reference lines drawn against
    the axes frame (`axvline`), and the inset's own parts are not drawn in
    the inset. Everything is recomputed each time the figure is drawn, so
    the window follows a target that moves, and survives layout, resizing,
    and new limits. The window is not clamped: a target near the panel edge
    is framed past it.

    The connectors are audited like a `link`, since the inset is a panel of
    its own: they cross no data of the panel and no text and take a neutral
    rather than a data color (the quarter-panel length of a `leader` does
    not apply to a line between panels). They are hidden while the frame
    box overlaps the inset, so place the inset in a corner the target does
    not visit, or give `loc` as ``"auto"``. The box, the inset's frame and
    the connectors each carry a halo in the panel's background color, so
    they read over a dark image and a light one.

    With ``loc="auto"`` the inset sits in the corner farthest from the
    target, measured as the larger of the horizontal and vertical gaps from
    the target to the inset's center, and moves to another corner only when
    that corner is farther by more than a quarter of the inset's side. The
    corner held is state kept between draws, so a target hovering where two
    corners tie does not make the inset flicker; it jumps once per crossing,
    never animated. For a `size` of 0.4 or less the target is never under
    the inset.

    ``update(u, level=1.0, time=None)`` fades the inset, the box and the
    connectors in with `u` (``u = 0`` draws nothing, as if no zoom were
    made; ``u = 1``, the rest state, draws them whole); `level` is their
    opacity, 1 as drawn, and never changes geometry. `time` sets the
    argument a callable target is called with, and keeps its value until
    changed.

    Args:
        ax: The panel to zoom into. Both axes must be linear.
        target: What the window follows: a callable ``f(time) -> (x, y)`` in
            the panel's data units, an `Anchor` or an artist whose position
            is read at draw time (an artist resolves at the center of its
            box), or a fixed ``(x, y)`` data pair.
        zoom: Magnification, greater than 1: pixels per data unit in the
            inset over pixels per data unit in the panel, exactly, on both
            axes.
        loc: Where the inset sits: ``"upper right"``, ``"upper left"``,
            ``"lower right"``, ``"lower left"``, a side, ``"right"``,
            ``"left"``, ``"upper"``, ``"lower"``, or ``"auto"``, the corner
            farthest from the target (see above).
        size: Side of the square inset, as a fraction of the panel's shorter
            side, in (0, 1].
        time: The initial argument of a callable target.
        color: Color of the box and the inset frame. None uses the text
            color at call time.
        exclude: Artists of the panel to leave out of the inset.
        ease: An easing kind for `eyepiece.ease`, or a callable, applied to
            `u`; the opacity is clipped to [0, 1].
        avoid: The artists the connectors must not cross; None audits every
            line, patch, and collection of the panel.
        strict: Whether `audit` raises when a connector breaks a rule.
        line_kw: Extra kwargs for the connector `Line2D`, applied last.

    Returns:
        A `PlotResult` on `ax`, with ``artists["ellipse"]`` the frame box (a
        `Rectangle` on the figure, not in the panel's autoscaling),
        ``artists["lines"]`` the two connectors (whose ``faults()`` list the
        rules they break), the inset axes in ``insets``, and
        ``update(u, level=1.0, time=None)``.

    Raises:
        ValueError: If `zoom` is not above 1 or not finite, `size` is not in
            (0, 1], `loc` is unknown, or an axis of `ax` is not linear.
        TypeError: If `ax` is not an Axes, or `target` is none of the forms
            above.

    Example::

        res = ep.follow_zoom(ax, lambda t: (np.cos(t), np.sin(t)), zoom=5.0)
        res.update(1.0, time=0.7)  # the window is centered on the target at 0.7
        ep.audit(fig)
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"follow_zoom needs an Axes, got {type(ax).__name__}")
    zoom = float(zoom)
    if not (math.isfinite(zoom) and zoom > 1.0):
        raise ValueError(f"follow_zoom zoom must be above 1, got {zoom}")
    size = float(size)
    if not 0.0 < size <= 1.0:
        raise ValueError(f"follow_zoom size must be in (0, 1], got {size}")
    if loc != "auto" and loc not in _LOCS:
        raise ValueError(f"unknown follow_zoom loc: {loc!r}; known: {[*_LOCS, 'auto']}")
    if ax.get_xscale() != "linear" or ax.get_yscale() != "linear":
        raise ValueError("follow_zoom needs linear axes")

    rc = matplotlib.rcParams
    state = _Zoom(ax, _target(target, ax), zoom, size, loc, time)
    edge = rc["text.color"] if color is None else color
    fig = ax.figure

    inset = ax.inset_axes([0.0, 0.0, 1.0, 1.0], zorder=6)
    inset.set_axes_locator(_Locator(state))
    inset.set_position(_Locator(state)(inset, None))
    inset.set_in_layout(False)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.patch.set_facecolor(_resolve_background(ax))
    # A halo in the panel's own background under every line, so the frame
    # reads on a dark image and on a light one alike.
    halo = matplotlib.colors.to_rgba(_resolve_background(ax))

    def haloed(artist, width, alpha=1.0):
        artist.set_path_effects(
            [
                pe.Stroke(
                    linewidth=width + 2.4, foreground=(*halo[:3], halo[3] * alpha)
                ),
                pe.Normal(),
            ]
        )

    for spine in inset.spines.values():
        spine.set_edgecolor(edge)
        spine.set_linewidth(1.0)
        haloed(spine, 1.0)
    mirror = _Mirror(state, inset, exclude)
    inset.add_artist(mirror)
    mirror.sync()

    box = _ZoomBox(
        state, fill=False, edgecolor=edge, linewidth=1.0, zorder=15, clip_on=False
    )
    box.set_in_layout(False)
    haloed(box, 1.0)
    fig.add_artist(box)

    lkw = {
        "color": _style.neutral(0.6),
        "lw": 0.6 * float(rc["lines.linewidth"]),
        "solid_capstyle": "butt",
        "zorder": 15,
        **(line_kw or {}),
    }
    links = []
    for index in (0, 1):
        line = _zoom_link(
            state,
            index,
            avoid=None if avoid is None else list(avoid),
            strict=strict,
            **lkw,
        )
        line.set_clip_on(False)
        line.set_in_layout(False)
        haloed(line, line.get_linewidth())
        fig.add_artist(line)
        links.append(line)

    bases = {
        "box": box.get_alpha(),
        "links": [ln.get_alpha() for ln in links],
        "face": inset.patch.get_alpha(),
        "spines": [s.get_alpha() for s in inset.spines.values()],
    }

    def update(u, level=1.0, *, time=None):
        u = _check_u(u)
        level = _check_u(level)
        if time is not None:
            state.time = float(time)
        a = min(1.0, max(0.0, _ease(u, ease))) if u < 1.0 else 1.0
        a *= level
        state.alpha = a
        shown = u > 0.0 and a > 0.0
        for part in (inset, box, *links):
            part.set_visible(shown)

        def scaled(base):
            return base if a >= 1.0 else (1.0 if base is None else base) * a

        box.set_alpha(scaled(bases["box"]))
        for line, base in zip(links, bases["links"], strict=True):
            line.set_alpha(scaled(base))
        inset.patch.set_alpha(scaled(bases["face"]))
        for spine, base in zip(inset.spines.values(), bases["spines"], strict=True):
            spine.set_alpha(scaled(base))
        for part in (box, *links, *inset.spines.values()):
            haloed(part, part.get_linewidth(), a)

    update(1.0)
    return PlotResult(
        ax=ax,
        artists={"ellipse": box, "lines": links},
        update=update,
        insets=(inset,),
    )
