"""Attention moves: anchors, a mark carried between panels, captions at their subject.

An explanatory animation has to say where to look. Three moves do most of
that work, and none of them is data: `flight` carries a copy of a mark from
one panel onto another axis, or into its label, so the viewer sees one
quantity become another; `callout` writes a short caption beside the thing
it names; and `leader` joins a caption to its mark when the two cannot
touch.

Every position is an `Anchor`, a data point of an axes or a part of a drawn
artist, resolved in display space each time the figure is drawn, so the
moves follow layout, resizing, and per-frame data changes the way
`curve_insets` does. Their artists belong to the figure rather than to an
axes: inside an axes they would join its autoscaling and could change the
data limits, which a recording reports as a drifting scale.

Each move returns a result whose `update(u)` sets its progress, `u` from 0
to 1, absolutely: the same `u` always draws the same picture, so an
animation frame still draws itself from its own state. Each has a rest
state, which is the state it is in when it is made: a flight is absent (its
copy hidden, its landing label shown), and a callout or a leader is fully
drawn. A still that never calls `update` shows rest states only.
"""

import math

import matplotlib
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.transforms import Bbox, IdentityTransform

from eyepiece import _style
from eyepiece._motion import ease as _ease
from eyepiece._result import MosaicResult, PlotResult

# Where on an artist's box `Anchor.from_artist` resolves, as box fractions.
_LOCS = {
    "center": (0.5, 0.5),
    "left": (0.0, 0.5),
    "right": (1.0, 0.5),
    "top": (0.5, 1.0),
    "bottom": (0.5, 0.0),
    "upper left": (0.0, 1.0),
    "upper right": (1.0, 1.0),
    "lower left": (0.0, 0.0),
    "lower right": (1.0, 0.0),
}

# A caption's side of its subject, and the alignment that puts it there.
_SIDES = ("right", "left", "above", "below")
_ALIGN = {
    "right": ("left", "center"),
    "left": ("right", "center"),
    "above": ("center", "bottom"),
    "below": ("center", "top"),
}

# Display pixels a leader keeps clear of its own end when it is audited, so
# a mark sitting on a curve does not count as the leader crossing the curve.
_END_CLEAR_PX = 2.0


def _check_u(u):
    u = float(u)
    if not math.isfinite(u):
        raise ValueError(f"progress must be finite, got {u}")
    return min(1.0, max(0.0, u))


def _root_figure(artist):
    return artist.get_figure(root=True)


def _pixels(renderer, figure, points):
    """`points` in display pixels at the dpi the figure is drawn at."""
    if renderer is not None:
        return float(renderer.points_to_pixels(points))
    return float(points) * figure.dpi / 72.0


class Anchor:
    """A point of a figure, resolved in display space when it is drawn.

    Make one with `from_data` (a data point of an axes) or `from_artist`
    (a part of a drawn artist's box). Nothing is computed when the anchor is
    made: `display` reads the transforms and the artist as they are when it
    is called, so an anchor stays on its point through layout changes,
    resizing, new limits, and an artist whose data change every frame.

    Attributes:
        axes: The axes the point belongs to: the `ax` of `from_data`, or the
            artist's axes. None for an artist drawn on the figure itself.
        artist: The artist of `from_artist`, or None.
        figure: The root figure the point is drawn in.
    """

    def __init__(self, *, ax=None, xy=None, artist=None, loc="center"):
        """Use `from_data` or `from_artist` rather than this."""
        self._ax = ax
        self._xy = None if xy is None else (float(xy[0]), float(xy[1]))
        self._artist = artist
        self._loc = loc

    @classmethod
    def from_data(cls, ax, xy):
        """A data point of an axes, read through its data transform.

        Args:
            ax: The axes whose data coordinates `xy` is in.
            xy: ``(x, y)`` in data units.

        Returns:
            An `Anchor`.

        Raises:
            TypeError: If `ax` is not an Axes.
        """
        if not isinstance(ax, Axes):
            raise TypeError(f"Anchor.from_data needs an Axes, got {type(ax).__name__}")
        return cls(ax=ax, xy=xy)

    @classmethod
    def from_artist(cls, artist, loc="center"):
        """A point of a drawn artist's box in display space.

        A marker drawn as a one-point `Line2D` resolves to the point itself
        at `"center"`, since its box is centered there. A text resolves on
        the box of its glyphs, so keep it visible (an alpha of 0 is fine)
        while it serves as an anchor.

        Args:
            artist: Any drawn artist with a window extent.
            loc: Where on the artist's box: ``"center"``, ``"left"``,
                ``"right"``, ``"top"``, ``"bottom"``, ``"upper left"``,
                ``"upper right"``, ``"lower left"`` or ``"lower right"``.

        Returns:
            An `Anchor`.

        Raises:
            TypeError: If `artist` is not an Artist.
            ValueError: If `loc` is unknown.
        """
        if not isinstance(artist, Artist):
            raise TypeError(
                f"Anchor.from_artist needs an Artist, got {type(artist).__name__}"
            )
        if loc not in _LOCS:
            raise ValueError(f"unknown anchor loc: {loc!r}; known: {list(_LOCS)}")
        return cls(artist=artist, loc=loc)

    @property
    def axes(self):
        """The axes the point belongs to, or None."""
        if self._ax is not None:
            return self._ax
        return getattr(self._artist, "axes", None)

    @property
    def artist(self):
        """The anchored artist, or None for a data point."""
        return self._artist

    @property
    def figure(self):
        """The root figure the point is drawn in."""
        owner = self._ax if self._ax is not None else self._artist
        figure = owner.get_figure(root=True)
        if figure is None:
            raise ValueError(
                f"anchored {type(owner).__name__} is not in a figure yet; add it first"
            )
        return figure

    def box(self, renderer=None):
        """The anchored artist's box in display pixels.

        A data point has no extent, so its box is that point, with zero
        width and height.

        Args:
            renderer: The renderer drawing the figure, or None outside a
                draw.

        Returns:
            A `Bbox` in display pixels.
        """
        if self._artist is None:
            x, y = self._ax.transData.transform(self._xy)
            return Bbox.from_bounds(float(x), float(y), 0.0, 0.0)
        return Bbox(self._artist.get_window_extent(renderer).get_points())

    def display(self, renderer=None):
        """The point in display pixels, as the figure is now.

        Args:
            renderer: The renderer drawing the figure, or None outside a
                draw.

        Returns:
            A float array ``[x, y]`` in display pixels.
        """
        if self._artist is None:
            return np.asarray(self._ax.transData.transform(self._xy), dtype=float)
        box = self.box(renderer)
        fx, fy = _LOCS[self._loc]
        return np.array([box.x0 + fx * box.width, box.y0 + fy * box.height])


def _as_anchor(target, name):
    if isinstance(target, Anchor):
        return target
    if isinstance(target, Artist):
        return Anchor.from_artist(target)
    raise TypeError(
        f"{name} must be an Anchor or an Artist, got {type(target).__name__}"
    )


class _TextFade:
    """Absolute opacity for a text and its backing box, from where it started."""

    def __init__(self, text):
        self._text = text
        self._alpha = text.get_alpha()
        patch = text.get_bbox_patch()
        self._face = None if patch is None else to_rgba(patch.get_facecolor())
        self._edge = None if patch is None else to_rgba(patch.get_edgecolor())

    def set(self, level):
        """Show the text at `level` of its starting opacity, box included."""
        text = self._text
        if level >= 1.0:
            text.set_alpha(self._alpha)
        else:
            text.set_alpha(level * (1.0 if self._alpha is None else self._alpha))
        patch = text.get_bbox_patch()
        if patch is not None and self._face is not None:
            k = min(1.0, level)
            patch.set_facecolor((*self._face[:3], self._face[3] * k))
            patch.set_edgecolor((*self._edge[:3], self._edge[3] * k))


# Flight


class _GhostLine(Line2D):
    """A copy of a marker that flies between two anchors."""

    def __init__(self, mark, path):
        super().__init__(
            [np.nan],
            [np.nan],
            transform=IdentityTransform(),
            linestyle="none",
            marker=mark.get_marker(),
            markersize=mark.get_markersize(),
            markerfacecolor=mark.get_markerfacecolor(),
            markeredgecolor=mark.get_markeredgecolor(),
            markeredgewidth=mark.get_markeredgewidth(),
            color=mark.get_color(),
            alpha=mark.get_alpha(),
        )
        self._route = path

    def draw(self, renderer):
        """Put the copy where the flight's progress says, then draw it."""
        if not self.get_visible():
            return
        x, y = self._route(renderer)
        self.set_data([x], [y])
        super().draw(renderer)


class _GhostText(Text):
    """A copy of a label that flies between two anchors."""

    def __init__(self, mark, path):
        super().__init__(
            0.0,
            0.0,
            mark.get_text(),
            transform=IdentityTransform(),
            color=mark.get_color(),
            fontproperties=mark.get_fontproperties(),
            alpha=mark.get_alpha(),
            ha="center",
            va="center",
        )
        patch = mark.get_bbox_patch()
        if patch is not None:
            self.set_bbox(
                {
                    "boxstyle": patch.get_boxstyle(),
                    "facecolor": patch.get_facecolor(),
                    "edgecolor": patch.get_edgecolor(),
                    "linewidth": patch.get_linewidth(),
                }
            )
        self._route = path

    def draw(self, renderer):
        """Put the copy where the flight's progress says, then draw it."""
        if not self.get_visible():
            return
        self.set_position(tuple(self._route(renderer)))
        super().draw(renderer)


def flight(
    source,
    destination,
    *,
    fig,
    arc=0.0,
    crossfade=None,
    mark=None,
    ease="smoothstep",
    fade=0.3,
    zorder=20,
):
    """Carry a copy of a mark from one place to another, across panels.

    The move shows that one quantity has become another: a dot read off a
    curve in one panel is carried onto the axis of the next, or a value is
    carried into the label that names it. The source stays where it is; a
    copy of it, the ghost, travels from the source anchor to the destination
    anchor and is hidden again when it lands, where the caller shows the
    destination mark. Both ends are resolved when the figure is drawn, so a
    flight lands on the destination as it is drawn then.

    The ghost is a figure-level artist. Inside an axes it would join the
    axes' autoscaling, and a recording would report the data limits it
    moved as a drifting scale.

    The ghost is visible only for ``0 < u < 1``. At ``u = 0`` and ``u = 1``
    it draws nothing, so those frames equal the frame without the flight;
    the rest state, when the flight is made, is ``u = 1``.

    Args:
        source: Where the ghost starts: an `Anchor`, or an artist (anchored
            at its center).
        destination: Where it lands: an `Anchor`, or an artist. To land on
            a mark the caller has not drawn yet, anchor the data point it
            will be drawn at with `Anchor.from_data`.
        fig: The figure that owns the ghost, normally the root figure.
        arc: How far the path bows away from the straight line, as a
            fraction of the distance flown, to the left of the direction of
            travel; negative bows right. 0 flies straight.
        crossfade: Optional `Text` the ghost lands into. Over the last
            `fade` of the flight the ghost fades out while this text fades
            in; it is at full opacity at ``u = 1`` and at none for progress
            before the fade starts. Keep it visible: its opacity carries the
            crossfade.
        mark: The artist the ghost copies the look of: a `Line2D` marker
            (marker, size, colors) or a `Text` (string, font, color, box).
            None copies the source anchor's artist.
        ease: An easing kind for `eyepiece.ease`, applied to `u` along the
            path.
        fade: Fraction of the flight, at its end, over which `crossfade`
            runs, in (0, 1].
        zorder: Drawing order of the ghost among the figure's children.

    Returns:
        A `MosaicResult` over the axes the two anchors belong to (source
        first), with ``artists["line"]`` (a `Line2D` ghost) or
        ``artists["text"]`` (a `Text` ghost), and ``update(u)`` that sets
        the progress; values outside [0, 1] are clipped.

    Raises:
        TypeError: If an anchor is neither an `Anchor` nor an artist, or the
            mark is neither a `Line2D` nor a `Text`.
        ValueError: If there is no mark to copy, or `fade` is outside
            (0, 1].

    Example::

        top_dot, = top.plot([0.35], [2.41], "o")
        carry = ep.flight(top_dot, ep.Anchor.from_data(bottom, (0.25, 2.41)),
                          fig=fig, arc=0.15)
        for u in np.linspace(0.0, 1.0, 30):
            carry.update(u)
            rec.frame()
    """
    source = _as_anchor(source, "source")
    destination = _as_anchor(destination, "destination")
    mark = source.artist if mark is None else mark
    if mark is None:
        raise ValueError("flight has no mark to copy: pass mark= or anchor an artist")
    fade = float(fade)
    if not 0.0 < fade <= 1.0:
        raise ValueError(f"flight fade must be in (0, 1], got {fade}")
    arc = float(arc)
    state = {"u": 1.0}

    def where(renderer):
        p0 = source.display(renderer)
        p1 = destination.display(renderer)
        e = _ease(state["u"], ease)
        step = p1 - p0
        # The control point sits twice as far out as the path's widest bow.
        control = 0.5 * (p0 + p1) + 2.0 * arc * np.array([-step[1], step[0]])
        return (1.0 - e) ** 2 * p0 + 2.0 * (1.0 - e) * e * control + e**2 * p1

    if isinstance(mark, Line2D):
        ghost, key = _GhostLine(mark, where), "line"
    elif isinstance(mark, Text):
        ghost, key = _GhostText(mark, where), "text"
    else:
        raise TypeError(f"flight copies a Line2D or a Text, got {type(mark).__name__}")
    ghost.set_zorder(zorder)
    ghost.set_clip_on(False)
    ghost.set_in_layout(False)
    fig.add_artist(ghost)
    if key == "text":
        ghost_fade = _TextFade(ghost).set
    else:
        alpha = 1.0 if ghost.get_alpha() is None else ghost.get_alpha()

        def ghost_fade(level):
            ghost.set_alpha(alpha * level)

    if crossfade is not None and crossfade.get_alpha() == 0.0:
        raise ValueError(
            "flight crossfade text has alpha 0, which is its landed opacity; "
            "leave its alpha as drawn and let the flight hide it"
        )
    landing = None if crossfade is None else _TextFade(crossfade)

    def update(u):
        u = _check_u(u)
        state["u"] = u
        ghost.set_visible(0.0 < u < 1.0)
        if landing is not None:
            w = min(1.0, max(0.0, (u - (1.0 - fade)) / fade))
            ghost_fade(1.0 - w)
            landing.set(w)

    update(1.0)
    axes = [a for a in (source.axes, destination.axes) if a is not None]
    axes = list(dict.fromkeys(axes)) or list(fig.axes)
    return MosaicResult(
        axes=np.array(axes, dtype=object), artists={key: ghost}, update=update
    )


# Leader


def _exit(box, inside, toward):
    """Where the ray from `inside` toward `toward` leaves `box`, or None.

    None when `toward` is itself inside the box, so the ray never leaves it
    before reaching its target.
    """
    d = toward - inside
    ts = []
    for k, (lo, hi) in enumerate(((box.x0, box.x1), (box.y0, box.y1))):
        if d[k] > 0.0:
            ts.append((hi - inside[k]) / d[k])
        elif d[k] < 0.0:
            ts.append((lo - inside[k]) / d[k])
    ts = [t for t in ts if t >= 0.0]
    if not ts:
        return None
    t = min(ts)
    if t >= 1.0:
        return None
    return inside + t * d


def _proper_crossings(p, q, a, b):
    """Whether segment p-q crosses any segment a[i]-b[i] at an interior point."""

    def orient(o, s, t):
        return (s[..., 0] - o[..., 0]) * (t[..., 1] - o[..., 1]) - (
            s[..., 1] - o[..., 1]
        ) * (t[..., 0] - o[..., 0])

    ok = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    a, b = a[ok], b[ok]
    if not len(a):
        return False
    p = np.broadcast_to(p, a.shape)
    q = np.broadcast_to(q, a.shape)
    d1 = orient(a, b, p)
    d2 = orient(a, b, q)
    d3 = orient(p, q, a)
    d4 = orient(p, q, b)
    return bool(np.any((d1 * d2 < 0.0) & (d3 * d4 < 0.0)))


def _polyline_crosses(points, p, q):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return False
    return _proper_crossings(p, q, points[:-1], points[1:])


def _near_points(points, p, q, radius):
    """Whether segment p-q passes within `radius` of any of `points`."""
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return False
    d = q - p
    length2 = float(d @ d)
    if length2 == 0.0:
        dist = np.hypot(*(points - p).T)
    else:
        t = np.clip((points - p) @ d / length2, 0.0, 1.0)
        dist = np.hypot(*(points - (p + t[:, None] * d)).T)
    return bool(np.any(dist < radius))


def _crosses(artist, p, q, renderer):
    """Whether the segment p-q (display pixels) crosses a data artist."""
    figure = _root_figure(artist)
    if isinstance(artist, Line2D):
        xy = artist.get_transform().transform(artist.get_xydata())
        if artist.get_linestyle() in ("None", "none", "", " "):
            radius = 0.5 * _pixels(renderer, figure, artist.get_markersize())
            return _near_points(xy, p, q, radius)
        return _polyline_crosses(xy, p, q)
    if isinstance(artist, Patch):
        polygons = artist.get_path().to_polygons(
            artist.get_transform(), closed_only=False
        )
        return any(_polyline_crosses(poly, p, q) for poly in polygons)
    if isinstance(artist, PathCollection):
        offsets = artist.get_offset_transform().transform(artist.get_offsets())
        sizes = artist.get_sizes()
        size = float(np.max(sizes)) if len(sizes) else 0.0
        radius = 0.5 * _pixels(renderer, figure, math.sqrt(size))
        return _near_points(offsets, p, q, radius)
    if isinstance(artist, Collection):
        transform = artist.get_transform()
        for path in artist.get_paths():
            for poly in path.to_polygons(transform, closed_only=False):
                if _polyline_crosses(poly, p, q):
                    return True
    return False


def _ends_on(artist, end, renderer):
    """Whether `end` is one of the marks a markers-only line draws.

    A leader anchored on a data point that the caller plots as a marker
    ends on that marker, which is the mark it names, not data it crosses.
    """
    if not isinstance(artist, Line2D):
        return False
    if artist.get_linestyle() not in ("None", "none", "", " "):
        return False
    xy = artist.get_transform().transform(artist.get_xydata())
    radius = 0.5 * _pixels(renderer, _root_figure(artist), artist.get_markersize())
    return _near_points(xy, end, end, radius + _END_CLEAR_PX)


def _data_colors(artist):
    if isinstance(artist, Line2D):
        return [to_rgba(artist.get_color())]
    if isinstance(artist, Patch):
        return [to_rgba(artist.get_edgecolor()), to_rgba(artist.get_facecolor())]
    if isinstance(artist, Collection):
        return [
            *map(to_rgba, artist.get_edgecolor()),
            *map(to_rgba, artist.get_facecolor()),
        ]
    return []


def _is_reference(artist, ax):
    """Whether an artist is drawn against the axes frame, like `axvline`.

    Reference lines and spans (`axhline`, `axvline`, `axvspan`) are scenery,
    not data: at least one of their coordinates is in axes units, so their
    transform does not run through the data transform.
    """
    return not artist.get_transform().contains_branch(ax.transData)


def _default_avoid(ax, exclude):
    """The data a leader in `ax` may not cross: lines, patches, collections."""
    out = []
    for child in ax.get_children():
        if child is ax.patch or isinstance(child, Spine) or child in exclude:
            continue
        if _is_reference(child, ax):
            continue
        if isinstance(child, (Line2D, Patch, LineCollection, PathCollection)) or (
            isinstance(child, Collection) and child.get_array() is None
        ):
            out.append(child)
    return out


class _Leader(Line2D):
    """A connector from a text's box to a mark, recomputed at every draw.

    Its geometry is audited against the leader rules; `faults` lists the
    rules the fully grown leader breaks as the figure is now.
    """

    def __init__(self, text, target, *, panel, avoid, min_gap_em, strict, **kw):
        super().__init__([], [], transform=IdentityTransform(), **kw)
        self._text = text
        self._target = target
        self._panel = panel
        self._avoid = avoid
        self._min_gap_em = float(min_gap_em)
        self._strict = bool(strict)
        self._grow = 1.0

    def _ends(self, renderer=None):
        """Start on the text's padded box and end on the mark's box, or None."""
        text = self._text
        if not (text.get_visible() and text.get_text()):
            return None
        place = getattr(text, "place", None)
        if place is not None:
            place(renderer)
        figure = _root_figure(text)
        em = _pixels(renderer, figure, text.get_fontsize())
        tbox = text.get_window_extent(renderer).padded(0.25 * em)
        target = self._target.display(renderer)
        center = np.array([0.5 * (tbox.x0 + tbox.x1), 0.5 * (tbox.y0 + tbox.y1)])
        start = _exit(tbox, center, target)
        if start is None:
            return None
        end = target
        if self._target.artist is not None:
            mbox = self._target.box(renderer)
            # Only an anchor strictly inside the box (its center, say) is
            # moved to the box edge; one on the edge is where the caller
            # asked the leader to end.
            inside = mbox.x0 < target[0] < mbox.x1 and mbox.y0 < target[1] < mbox.y1
            edge = _exit(mbox, target, start) if inside else None
            if edge is not None:
                end = edge
        if np.hypot(*(end - start)) <= self._min_gap_em * em:
            return None
        return start, end

    def _avoided(self):
        if self._avoid is not None:
            return [a for a in self._avoid if a.get_visible()]
        if self._panel is None:
            return []
        exclude = [self._target.artist] if self._target.artist is not None else []
        return [a for a in _default_avoid(self._panel, exclude) if a.get_visible()]

    def faults(self, renderer=None):
        """The leader rules the fully grown leader breaks as drawn now.

        The rules are those of a connector in the evidence guide: the leader
        ends on the mark it names and does not run through it, it is shorter
        than a quarter of its panel (the panel's longer side), it crosses no
        data and no other text, and it is not drawn in a color the data in
        its panel use. Reference lines and spans drawn against the axes
        frame (`axvline`, `axhline`, `axvspan`) are scenery, not data, and a
        marker the leader ends on is the mark it names. Images are not
        audited, since any line over an image crosses pixels.

        Args:
            renderer: The renderer drawing the figure, or None outside a
                draw.

        Returns:
            A list of strings, one per broken rule; empty when the leader
            keeps every rule or is not drawn at all.
        """
        ends = self._ends(renderer)
        if ends is None:
            return []
        start, end = ends
        out = []
        length = float(np.hypot(*(end - start)))
        if self._panel is not None:
            box = self._panel.bbox
            limit = 0.25 * max(box.width, box.height)
            if length > limit:
                out.append(
                    f"leader is {length:.0f} px long, more than a quarter of its "
                    f"panel ({limit:.0f} px)"
                )
        if length > _END_CLEAR_PX:
            short = end - _END_CLEAR_PX * (end - start) / length
        else:
            short = end
        if self._target.artist is not None:
            mbox = self._target.box(renderer)
            if mbox.width > 0.0 and mbox.height > 0.0:
                edges = np.array(
                    [
                        [mbox.x0, mbox.y0],
                        [mbox.x1, mbox.y0],
                        [mbox.x1, mbox.y1],
                        [mbox.x0, mbox.y1],
                        [mbox.x0, mbox.y0],
                    ]
                )
                if _polyline_crosses(edges, start, short):
                    out.append("leader runs through the mark it names")
        own = np.array(to_rgb(self.get_color()))
        for artist in self._avoided():
            if _ends_on(artist, end, renderer):
                continue
            if _crosses(artist, start, short, renderer):
                name = artist.get_label() or type(artist).__name__
                out.append(f"leader crosses data: {name}")
            for rgba in _data_colors(artist):
                if rgba[3] > 0.0 and np.allclose(own, rgba[:3], atol=2e-3):
                    name = artist.get_label() or type(artist).__name__
                    out.append(f"leader takes a data color of its panel: {name}")
                    break
        for other in self._texts():
            box = other.get_window_extent(renderer)
            edges = np.array(
                [[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]]
            )
            if _polyline_crosses(np.vstack([edges, edges[:1]]), start, short):
                out.append(f"leader crosses text: {other.get_text()!r}")
        return out

    def _texts(self):
        """Other visible texts of the panel and of the figure the leader runs by."""
        figure = _root_figure(self)
        pool = list(figure.texts) + [a for a in figure.artists if isinstance(a, Text)]
        if self._panel is not None:
            pool += list(self._panel.texts)
        return [
            t
            for t in dict.fromkeys(pool)
            if t is not self._text and t.get_visible() and t.get_text()
        ]

    def draw(self, renderer):
        """Grow from the text toward the mark, after auditing the full leader."""
        if not self.get_visible() or self._grow <= 0.0:
            return
        ends = self._ends(renderer)
        if ends is None:
            return
        if self._strict:
            broken = self.faults(renderer)
            if broken:
                raise ValueError("leader breaks the leader rules: " + "; ".join(broken))
        start, end = ends
        tip = start + self._grow * (end - start)
        self.set_data([start[0], tip[0]], [start[1], tip[1]])
        super().draw(renderer)


def _leader_line(
    text, target, *, color, lw, avoid, min_gap_em, strict, zorder, line_kw
):
    rc = matplotlib.rcParams
    kw = {
        "color": _style.neutral(0.6) if color is None else color,
        "lw": 0.6 * float(rc["lines.linewidth"]) if lw is None else lw,
        "solid_capstyle": "butt",
        "zorder": zorder,
        **(line_kw or {}),
    }
    line = _Leader(
        text,
        target,
        panel=target.axes,
        avoid=None if avoid is None else list(avoid),
        min_gap_em=min_gap_em,
        strict=strict,
        **kw,
    )
    line.set_clip_on(False)
    line.set_in_layout(False)
    target.figure.add_artist(line)
    return line


def leader(text, target, *, color=None, lw=None, avoid=None, strict=True, line_kw=None):
    """Draw a leader line from a text to the mark it names.

    A leader is a connector, and the evidence guide's rules for connectors
    apply: move the words to the mark where you can. Where you cannot, the
    leader starts at the text's box and ends on the mark (on the edge of an
    anchored artist's box, or at a data point), runs shorter than a quarter
    of its panel, crosses no data, and takes a neutral rather than a color
    that means something in the panel. The geometry is recomputed at every
    draw, so it follows the text and the mark.

    With `strict` (the default) drawing a leader that breaks a rule raises,
    so a recording fails instead of shipping the frame; the leader's
    `faults()` lists the broken rules without drawing.

    Args:
        text: The `Text` artist the leader starts from.
        target: The mark it names: an `Anchor`, or an artist.
        color: Line color. None uses a neutral two thirds of the way from
            the background to the text color, read now.
        lw: Line width in points. None uses 0.6 of
            ``rcParams["lines.linewidth"]``.
        avoid: The artists the leader must not cross. None audits every
            line, patch, and collection in the target's axes except the
            target itself, spines, and the axes background.
        strict: Raise on drawing a leader that breaks a rule.
        line_kw: Extra kwargs for the `Line2D`, applied last.

    Returns:
        A `PlotResult` on the target's axes (None for a target drawn on the
        figure), with ``artists["line"]``, the leader, whose ``faults()``
        lists the rules it breaks as the figure is now, and ``update(u)``
        that grows it from the text, ``u = 0`` hidden and ``u = 1`` (the
        rest state) reaching the mark.

    Raises:
        TypeError: If `text` is not a `Text`, or `target` is neither an
            `Anchor` nor an artist.

    Example::

        note = ax.text(0.8, 0.9, "first null", transform=ax.transAxes)
        res = ep.leader(note, ep.Anchor.from_data(ax, (1.22, 0.0)))
        res.update(0.5)  # halfway grown
    """
    if not isinstance(text, Text):
        raise TypeError(f"leader starts from a Text, got {type(text).__name__}")
    target = _as_anchor(target, "target")
    line = _leader_line(
        text,
        target,
        color=color,
        lw=lw,
        avoid=avoid,
        min_gap_em=0.0,
        strict=strict,
        zorder=text.get_zorder() - 0.01,
        line_kw=line_kw,
    )

    def update(u):
        line._grow = _check_u(u)

    return PlotResult(ax=target.axes, artists={"line": line}, update=update)


# Callout


class _AnchoredText(Text):
    """A caption placed beside an anchor, or at its own anchor, at every draw."""

    def __init__(self, text, *, target, side, at, offset_pt, **kw):
        ha, va = _ALIGN[side]
        super().__init__(
            0.0, 0.0, text, transform=IdentityTransform(), ha=ha, va=va, **kw
        )
        self._target = target
        self._side = side
        self._at = at
        self._offset_pt = float(offset_pt)

    def place(self, renderer=None):
        """Move the caption to its place as the figure is now."""
        if self._at is not None:
            x, y = self._at.display(renderer)
        else:
            box = self._target.box(renderer)
            gap = _pixels(renderer, _root_figure(self), self._offset_pt)
            xc, yc = 0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)
            x, y = {
                "right": (box.x1 + gap, yc),
                "left": (box.x0 - gap, yc),
                "above": (xc, box.y1 + gap),
                "below": (xc, box.y0 - gap),
            }[self._side]
        self.set_position((float(x), float(y)))

    def set_side(self, side):
        """Put the caption on another side of its subject."""
        ha, va = _ALIGN[side]
        self._side = side
        self.set_horizontalalignment(ha)
        self.set_verticalalignment(va)

    def draw(self, renderer):
        """Place the caption, then draw it."""
        if not self.get_visible():
            return
        self.place(renderer)
        super().draw(renderer)


def _fits(text, panel, mbox):
    """Whether the caption's box lies inside the panel and off the mark's box."""
    tb = text.get_window_extent()
    inside = (
        tb.x0 >= panel.x0
        and tb.x1 <= panel.x1
        and tb.y0 >= panel.y0
        and tb.y1 <= panel.y1
    )
    return inside and not tb.overlaps(mbox)


def _word_count(text):
    return len(str(text).split())


def callout(
    target,
    text,
    *,
    side="auto",
    leader=True,
    box=None,
    at=None,
    offset_pt=None,
    max_words=8,
    color=None,
    fontsize=None,
    leader_color=None,
    avoid=None,
    strict=True,
    zorder=9,
    text_kw=None,
    line_kw=None,
):
    """Write a short caption beside the thing it describes.

    The caption is anchored to its subject, so it moves with it, and it is
    short: a reader takes it in at a glance, at the place they are already
    looking. It sits `offset_pt` off the subject's box on one `side`; given
    `at`, it sits at that anchor instead, and a leader joins it to the
    subject. A leader is drawn only when the gap between the caption and
    the subject is more than one em, so a caption that touches its subject
    has none. The leader keeps the rules of `leader`.

    `update(u)` brings the caption in: its text and box fade in over the
    first half of `u`, then the leader grows from the caption to the
    subject over the second half (over all of it, without a leader). The
    rest state, when the callout is made, is ``u = 1``, fully drawn; at
    ``u = 0`` it draws nothing. ``update(u, level)`` also dims the whole
    callout to `level` of its opacity, for a caption that has been read and
    must stop competing with the next event; ``level = 0`` hides it.

    `side="auto"` is judged once, when the callout is made, from the
    geometry then: draw the figure first under a layout engine. The caption
    is a figure artist outside every layout, so a ``bbox_inches="tight"``
    save does not grow to hold it, and it is not animated, so a blitting
    loop does not repaint it.

    Args:
        target: The subject: an `Anchor`, or an artist.
        text: The caption, at most `max_words` words.
        side: ``"right"``, ``"left"``, ``"above"``, ``"below"``, or
            ``"auto"``, the first of those in which the caption fits inside
            the subject's panel without covering the subject, judged once,
            from the geometry when the callout is made (lay the figure out
            first, or name the side).
        leader: Draw a leader when the caption does not touch its subject.
        box: The caption's ``bbox``. None uses `eyepiece.backing`, read
            now; False draws no box.
        at: Optional `Anchor` for the caption itself; `side` then sets its
            alignment, and "auto" means "right".
        offset_pt: Gap between the subject's box and the caption, in points.
            None uses 0.4 of the font size.
        max_words: Longest caption accepted, in whitespace-separated words.
        color: Text color. None uses ``rcParams["text.color"]``.
        fontsize: Font size in points. None uses ``rcParams["font.size"]``.
        leader_color: Leader color, as in `leader`.
        avoid: Artists the leader must not cross, as in `leader`.
        strict: Raise on drawing a leader that breaks a rule.
        zorder: Drawing order of the caption among the figure's children;
            the leader draws just below it.
        text_kw: Extra kwargs for the `Text`, applied last.
        line_kw: Extra kwargs for the leader's `Line2D`, applied last.

    Returns:
        A `PlotResult` on the subject's axes (None for a subject drawn on
        the figure), with ``artists["text"]``, the caption, and, when
        `leader`, ``artists["line"]``, the leader (see `leader` for its
        ``faults()``), and ``update(u, level=1.0)`` as above.

    Raises:
        TypeError: If `target` or `at` is neither an `Anchor` nor an artist.
        ValueError: If the caption is longer than `max_words` words, or
            `side` is unknown.

    Example::

        (dot,) = ax.plot([0.35], [2.41], "o")
        res = ep.callout(dot, "the height at y", side="right")
        res.update(0.0)  # hidden until its beat
    """
    target = _as_anchor(target, "target")
    at = None if at is None else _as_anchor(at, "at")
    if side not in (*_SIDES, "auto"):
        raise ValueError(f"unknown callout side: {side!r}; known: {[*_SIDES, 'auto']}")
    words = _word_count(text)
    if words > int(max_words):
        raise ValueError(
            f"callout caption has {words} words, more than {max_words}: {text!r}"
        )
    rc = matplotlib.rcParams
    fontsize = float(rc["font.size"]) if fontsize is None else float(fontsize)
    offset_pt = 0.4 * fontsize if offset_pt is None else float(offset_pt)
    fig = target.figure
    first = "right" if side == "auto" else side
    caption = _AnchoredText(
        text,
        target=target,
        side=first,
        at=at,
        offset_pt=offset_pt,
        color=rc["text.color"] if color is None else color,
        fontsize=fontsize,
        zorder=zorder,
        **(text_kw or {}),
    )
    if box is None:
        caption.set_bbox(_style.backing())
    elif box:
        caption.set_bbox(dict(box))
    caption.set_clip_on(False)
    caption.set_in_layout(False)
    fig.add_artist(caption)
    if side == "auto" and at is None and target.axes is not None:
        panel = target.axes.bbox
        mbox = target.box()
        for candidate in _SIDES:
            caption.set_side(candidate)
            caption.place()
            if _fits(caption, panel, mbox):
                break
        else:
            caption.set_side(first)
    line = None
    if leader:
        line = _leader_line(
            caption,
            target,
            color=leader_color,
            lw=None,
            avoid=avoid,
            min_gap_em=1.0,
            strict=strict,
            zorder=zorder - 0.01,
            line_kw=line_kw,
        )
    fade = _TextFade(caption)

    def update(u, level=1.0):
        u = _check_u(u)
        level = _check_u(level)
        if line is None:
            shown, grow = u, 0.0
        else:
            shown, grow = min(1.0, 2.0 * u), max(0.0, 2.0 * u - 1.0)
        caption.set_visible(u > 0.0 and level > 0.0)
        opacity = _ease(shown) if shown < 1.0 else 1.0
        fade.set(opacity * level if level < 1.0 else opacity)
        if line is not None:
            line.set_visible(grow > 0.0 and level > 0.0)
            line._grow = _ease(grow) if grow < 1.0 else 1.0
            line.set_alpha(None if level >= 1.0 else level)

    update(1.0)
    artists = {"text": caption}
    if line is not None:
        artists["line"] = line
    return PlotResult(ax=target.axes, artists=artists, update=update)
