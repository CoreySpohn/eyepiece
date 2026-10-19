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

Other moves say "this one" or join panels: `indicate` pulses an artist and
returns it, `circumscribe` draws an outline on around it and lets it fade
(both leave the frame as it was at either end), `term_row` lays an equation
out as one text per term, `readout` counts a number to a new value in a fixed
layout, `words` sets a line as one text per word to color or reveal alone,
`link` grows a line between panels, `morph` carries one path into another,
and `write` traces a text's glyph outlines and fills them. Drawing never
raises; `audit` is the explicit
check of the leaders and links.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform

from eyepiece import _style
from eyepiece._motion import DEFAULT_EASE, _unsnapped
from eyepiece._motion import _apply_ease as _ease
from eyepiece._result import MosaicResult, MotionContract, PlotResult
from eyepiece.emphasis import _own_background, _resolve_background

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
        """Show the text at `level` of its starting opacity, box included.

        The box fades as the square of `level`, so while the text comes or
        goes its box is always fainter than it: no empty box is left
        standing as the words fade out.
        """
        text = self._text
        if level >= 1.0:
            text.set_alpha(self._alpha)
        else:
            text.set_alpha(level * (1.0 if self._alpha is None else self._alpha))
        patch = text.get_bbox_patch()
        if patch is not None and self._face is not None:
            k = min(1.0, level) ** 2
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


def _turned(offsets, angle):
    """`offsets` (N, 2) rotated counterclockwise by `angle` radians."""
    c, s = math.cos(angle), math.sin(angle)
    return offsets @ np.array([[c, s], [-s, c]])


def _display_points(artist):
    xy = np.asarray(artist.get_transform().transform(artist.get_xydata()), dtype=float)
    return xy[np.all(np.isfinite(xy), axis=1)]


def _center(points):
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


class _GhostPath(Line2D):
    """A copy of a line segment (or polyline) that flies and turns."""

    def __init__(self, mark, path, spin):
        super().__init__(
            [np.nan],
            [np.nan],
            transform=IdentityTransform(),
            linestyle=mark.get_linestyle(),
            linewidth=mark.get_linewidth(),
            color=mark.get_color(),
            alpha=mark.get_alpha(),
            solid_capstyle=mark.get_solid_capstyle(),
            dash_capstyle=mark.get_dash_capstyle(),
            solid_joinstyle=mark.get_solid_joinstyle(),
            drawstyle=mark.get_drawstyle(),
        )
        self._mark = mark
        self._route = path
        self._spin = spin

    def draw(self, renderer):
        """Put the copy where the flight's progress says, turned, then draw it."""
        if not self.get_visible():
            return
        points = _display_points(self._mark)
        if not len(points):
            return
        shape = _turned(points - _center(points), self._spin())
        xy = shape + self._route(renderer)
        self.set_data(xy[:, 0], xy[:, 1])
        super().draw(renderer)


class _GhostPatch(PathPatch):
    """A copy of a patch (a bar, say) that flies and turns."""

    def __init__(self, mark, path, spin):
        super().__init__(
            Path(np.zeros((1, 2)), [Path.MOVETO]),
            transform=IdentityTransform(),
            facecolor=mark.get_facecolor(),
            edgecolor=mark.get_edgecolor(),
            linewidth=mark.get_linewidth(),
            linestyle=mark.get_linestyle(),
            alpha=mark.get_alpha(),
            joinstyle=mark.get_joinstyle(),
            capstyle=mark.get_capstyle(),
        )
        self._mark = mark
        self._route = path
        self._spin = spin

    def draw(self, renderer):
        """Put the copy where the flight's progress says, turned, then draw it."""
        if not self.get_visible():
            return
        mark = self._mark
        path = mark.get_path()
        points = np.asarray(mark.get_transform().transform(path.vertices), dtype=float)
        finite = points[np.all(np.isfinite(points), axis=1)]
        if not len(finite):
            return
        shape = _turned(points - _center(finite), self._spin())
        self.set_path(Path(shape + self._route(renderer), path.codes))
        super().draw(renderer)


class _GhostText(Text):
    """A copy of a label that flies between two anchors."""

    def __init__(self, mark, path, spin=None):
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
        self._spin = spin
        self._turn0 = mark.get_rotation()

    def draw(self, renderer):
        """Put the copy where the flight's progress says, then draw it."""
        if not self.get_visible():
            return
        if self._spin is not None:
            self.set_rotation(self._turn0 + math.degrees(self._spin()))
        self.set_position(tuple(self._route(renderer)))
        super().draw(renderer)


def _is_path(line):
    """Whether a line is drawn as a line (two or more points, no marker)."""
    return (
        len(line.get_xydata()) >= 2
        and line.get_linestyle() not in ("None", "none", "", " ")
        and line.get_marker() in (None, "None", "none", "", " ")
    )


def flight(
    source,
    destination,
    *,
    fig,
    arc=0.0,
    turn_deg=0.0,
    crossfade=None,
    mark=None,
    ease=DEFAULT_EASE,
    fade=0.3,
    zorder=20,
):
    """Carry a copy of a mark from one place to another, across panels.

    Its ghost travels; the move leaves the source and destination geometry alone.

    The move shows that one quantity has become another: a dot read off a
    curve in one panel is carried onto the axis of the next, or a value is
    carried into the label that names it. The source stays where it is; a
    copy of it, the ghost, travels from the source anchor to the destination
    anchor and is hidden again when it lands, where the caller shows the
    destination mark. Both ends are resolved when the figure is drawn, so a
    flight lands on the destination as it is drawn then.

    The mark can be a marker (a dot), a line segment or polyline, a patch (a
    bar), or a text. A segment, polyline or patch is carried at its size in
    display pixels, centered on the path, and `turn_deg` turns it as it
    travels: a short segment beside a star in one panel is carried and turned
    a quarter turn onto the velocity axis of another, where it becomes the
    bar or the dot the viewer is meant to read there. A text turns too; a
    marker is a point and does not.

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
        turn_deg: How far the ghost turns by the time it lands, in degrees,
            counterclockwise on the screen (negative turns clockwise); it
            turns about its center, in step with the eased travel. 0 keeps
            its orientation. Not for a marker.
        crossfade: Optional `Text` the ghost lands into. Over the last
            `fade` of the flight the ghost fades out while this text fades
            in; it is at full opacity at ``u = 1`` and at none for progress
            before the fade starts. Keep it visible: its opacity carries the
            crossfade.
        mark: The artist the ghost copies: a `Line2D` marker (marker, size,
            colors), a `Line2D` of two or more points drawn as a line
            without a marker (its shape and style), a `Patch` (its shape and
            style) or a `Text` (string, font, color, box). None copies the
            source anchor's artist.
        ease: An easing kind for `eyepiece.ease`, or a callable from progress to
            eased progress such as ``functools.partial(ep.ease, kind="back_out",
            strength=0.2)``, applied to `u` along the
            path.
        fade: Fraction of the flight, at its end, over which `crossfade`
            runs, in (0, 1].
        zorder: Drawing order of the ghost among the figure's children.

    Returns:
        A `MosaicResult` over the axes the two anchors belong to (source
        first), with ``artists["line"]`` (a `Line2D` ghost), ``artists["fill"]``
        (a patch ghost) or ``artists["text"]`` (a `Text` ghost), and
        ``update(u)`` that sets the progress; values outside [0, 1] are
        clipped.

    Raises:
        TypeError: If an anchor is neither an `Anchor` nor an artist, or the
            mark is neither a `Line2D`, a `Patch` nor a `Text`.
        ValueError: If there is no mark to copy, `fade` is outside (0, 1],
            or `turn_deg` is not zero for a marker.

    Example::

        top_dot, = top.plot([0.35], [2.41], "o")
        carry = ep.flight(top_dot, ep.Anchor.from_data(bottom, (0.25, 2.41)),
                          fig=fig, arc=0.15)
        for u in np.linspace(0.0, 1.0, 30):
            carry.update(u)
            rec.frame()

    A segment turned a quarter turn onto another axis::

        stub, = star_panel.plot([x0, x1], [y0, y1], lw=3)
        carry = ep.flight(stub, ep.Anchor.from_data(velocity_panel, (0.0, 1.2)),
                          fig=fig, turn_deg=90.0)
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
    turn = math.radians(float(turn_deg))
    state = {"u": 1.0}

    def where(renderer):
        p0 = source.display(renderer)
        p1 = destination.display(renderer)
        e = _ease(state["u"], ease)
        step = p1 - p0
        # The control point sits twice as far out as the path's widest bow.
        control = 0.5 * (p0 + p1) + 2.0 * arc * np.array([-step[1], step[0]])
        return (1.0 - e) ** 2 * p0 + 2.0 * (1.0 - e) * e * control + e**2 * p1

    def spin():
        return turn * _ease(state["u"], ease)

    if isinstance(mark, Line2D) and _is_path(mark):
        ghost, key = _GhostPath(mark, where, spin), "line"
    elif isinstance(mark, Line2D):
        if turn != 0.0:
            raise ValueError(
                "a marker is a point and does not turn: turn_deg must be 0"
            )
        ghost, key = _GhostLine(mark, where), "line"
    elif isinstance(mark, Text):
        ghost, key = _GhostText(mark, where, spin), "text"
    elif isinstance(mark, Patch):
        ghost, key = _GhostPatch(mark, where, spin), "fill"
    else:
        raise TypeError(
            f"flight copies a Line2D, a Patch or a Text, got {type(mark).__name__}"
        )
    _unsnapped(ghost)
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
    overlaps = []
    for item in dict.fromkeys((mark, destination.artist, crossfade)):
        if item is None:
            continue
        # Equation terms are layout anchors; their glyphs are the drawn marks.
        for part in getattr(item, "_pieces", (item,)):
            if _root_figure(part) is fig and part.get_visible():
                overlaps.append((ghost, part))
    return MosaicResult(
        axes=np.array(axes, dtype=object),
        artists={key: ghost},
        update=update,
        contract=MotionContract(
            moves=(ghost,),
            may_overlap=tuple(dict.fromkeys(overlaps)),
        ),
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
        """Grow from the text toward the mark; drawing never raises."""
        if not self.get_visible() or self._grow <= 0.0:
            return
        ends = self._ends(renderer)
        if ends is None:
            return
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

    Drawing never raises, so a still or a PDF save is not aborted half way.
    The leader's `faults()` lists the rules it breaks as the figure is now,
    and `audit` checks every leader and link of a figure at once and, for
    those made with `strict` (the default), raises `AttentionFault`; a
    recording calls it on the composite, or on any frame it wants to vet.

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
        strict: Whether `audit` raises for this leader when it breaks a
            rule; False leaves the faults for the caller to read.
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


class AttentionFault(ValueError):
    """A leader or link of a figure breaks the connector rules.

    Attributes:
        faults: ``{line: [message, ...]}`` for every strict line that
            breaks a rule.
    """

    def __init__(self, faults):
        """Report the faults of every strict line that broke a rule."""
        self.faults = faults
        lines = [m for msgs in faults.values() for m in msgs]
        super().__init__(
            "attention lines break the connector rules: " + "; ".join(lines)
        )


def audit(fig, *, raise_on_fault=True):
    """Check every leader and link of a figure against the connector rules.

    Drawing a leader or a link never raises, so a still or a PDF save is
    never aborted half way. This is the explicit check: call it after the
    figure is laid out, on the composite or on any frame to be vetted.
    Lines that are hidden (a callout before its beat) are skipped; the
    others are audited as fully grown.

    Args:
        fig: The figure the lines were made in.
        raise_on_fault: Raise `AttentionFault` when a line made with
            ``strict=True`` breaks a rule.

    Returns:
        ``{line: [message, ...]}`` for every line, strict or not, that
        breaks a rule; empty when all keep them.

    Raises:
        AttentionFault: If `raise_on_fault` and a strict line breaks a rule.

    Example::

        res = ep.leader(note, ep.Anchor.from_data(ax, (1.22, 0.0)))
        ep.audit(fig)  # raises AttentionFault if the leader crosses data
    """
    root = fig.get_figure(root=True) or fig
    renderer = _renderer(root)
    found = {}
    for artist in root.artists:
        if isinstance(artist, (_Leader, _Link)) and artist.get_visible():
            broken = artist.faults(renderer)
            if broken:
                found[artist] = broken
    strict = {a: m for a, m in found.items() if a._strict}
    if raise_on_fault and strict:
        raise AttentionFault(strict)
    return found


# Callout


class _AnchoredText(Text):
    """A caption placed beside an anchor, or at its own anchor, at every draw.

    A caption given as pieces draws each piece in its own color, left to
    right on one baseline; the caption's own string is the pieces joined,
    so its box, extent, and leader are those of the whole line.
    """

    def __init__(self, text, *, target, side, at, offset_pt, pieces=None, **kw):
        ha, va = _ALIGN[side]
        super().__init__(
            0.0, 0.0, text, transform=IdentityTransform(), ha=ha, va=va, **kw
        )
        self._target = target
        self._side = side
        self._at = at
        self._offset_pt = float(offset_pt)
        self._lift_pt = 0.0
        self._pieces = []
        for piece, color, weight in pieces or ():
            font = self.get_fontproperties().copy()
            if weight is not None:
                font.set_weight(weight)
            part = Text(
                0.0,
                0.0,
                piece,
                transform=IdentityTransform(),
                fontproperties=font,
                color=self.get_color() if color is None else color,
                ha="left",
                va="baseline",
            )
            self._pieces.append(part)

    def get_children(self):
        """The caption's colored pieces, when it has them."""
        return [*super().get_children(), *self._pieces]

    def set_figure(self, fig):
        """Put the caption, and each of its pieces, in a figure."""
        super().set_figure(fig)
        for part in getattr(self, "_pieces", ()):
            part.set_figure(fig)

    def set_visible(self, b):
        """Show or hide the caption with its pieces."""
        super().set_visible(b)
        for part in getattr(self, "_pieces", ()):
            part.set_visible(b)

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
        y -= _pixels(renderer, _root_figure(self), self._lift_pt)
        self.set_position((float(x), float(y)))

    def set_side(self, side):
        """Put the caption on another side of its subject."""
        ha, va = _ALIGN[side]
        self._side = side
        self.set_horizontalalignment(ha)
        self.set_verticalalignment(va)

    def _draw_pieces(self, renderer):
        """The box, then each piece in its color along the line's baseline."""
        box = self.get_window_extent(renderer)
        figure = self.get_figure(root=False)
        # A heavier piece makes the line wider than the plain text it was
        # laid out as; the line and its box grow away from the subject:
        # rightward when left-aligned, leftward when right-aligned (a
        # caption on the subject's left), both ways when centered.
        for part in self._pieces:
            part.set_figure(figure)
        wide = sum(p.get_window_extent(renderer).width for p in self._pieces)
        extra = max(0.0, wide - box.width)
        shift = extra * _GROW_BACK[self.get_horizontalalignment()]
        patch = self.get_bbox_patch()
        if patch is not None:
            self.update_bbox_position_size(renderer)
            patch.set_x(patch.get_x() - shift)
            patch.set_width(patch.get_width() + extra)
            patch.draw(renderer)
        # The whole line set on a baseline at 0 measures how far its box
        # reaches below the baseline.
        probe = Text(
            0.0,
            0.0,
            self.get_text(),
            transform=IdentityTransform(),
            fontproperties=self.get_fontproperties(),
            va="baseline",
        )
        probe.set_figure(figure)
        x, y = box.x0 - shift, box.y0 - probe.get_window_extent(renderer).y0
        for part in self._pieces:
            part.set_alpha(self.get_alpha())
            part.set_position((x, y))
            part.draw(renderer)
            x += part.get_window_extent(renderer).width

    def draw(self, renderer):
        """Place the caption, then draw it."""
        if not self.get_visible():
            return
        self.place(renderer)
        if self._pieces:
            self._draw_pieces(renderer)
            self.stale = False
            return
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


# How much of a heavier line's extra width goes back past its anchor, by
# horizontal alignment.
_GROW_BACK = {"left": 0.0, "center": 0.5, "right": 1.0}


def _caption_pieces(text):
    """The caption's string and its colored pieces, or None for plain text."""
    if isinstance(text, str):
        return text, None
    pieces = []
    for piece in text:
        if isinstance(piece, str):
            pieces.append((piece, None, None))
        elif len(piece) == 2:
            words, color = piece
            pieces.append((str(words), color, None))
        else:
            words, color, weight = piece
            pieces.append((str(words), color, weight))
    if not pieces:
        raise ValueError("callout caption has no pieces")
    return "".join(words for words, _, _ in pieces), pieces


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
    rise_pt=0.0,
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
    ``u = 0`` it draws nothing. With `rise_pt` the caption also rises that
    far into its place as it fades in, so its arrival reads as motion.
    ``update(u, level)`` also dims the whole callout to `level` of its
    opacity, for a caption that has been read and must stop competing with
    the next event; ``level = 0`` hides it.

    A caption can name its key term in the color of the mark the term
    names, so the words and the graphic are visibly one thing: give `text`
    as pieces, each a string or a ``(string, color)`` pair, drawn in order
    on one baseline; a plain string piece takes `color`. A piece given as
    ``(string, color, weight)`` is also set in that font weight, for a key
    term whose mark is drawn in the text color itself (a gray map), where
    only weight can set it apart.

    `side="auto"` is judged once, when the callout is made, from the
    geometry then: draw the figure first under a layout engine. The caption
    is a figure artist outside every layout, so a ``bbox_inches="tight"``
    save does not grow to hold it, and it is not animated, so a blitting
    loop does not repaint it.

    Args:
        target: The subject: an `Anchor`, or an artist.
        text: The caption, at most `max_words` words: a string, or a
            sequence of pieces, each a string, a ``(string, color)`` pair,
            or a ``(string, color, weight)`` triple, joined in order.
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
        rise_pt: How far below its place, in points, the caption starts as
            it fades in; 0 fades it in place.
        leader_color: Leader color, as in `leader`.
        avoid: Artists the leader must not cross, as in `leader`.
        strict: Whether `audit` raises for the leader when it breaks a rule.
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
    text, pieces = _caption_pieces(text)
    if side not in (*_SIDES, "auto"):
        raise ValueError(f"unknown callout side: {side!r}; known: {[*_SIDES, 'auto']}")
    words = _word_count(text)
    if words > int(max_words):
        raise ValueError(
            f"callout caption has {words} words, more than {max_words}: {text!r}"
        )
    rc = matplotlib.rcParams
    fontsize = float(rc["font.size"]) if fontsize is None else float(fontsize)
    rise_pt = float(rise_pt)
    offset_pt = 0.4 * fontsize if offset_pt is None else float(offset_pt)
    fig = target.figure
    first = "right" if side == "auto" else side
    caption = _AnchoredText(
        text,
        target=target,
        side=first,
        at=at,
        offset_pt=offset_pt,
        pieces=pieces,
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
        caption._lift_pt = rise_pt * (1.0 - opacity)
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


# Stage


class _Look:
    """One drawn artist's colors and opacity as made, set to absolute levels.

    `presence` is how far the artist has arrived (its opacity, 0 to 1) and
    `contrast` how far its colors stand from the background it sits on (1
    as drawn, 0 the background itself). Every write starts from the values
    read when the look was made, so levels never compound.
    """

    def __init__(self, artist, background):
        self.artist = artist
        self.background = background
        self.alpha = artist.get_alpha()
        if isinstance(artist, Line2D):
            self.colors = [
                (artist.get_color(), artist.set_color),
                (artist.get_markerfacecolor(), artist.set_markerfacecolor),
                (artist.get_markeredgecolor(), artist.set_markeredgecolor),
                (artist.get_markerfacecoloralt(), artist.set_markerfacecoloralt),
            ]
        elif isinstance(artist, Text):
            self.colors = [(artist.get_color(), artist.set_color)]
        elif isinstance(artist, Patch):
            self.colors = [
                (artist.get_facecolor(), artist.set_facecolor),
                (artist.get_edgecolor(), artist.set_edgecolor),
            ]
        elif isinstance(artist, Collection) and artist.get_array() is None:
            self.colors = [
                (artist.get_facecolor().copy(), artist.set_facecolor),
                (artist.get_edgecolor().copy(), artist.set_edgecolor),
            ]
        else:
            # A colormapped collection or an image: opacity carries both
            # knobs, which equals the color blend over a uniform background.
            self.colors = None

    def set(self, presence, contrast):
        """Draw the artist at `presence` of its opacity and `contrast` of its color."""
        artist = self.artist
        if presence >= 1.0 and contrast >= 1.0:
            artist.set_alpha(self.alpha)
            for value, setter in self.colors or ():
                setter(value)
            return
        base = 1.0 if self.alpha is None else self.alpha
        if self.colors is None:
            artist.set_alpha(base * presence * contrast)
            return
        own = self.alpha is not None
        if own:
            artist.set_alpha(base * presence)
        for value, setter in self.colors:
            setter(self._shade(value, presence, contrast, own))

    def _shade(self, value, presence, contrast, own):
        if isinstance(value, str) and value.lower() == "none":
            return value
        scale = 1.0 if own else presence
        rgba = np.asarray(to_rgba_array(value), dtype=float)
        bg = np.asarray(to_rgba(self.background), dtype=float)[:3]
        out = rgba.copy()
        out[:, :3] = bg + contrast * (rgba[:, :3] - bg)
        out[:, 3] = rgba[:, 3] * scale
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return out
        return tuple(out[0])


def _looks(target, background, seen, out):
    """Every leaf artist under `target`, each with the background it sits on."""
    if id(target) in seen:
        return
    seen.add(id(target))
    if isinstance(target, Axes):
        face = _own_background(target) if background is None else background
        for child in target.get_children():
            if child is not target.patch:
                _looks(child, face, seen, out)
        return
    leaf = isinstance(target, (Text, Line2D, Patch, Collection)) or hasattr(
        target, "get_array"
    )
    if not leaf:
        for child in target.get_children():
            _looks(child, background, seen, out)
        return
    bg = background if background is not None else _resolve_background(target)
    out.append(_Look(target, bg))
    # `_pieces` is the hook for `_equation._Term`, whose glyphs it dims.
    for part in getattr(target, "_pieces", ()):
        _looks(part, background, seen, out)
    if isinstance(target, Text):
        for part in (
            target.get_bbox_patch(),
            getattr(target, "arrow_patch", None),
            *target.get_children(),
        ):
            if part is not None:
                _looks(part, background, seen, out)


class Stage:
    """Groups of drawn artists set to absolute levels: shown, dimmed, or absent.

    Its artists change opacity and color; their geometry stays in place.

    A staged explanation brings in one idea per beat and quiets what the
    viewer has already read. A `Stage` holds named groups of artists (an
    artist, an Axes, which stages everything it holds except its
    background, or an iterable of these) and sets every group, every frame,
    to two absolute knobs: its presence, how far it has arrived (its
    opacity), and its contrast, how far its colors stand from the
    background (a dimmed group blends toward the background it sits on, so
    it recedes the same way on a light figure and a dark one, and prints).

    Every level is computed from the colors and opacities the artists had
    when the stage was made, so setting a state twice draws the same picture
    as setting it once, and any frame can be drawn in any order. The rest
    state, which is the state the stage is made in, is every group as
    drawn; a still that never calls `set` shows it, and `rest` returns to
    it exactly.

    A stage owns the colors and opacity of the artists it holds: make it
    after they take their final colors, and do not also fade them with
    `fade` or by hand. Data may change freely between frames.
    Keep artists recolored by a draw function on every frame, such as bars,
    out of Stage groups, because the stage overwrites their colors.

    Args:
        groups: Mapping of group name to an artist, an Axes, or an iterable
            of these.
        dim: Contrast a dimmed group keeps, in [0, 1]. None resolves it by
            the active mode when the stage is made: 0.35 in dark mode and 0.5
            in light and paper, where dimmed hues separate less.
        background: Color every group blends toward. None resolves it per
            artist, as `fade` does: the face of the axes it is drawn in,
            else the nearest opaque figure face.

    Raises:
        ValueError: If `dim` is outside [0, 1] or a group is empty.

    Example::

        stage = ep.Stage({"scenery": [ax_left], "row": [row_line]})
        stage.set(shown=["row"], dimmed=["scenery"])
        stage.rest()  # the still
    """

    def __init__(self, groups, *, dim=None, background=None):
        """Read every group's artists as drawn now; that is the rest state."""
        dim = float(_style.dim_level() if dim is None else dim)
        if not 0.0 <= dim <= 1.0:
            raise ValueError(f"dim must be in [0, 1], got {dim}")
        self.dim = dim
        self._looks = {}
        seen = set()
        for name, members in dict(groups).items():
            items = [members] if isinstance(members, Artist) else list(members)
            looks = []
            for item in items:
                _looks(item, background, seen, looks)
            if not looks:
                raise ValueError(f"stage group {name!r} holds no drawn artist")
            self._looks[name] = looks
        # Axes decorations and virtual equation terms are not drawn leaves
        # that the motion checker can trace. Their real glyphs are included.
        rest = []
        for looks in self._looks.values():
            for look in looks:
                artist = look.artist
                owner = artist.axes if artist.axes is not None else artist.get_figure()
                if owner is None or artist not in owner.get_children():
                    continue
                if (
                    isinstance(artist, Spine)
                    or getattr(artist, "_pieces", ())
                    or not artist.get_visible()
                    or np.all(np.asarray(artist.get_alpha()) == 0.0)
                ):
                    continue
                if isinstance(artist, Text) and not artist.get_text().strip():
                    continue
                rest.append(artist)
        self.contract = MotionContract(rest=tuple(rest))

    @property
    def names(self):
        """The group names, in the order given."""
        return list(self._looks)

    def state(self, shown=None, dimmed=()):
        """The levels of a state, as ``{name: (presence, contrast)}``.

        Args:
            shown: Groups at full strength. None means every group not in
                `dimmed`.
            dimmed: Groups present but blended down to the stage's `dim`.

        Returns:
            A dict over every group: shown groups at ``(1, 1)``, dimmed ones
            at ``(1, dim)``, and the rest absent at ``(0, 1)``.

        Raises:
            KeyError: If a name is not a group of the stage.
        """
        dimmed = list(dimmed)
        shown = [n for n in self._looks if n not in dimmed] if shown is None else shown
        for name in (*shown, *dimmed):
            if name not in self._looks:
                raise KeyError(f"no stage group {name!r}; groups are {self.names}")
        out = {}
        for name in self._looks:
            if name in dimmed:
                out[name] = (1.0, self.dim)
            elif name in shown:
                out[name] = (1.0, 1.0)
            else:
                out[name] = (0.0, 1.0)
        return out

    def weigh(self, levels, *, ease=None):
        """Set the named groups to absolute levels; leave the others alone.

        A group not named keeps whatever its artists show, so a group can be
        handed back to another move (an axis build, say) while the stage
        sets the rest; `rest` returns every group to how it was made.

        Args:
            levels: Mapping of group name to a presence in [0, 1], or to a
                ``(presence, contrast)`` pair.
            ease: An easing kind for `eyepiece.ease`, or a callable from progress
                to eased progress such as ``functools.partial(ep.ease,
                kind="back_out", strength=0.2)``, or None for the levels
                as given. It shapes presence only, so a caller that ramps a
                presence linearly in time gets a fade that starts and stops
                gently, and presence 0 and 1 stay where they are. Contrast is
                always taken as given: a caller that ramps a contrast applies
                `eyepiece.ease` to its own progress. Pass the uneased
                progress, never a value `weigh` or `ease` has already shaped,
                or the curve is applied twice.

        Raises:
            KeyError: If a name is not a group of the stage.
            ValueError: If ``ease`` is not an easing kind.
        """
        for name, level in levels.items():
            if name not in self._looks:
                raise KeyError(f"no stage group {name!r}; groups are {self.names}")
            if not isinstance(level, (tuple, list)):
                level = (level, 1.0)
            presence, contrast = (_check_u(v) for v in level)
            if ease is not None:
                presence = min(1.0, max(0.0, _ease(presence, ease)))
            for look in self._looks[name]:
                look.set(presence, contrast)

    def set(self, shown=None, dimmed=(), *, start=None, u=1.0, ease=DEFAULT_EASE):
        """Go to a state, or part of the way there from another.

        Args:
            shown: As in `state`.
            dimmed: As in `state`.
            start: The state the move starts from, a dict as `state`
                returns; None starts from the rest state.
            u: Progress from `start` to the state, from 0 to 1; values
                outside are clipped. At 1 (the default) the stage is exactly
                in the state.
            ease: An easing kind for `eyepiece.ease`, or a callable from progress
                to eased progress such as ``functools.partial(ep.ease,
                kind="back_out", strength=0.2)``, applied to `u`.

        Returns:
            The levels drawn, ``{name: (presence, contrast)}``.
        """
        end = self.state(shown, dimmed)
        u = _check_u(u)
        if u >= 1.0:
            levels = end
        else:
            begin = start or {name: (1.0, 1.0) for name in self._looks}
            e = min(1.0, max(0.0, _ease(u, ease)))  # a level cannot overshoot
            levels = {
                name: tuple(
                    a + e * (b - a) for a, b in zip(begin[name], end[name], strict=True)
                )
                for name in self._looks
            }
        self.weigh(levels)
        return levels

    def rest(self):
        """Draw every group exactly as it was when the stage was made."""
        self.weigh(dict.fromkeys(self._looks, (1.0, 1.0)))


# Axes build


class _BuildPart:
    """One artist's opacity and clipping as made, so a build can return to it."""

    def __init__(self, artist):
        self.artist = artist
        self.alpha = artist.get_alpha()
        self.clip_on = artist.get_clip_on()
        self.clip_path = artist.get_clip_path()
        self.clip_box = artist.clipbox

    def fade(self, level):
        if level >= 1.0:
            self.artist.set_alpha(self.alpha)
        else:
            base = 1.0 if self.alpha is None else self.alpha
            self.artist.set_alpha(base * level)

    def clip(self, patch):
        self.artist.set_clip_on(True)
        self.artist.set_clip_path(patch)

    def restore(self):
        artist = self.artist
        artist.set_alpha(self.alpha)
        artist.set_clip_path(self.clip_path)
        artist.set_clip_box(self.clip_box)
        artist.set_clip_on(self.clip_on)


def _renderer(figure):
    canvas = figure.canvas
    get = getattr(canvas, "get_renderer", None)
    return get() if get is not None else figure._get_renderer()


_SPINES = {"x": ("bottom", "top"), "y": ("left", "right")}


class _AxisBuild:
    """The build of one axis of an axes: spine, ticks, tick labels, label."""

    def __init__(self, ax, which, *, label, spine_end, tick_fade, label_lag):
        self.ax = ax
        self.which = which
        self.axis = ax.xaxis if which == "x" else ax.yaxis
        self.spine_end = spine_end
        self.tick_fade = tick_fade
        self.label_lag = label_lag
        self.write_label = label
        self.spines = [_BuildPart(ax.spines[s]) for s in _SPINES[which]]
        self.ticks = []
        for major in (True, False):
            locs = (
                self.axis.get_majorticklocs()
                if major
                else self.axis.get_minorticklocs()
            )
            ticks = (
                self.axis.get_major_ticks(len(locs))
                if major
                else self.axis.get_minor_ticks(len(locs))
            )
            for tick, loc in zip(ticks, locs, strict=False):
                marks = [_BuildPart(tick.tick1line), _BuildPart(tick.tick2line)]
                texts = [_BuildPart(tick.label1), _BuildPart(tick.label2)]
                grid = _BuildPart(tick.gridline)
                self.ticks.append((float(loc), marks, texts, grid))
        self.label = _BuildPart(self.axis.label)
        figure = ax.get_figure(root=True)
        self._figure = figure
        self._spine_clip = Rectangle(
            (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, visible=False
        )
        self._label_clip = Rectangle(
            (0.0, 0.0), 0.0, 0.0, transform=figure.transFigure, visible=False
        )

    def _fraction(self, loc):
        """Where a tick location sits along the axis, as an axes fraction."""
        to_axes = self.ax.transScale + self.ax.transLimits
        point = (loc, 0.0) if self.which == "x" else (0.0, loc)
        with np.errstate(all="ignore"):
            frac = to_axes.transform(np.array([point], dtype=float))[0]
        return float(frac[0] if self.which == "x" else frac[1])

    def update(self, u):
        if u >= 1.0:
            for part in [*self.spines, self.label]:
                part.restore()
            for _loc, marks, texts, grid in self.ticks:
                for part in (*marks, *texts, grid):
                    part.restore()
            return
        # The front runs from the origin to past the far end, so the last
        # tick label has finished fading in when the spine is drawn.
        reach = 1.0 + self.tick_fade + self.label_lag
        front = reach * _ease(min(1.0, u / self.spine_end))
        drawn = min(1.0, front)
        big = 1e3
        if self.which == "x":
            self._spine_clip.set_bounds(-big, -big, big + drawn, 3.0 * big)
        else:
            self._spine_clip.set_bounds(-big, -big, 3.0 * big, big + drawn)
        for part in self.spines:
            part.clip(self._spine_clip)
            # A spine clipped to nothing would still show its line cap.
            part.fade(1.0 if drawn > 0.0 else 0.0)
        for loc, marks, texts, grid in self.ticks:
            f = self._fraction(loc)
            mark = 1.0 if drawn >= 1.0 or f < drawn else 0.0
            for part in (*marks, grid):
                part.fade(mark)
            words = min(1.0, max(0.0, (front - f - self.label_lag) / self.tick_fade))
            for part in texts:
                part.fade(words)
        if not self.write_label:
            self.label.restore()
            return
        write = max(0.0, (u - self.spine_end) / (1.0 - self.spine_end))
        text = self.axis.label
        box = text.get_window_extent(_renderer(self._figure))
        inverse = self._figure.transFigure.inverted()
        (x0, y0), (x1, y1) = inverse.transform(box.get_points())
        pad = 0.02
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        e = _ease(write)
        if self.which == "x":
            self._label_clip.set_bounds(x0, y0, e * (x1 - x0), y1 - y0)
        else:
            self._label_clip.set_bounds(x0, y0, x1 - x0, e * (y1 - y0))
        self.label.clip(self._label_clip)


def build_axes(ax, *, axis="both", label=True, spine_end=0.7, title=False):
    """Build an axes the way a teacher draws one: line, ticks, numbers, name.

    An explanation that introduces a quantity draws its axis in front of
    the viewer rather than showing it finished: the spine draws on from the
    origin, each tick appears as the spine reaches it, its number fades in
    just after, and the axis label writes on last, from its first letter.
    One axis at a time keeps one new quantity at a time; ``axis="both"``
    builds the horizontal axis and then the vertical one.

    Nothing about the axes' limits, ticks, or layout changes: the build only
    clips and fades what the axes already draws, so it moves no data limit
    and a recording sees no drifting scale. Make the axes final first
    (limits, ticks, labels, and a layout that will not move), then build it.

    Args:
        ax: The axes to build.
        axis: ``"x"``, ``"y"``, or ``"both"`` (x over the first half of the
            progress, then y).
        label: Write the axis label on after the spine; False shows it
            throughout.
        spine_end: Fraction of each axis's progress the spine, ticks, and
            numbers take; the label writes on over the rest.
        title: Fade the center, left, and right axes titles in with the last
            axis label. False leaves the titles as made throughout. True
            uses the final label stage even when `label` is False.

    Returns:
        A `PlotResult` on `ax` with ``update(u)``, `u` from 0 (nothing of
        the axis drawn) to 1 (the rest state: the axes exactly as made).
        The result is made in its rest state.

    Raises:
        ValueError: If `axis` is unknown or `spine_end` is outside (0, 1).

    Example::

        build = ep.build_axes(ax, axis="x")
        for u in np.linspace(0.0, 1.0, 24):
            build.update(u)
            rec.frame()
    """
    if axis not in ("x", "y", "both"):
        raise ValueError(f"unknown axis: {axis!r}; known: ['x', 'y', 'both']")
    spine_end = float(spine_end)
    if not 0.0 < spine_end < 1.0:
        raise ValueError(f"spine_end must be in (0, 1), got {spine_end}")
    which = ("x", "y") if axis == "both" else (axis,)
    parts = [
        _AxisBuild(
            ax,
            w,
            label=label,
            spine_end=spine_end,
            tick_fade=0.25,
            label_lag=0.05,
        )
        for w in which
    ]
    titles = (
        [_BuildPart(text) for text in (ax.title, ax._left_title, ax._right_title)]
        if title
        else []
    )

    def update(u):
        u = _check_u(u)
        n = len(parts)
        for i, part in enumerate(parts):
            part.update(min(1.0, max(0.0, n * u - i)))
        last = max(0.0, n * u - (n - 1))
        reveal = _ease(max(0.0, (last - spine_end) / (1.0 - spine_end)))
        for text in titles:
            if u >= 1.0:
                text.restore()
            else:
                text.fade(reveal)

    return PlotResult(
        ax=ax,
        artists={"line": [p.artist for b in parts for p in b.spines]},
        update=update,
    )


# Collapse


class _CollapseGhost(LineCollection):
    """Cells of a row drawn as segments moving into a bar, and the bar moving on."""

    def __init__(self, route, colors, **kw):
        super().__init__([], transform=IdentityTransform(), **kw)
        self._route = route
        self._colors = colors

    def draw(self, renderer):
        """Place every segment where the progress says, then draw them."""
        if not self.get_visible():
            return
        segments, colors = self._route(renderer)
        self.set_segments(segments)
        self.set_color(colors)
        super().draw(renderer)


def collapse(
    ax,
    segments,
    weights,
    start,
    end,
    *,
    fig,
    colors,
    color=None,
    to=None,
    width_pt=None,
    ease=DEFAULT_EASE,
    zorder=20,
):
    """Gather the cells of a row into one bar whose length is their sum.

    Its cell copy gathers and travels; the original cells stay in place.

    The move shows that a set adds up to one value: each cell, drawn as a
    segment where it sits, slides into a bar between `start` and `end` and
    takes a share of the bar's length in proportion to its weight, so the
    bar's whole length is the sum and a heavier cell takes more of it. With
    `to`, the formed bar is then carried to another place, for example
    onto the axis of a panel that shows the sums, the way `flight` carries a
    mark.

    The copy is a figure-level artist, so it never joins an axes'
    autoscaling. ``update(u, carry=0.0)`` sets the progress absolutely: `u`
    gathers the cells (0 drawn nothing, the cells are the caller's) and
    `carry` moves the formed bar to `to`. Without `to` the copy is hidden at
    ``u = 1``, where the caller draws its own bar; with `to`, it is visible
    from ``u > 0`` until ``carry = 1``, where the caller draws the landed
    bar. The rest state, when it is made, is the move finished: nothing
    drawn. The copy fades in over the first fifth of the gather and fades
    out over the last fifth of the gather, or of the carry when `to` is given.

    Args:
        ax: The axes the cells are drawn in.
        segments: The cells along the row, an array of shape ``(N, 2, 2)``:
            each cell's two ends in `ax` data coordinates.
        weights: The cells' values, ``N`` of them, not negative.
        start: Where the bar starts: an `Anchor` (or an artist).
        end: Where the bar ends: an `Anchor` (or an artist).
        fig: The figure that owns the copy, normally the root figure.
        colors: The cells' colors, one color or ``N`` of them.
        color: Optional color the cells take on as they gather, so the bar
            arrives in the color of the mark it becomes; None keeps theirs.
        to: Optional ``(start, end)`` pair of anchors the bar is carried to.
        width_pt: Line width of the segments in points. None uses twice
            ``rcParams["lines.linewidth"]``.
        ease: An easing kind for `eyepiece.ease`, or a callable from progress to
            eased progress such as ``functools.partial(ep.ease, kind="back_out",
            strength=0.2)``.
        zorder: Drawing order of the copy among the figure's children.

    Returns:
        A `MosaicResult` over the axes involved, with ``artists["line"]``
        (the copy) and ``update(u, carry=0.0)``.

    Raises:
        ValueError: If the shapes disagree, a weight is negative, or every
            weight is zero.

    Example::

        cells = np.stack([np.c_[edges[:-1], [r] * n], np.c_[edges[1:], [r] * n]], 1)
        sums = ep.collapse(left, cells, row, ep.Anchor.from_data(left, (0.1, r)),
                           ep.Anchor.from_data(left, (0.1 + total, r)), fig=fig,
                           colors=cmap(norm(row)), color=pink,
                           to=(ep.Anchor.from_data(right, (0, r)), tip))
        sums.update(0.5)            # halfway gathered
        sums.update(1.0, carry=0.5)  # the bar halfway to the other panel
    """
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 3 or segments.shape[1:] != (2, 2):
        raise ValueError(f"segments must have shape (N, 2, 2), got {segments.shape}")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(segments),):
        raise ValueError(
            f"weights must have shape ({len(segments)},), got {weights.shape}"
        )
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("collapse weights must be finite and not negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("collapse weights sum to zero: there is no bar to gather")
    cells = np.asarray(to_rgba_array(colors), dtype=float)
    if len(cells) == 1:
        cells = np.repeat(cells, len(segments), axis=0)
    if len(cells) != len(segments):
        raise ValueError(f"colors must be one color or {len(segments)} of them")
    target = None if color is None else np.asarray(to_rgba(color), dtype=float)
    start = _as_anchor(start, "start")
    end = _as_anchor(end, "end")
    landing = None if to is None else tuple(_as_anchor(a, "to") for a in to)
    edges = np.concatenate([[0.0], np.cumsum(weights) / total])
    share = np.stack([edges[:-1], edges[1:]], axis=1)  # (N, 2)
    state = {"u": 1.0, "carry": 1.0}

    def bar(a, b, renderer):
        p0, p1 = a.display(renderer), b.display(renderer)
        return p0 + share[..., None] * (p1 - p0)  # (N, 2, 2)

    def route(renderer):
        source = ax.transData.transform(segments.reshape(-1, 2)).reshape(-1, 2, 2)
        e = _ease(state["u"], ease)
        place = source + e * (bar(start, end, renderer) - source)
        if landing is not None and state["carry"] > 0.0:
            c = _ease(state["carry"], ease)
            place = place + c * (bar(*landing, renderer) - place)
        shade = cells
        if target is not None:
            blend = min(1.0, max(0.0, e))  # a color cannot overshoot
            shade = cells + blend * (target - cells)
        opacity = _ease(min(1.0, state["u"] / 0.2), "smoothstep")
        finish = state["u"] if landing is None else state["carry"]
        opacity *= _ease(min(1.0, (1.0 - finish) / 0.2), "smoothstep")
        shade = shade.copy()
        shade[:, 3] *= opacity
        return place, shade

    rc = matplotlib.rcParams
    width = 2.0 * float(rc["lines.linewidth"]) if width_pt is None else float(width_pt)
    # Without antialiasing the cells meet without a seam between them.
    ghost = _CollapseGhost(
        route, cells, linewidths=width, capstyle="butt", antialiaseds=False
    )
    _unsnapped(ghost)
    ghost.set_zorder(zorder)
    ghost.set_clip_on(False)
    ghost.set_in_layout(False)
    fig.add_artist(ghost)

    def update(u, carry=0.0):
        u = _check_u(u)
        carry = _check_u(carry)
        state["u"], state["carry"] = u, carry
        if landing is None:
            ghost.set_visible(0.0 < u < 1.0)
        else:
            ghost.set_visible(u > 0.0 and carry < 1.0)

    update(1.0, carry=1.0)
    axes = [a for a in (ax, start.axes, *(x.axes for x in landing or ())) if a]
    axes = list(dict.fromkeys(axes))
    return MosaicResult(
        axes=np.array(axes, dtype=object),
        artists={"line": ghost},
        update=update,
        contract=MotionContract(moves=(ghost,)),
    )


# Indicate


def _same(a, b):
    """Whether two artist property values are equal, arrays and strings alike."""
    try:
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except (TypeError, ValueError):
        return a == b


class _Channel:
    """One property a pulse moves, with the value it started from.

    The pulse writes the property from `base` every frame. Whatever the
    property holds at the start of a frame is compared with what the pulse
    last wrote: if they are equal the pulse's own write is still there and
    `base` stands, and if not, something else (a `Stage`, a fade, the
    caller) set the property since, and what it holds now is the new base.
    """

    def __init__(self, get, set_, shape):
        self.get = get
        self.set = set_
        self.shape = shape
        self.base = None
        self.written = None

    def apply(self, k):
        current = self.get()
        ours = self.written is not None and _same(current, self.written)
        base = self.base if ours else current
        if k <= 0.0:
            if ours:
                self.set(base)
            self.base = self.written = None
            return
        self.set(self.shape(base, k))
        self.base = base
        self.written = self.get()


def _is_none_color(value):
    return isinstance(value, str) and value.lower() == "none"


def _channels(artist, scale, strength):
    """The size and color properties of one artist that a pulse moves."""
    grow = scale - 1.0

    def size(power=1.0):
        return lambda value, k: (
            np.asarray(value, dtype=float) * (1.0 + grow * k) ** power
        )

    def scalar(value, k):
        return float(value) * (1.0 + grow * k)

    def color(value, k):
        if _is_none_color(value):
            return value
        target = np.array(to_rgb(matplotlib.rcParams["text.color"]))
        rgba = np.asarray(to_rgba_array(value), dtype=float).copy()
        rgba[:, :3] += strength * k * (target - rgba[:, :3])
        if isinstance(value, np.ndarray) and value.ndim == 2:
            return rgba
        return tuple(rgba[0])

    out = []
    if isinstance(artist, Line2D):
        out += [_Channel(artist.get_linewidth, artist.set_linewidth, scalar)]
        if artist.get_marker() not in (None, "None", "none", "", " "):
            out += [
                _Channel(artist.get_markersize, artist.set_markersize, scalar),
                _Channel(
                    artist.get_markeredgewidth, artist.set_markeredgewidth, scalar
                ),
                _Channel(artist.get_markerfacecolor, artist.set_markerfacecolor, color),
                _Channel(artist.get_markeredgecolor, artist.set_markeredgecolor, color),
            ]
        out += [_Channel(artist.get_color, artist.set_color, color)]
    elif isinstance(artist, Text):
        out += [
            _Channel(artist.get_fontsize, artist.set_fontsize, scalar),
            _Channel(artist.get_color, artist.set_color, color),
        ]
    elif isinstance(artist, Patch):
        out += [
            _Channel(artist.get_linewidth, artist.set_linewidth, scalar),
            _Channel(artist.get_facecolor, artist.set_facecolor, color),
            _Channel(artist.get_edgecolor, artist.set_edgecolor, color),
        ]
    elif isinstance(artist, Collection):
        if len(artist.get_linewidth()):
            out += [_Channel(artist.get_linewidth, artist.set_linewidth, size())]
        if isinstance(artist, PathCollection) and len(artist.get_sizes()):
            # Sizes are areas, so a 1.15 stretch of the mark is its square.
            out += [_Channel(artist.get_sizes, artist.set_sizes, size(2.0))]
        if artist.get_array() is None:
            out += [
                _Channel(artist.get_facecolor, artist.set_facecolor, color),
                _Channel(artist.get_edgecolor, artist.set_edgecolor, color),
            ]
    return out


def _pulse_parts(artist):
    """The artists a pulse on `artist` moves: itself and a caption's pieces."""
    return [artist, *getattr(artist, "_pieces", ())]


def indicate(artists, *, scale=1.15, strength=0.6):
    """Say "this one": swell an artist and tint it toward the text color, then return.

    A pulse is a transient emphasis. For its duration the artist's line
    width, marker size or font size grows by `scale` and its colors move
    toward the text color by `strength`, both peaking half way through, and
    then everything returns. It adds no hue: a tint toward the text color is
    a darker mark on a light figure and a lighter one on a dark figure, so
    the pulse reads on either. It works on lines, texts (a callout or a
    label with a backing box swells with its box), patches, and collections
    (a colormapped collection swells only, since its colors come from its
    data).

    ``update(u)`` is absolute. At ``u = 0`` and ``u = 1``, the rest state
    in which the move is made, the artist is exactly as it was and the frame
    is pixel-identical to the frame without the move.

    A pulse reads each property as it is at the start of every update, so it
    pulses from wherever the artist is. When a `Stage` (or any other fade)
    owns the artist's colors, set the stage first and then update the pulse
    in the same frame: the pulse swells from the stage's level, a dimmed
    artist stays dimmed under it, and at its ends the artist is at the
    stage's level rather than at the colors it was made with. If the stage
    is set after the pulse in a frame it overwrites the tint, so the pulse is
    seen only in size. Do not run two pulses on one artist at once.

    Args:
        artists: An artist, or an iterable of them (`Line2D`, `Text`,
            `Patch`, `Collection`).
        scale: Peak growth of line width, marker size, and font size, a
            factor of at least 1. Manim's `Indicate` uses 1.2; 1.15 reads as
            emphasis without moving the layout.
        strength: Peak fraction of the way from the artist's color to the
            text color, in [0, 1].

    Returns:
        A `PlotResult` on the first artist's axes (None if it has none) with
        ``artists`` grouped by kind (``"line"``, ``"text"``, ``"fill"``,
        ``"collection"``) and ``update(u)``, `u` from 0 to 1; values outside
        are clipped.

    Raises:
        TypeError: If an artist is not a line, text, patch, or collection.
        ValueError: If `scale` is below 1, `strength` is outside [0, 1], or
            there is nothing to pulse.

    Example::

        pulse = ep.indicate([curve, note])
        for u in np.linspace(0.0, 1.0, 20):
            pulse.update(u)
            rec.frame()
    """
    scale, strength = float(scale), float(strength)
    if scale < 1.0:
        raise ValueError(f"indicate scale must be at least 1, got {scale}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"indicate strength must be in [0, 1], got {strength}")
    items = [artists] if isinstance(artists, Artist) else list(artists)
    if not items:
        raise ValueError("indicate has no artist to pulse")
    channels, groups = [], {}
    for item in items:
        if not isinstance(item, (Line2D, Text, Patch, Collection)):
            raise TypeError(
                f"indicate pulses a line, text, patch, or collection, got "
                f"{type(item).__name__}"
            )
        for part in _pulse_parts(item):
            channels += _channels(part, scale, strength)
        key = (
            "line"
            if isinstance(item, Line2D)
            else "text"
            if isinstance(item, Text)
            else "fill"
            if isinstance(item, Patch)
            else "collection"
        )
        groups.setdefault(key, []).append(item)

    def update(u):
        u = _check_u(u)
        k = 0.0 if u <= 0.0 or u >= 1.0 else math.sin(math.pi * u) ** 2
        for channel in channels:
            channel.apply(k)

    return PlotResult(ax=getattr(items[0], "axes", None), artists=groups, update=update)


# Circumscribe

_ELLIPSE_STEPS = 160


def _cut(vertices, fraction):
    """The first `fraction` of a polyline's length, as vertices."""
    seg = np.hypot(*np.diff(vertices, axis=0).T)
    total = float(seg.sum())
    if total <= 0.0 or fraction >= 1.0:
        return vertices
    reach = fraction * total
    done = np.concatenate([[0.0], np.cumsum(seg)])
    k = int(np.searchsorted(done, reach, side="right")) - 1
    k = min(k, len(seg) - 1)
    t = (reach - done[k]) / seg[k] if seg[k] > 0.0 else 0.0
    tip = vertices[k] + t * (vertices[k + 1] - vertices[k])
    return np.vstack([vertices[: k + 1], tip])


class _DrawOn(Line2D):
    """A line's current path, cut in display space when it is drawn."""

    def __init__(self, source):
        super().__init__([], [])
        self._source = source
        self._fraction = 1.0
        self.set_zorder(source.get_zorder())
        self.set_in_layout(False)
        self.set_visible(False)

    def draw(self, renderer):
        """Read the current data and transform, then draw the path's prefix."""
        if not self.get_visible() or self._fraction <= 0.0:
            return
        source = self._source
        path = source.get_transform().transform_path(source.get_path())
        parts = [points for points, _ in _line_parts(path.vertices, False)]
        lengths = [np.hypot(*np.diff(p, axis=0).T).sum() for p in parts]
        reach = self._fraction * sum(lengths)
        shown = []
        for points, length in zip(parts, lengths, strict=True):
            if shown:
                shown.append(np.full((1, 2), np.nan))
            shown.append(_cut(points, min(1.0, reach / length) if length else 1.0))
            reach -= length
            if reach <= 0.0:
                break
        self.update_from(source)
        self.set_visible(True)  # update_from copied the hidden source's flag
        self.set_transform(IdentityTransform())
        self.set_drawstyle("default")
        self.set_antialiased(source.get_antialiased())
        self.set_markevery(source.get_markevery())
        self.set_snap(source.get_snap())
        self.set_label("_nolegend_")
        vertices = np.vstack(shown) if shown else np.empty((0, 2))
        self.set_data(vertices[:, 0], vertices[:, 1])
        super().draw(renderer)


def draw_on(lines, *, ease="linear") -> PlotResult:
    """Reveal lines along their current paths at a steady display-space speed.

    ``update(u, level=1.0)`` draws the first eased fraction of each path's
    arc length in display pixels. Data and transforms are read when the
    figure is drawn, so ``set_data``, new limits, layout changes, and log
    axes all work between frames. NaN gaps contribute no length.

    A temporary line draws the cut path while the source is hidden. Its
    markers sit at the retained vertices, including the cut tip, and its
    dashes start at the beginning of each connected run. The source data,
    transform, and drawing method are never replaced. At ``u = 0`` nothing
    is drawn; at ``u = 1``, the rest state, the source draws normally, with
    exactly its original markers and dashes. With the default level this
    frame is pixel-identical to the frame without the move. ``level`` only
    multiplies the original opacity and never changes the cut length.

    Args:
        lines: One `Line2D` or an iterable of lines attached to axes.
        ease: An easing kind for `eyepiece.ease`, or a callable mapping
            progress to eased progress. The path fraction is clipped to
            [0, 1]; monotone easing gives a monotone reveal.

    Returns:
        A `PlotResult` on the first line's axes, with the source lines and
        their temporary copies in ``artists["lines"]``, and ``update``.
        Progress and level are clipped to [0, 1].

    Raises:
        TypeError: If an item is not a `Line2D`.
        ValueError: If there are no lines, a line has no axes, or a line
            uses a drawstyle other than ``"default"``, whose stepped path
            the reveal cannot follow.
    """
    items = [lines] if isinstance(lines, Line2D) else list(lines)
    if not items:
        raise ValueError("draw_on needs at least one line")
    for line in items:
        if not isinstance(line, Line2D):
            raise TypeError(f"draw_on needs a Line2D, got {type(line).__name__}")
        if line.axes is None:
            raise ValueError("draw_on needs lines attached to axes")
        if line.get_drawstyle() != "default":
            raise ValueError("draw_on does not support stepped drawstyles")
    copies = [_DrawOn(line) for line in items]
    states = [(line.get_visible(), line.get_alpha()) for line in items]
    for line, copy in zip(items, copies, strict=True):
        line.axes.add_artist(copy)

    def update(u, level=1.0):
        u, level = _check_u(u), _check_u(level)
        fraction = u if u in (0.0, 1.0) else _check_u(_ease(u, ease))
        for line, copy, (visible, alpha) in zip(items, copies, states, strict=True):
            base = 1.0 if alpha is None else alpha
            line.set_alpha(alpha if level == 1.0 else level * base)
            line.set_visible(visible and u >= 1.0)
            copy._fraction = fraction
            copy.set_zorder(line.get_zorder())
            copy.set_visible(visible and 0.0 < fraction and u < 1.0)

    return PlotResult(
        ax=items[0].axes,
        artists={"lines": [*items, *copies]},
        update=update,
        contract=MotionContract(moves=tuple(copies)),
    )


class _Outline(Line2D):
    """A closed outline around a target's box that draws on around its perimeter."""

    def __init__(self, target, *, shape, pad_pt, **kw):
        super().__init__([], [], transform=IdentityTransform(), **kw)
        self._target = target
        self._shape = shape
        self._pad_pt = float(pad_pt)
        self._fraction = 1.0

    def outline(self, renderer=None):
        """The whole outline, closed, in display pixels, as the figure is now."""
        box = self._target.box(renderer)
        pad = _pixels(renderer, _root_figure(self), self._pad_pt)
        cx, cy = 0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1)
        if self._shape == "rect":
            x0, x1, y0, y1 = box.x0 - pad, box.x1 + pad, box.y0 - pad, box.y1 + pad
            return np.array(
                [[cx, y1], [x1, y1], [x1, y0], [x0, y0], [x0, y1], [cx, y1]],
                dtype=float,
            )
        # An ellipse through the corners of the padded box.
        a = math.sqrt(0.5) * box.width + pad
        b = math.sqrt(0.5) * box.height + pad
        t = 0.5 * math.pi - np.linspace(0.0, 2.0 * math.pi, _ELLIPSE_STEPS + 1)
        return np.c_[cx + a * np.cos(t), cy + b * np.sin(t)]

    def draw(self, renderer):
        """Draw the first part of the outline, from the top, clockwise."""
        if not self.get_visible() or self._fraction <= 0.0:
            return
        points = _cut(self.outline(renderer), self._fraction)
        self.set_data(points[:, 0], points[:, 1])
        super().draw(renderer)


def circumscribe(
    target,
    *,
    shape="rect",
    pad_pt=4.0,
    color=None,
    lw=None,
    draw=0.6,
    ease=DEFAULT_EASE,
    zorder=21,
    line_kw=None,
):
    """Draw an outline around a thing, then let it go.

    The move says "this one" for a thing the viewer has to find in a busy
    panel: an outline draws on around the target's perimeter, from the top
    and clockwise, and then fades. It is a figure-level artist resolved in
    display space when the figure is drawn, so it follows its target and
    never joins an axes' autoscaling. A data-point `Anchor` has no extent,
    so its outline is a ring of `pad_pt` around the point.

    ``update(u)`` is absolute: over the first `draw` of ``u`` the outline
    draws on, and over the rest it fades. At ``u = 0`` and ``u = 1``, the
    rest state, nothing is drawn and the frame is pixel-identical to the
    frame without the move.

    Args:
        target: The thing to outline: an `Anchor` (a data point, or a part
            of an artist's box) or an artist, outlined on its box.
        shape: ``"rect"`` or ``"ellipse"`` (through the corners of the
            padded box).
        pad_pt: Gap between the target's box and the outline, in points.
        color: Outline color. None uses the text color, read now.
        lw: Line width in points. None uses ``rcParams["lines.linewidth"]``.
        draw: Fraction of the progress spent drawing the outline on, in
            (0, 1); the rest fades it.
        ease: An easing kind for `eyepiece.ease`, or a callable from progress to
            eased progress such as ``functools.partial(ep.ease, kind="back_out",
            strength=0.2)``, applied to the draw-on and
            to the fade.
        zorder: Drawing order among the figure's children.
        line_kw: Extra kwargs for the `Line2D`, applied last.

    Returns:
        A `PlotResult` on the target's axes (None for a target drawn on the
        figure), with ``artists["line"]`` and ``update(u)``, `u` from 0 to 1;
        values outside are clipped.

    Raises:
        TypeError: If `target` is neither an `Anchor` nor an artist.
        ValueError: If `shape` is unknown or `draw` is outside (0, 1).

    Example::

        ring = ep.circumscribe(ep.Anchor.from_artist(label), pad_pt=6)
        for u in np.linspace(0.0, 1.0, 24):
            ring.update(u)
            rec.frame()
    """
    if shape not in ("rect", "ellipse"):
        raise ValueError(f"unknown shape: {shape!r}; known: ['rect', 'ellipse']")
    draw = float(draw)
    if not 0.0 < draw < 1.0:
        raise ValueError(f"circumscribe draw must be in (0, 1), got {draw}")
    target = _as_anchor(target, "target")
    rc = matplotlib.rcParams
    kw = {
        "color": _style.neutral(1.0) if color is None else color,
        "lw": float(rc["lines.linewidth"]) if lw is None else lw,
        "solid_capstyle": "round",
        "solid_joinstyle": "round",
        "zorder": zorder,
        **(line_kw or {}),
    }
    line = _Outline(target, shape=shape, pad_pt=pad_pt, **kw)
    line.set_clip_on(False)
    line.set_in_layout(False)
    target.figure.add_artist(line)
    alpha = line.get_alpha()

    def update(u):
        u = _check_u(u)
        if u <= 0.0 or u >= 1.0:
            line.set_visible(False)
            return
        line.set_visible(True)
        if u < draw:
            line._fraction = min(1.0, max(0.0, _ease(u / draw, ease)))
            line.set_alpha(alpha)
        else:
            line._fraction = 1.0
            level = 1.0 - min(1.0, max(0.0, _ease((u - draw) / (1.0 - draw), ease)))
            line.set_alpha(level * (1.0 if alpha is None else alpha))

    update(1.0)
    return PlotResult(ax=target.axes, artists={"line": line}, update=update)


# Morph

_MORPH_POINTS = 200


def _path_parts(path):
    """A `Path` as ``[(vertices, closed), ...]``, curves flattened, NaNs split."""
    parts, current, closed = [], [], False

    def finish():
        if len(current) and not np.isnan(np.asarray(current)).any():
            parts.append((np.array(current, dtype=float), closed))

    for xy, code in path.iter_segments(curves=False, simplify=False):
        if code == Path.MOVETO:
            finish()
            current, closed = [tuple(xy[-2:])], False
        elif code == Path.CLOSEPOLY:
            closed = True
        else:
            current.append(tuple(xy[-2:]))
    finish()
    return parts


def _line_parts(vertices, closed):
    """The runs of finite points in `vertices`, split at NaN rows."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 2)
    good = np.all(np.isfinite(vertices), axis=1)
    parts, start = [], None
    for i, ok in enumerate([*good, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            run = vertices[start:i]
            ring = closed is None and len(run) > 2 and np.allclose(run[0], run[-1])
            if ring or (closed and len(run) > 2):
                run = run[:-1] if np.allclose(run[0], run[-1]) else run
                parts.append((run, True))
            else:
                parts.append((run, False))
            start = None
    return parts


class _MorphLook:
    """Stroke and fill of one end of a morph, as RGBA tuples and a width."""

    def __init__(self, edge, face, lw, ls):
        self.edge, self.face, self.lw, self.ls = edge, face, lw, ls


def _rgba(color, alpha):
    if color is None or _is_none_color(color):
        return None
    rgba = list(to_rgba(color))
    if alpha is not None:
        rgba[3] *= float(alpha)
    return tuple(rgba)


def _morph_source(item, ax, closed, kw):
    """One end of a morph: its parts, the transform to display, and its look."""
    kw = dict(kw or {})
    if isinstance(item, Line2D):
        parts = _line_parts(item.get_xydata(), closed)
        edge = _rgba(item.get_color(), item.get_alpha())
        face, lw, ls = None, item.get_linewidth(), item.get_linestyle()
        transform = item.get_transform()
    elif isinstance(item, PathPatch):
        parts = _path_parts(item.get_path())
        # A patch's colors already carry its alpha.
        edge = _rgba(item.get_edgecolor(), None) if item.get_linewidth() > 0 else None
        face = _rgba(item.get_facecolor(), None) if item.get_fill() else None
        lw, ls = item.get_linewidth(), item.get_linestyle()
        transform = item.get_transform()
    else:
        if ax is None:
            raise ValueError("morph needs ax= (the data axes) for an array path")
        parts = _line_parts(item, closed)
        edge = _rgba(_style.neutral(1.0), None)
        face, lw, ls = None, float(matplotlib.rcParams["lines.linewidth"]), "-"
        transform = ax.transData
    if "edgecolor" in kw:
        edge = _rgba(kw.pop("edgecolor"), None)
    if "facecolor" in kw:
        face = _rgba(kw.pop("facecolor"), None)
    if "linewidth" in kw:
        lw = float(kw.pop("linewidth"))
    alpha = kw.pop("alpha", None)
    if kw:
        raise TypeError(f"unknown morph style keys: {sorted(kw)}")
    if alpha is not None:
        edge = None if edge is None else (*edge[:3], edge[3] * float(alpha))
        face = None if face is None else (*face[:3], face[3] * float(alpha))
    if not parts:
        raise ValueError("morph has no finite points in one of its paths")
    return parts, transform, _MorphLook(edge, face, float(lw), ls)


def _resample(vertices, closed, n):
    """`n` points spaced evenly in arc length along a path.

    A closed path returns `n` distinct points (the first is not repeated at
    the end); an open path returns `n` points from its first vertex to its
    last. A path of zero length returns its one point `n` times.
    """
    v = np.asarray(vertices, dtype=float)
    if closed or len(v) < 2:
        ring = np.vstack([v, v[:1]]) if len(v) > 1 else v
    else:
        ring = v
    seg = np.hypot(*np.diff(ring, axis=0).T) if len(ring) > 1 else np.zeros(0)
    total = float(seg.sum())
    if total <= 0.0:
        return np.repeat(v[:1], n, axis=0)
    at = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.linspace(0.0, total, n, endpoint=not closed)
    return np.c_[np.interp(s, at, ring[:, 0]), np.interp(s, at, ring[:, 1])]


def _align(a, b):
    """`b` rolled, and reversed if that is shorter, to sit on closed path `a`.

    Both are (n, 2) rings of corresponding points. Every start index of `b`
    in both directions is tried, and the one with the least summed squared
    distance to `a` wins; ties go to the unreversed ring and the smaller
    shift.
    """
    n = len(a)
    shifts = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    best, pick = np.inf, b
    for ring in (b, b[::-1]):
        cost = ((a[None, :, :] - ring[shifts]) ** 2).sum(axis=(1, 2))
        k = int(np.argmin(cost))
        if cost[k] < best * (1.0 - 1e-12):
            best, pick = float(cost[k]), ring[shifts[k]]
    return pick


def _size(vertices):
    return float(np.ptp(vertices[:, 0]) + np.ptp(vertices[:, 1]))


def _morph_pairs(parts_a, parts_b, n, align):
    """Paired point arrays ``(pa, pb, closed)``, one per output subpath.

    Subpaths are ranked by size (the half perimeter of their box, largest
    first) and the ranks paired. A path with no partner is paired with a
    point at its own center, so it shrinks to that point or grows from it.
    """
    order_a = sorted(parts_a, key=lambda p: -_size(p[0]))
    order_b = sorted(parts_b, key=lambda p: -_size(p[0]))
    out = []
    for i in range(max(len(order_a), len(order_b))):
        va = order_a[i] if i < len(order_a) else None
        vb = order_b[i] if i < len(order_b) else None
        if va is not None and vb is not None:
            closed = bool(va[1] and vb[1])
            pa = _resample(va[0], closed, n)
            pb = _resample(vb[0], closed, n)
            if closed and align:
                pb = _align(pa, pb)
        elif va is not None:
            closed = va[1]
            pa = _resample(va[0], closed, n)
            pb = np.repeat(0.5 * (va[0].min(axis=0) + va[0].max(axis=0))[None], n, 0)
        else:
            closed = vb[1]
            pb = _resample(vb[0], closed, n)
            pa = np.repeat(0.5 * (vb[0].min(axis=0) + vb[0].max(axis=0))[None], n, 0)
        out.append((pa, pb, closed))
    return out


def _lerp_color(c0, c1, e):
    """Straight RGBA blend; a missing end is the other end's color, clear."""
    if c0 is None and c1 is None:
        return None
    if c0 is None:
        c0 = (*c1[:3], 0.0)
    if c1 is None:
        c1 = (*c0[:3], 0.0)
    return tuple((1.0 - e) * np.asarray(c0) + e * np.asarray(c1))


class _GhostMorph(PathPatch):
    """The ghost of a morph: one patch carrying every subpath."""

    def __init__(self, ends, *, n, align, ease):
        super().__init__(
            Path(np.zeros((1, 2)), [Path.MOVETO]), transform=IdentityTransform()
        )
        self._ends = ends
        self._n = n
        self._align = align
        self._ease = ease
        self._u = 1.0
        self._level = 1.0
        self.eased = 1.0
        self._key = None
        self._pairs = None
        self.set_linestyle(ends[0][2].ls)

    def _display(self, end):
        parts, transform, _ = end
        return [(transform.transform(v), c) for v, c in parts]

    def pairs(self):
        """The paired, aligned point arrays in display pixels, cached by geometry."""
        parts_a, parts_b = (self._display(end) for end in self._ends)
        key = b"".join(v.tobytes() for v, _ in parts_a + parts_b)
        if key != self._key:
            self._pairs = _morph_pairs(parts_a, parts_b, self._n, self._align)
            self._key = key
        return self._pairs

    def shape(self, u):
        """The ghost's subpaths ``[(vertices, closed), ...]`` at progress `u`.

        Vertices are in display pixels: the straight blend of the resampled,
        aligned ends at the eased progress.
        """
        e = _ease(_check_u(u), self._ease)
        return [((1.0 - e) * pa + e * pb, c) for pa, pb, c in self.pairs()]

    def draw(self, renderer):
        """Blend the two ends at the current progress, then draw."""
        if not self.get_visible():
            return
        e = _ease(self._u, self._ease)
        self.eased = e
        vertices, codes = [], []
        for pa, pb, closed in self.pairs():
            v = (1.0 - e) * pa + e * pb
            vertices += [v, v[:1]] if closed else [v]
            codes += [Path.MOVETO, *[Path.LINETO] * (len(v) - 1)]
            if closed:
                codes.append(Path.CLOSEPOLY)
        self.set_path(Path(np.vstack(vertices), codes))
        look_a, look_b = self._ends[0][2], self._ends[1][2]
        blend = min(1.0, max(0.0, e))  # colors and widths cannot overshoot
        edge = _lerp_color(look_a.edge, look_b.edge, blend)
        face = _lerp_color(look_a.face, look_b.face, blend)
        self.set_linewidth((1.0 - blend) * look_a.lw + blend * look_b.lw)
        for color, setter in ((edge, self.set_edgecolor), (face, self.set_facecolor)):
            if color is None:
                setter("none")
            else:
                setter((*color[:3], color[3] * self._level))
        super().draw(renderer)


def morph(
    a,
    b,
    *,
    ax=None,
    closed=None,
    n=_MORPH_POINTS,
    align=True,
    ease=DEFAULT_EASE,
    a_kw=None,
    b_kw=None,
    zorder=20,
):
    """Carry one path into another, keeping its identity on the way.

    Its ghost changes shape; both original paths stay in place.

    The move shows that one shape has become another: a circle that becomes a
    square, a curve that becomes a different curve, an outline that becomes
    a smaller one. A ghost is drawn in between, and each point of the first
    path travels to a point of the second, so the viewer can follow the
    shape rather than see one disappear and another appear. The ends are
    paths in the sense of matplotlib: open polylines, closed rings, or
    several subpaths of either.

    Both paths are resampled to `n` points evenly spaced in arc length. For
    a closed pair the start point and the direction of the second are
    chosen so that the summed squared distance between corresponding points
    is least, so a ring does not turn inside out or spin when it could
    simply grow; an open pair keeps its drawn order. The ghost is the
    straight blend of the two samplings, and its stroke color, fill color,
    opacity (all through RGBA) and line width blend the same way; an end
    with no fill blends toward the other end's fill color from clear. The
    blend is straight in display space, so two simple closed paths that are
    star-shaped about a common point stay simple, but a pair with a very
    different winding can cross itself on the way.

    Subpaths are ranked by size (the half perimeter of the box) and paired
    in rank. A path with no partner shrinks to a point at its center, or
    grows from one. A subpath is closed on the way only when both of its
    ends are closed; otherwise the closed end is traversed as an open path
    from its first vertex.

    The ghost is a figure-level patch, so it never joins an axes'
    autoscaling. It is visible only for ``0 < u < 1``: at ``u = 0`` and
    ``u = 1`` it draws nothing, so those frames are the frames without the
    move and the caller shows the first and the second path at their ends.
    The ghost fades in over the first fifth of the move and fades out over
    the last fifth, continuously reaching its absent rest state at ``u = 1``.

    Args:
        a: The path to start from: a `Line2D` (a run is closed if its first
            and last points coincide), a `PathPatch`, or an array of
            vertices (N, 2) in the data coordinates of `ax`. NaN rows split
            a line or array into subpaths.
        b: The path to end on, of the same kinds.
        ax: The axes whose data coordinates an array end is in, and whose
            figure owns the ghost. Required for an array end; otherwise the
            figure of the first artist end owns it.
        closed: True to treat every run of a line or an array end as a closed
            ring, False as open; None infers it per run, closed when the
            first and last points coincide. A `PathPatch` always follows its
            own path codes.
        n: Points per subpath after resampling, at least 3.
        align: Whether closed pairs search the start point and direction.
            False pairs the resampled points as drawn.
        ease: An easing kind for `eyepiece.ease`, or a callable from progress to
            eased progress such as ``functools.partial(ep.ease, kind="back_out",
            strength=0.2)``, applied to the progress.
        a_kw: Overrides for the style of the first path: any of
            ``edgecolor``, ``facecolor``, ``linewidth`` and ``alpha``.
        b_kw: The same for the second path.
        zorder: Drawing order of the ghost among the figure's children.

    Returns:
        A `PlotResult` on `ax`, or the first artist's axes (None if it has none), with
        ``artists["fill"]`` the ghost patch and ``update(u, level=1.0)``,
        `u` from 0 to 1 (values outside are clipped) and `level` the
        opacity of the ghost as a fraction of its blended opacity, so a
        `Stage` can dim it. ``artists["fill"].shape(u)`` returns the ghost's
        subpaths at progress `u` as ``[(vertices, closed), ...]``.

    Raises:
        TypeError: If an end is neither an array nor a `Line2D` or a
            `PathPatch`, or a style key is unknown.
        ValueError: If a path has no finite point, `n` is below 3, or an
            array end is given without `ax`.

    Example::

        ring, = ax.plot(np.cos(t), np.sin(t))
        box = ax.add_patch(PathPatch(square_path, fc="none"))
        step = ep.morph(ring, box)
        for u in np.linspace(0.0, 1.0, 30):
            step.update(u)
            rec.frame()
    """
    n = int(n)
    if n < 3:
        raise ValueError(f"morph needs at least 3 points per path, got {n}")
    for item in (a, b):
        if isinstance(item, Artist) and not isinstance(item, (Line2D, PathPatch)):
            raise TypeError(
                f"morph carries a Line2D, a PathPatch or an array, got "
                f"{type(item).__name__}"
            )
    owner = (
        ax
        if ax is not None
        else next((i for i in (a, b) if isinstance(i, Artist)), None)
    )
    if owner is None:
        raise ValueError("morph needs ax= when both paths are arrays")
    axes = ax if ax is not None else owner.axes
    ends = [_morph_source(item, ax, closed, kw) for item, kw in ((a, a_kw), (b, b_kw))]
    ghost = _unsnapped(_GhostMorph(ends, n=n, align=bool(align), ease=ease))
    ghost.set_zorder(zorder)
    ghost.set_clip_on(False)
    ghost.set_in_layout(False)
    _root_figure(owner).add_artist(ghost)

    def update(u, level=1.0):
        u = _check_u(u)
        ghost._u = u
        envelope = _ease(min(1.0, u / 0.2, (1.0 - u) / 0.2), "smoothstep")
        ghost._level = min(1.0, max(0.0, float(level))) * envelope
        ghost.set_visible(0.0 < u < 1.0)

    update(1.0)
    return PlotResult(
        ax=axes,
        artists={"fill": ghost},
        update=update,
        contract=MotionContract(
            moves=(ghost,),
            may_overlap=tuple(
                (ghost, item)
                for item in (a, b)
                if isinstance(item, Artist)
                and _root_figure(item) is _root_figure(ghost)
                and item.get_visible()
            ),
        ),
    )


# Write

_WRITE_LW_EM = 0.06


def _glyph_outlines(text, renderer):
    """The glyphs of a `Text` as display-space paths, in reading order.

    Each glyph is a list of ``(vertices, closed)`` contours in display
    pixels, laid out exactly as the text draws: the same font properties,
    the same line origins and the same rotation. Mathtext rules (fraction
    bars, radical overlines) count as glyphs. Glyphs with no outline (a
    space) are left out.
    """
    from matplotlib.textpath import text_to_path

    figure = _root_figure(text)
    _, info, _ = text._get_layout(renderer)
    prop = text.get_fontproperties()
    posx, posy = text.get_transform().transform(
        (text.convert_xunits(text._x), text.convert_yunits(text._y))
    )
    size_px = _pixels(renderer, figure, prop.get_size_in_points())
    angle = text.get_rotation()
    glyphs = []
    for line, _wh, (x, y) in info:
        clean, ismath = text._preprocess_math(line)
        if ismath == "TeX":
            raise ValueError("write does not trace usetex text")
        if ismath:
            glyph_info, glyph_map, rects = text_to_path.get_glyphs_mathtext(prop, clean)
        else:
            font = text_to_path._get_font(prop)
            glyph_info, glyph_map, rects = text_to_path.get_glyphs_with_font(
                font, clean
            )
        moves = (
            Affine2D()
            .scale(size_px / text_to_path.FONT_SCALE)
            .rotate_deg(angle)
            .translate(x + posx, y + posy)
        )
        items = []
        for glyph_id, gx, gy, scale in glyph_info:
            verts, codes = glyph_map[glyph_id]
            items.append((np.asarray(verts) * scale + [gx, gy], codes))
        for verts, codes in rects:
            items.append((np.asarray(verts, dtype=float), codes))
        # Reading order is the left edge on the page. A rule's closing vertex
        # is a placeholder at the origin, so it must not set the key.
        placed = [
            (moves.transform(verts), codes) for verts, codes in items if len(verts)
        ]
        for verts, codes in sorted(placed, key=_left_edge):
            parts = _path_parts(Path(verts, codes))
            if parts:
                glyphs.append(parts)
    return glyphs


def _left_edge(item):
    verts, codes = item
    return float(verts[np.asarray(codes) != Path.CLOSEPOLY, 0].min())


def _glyph_length(glyph):
    return sum(float(np.hypot(*np.diff(_ring(v, c), axis=0).T).sum()) for v, c in glyph)


def _ring(vertices, closed):
    return np.vstack([vertices, vertices[:1]]) if closed else vertices


class _GhostWrite(Artist):
    """The ghost of a write: glyph outlines traced, then the text faded in."""

    def __init__(self, text, *, draw, overlap, lag, lw, ease):
        super().__init__()
        self._text = text
        self._draw = draw
        self._overlap = overlap
        self._lag = lag
        self._lw = lw
        self._ease = ease
        self._u = 1.0
        self._level = 1.0
        self._key = None
        self._glyphs = None
        self._lengths = None
        kw = {"transform": IdentityTransform(), "clip_on": False}
        self._stroke = PathPatch(
            Path(np.zeros((1, 2)), [Path.MOVETO]),
            fill=False,
            capstyle="round",
            joinstyle="round",
            **kw,
        )

    def glyphs(self, renderer=None):
        """Every glyph as ``[(vertices, closed), ...]`` in display pixels."""
        text = self._text
        figure = _root_figure(text)
        renderer = _renderer(figure) if renderer is None else renderer
        key = (
            text.get_text(),
            text.get_fontproperties().get_fontconfig_pattern(),
            text.get_rotation(),
            float(renderer.points_to_pixels(1.0)),
            tuple(np.ravel(text.get_transform().get_affine().get_matrix())),
            text.get_unitless_position(),
            text.get_ha(),
            text.get_va(),
            text.get_rotation_mode(),
            text.get_linespacing(),
            text.get_wrap(),
        )
        if key != self._key:
            self._glyphs = _glyph_outlines(text, renderer)
            self._lengths = [_glyph_length(g) for g in self._glyphs]
            self._key = key
        return self._glyphs

    def times(self, u):
        """``(traced, fill)`` at progress `u`.

        `traced` holds the eased fraction of each glyph's outline that is
        drawn, in reading order; `fill` is the eased opacity of the fill.
        """
        u = _check_u(u)
        n = len(self._glyphs) if self._glyphs is not None else 1
        span = 1.0 / (1.0 + (n - 1) * self._lag)
        g = min(1.0, u / self._draw)
        traced = [
            min(
                1.0,
                max(
                    0.0,
                    _ease(
                        min(1.0, max(0.0, (g - i * self._lag * span) / span)),
                        self._ease,
                    ),
                ),
            )
            for i in range(n)
        ]
        start = self._draw - self._overlap
        fill = _ease(min(1.0, max(0.0, (u - start) / (1.0 - start))), self._ease)
        fill = min(1.0, max(0.0, fill))
        return traced, fill

    def strokes(self, u, renderer=None):
        """The traced part of every glyph at progress `u`, in display pixels.

        Returns:
            One list per glyph, in reading order, of the polylines drawn so
            far (an untouched glyph has none).
        """
        glyphs = self.glyphs(renderer)
        traced, _ = self.times(u)
        out = []
        for glyph, length, frac in zip(glyphs, self._lengths, traced, strict=True):
            reach, lines = frac * length, []
            for vertices, closed in glyph:
                ring = _ring(vertices, closed)
                seg = float(np.hypot(*np.diff(ring, axis=0).T).sum())
                if reach > 0.0 and seg > 0.0:
                    lines.append(_cut(ring, min(1.0, reach / seg)))
                reach = max(0.0, reach - seg)
            out.append(lines)
        return out

    def draw(self, renderer):
        """Trace and fill the glyphs at the current progress."""
        if not self.get_visible():
            return
        text = self._text
        color = to_rgb(text.get_color())
        alpha = 1.0 if text.get_alpha() is None else float(text.get_alpha())
        alpha *= self._level
        _, fill = self.times(self._u)
        verts, codes = [], []
        for lines in self.strokes(self._u, renderer):
            for line in lines:
                verts.append(line)
                codes += [Path.MOVETO, *[Path.LINETO] * (len(line) - 1)]
        figure = _root_figure(text)
        size = text.get_fontproperties().get_size_in_points()
        lw = _WRITE_LW_EM * size if self._lw is None else self._lw
        for patch in (self._stroke,):
            patch.set_figure(figure)
            patch.set_clip_box(text.get_clip_box())
            patch.set_clip_path(text.get_clip_path())
            patch.set_clip_on(text.get_clip_on())
        if verts:
            self._stroke.set_path(Path(np.vstack(verts), codes))
            self._stroke.set_edgecolor((*color, alpha * (1.0 - fill)))
            self._stroke.set_linewidth(lw)
            self._stroke.draw(renderer)
        if fill > 0.0:
            # The text itself fades in, so the last frames are its own
            # rasterization. Its alpha and visibility are restored exactly.
            was_alpha, was_visible = text.get_alpha(), text.get_visible()
            text.set_alpha(alpha * fill)
            text.set_visible(True)
            try:
                text.draw(renderer)
            finally:
                text.set_alpha(was_alpha)
                text.set_visible(was_visible)


def write(
    text,
    *,
    draw=0.7,
    overlap=0.1,
    lag=0.5,
    lw=None,
    ease=DEFAULT_EASE,
    zorder=20,
):
    """Write a text out by hand: trace its glyph outlines, then fill them.

    Its ghost traces changing outlines and cannot be traced by the checker; the
    real text is hidden until the end, so the contract declares nothing.

    The move is for a label or a short equation that should arrive the way
    a hand would put it on a board, so the eye follows the writing. A ghost
    traces the outline of each glyph in reading order, one glyph starting
    before the one before it has finished, and then the text itself fades in
    as the stroke fades out. The outlines come from the font the text itself uses
    (its `FontProperties`, size, rotation, alignment and transform, laid out
    as the text lays itself out), so the ghost sits on the text. Mathtext
    works through the mathtext parser: every symbol is a glyph, and a
    fraction bar or a radical's overline counts as one. Several lines are
    written one after the other, top down.

    The real `Text` is hidden while the ghost is drawn and shown again at
    the end, so the rest state is the text as drawn, with its own colors,
    opacity and visibility. The ghost reads the text's color and opacity
    each time it is drawn, so a `Stage` that owns the text weighs it too.

    ``update(u, level=1.0)`` is absolute. Over ``0 < u < draw`` the strokes
    are traced; the fill starts at ``draw - overlap`` and is complete at
    ``u = 1``. At ``u = 0`` nothing is drawn, and at ``u = 1`` the text is
    drawn as it is without the move. The fill is the real text drawn at a
    fractional opacity, so the last frames converge on its rasterization
    and the swap at ``u = 1`` does not jump; the strokes are gone by then.

    Args:
        text: The `Text` artist to write. It must stay in its figure, and
            while the move is in use it should not be hidden or shown by
            hand.
        draw: Fraction of the progress spent tracing, in (0, 1).
        overlap: How much of that tracing the fill overlaps, as a fraction
            of the progress, in [0, draw): the fill starts at
            ``draw - overlap``. Zero starts the fill when the tracing ends.
        lag: How far a glyph's start is offset from the one before it, as a
            fraction of its own tracing time, in [0, 1]: 0 traces every
            glyph at once and 1 one after another.
        lw: Stroke width in points. None uses six hundredths of the font
            size.
        ease: An easing kind for `eyepiece.ease`, applied to each glyph's
            tracing and to the fill.
        zorder: Drawing order of the ghost among the figure's children.

    Returns:
        A `PlotResult` on the text's axes (None for text on the figure), with
        ``artists["fill"]`` the ghost and ``artists["text"]`` the real text,
        and ``update(u, level=1.0)``, `u` from 0 to 1 (values outside are
        clipped) and `level` the opacity of the ghost as a fraction of the
        text's own. ``artists["fill"].glyphs()`` returns every glyph as
        ``[(vertices, closed), ...]`` in display pixels, and
        ``artists["fill"].strokes(u)`` the traced part of each at `u`.

    Raises:
        TypeError: If `text` is not a `Text`.
        ValueError: If the text has no outline to trace (empty or only
            spaces), uses TeX, or `draw`, `overlap` or `lag` is out of range.

    Example::

        title = ax.text(0.5, 0.5, r"$E = mc^2$", fontsize=24, ha="center")
        step = ep.write(title)
        for u in np.linspace(0.0, 1.0, 36):
            step.update(u)
            rec.frame()
    """
    if not isinstance(text, Text):
        raise TypeError(f"write takes a Text artist, got {type(text).__name__}")
    draw, overlap, lag = float(draw), float(overlap), float(lag)
    if not 0.0 < draw < 1.0:
        raise ValueError(f"write draw must be in (0, 1), got {draw}")
    if not 0.0 <= overlap < draw:
        raise ValueError(f"write overlap must be in [0, draw), got {overlap}")
    if not 0.0 <= lag <= 1.0:
        raise ValueError(f"write lag must be in [0, 1], got {lag}")
    if text.get_usetex():
        raise ValueError("write does not trace usetex text")
    figure = _root_figure(text)
    ghost = _GhostWrite(text, draw=draw, overlap=overlap, lag=lag, lw=lw, ease=ease)
    if not ghost.glyphs():
        raise ValueError(f"write found no outline to trace in {text.get_text()!r}")
    ghost.set_zorder(zorder)
    ghost.set_in_layout(False)
    figure.add_artist(ghost)
    shown = text.get_visible()

    def update(u, level=1.0):
        u = _check_u(u)
        ghost._u = u
        ghost._level = min(1.0, max(0.0, float(level)))
        ghost.set_visible(shown and u < 1.0)
        text.set_visible(shown and u >= 1.0)

    update(1.0)
    return PlotResult(
        ax=text.axes,
        artists={"fill": ghost, "text": text},
        update=update,
        contract=MotionContract(),
    )


# Term row


class _Row:
    """The layout of a row of terms: one slot per term, side by side."""

    def __init__(self, at, *, gap_em, ha, fontsize):
        self.at = at
        self.gap_em = float(gap_em)
        self.ha = ha
        self.fontsize = float(fontsize)
        self.terms = []
        self._probes = []

    def add(self, term):
        probe = Text(0.0, 0.0, "", fontproperties=term.get_fontproperties().copy())
        probe.set_fontsize(self.fontsize)
        self.terms.append(term)
        self._probes.append(probe)

    def layout(self, renderer=None):
        """Put every term in its slot, as the text and the figure are now.

        A slot is as wide as its term set at the row's own font size, so a
        pulse that swells a term does not push its neighbors; the term is
        centered on its slot. All terms sit on one baseline.
        """
        figure = _root_figure(self.terms[0])
        em = _pixels(renderer, figure, self.fontsize)
        widths = []
        for term, probe in zip(self.terms, self._probes, strict=True):
            probe.set_figure(figure)
            probe.set_text(term.get_text())
            widths.append(
                Text.get_window_extent(probe, renderer).width
                if term.get_text()
                else 0.0
            )
        shown = [w for w in widths if w > 0.0]
        gap = self.gap_em * em
        total = sum(shown) + gap * max(0, len(shown) - 1)
        x0, y = self.at.display(renderer)
        x = x0 - {"left": 0.0, "center": 0.5, "right": 1.0}[self.ha] * total
        for term, width in zip(self.terms, widths, strict=True):
            term.set_position((float(x + 0.5 * width), float(y)))
            if width > 0.0:
                x += width + gap


class _RowTerm(Text):
    """One term of a row, laid out beside its neighbors whenever it is measured."""

    def __init__(self, row, string, **kw):
        super().__init__(
            0.0,
            0.0,
            string,
            transform=IdentityTransform(),
            ha="center",
            va="baseline",
            **kw,
        )
        self._row = row

    def get_window_extent(self, renderer=None, dpi=None):
        """The term's box in its slot, as the row is laid out now."""
        self._row.layout(renderer)
        return super().get_window_extent(renderer, dpi)

    def draw(self, renderer):
        """Lay the row out, then draw the term."""
        if not self.get_visible():
            return
        self._row.layout(renderer)
        super().draw(renderer)


def _write_on(texts, fades, u, ease=DEFAULT_EASE):
    """Fade the texts in one by one: text ``i`` of ``n`` is whole at ``(i + 1) / n``."""
    n = len(texts)
    for i, (text, fade) in enumerate(zip(texts, fades, strict=True)):
        level = min(1.0, max(0.0, n * u - i))
        text.set_visible(level > 0.0)
        shaped = min(1.0, max(0.0, _ease(level, ease)))  # opacity cannot overshoot
        fade.set(shaped if level < 1.0 else 1.0)


def term_row(
    at,
    terms,
    *,
    gap_em=0.4,
    ha="left",
    color=None,
    fontsize=None,
    zorder=10,
    text_kw=None,
):
    r"""Lay an equation out as one text per term, each term its own target.

    Each term is its own `Text` (a plain string, or math in ``$...$``), set
    left to right from an anchor on one baseline with a gap between
    neighbors, so that any term can be the target of an `Anchor.from_artist`,
    a member of a `Stage` group, the destination of a `flight`, the subject
    of a `callout`, or pulsed alone with `indicate`. The slots are measured
    and set when the figure is drawn (and when a term's box is asked for), so
    the row stays laid out when a term's text changes: a term the caller
    rewrites is ``result.artists["text"][i].set_text("x = 2")``, and the terms
    after it move over to make room. A slot is as wide as its term at the row's font
    size, so an `indicate` on one term swells it in place and moves none of
    its neighbors.

    The terms are figure-level artists in display space, so they never join
    an axes' autoscaling and move no data limit. An empty term has no slot
    and no gap, which lets a term start blank.

    Which tool lays out text: `term_row` is for terms whose text the caller
    chooses and whose neighbors may move; `readout` is for one number that
    counts to a new value and must hold its place (do not run a number
    through `term_row` terms); `words` colors or reveals words inside one
    ordinary line of text.

    ``update(u)`` writes the row on, one term after another: term ``i`` of
    ``n`` is at full opacity once ``u`` reaches ``(i + 1) / n``. At ``u = 1``,
    the rest state, every term is drawn as made. A row that a `Stage` owns
    is shown and hidden by the stage instead, and ``update`` is not called.

    Args:
        at: Where the row starts: an `Anchor`, or an artist. Its point is the
            baseline's left end, center, or right end, by `ha`.
        terms: The terms in order: strings, or ``(string, color)`` pairs.
        gap_em: Gap between neighbors, in font sizes.
        ha: Which end of the row `at` marks: ``"left"``, ``"center"`` or
            ``"right"``.
        color: Default term color. None uses ``rcParams["text.color"]``.
        fontsize: Font size in points. None uses ``rcParams["font.size"]``.
        zorder: Drawing order among the figure's children.
        text_kw: Extra kwargs for every `Text`, applied last.

    Returns:
        A `PlotResult` on the anchor's axes (None for an anchor drawn on the
        figure), with ``artists["text"]``, the terms in order, and
        ``update(u)`` as above.

    Raises:
        TypeError: If `at` is neither an `Anchor` nor an artist.
        ValueError: If there are no terms, `ha` is unknown, or `gap_em` is
            negative.

    Example::

        row = ep.term_row(ep.Anchor.from_data(ax, (0.1, 0.9)),
                          ["$\chi^2$", "=", "0.0"])
        row.artists["text"][2].set_text("2.41")  # a term is rewritten
        ep.Anchor.from_artist(row.artists["text"][0])  # a term is an anchor
    """
    at = _as_anchor(at, "at")
    if ha not in ("left", "center", "right"):
        raise ValueError(f"unknown ha: {ha!r}; known: ['left', 'center', 'right']")
    if float(gap_em) < 0.0:
        raise ValueError(f"gap_em must not be negative, got {gap_em}")
    pieces = [(t, None) if isinstance(t, str) else (str(t[0]), t[1]) for t in terms]
    if not pieces:
        raise ValueError("term_row has no terms")
    rc = matplotlib.rcParams
    size = float(rc["font.size"]) if fontsize is None else float(fontsize)
    row = _Row(at, gap_em=gap_em, ha=ha, fontsize=size)
    figure = at.figure
    made = []
    for string, own in pieces:
        kw = {
            "color": rc["text.color"]
            if color is None and own is None
            else own or color,
            "fontsize": size,
            "zorder": zorder,
            **(text_kw or {}),
        }
        term = _RowTerm(row, string, **kw)
        term.set_clip_on(False)
        term.set_in_layout(False)
        figure.add_artist(term)
        row.add(term)
        made.append(term)
    fades = [_TextFade(t) for t in made]

    def update(u):
        _write_on(made, fades, _check_u(u))

    return PlotResult(ax=at.axes, artists={"text": made}, update=update)


# Readout

# A readout format: a sign flag, a grouping mark, a precision, and a fixed-point
# type, so that the shown number has the same number of decimals every frame.
_READOUT_FORMAT = re.compile(r"([+\- ]?)([,_]?)(?:\.(\d+))?([fF%d])")


def _readout_spec(fmt):
    """The flag, grouping mark, decimals and type of a fixed-point `fmt`."""
    match = _READOUT_FORMAT.fullmatch(str(fmt))
    if match is None:
        raise ValueError(
            f"readout fmt {fmt!r} is not a fixed-point format spec such as "
            "'.2f', '+.1f', ',.0f', '.1%' or 'd'"
        )
    flag, group, decimals, kind = match.groups()
    if kind == "d" and decimals is not None:
        raise ValueError(f"readout fmt {fmt!r}: an integer format has no precision")
    if decimals is None:
        decimals = "0" if kind == "d" else "6"
    return flag, group, int(decimals), kind


class _Readout:
    """A number shown in fixed slots: prefix, integer part, fraction, tail.

    The decimal point sits at one place for the whole life of the readout.
    The integer part is right-aligned to it and the fraction left-aligned,
    whatever the font, and each slot is as wide as the widest string it can
    hold, so a count never moves the label.
    """

    def __init__(
        self,
        at,
        value,
        *,
        fmt,
        prefix,
        tail,
        ha,
        fontsize,
        digits,
        signed,
        ease,
        text_kw,
        color,
        zorder,
    ):
        self.at = at
        self.figure = at.figure
        self.flag, self.group, self.decimals, self.kind = _readout_spec(fmt)
        self.fmt = str(fmt)
        self.prefix = prefix
        self.tail = tail
        self.ha = ha
        self.ease = ease
        self.n_int = max(1, int(digits))
        self.signed = self.flag in ("+", " ") if signed is None else bool(signed)
        self.grow_sign = signed is None
        self.declined = signed is not None and not signed
        self.minus = "\u2212" if matplotlib.rcParams["axes.unicode_minus"] else "-"
        self.level = 1.0
        self._cache = {}
        kw = {"color": color, "fontsize": fontsize, "zorder": zorder, **text_kw}
        self.probe = Text(0.0, 0.0, "", **kw)
        self.probe.set_figure(self.figure)
        self.reserve(value)
        self.start = self.target = float(value)
        self.parts = []
        int_s, frac_s = self.strings(value)
        spec = [("prefix", prefix, "left"), ("int", int_s, "right")]
        if self.decimals > 0:
            spec.append(("frac", frac_s, "left"))
        spec.append(("tail", tail, "left"))
        for role, string, align in spec:
            if role in ("prefix", "tail") and not string:
                continue
            part = _ReadoutPart(self, role, string, align, **kw)
            part.set_clip_on(False)
            part.set_in_layout(False)
            self.figure.add_artist(part)
            self.parts.append(part)
        self.fades = [_TextFade(p) for p in self.parts]

    def strings(self, value):
        """The integer part and the fraction part of `value` as shown."""
        value = float(value)
        if self.kind == "d":
            text = format(round(value), self.fmt)
        else:
            text = format(value, self.fmt)
        # A value that rounds to zero shows no minus, so a count through zero
        # does not flash a negative zero.
        if text[:1] == "-" and not any(c in "123456789" for c in text):
            text = "+" + text[1:] if self.flag == "+" else text[1:]
        text = text.removeprefix(" ")
        text = text.replace("-", self.minus)
        if self.decimals == 0:
            return text, ""
        point = text.index(".")
        return text[:point], text[point:]

    def reserve(self, value):
        """Grow the slots, if need be, to hold `value`."""
        if not math.isfinite(float(value)):
            raise ValueError(f"readout value must be finite, got {value}")
        int_s, _ = self.strings(value)
        self.n_int = max(self.n_int, sum(c.isdigit() for c in int_s))
        if self.grow_sign and int_s[:1] in (self.minus, "+"):
            self.signed = True
        if self.declined and int_s[:1] == self.minus:
            raise ValueError(
                f"readout value {value} is negative but signed=False reserves "
                "no room for the sign"
            )

    def width(self, string, renderer):
        """The width in pixels of `string` in the readout's font."""
        if not string:
            return 0.0
        key = (self.figure.dpi, type(renderer).__name__, string)
        if key not in self._cache:
            self.probe.set_text(string)
            self._cache[key] = float(Text.get_window_extent(self.probe, renderer).width)
        return self._cache[key]

    def layout(self, renderer=None):
        """Put every part in its slot, as the figure is now."""
        widest = max("0123456789", key=lambda d: self.width(d, renderer))
        sign = 0.0
        if self.signed:
            sign = max(self.width(self.minus, renderer), self.width("+", renderer))
        whole = format(10**self.n_int - 1, f"{self.group}d")
        whole = re.sub(r"\d", widest, whole)
        if self.decimals == 0 and self.kind == "%":
            whole += "%"
        w_int = sign + self.width(whole, renderer)
        frac = ""
        if self.decimals > 0:
            frac = "." + widest * self.decimals + ("%" if self.kind == "%" else "")
        w_frac = self.width(frac, renderer)
        w_prefix = self.width(self.prefix, renderer)
        w_tail = self.width(self.tail, renderer)
        x0, y = self.at.display(renderer)
        total = w_prefix + w_int + w_frac + w_tail
        if self.ha == "point":
            left = x0 - w_prefix - w_int
        else:
            left = x0 - {"left": 0.0, "center": 0.5, "right": 1.0}[self.ha] * total
        point = left + w_prefix + w_int
        for part in self.parts:
            if part.role == "prefix":
                part.set_position((float(left), float(y)))
            elif part.role == "int":
                part.set_position((float(point), float(y)))
                part.slot = (float(point - w_int), float(point))
            elif part.role == "frac":
                part.set_position((float(point), float(y)))
                part.slot = (float(point), float(point + w_frac))
            else:
                part.set_position((float(point + w_frac), float(y)))

    def show(self, value):
        """Write `value` into the integer and fraction parts."""
        int_s, frac_s = self.strings(value)
        for part in self.parts:
            if part.role == "int":
                part.set_text(int_s)
            elif part.role == "frac":
                part.set_text(frac_s)


class _ReadoutPart(Text):
    """One slot of a readout, laid out beside the others whenever it is measured.

    The integer part and the fraction report their reserved slot as their
    width, so the box of a readout does not move while its digits change.
    """

    def __init__(self, readout, role, string, ha, **kw):
        super().__init__(
            0.0,
            0.0,
            string,
            transform=IdentityTransform(),
            ha=ha,
            va="baseline",
            **kw,
        )
        self._readout = readout
        self.role = role
        self.slot = None

    def get_window_extent(self, renderer=None, dpi=None):
        """The part's box, its whole reserved slot for a number part."""
        self._readout.layout(renderer)
        box = super().get_window_extent(renderer, dpi)
        if self.slot is None:
            return box
        return Bbox([[self.slot[0], box.y0], [self.slot[1], box.y1]])

    def draw(self, renderer):
        """Lay the readout out, then draw the part."""
        if not self.get_visible():
            return
        self._readout.layout(renderer)
        super().draw(renderer)


@dataclass(frozen=True, eq=False)
class _ReadoutResult(PlotResult):
    """A `PlotResult` that also holds the readout's `set`."""

    set: Callable | None = None


def readout(
    at,
    value,
    *,
    fmt=".1f",
    prefix="",
    suffix="",
    unit="",
    ha="left",
    digits=1,
    signed=None,
    ease=DEFAULT_EASE,
    color=None,
    fontsize=None,
    zorder=10,
    text_kw=None,
):
    r"""A running value on screen that counts to a new value without jitter.

    Its digits change in fixed slots; the prefix and tail stay in place.

    Use it for a number that changes while the viewer watches, such as
    ``t = 3.2 h`` or ``Pr = 0.41``. The format is fixed: a fixed-point `fmt`
    (so the number of decimals never changes), a `prefix`, a `suffix` and a
    `unit`. The layout is fixed too. The decimal point stays at one place for
    the life of the readout; the integer part is right-aligned to it and the
    fraction left-aligned from it, and each slot is as wide as the widest
    string it will hold. This holds in any font, tabular digits or not, and
    it is why the label never jitters while the digits run. The prefix and
    the tail (suffix and unit) are laid beside the slots and never move.

    ``update(u)`` shows the value ``u`` of the way from the old value to the
    new one under the ease, and ``set(value)`` chains the next move: the old
    value becomes the value last set, the new one is `value`, and the readout
    shows the old value until ``update`` is called. The shown number is the
    interpolated value rounded to the precision of `fmt`. Rounding is
    monotone, so a count never goes back, and a value that rounds to zero
    shows no minus sign. At ``u = 1`` the readout shows the exact formatted
    new value; this is its rest state, so a still shows the last value set (or
    the first value, if none was set).

    `set` grows the slots if the new value needs more digits or a sign than
    any value so far. That is a change of layout and moves the label once,
    when `set` is called, never during a count. To reserve room in advance,
    pass `digits` and `signed`.

    The parts are `Text` artists of the figure in display space, so they
    never join an axes' autoscaling. The integer part and the fraction report
    their whole slot as their box, which makes them steady anchors for a
    `callout` or a `Stage` group. A `term_row` term set from the loop (see its
    docstring) changes width as its text changes and pushes its neighbors; a
    readout keeps one place for the point instead, and is the move for a
    number that runs.

    Args:
        at: Where the readout is: an `Anchor`, or an artist. Its point is the
            baseline's left end, center or right end by `ha`, or the decimal
            point itself for ``ha="point"``.
        value: The first value shown (finite).
        fmt: A fixed-point format spec: an optional sign flag (``"+"`` or
            ``" "``), an optional grouping mark (``","`` or ``"_"``), a
            precision, and one of ``f``, ``F``, ``%`` or ``d``, such as
            ``".2f"``, ``"+.1f"``, ``".0%"`` or ``"d"``.
        prefix: Text before the number, for example ``"t = "`` or ``"$Pr$ = "``.
        suffix: Text glued to the number, for example ``"x"``.
        unit: Text after the suffix, set off by a space, for example ``"h"``
            or ``r"$\mathrm{km\,s^{-1}}$"``.
        ha: Which end of the readout `at` marks: ``"left"``, ``"center"``,
            ``"right"``, or ``"point"``, the decimal point.
        digits: Integer digits to reserve at least. The slots always hold
            the digits of every value given.
        signed: Whether to reserve room for a sign. None reserves it for a
            sign flag in `fmt` and once a negative value is given; False
            declines, and a negative value then raises.
        ease: The easing kind of `eyepiece.ease` for ``update(u)``. An
            ease that leaves [0, 1] (an overshoot) is clamped here, so a
            count never passes its target and never goes back.
        color: Text color. None uses ``rcParams["text.color"]``.
        fontsize: Font size in points. None uses ``rcParams["font.size"]``.
        zorder: Drawing order among the figure's children.
        text_kw: Extra kwargs for every `Text`, applied last.

    Returns:
        A `PlotResult` on the anchor's axes (None for an anchor drawn on the
        figure), with ``artists["text"]``, the parts left to right (prefix,
        integer part, fraction, tail; the empty ones are left out),
        ``update(u, level=1.0)`` as above, where ``level`` dims the whole
        readout, and ``set(value)``, which starts the next move.

    Raises:
        TypeError: If `at` is neither an `Anchor` nor an artist.
        ValueError: If `fmt` is not a fixed-point spec, `ha` or `ease` is
            unknown, a value is not finite, or a value is negative while
            ``signed=False``.

    Example::

        rd = ep.readout(ep.Anchor.from_data(ax, (0.1, 0.9)), 0.0,
                        prefix="t = ", unit="h")
        rd.set(3.2)
        rd.update(0.5)  # t = 1.6 h
    """
    at = _as_anchor(at, "at")
    if ha not in ("left", "center", "right", "point"):
        raise ValueError(
            f"unknown ha: {ha!r}; known: ['left', 'center', 'right', 'point']"
        )
    _ease(0.5, ease)
    rc = matplotlib.rcParams
    size = float(rc["font.size"]) if fontsize is None else float(fontsize)
    tail = str(suffix) + (f" {unit}" if unit else "")
    state = _Readout(
        at,
        value,
        fmt=fmt,
        prefix=str(prefix),
        tail=tail,
        ha=ha,
        fontsize=size,
        digits=digits,
        signed=signed,
        ease=ease,
        text_kw=dict(text_kw or {}),
        color=rc["text.color"] if color is None else color,
        zorder=zorder,
    )
    parts = state.parts

    def update(u, level=1.0):
        u = _check_u(u)
        level = _check_u(level)
        if u >= 1.0:
            shown = state.target
        elif u <= 0.0:
            shown = state.start
        else:
            # An overshooting ease is clamped, so a count never passes its
            # target and never goes back.
            e = min(1.0, max(0.0, _ease(u, state.ease)))
            shown = state.start + e * (state.target - state.start)
        state.show(shown)
        # Opacity is written only while dimmed and once on the way back, so a
        # readout held by a Stage keeps the stage's levels.
        if level < 1.0 or state.level < 1.0:
            for part, fade in zip(parts, state.fades, strict=True):
                part.set_visible(level > 0.0)
                fade.set(level)
        state.level = level

    def set_value(new):
        state.reserve(new)
        state.start = state.target
        state.target = float(new)
        update(0.0, state.level)

    return _ReadoutResult(
        ax=at.axes,
        artists={"text": parts},
        update=update,
        set=set_value,
        contract=MotionContract(rest=tuple(parts)),
    )


# Words


def _has_math(string):
    """Whether matplotlib reads `string` as mathtext (an even count of bare ``$``)."""
    count = string.count("$") - string.count("\\$")
    return count > 0 and count % 2 == 0


def _word_spans(string):
    """The ``(start, end)`` of each word of `string`, math in ``$...$`` kept whole."""
    spans = []
    start = None
    in_math = False
    math_line = _has_math(string)
    i = 0
    while i < len(string):
        ch = string[i]
        if ch == "\\" and i + 1 < len(string):
            if start is None:
                start = i
            i += 2
            continue
        if ch == "$" and math_line:
            in_math = not in_math
        if ch.isspace() and not in_math:
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
        i += 1
    if start is not None:
        spans.append((start, len(string)))
    return spans


def _text_props(word):
    """The properties that decide a word's size on the page, for a measuring probe."""
    return {
        "fontproperties": word.get_fontproperties().copy(),
        "parse_math": word.get_parse_math(),
        "usetex": word.get_usetex(),
    }


class _Line:
    """The layout of one line of text cut into words, laid out once as a string.

    A word starts where the line up to the end of that word, measured as one
    string, ends less the word's own width, so the spacing and the kerning
    inside each word are those of a single text. Widths depend only on the
    font and the dpi, so they are cached by the size in pixels.

    One engine measures and draws every word. A string with math is drawn
    entirely as mathtext, plain parts included, whose spacing differs from
    plain text, so on such a line the plain words are measured and drawn as
    mathtext too (see `_WordText`).
    """

    def __init__(self, at, string, spans, *, ha):
        self.at = at
        self.string = string
        self.spans = spans
        self.ha = ha
        self.as_math = _has_math(string)
        self.words = []
        self._probe = None
        self._cache = {}

    def add(self, word):
        self.words.append(word)
        if self._probe is None:
            # The probe has the properties the words are drawn with, the font
            # size included, however the caller set them.
            self._probe = _WordText(None, "", **_text_props(word))
            self._probe.set_figure(_root_figure(word))
            self._probe._as_math = self.as_math

    def _width(self, text, renderer):
        probe = self._probe
        probe.set_text(text)
        return float(Text.get_window_extent(probe, renderer).width)

    def _measure(self, renderer, figure):
        key = round(_pixels(renderer, figure, self._probe.get_fontsize()), 6)
        if key not in self._cache:
            self._probe.set_figure(figure)
            starts, total = [], 0.0
            for a, b in self.spans:
                total = self._width(self.string[:b], renderer)
                starts.append(total - self._width(self.string[a:b], renderer))
            self._cache = {key: (starts, total)}
        return self._cache[key]

    def layout(self, renderer=None):
        """Put every word at its place on the line, as the figure is now."""
        figure = _root_figure(self.words[0])
        starts, total = self._measure(renderer, figure)
        x0, y = self.at.display(renderer)
        x0 -= {"left": 0.0, "center": 0.5, "right": 1.0}[self.ha] * total
        for word, start in zip(self.words, starts, strict=True):
            word.set_position((float(x0 + start), float(y)))


class _WordText(Text):
    """One word of a line, placed on the line whenever it is measured or drawn."""

    def __init__(self, line, string, **kw):
        super().__init__(
            0.0,
            0.0,
            string,
            transform=IdentityTransform(),
            ha="left",
            va="baseline",
            **kw,
        )
        self._line = line
        self._as_math = line is not None and line.as_math

    def _preprocess_math(self, s):
        """Read a plain word as mathtext on a line that has math in it."""
        if self._as_math and not self.get_usetex() and self.get_parse_math():
            return s, True
        return super()._preprocess_math(s)

    def get_window_extent(self, renderer=None, dpi=None):
        """The word's box on the line, as the line is laid out now."""
        self._line.layout(renderer)
        return super().get_window_extent(renderer, dpi)

    def draw(self, renderer):
        """Lay the line out, then draw the word."""
        if not self.get_visible():
            return
        self._line.layout(renderer)
        super().draw(renderer)


@dataclass(frozen=True, eq=False)
class _WordsResult(PlotResult):
    """A `PlotResult` of `words`, which also carries ``reveal(u, by=...)``."""

    reveal: Callable | None = None


def words(
    at,
    string,
    *,
    by="word",
    ease=DEFAULT_EASE,
    ha="left",
    color=None,
    colors=None,
    fontsize=None,
    zorder=10,
    text_kw=None,
):
    r"""Set one line of text as one text per word, to color or reveal alone.

    Its words reveal in place; the line layout stays fixed.

    The line is laid out once, as a single string: each word sits where it
    would in one `Text` of the whole line, with that text's spacing, and its
    extent comes from the renderer, so no gap is chosen by hand. A word in
    ``$...$`` (math may hold spaces) is kept whole. Every word is a
    figure-level `Text`, so it can be the target of an `Anchor.from_artist`,
    a member of a `Stage` group, the destination of a `flight`, the subject
    of a `callout`, or pulsed alone with `indicate`, and the line is laid out
    again whenever the figure is drawn, so it follows resizing and a new dpi.

    `words` colors or reveals words inside one ordinary line, which keeps the
    spacing of one text and does not re-lay when a word changes. The `term_row`
    docstring says which tool to use for terms the caller spaces and for a
    number that counts.

    The reveal is in place: a word comes in where it will stay, by opacity
    alone, never by moving, so text does not arrive with motion. By
    ``"word"`` the words come in left to right, each fading in as the one
    before it finishes. By ``"letter"`` the letters of each word come in
    left to right, one after another with no fade (a math word counts as one
    letter, and is whole when it arrives); the box of a word that is partly
    spelled is the box of the letters shown, so anchor on a finished line.
    At ``u = 1``, the rest state, every word is drawn whole as made. A line
    that a `Stage` owns is shown and hidden by the stage instead, and the
    reveal is not called. ``level`` below 1 dims every word by opacity only,
    never by geometry, and is written only while it is below 1.

    One engine measures and draws every word. A string that has math is
    rendered by matplotlib entirely as mathtext, plain parts included, so
    on such a line the plain words are set as mathtext too. At ``u = 1`` the
    joined line is then pixel-identical to one `Text` with the same
    properties, with or without math, in the tests' fonts, sizes and dpis;
    where a font rounds a glyph to a different subpixel position the
    difference stays within one pixel. Trailing spaces of the string are not
    part of the line, so ``ha="right"`` ends at the last word. A line with
    an odd number of unescaped ``$`` has no math, as in matplotlib.

    Args:
        at: Where the line starts: an `Anchor`, or an artist. Its point is
            the baseline's left end, center, or right end, by `ha`.
        string: The line, on one line (no newline).
        by: What ``update`` reveals by: ``"word"`` or ``"letter"``.
        ease: An easing kind for `eyepiece.ease`, or a callable taking the
            clipped progress, shaping how each word fades in by word. Only
            its input is clipped; an overshoot is clamped to [0, 1] where it
            becomes opacity, and the end states are exact. A reveal by
            letter counts letters evenly in ``u``.
        ha: Which end of the line `at` marks: ``"left"``, ``"center"`` or
            ``"right"``.
        color: Default color of the words. None uses ``rcParams["text.color"]``.
        colors: Color of individual words: a dict whose keys are word
            indices or word strings, or a sequence with one color (or None
            for the default) per word. A string key colors every occurrence
            of that word, and a math word is matched by its full ``$...$``
            text. A key that matches no word, or an index and a string that
            reach one word with different colors, raises.
        fontsize: Font size in points. None uses ``rcParams["font.size"]``.
        zorder: Drawing order among the figure's children.
        text_kw: Extra kwargs for every `Text`, applied last. Keep them to
            properties shared by the whole line (font family, weight, size);
            the layout reads the size from the words, so a size here is used.

    Returns:
        A `PlotResult` on the anchor's axes (None for an anchor drawn on the
        figure), with ``artists["text"]``, the words in order. ``update(u,
        level=1.0)`` reveals by `by`, and ``reveal(u, by="word", level=1.0, ease=...)``
        reveals by either.

    Raises:
        TypeError: If `at` is neither an `Anchor` nor an artist.
        ValueError: If the line has no words or a newline, `ha` or `by` is
            unknown, or `colors` does not match the words (a key naming no
            word, a wrong length, or two colors for one word).

    Example::

        line = ep.words(ep.Anchor.from_data(ax, (0.1, 0.5)),
                        "the prior times the likelihood", colors={1: "C0", 4: "C1"})
        line.reveal(0.5, by="letter")
        ep.Anchor.from_artist(line.artists["text"][1], loc="bottom")  # a word
    """
    at = _as_anchor(at, "at")
    if ha not in ("left", "center", "right"):
        raise ValueError(f"unknown ha: {ha!r}; known: ['left', 'center', 'right']")
    if by not in ("word", "letter"):
        raise ValueError(f"unknown by: {by!r}; known: ['word', 'letter']")
    string = str(string)
    if "\n" in string:
        raise ValueError("words sets one line; split the text into lines first")
    spans = _word_spans(string)
    if not spans:
        raise ValueError("words has no words")
    n = len(spans)
    if colors is None:
        own = [None] * n
    elif isinstance(colors, dict):
        tokens = [string[a:b] for a, b in spans]
        own = [None] * n
        for key, value in colors.items():
            if isinstance(key, str):
                hits = [i for i, t in enumerate(tokens) if t == key]
                if not hits:
                    raise ValueError(f"colors key {key!r} matches no word of the line")
            elif isinstance(key, int) and not isinstance(key, bool) and -n <= key < n:
                hits = [key % n]
            else:
                raise ValueError(f"colors has no word {key!r}; the line has {n} words")
            for i in hits:
                if own[i] is not None and to_rgba(own[i]) != to_rgba(value):
                    raise ValueError(
                        f"colors gives word {i} ({tokens[i]!r}) two colors: "
                        f"{own[i]!r} and {value!r}"
                    )
                own[i] = value
    else:
        own = list(colors)
        if len(own) != n:
            raise ValueError(f"colors has {len(own)} entries for {n} words")
    rc = matplotlib.rcParams
    size = float(rc["font.size"]) if fontsize is None else float(fontsize)
    base = rc["text.color"] if color is None else color
    line = _Line(at, string, spans, ha=ha)
    figure = at.figure
    made = []
    for (a, b), mine in zip(spans, own, strict=True):
        kw = {
            "color": base if mine is None else mine,
            "fontsize": size,
            "zorder": zorder,
            **(text_kw or {}),
        }
        word = _WordText(line, string[a:b], **kw)
        word.set_clip_on(False)
        word.set_in_layout(False)
        figure.add_artist(word)
        line.add(word)
        made.append(word)
    fades = [_TextFade(t) for t in made]
    whole = [t.get_text() for t in made]
    units = [1 if "$" in w else len(w) for w in whole]

    def reveal(u, by="word", level=1.0, ease=ease):
        u = _check_u(u)
        level = _check_u(level)
        if by not in ("word", "letter"):
            raise ValueError(f"unknown by: {by!r}; known: ['word', 'letter']")
        if by == "word":
            for word, text in zip(made, whole, strict=True):
                word.set_text(text)
            _write_on(made, fades, u, ease)
        else:
            shown = math.floor(sum(units) * u + 1e-9)
            for word, fade, text, count in zip(made, fades, whole, units, strict=True):
                k = min(count, max(0, shown))
                shown -= k
                word.set_text(text[:k] if 0 < k < count else text)
                word.set_visible(k > 0)
                fade.set(1.0)
        if level < 1.0:
            for word in made:
                base = word.get_alpha()
                word.set_alpha(level * (1.0 if base is None else base))

    def update(u, level=1.0):
        reveal(u, by, level, ease)

    return _WordsResult(
        ax=at.axes,
        artists={"text": made},
        update=update,
        reveal=reveal,
        # reveal() can choose letters at call time, changing the visible box.
        contract=MotionContract(moves=tuple(made)),
    )


# Link


def _inside(p, q, box):
    """The part of the segment p-q inside `box`, as ``(a, b)``, or None."""
    d = q - p
    lo, hi = 0.0, 1.0
    for k, (a, b) in enumerate(((box.x0, box.x1), (box.y0, box.y1))):
        if d[k] == 0.0:
            if not a <= p[k] <= b:
                return None
            continue
        t0, t1 = sorted(((a - p[k]) / d[k], (b - p[k]) / d[k]))
        lo, hi = max(lo, t0), min(hi, t1)
        if lo >= hi:
            return None
    return p + lo * d, p + hi * d


class _Link(Line2D):
    """A connector from a mark in one panel to a mark in another, grown from the first.

    It is audited where it touches data: the part of it inside each panel it
    passes through. `faults` lists the rules it breaks as the figure is now.
    """

    def __init__(self, source, target, *, avoid, strict, **kw):
        super().__init__([], [], transform=IdentityTransform(), **kw)
        self._source = source
        self._target = target
        self._avoid = avoid
        self._strict = bool(strict)
        self._grow = 1.0

    def _ends(self, renderer=None):
        """From the source's mark to the target's, each on its mark's box edge."""
        p = self._source.display(renderer)
        q = self._target.display(renderer)
        if np.hypot(*(q - p)) <= 0.0:
            return None
        start, end = p, q
        for anchor, here, there in ((self._source, p, q), (self._target, q, p)):
            if anchor.artist is None:
                continue
            box = anchor.box(renderer)
            if box.x0 < here[0] < box.x1 and box.y0 < here[1] < box.y1:
                edge = _exit(box, here, there)
                if edge is not None:
                    if anchor is self._source:
                        start = edge
                    else:
                        end = edge
        return start, end

    def faults(self, renderer=None):
        """The rules the fully grown link breaks as the figure is now.

        A link joins two panels, so the quarter-panel length rule of a
        `leader` does not apply. What matters is where it touches data: the
        part of the link inside each panel it passes through (its two ends,
        and any panel between) crosses no line, patch, or collection of that
        panel, and the link is not drawn in a color the data of those panels
        use. A marker the link ends on is the mark it names, reference lines
        drawn against the axes frame are scenery, and images are not
        audited, since any line over an image crosses pixels. It also
        crosses no other text.

        Args:
            renderer: The renderer drawing the figure, or None outside a
                draw.

        Returns:
            A list of strings, one per broken rule; empty when the link keeps
            every rule or is not drawn at all.
        """
        ends = self._ends(renderer)
        if ends is None:
            return []
        start, end = ends
        figure = self._source.figure
        out = []
        own = np.array(to_rgb(self.get_color()))
        exclude = [a for a in (self._source.artist, self._target.artist) if a]
        for panel in figure.axes:
            part = _inside(start, end, panel.bbox)
            if part is None:
                continue
            a, b = part
            length = float(np.hypot(*(b - a)))
            if length > 2.0 * _END_CLEAR_PX:
                unit = (b - a) / length
                a, b = a + _END_CLEAR_PX * unit, b - _END_CLEAR_PX * unit
            if self._avoid is not None:
                pool = [x for x in self._avoid if x.axes is panel and x.get_visible()]
            else:
                pool = [x for x in _default_avoid(panel, exclude) if x.get_visible()]
            for artist in pool:
                name = artist.get_label() or type(artist).__name__
                ends_on = _ends_on(artist, start, renderer) or _ends_on(
                    artist, end, renderer
                )
                if not ends_on and _crosses(artist, a, b, renderer):
                    out.append(f"link crosses data: {name}")
                for rgba in _data_colors(artist):
                    if rgba[3] > 0.0 and np.allclose(own, rgba[:3], atol=2e-3):
                        out.append(f"link takes a data color of its panel: {name}")
                        break
        pool = list(figure.texts) + [x for x in figure.artists if isinstance(x, Text)]
        for panel in figure.axes:
            pool += list(panel.texts)
            pool += [panel.title, panel._left_title, panel._right_title]
            pool += [panel.xaxis.label, panel.yaxis.label]
        for other in dict.fromkeys(pool):
            if not (other.get_visible() and other.get_text()):
                continue
            box = other.get_window_extent(renderer)
            edges = np.array(
                [[box.x0, box.y0], [box.x1, box.y0], [box.x1, box.y1], [box.x0, box.y1]]
            )
            if _polyline_crosses(np.vstack([edges, edges[:1]]), start, end):
                out.append(f"link crosses text: {other.get_text()!r}")
        return out

    def draw(self, renderer):
        """Grow from the source toward the target; drawing never raises."""
        if not self.get_visible() or self._grow <= 0.0:
            return
        ends = self._ends(renderer)
        if ends is None:
            return
        start, end = ends
        tip = start + self._grow * (end - start)
        self.set_data([start[0], tip[0]], [start[1], tip[1]])
        super().draw(renderer)


def link(
    source,
    target,
    *,
    fig=None,
    color=None,
    lw=None,
    avoid=None,
    strict=True,
    ease=DEFAULT_EASE,
    zorder=15,
    line_kw=None,
):
    """Draw a line that grows from a mark in one panel to a mark in another.

    The move shows that two views hold the same object: from "this pixel" of
    an image, a line grows to the origin of the phasor panel that explains
    it. A `leader` joins a caption to a mark in its own panel and keeps a
    quarter-panel length rule that a line between panels breaks by
    construction; a link is for the case where the line is the point. It is
    a figure-level line in a neutral tone, resolved in display space when
    the figure is drawn, so it follows layout and limits and never joins an
    axes' autoscaling.

    What a link is audited for is where it touches data (see
    ``faults()``): the part of it inside each panel it passes through must
    cross no line, patch, or collection, and the link takes no data color.
    The part between panels is empty by construction. Drawing never
    raises; `audit` raises `AttentionFault` for a strict link that breaks a
    rule.

    ``update(u, level=1.0)`` grows the line from the source: ``u = 0``
    draws nothing and ``u = 1`` (the rest state, when the link is made)
    reaches the target; ``level`` fades the whole line (1 as drawn, 0 gone),
    so a link can recede with the caption it serves.

    The audit reads the figure when ``faults()`` runs, not when the link is
    made: every text, panel title and axis label present then counts, so
    audit the composite after everything is drawn.

    Args:
        source: Where the link starts: an `Anchor`, or an artist. An
            anchored artist's box is left at its edge, toward the target.
        target: Where it ends, the same way.
        fig: The figure that owns the line. None uses the source's figure.
        color: Line color. None uses a neutral two thirds of the way from
            the background to the text color, read now.
        lw: Line width in points. None uses 0.6 of
            ``rcParams["lines.linewidth"]``.
        avoid: The artists the link must not cross. None audits every line,
            patch, and collection of every panel the link passes through,
            except the anchored artists, spines, and the axes backgrounds.
        strict: Whether `audit` raises for the link when it breaks a rule;
            False leaves the faults for the caller to read.
        ease: An easing kind for `eyepiece.ease`, or a callable from progress to
            eased progress such as ``functools.partial(ep.ease, kind="back_out",
            strength=0.2)``, applied to `u`.
        zorder: Drawing order among the figure's children.
        line_kw: Extra kwargs for the `Line2D`, applied last.

    Returns:
        A `MosaicResult` over the axes of the two anchors (source first),
        with ``artists["line"]``, the link, whose ``faults()`` lists the
        rules it breaks as the figure is now, and ``update(u, level)`` as
        above; values outside [0, 1] are clipped.

    Raises:
        TypeError: If `source` or `target` is neither an `Anchor` nor an
            artist.

    Example::

        res = ep.link(ep.Anchor.from_data(image_panel, (12, 30)),
                      ep.Anchor.from_data(phasor_panel, (0.0, 0.0)))
        res.update(0.5)  # halfway there
        ep.audit(fig)
    """
    source = _as_anchor(source, "source")
    target = _as_anchor(target, "target")
    rc = matplotlib.rcParams
    kw = {
        "color": _style.neutral(0.6) if color is None else color,
        "lw": 0.6 * float(rc["lines.linewidth"]) if lw is None else lw,
        "solid_capstyle": "butt",
        "zorder": zorder,
        **(line_kw or {}),
    }
    line = _Link(
        source,
        target,
        avoid=None if avoid is None else list(avoid),
        strict=strict,
        **kw,
    )
    line.set_clip_on(False)
    line.set_in_layout(False)
    owner = source.figure if fig is None else fig
    owner.add_artist(line)

    alpha = line.get_alpha()

    def update(u, level=1.0):
        u = _check_u(u)
        level = _check_u(level)
        line.set_visible(u > 0.0 and level > 0.0)
        line._grow = min(1.0, max(0.0, _ease(u, ease))) if u < 1.0 else 1.0
        base = 1.0 if alpha is None else alpha
        line.set_alpha(alpha if level >= 1.0 else base * level)

    update(1.0)
    axes = [a for a in (source.axes, target.axes) if a is not None]
    axes = list(dict.fromkeys(axes)) or list(owner.axes)
    return MosaicResult(
        axes=np.array(axes, dtype=object), artists={"line": line}, update=update
    )
