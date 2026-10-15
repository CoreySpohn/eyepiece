"""Updatable arrows: a vector, an angle arc, and a compass.

An explanation points with arrows, and a build-up wants them to grow, move,
and turn rather than be redrawn. `arrow` is one `FancyArrowPatch` whose ends
and growth are set by `update`; `angle_arc` is one patch along a circular
arc, with its head tangent to the arc, that sweeps; `compass` is a north
arrow and an east arrow built on `arrow` that turn together.

Each returns a `PlotResult` whose `update` sets the state absolutely and
reuses the same artists, so the same arguments always draw the same picture
and an animation frame draws itself from its own state. The rest state, the
one a still shows, is the figure as drawn: ``u = 1``.
"""

import math

import matplotlib
import numpy as np
from matplotlib.path import Path

from eyepiece import _style
from eyepiece._phasor import HEAD_LENGTH, HEAD_PER_MARKER, HEAD_WIDTH, _Arrow
from eyepiece._result import PlotResult
from eyepiece.attention import _check_u, _TextFade

# Arrow kinds: the matplotlib arrow style and the number of heads.
_KINDS = {
    "vector": ("-|>", 1),
    "double": ("<|-|>", 2),
    "dimension": ("<->", 2),
}

# The dimension kind draws a thinner line and a smaller head than a vector.
_DIMENSION_HEAD = 0.75
_DIMENSION_WIDTH = 0.75

# Points a label stands off a vector or a double arrow.
_LABEL_PAD_PT = 6.0

# Segments in the polyline that stands for an arc when measuring its length.
_ARC_SAMPLES = 64


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite (x, y) pair, got {value!r}")
    return point


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value


def _label_text(ax, text, kw, zorder, boxed):
    """An annotation used as a text: it never clips at the axes' edge."""
    options = {
        "ha": "center",
        "va": "center",
        "color": matplotlib.rcParams["text.color"],
        "zorder": zorder,
        **(kw or {}),
    }
    if boxed and "bbox" not in options:
        options["bbox"] = _style.backing()
    return ax.annotate(
        text,
        xy=(0.0, 0.0),
        xytext=(0.0, 0.0),
        textcoords="offset points",
        annotation_clip=False,
        **options,
    )


def _display_unit(ax, start, end):
    """Unit vector from `start` to `end` in display space; zero if they meet."""
    a, b = ax.transData.transform(np.array([start, end], dtype=float))
    step = b - a
    size = float(np.hypot(*step))
    return step / size if size > 0.0 else np.zeros(2)


def _set_level(patch, base_alpha, level):
    if level >= 1.0:
        patch.set_alpha(base_alpha)
    else:
        patch.set_alpha(level * (1.0 if base_alpha is None else base_alpha))


def arrow(
    tail,
    head,
    *,
    ax,
    kind="vector",
    label=None,
    color=None,
    lw=None,
    head_size=1.0,
    label_offset=None,
    label_kw=None,
    arrow_kw=None,
    zorder=6,
):
    """Draw an arrow that can grow and move without being redrawn.

    One `FancyArrowPatch` carries the shaft and the head or heads. `update`
    sets its ends and its growth, so a build-up can lengthen a vector, slide
    it along an axis, or open a dimension line across a gap, all on the same
    artist. The head is a fixed size in points, shrunk to fit when the arrow
    is shorter than it, and an arrow too short to see draws nothing.

    The kinds are ``"vector"``, a shaft with a head at `head`;
    ``"double"``, a head at each end; and ``"dimension"``, a thinner line
    with a small open head at each end for measuring a gap, whose label sits
    on the line in a backing box. The arrow grows from `tail` for a vector
    and outward from the middle for the two-headed kinds, so a dimension
    opens from its center.

    Args:
        tail: Start of the arrow, ``(x, y)`` in the data coordinates of `ax`.
        head: End of the arrow, in the same coordinates.
        ax: The axes whose data coordinates the ends are in. The arrow is
            added without joining the axes' autoscaling.
        kind: ``"vector"``, ``"double"`` or ``"dimension"``.
        label: Text at the middle of the arrow. A vector or a double arrow
            keeps it `label_offset` points to the left of the direction from
            tail to head; a dimension puts it on the line.
        color: Arrow and label color. None uses ``rcParams["text.color"]``.
        lw: Line width in points. None uses ``rcParams["lines.linewidth"]``
            (three quarters of it for a dimension).
        head_size: Head size as a multiple of the default, which grows with
            ``rcParams["lines.markersize"]``.
        label_offset: Points between the middle and the label, positive to
            the left of the direction of travel. None is 6 for a vector or a
            double arrow and 0 for a dimension.
        label_kw: Extra `Annotation` keywords for the label (for example
            ``fontsize``, ``bbox``); a ``bbox`` of None draws no box.
        arrow_kw: Extra `FancyArrowPatch` keywords, applied last.
        zorder: Drawing order of the arrow; its label is just above.

    Returns:
        A `PlotResult` on `ax` with ``artists["arrow"]`` the patch and, with a
        label, ``artists["text"]`` the label, and
        ``update(u=1.0, *, tail=None, head=None, level=1.0)``. `u`, from 0 to
        1 (values outside are clipped), is how much of the arrow is drawn;
        `tail` and `head` move the full-length ends and are kept for later
        calls, as `u` and `level` are not; `level` is the opacity as a
        fraction of the drawn one, so a `Stage` can dim it. The label is
        absent at ``u = 0``. The rest state is ``u = 1``.

    Raises:
        ValueError: If `kind` is unknown, or an end or `head_size` is not
            finite.

    Example::

        gap = ep.arrow((0.0, 0.0), (2.0, 0.0), ax=ax, kind="dimension", label="d")
        for u in np.linspace(0.0, 1.0, 20):
            gap.update(u)
            rec.frame()
        gap.update(tail=(0.0, 0.0), head=(3.0, 0.0))  # the gap widens
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    style, heads = _KINDS[kind]
    ends = [_point(tail, "tail"), _point(head, "head")]
    size = _finite(head_size, "head_size")
    thin = kind == "dimension"
    width = (
        float(matplotlib.rcParams["lines.linewidth"])
        * (_DIMENSION_WIDTH if thin else 1)
        if lw is None
        else float(lw)
    )
    mutation = (
        HEAD_PER_MARKER
        * float(matplotlib.rcParams["lines.markersize"])
        * size
        * (_DIMENSION_HEAD if thin else 1.0)
    )
    kw = {
        "arrowstyle": f"{style},head_length={HEAD_LENGTH},head_width={HEAD_WIDTH}",
        "mutation_scale": mutation,
        "color": matplotlib.rcParams["text.color"] if color is None else color,
        "linewidth": width,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
        "zorder": zorder,
        "clip_on": False,
        **(arrow_kw or {}),
    }
    patch = _Arrow(tuple(ends[0]), tuple(ends[1]), **kw)
    patch.heads = heads
    ax.add_artist(patch)
    offset = (0.0 if thin else _LABEL_PAD_PT) if label_offset is None else label_offset
    offset = _finite(offset, "label_offset")
    note = None
    if label is not None:
        note = _label_text(
            ax,
            label,
            {"color": kw["color"], **(label_kw or {})},
            zorder + 0.1,
            boxed=thin,
        )
    base_alpha = patch.get_alpha()
    fade = None if note is None else _TextFade(note)

    def update(u=1.0, *, tail=None, head=None, level=1.0):
        u = _check_u(u)
        level = _check_u(level)
        if tail is not None:
            ends[0] = _point(tail, "tail")
        if head is not None:
            ends[1] = _point(head, "head")
        a, b = ends
        if heads == 1:
            start, stop = a, a + u * (b - a)
        else:
            mid = 0.5 * (a + b)
            start, stop = mid + u * (a - mid), mid + u * (b - mid)
        patch.set_positions(tuple(start), tuple(stop))
        _set_level(patch, base_alpha, level)
        if note is not None:
            mid = 0.5 * (a + b)
            direction = _display_unit(ax, a, b)
            normal = np.array([-direction[1], direction[0]])
            note.xy = tuple(mid)
            note.xyann = tuple(offset * normal)
            note.set_visible(u > 0.0 and level > 0.0)
            fade.set(level)

    update(1.0)
    artists = {"arrow": patch}
    if note is not None:
        artists["text"] = note
    return PlotResult(ax=ax, artists=artists, update=update)


class _ArcArrow(_Arrow):
    """An arrow along a circular arc, its head capped by the arc's length.

    The path is set from the arc's geometry, so the head is tangent to the arc
    at the end it sits on, and the cap measures the arc rather than the chord.
    """

    def __init__(self, **kwargs):
        super().__init__(None, None, path=Path([(0.0, 0.0)] * 2), **kwargs)
        self._arc = (np.zeros(2), 1.0, 0.0, 0.0)

    def set_arc(self, center, radius, start, sweep):
        """Lay the arc out: angles in degrees, `sweep` signed (positive is ccw)."""
        self._arc = (np.asarray(center, dtype=float), float(radius), start, sweep)
        if sweep == 0.0:
            path = None
        elif sweep > 0.0:
            path = Path.arc(start, start + sweep)
        else:
            # An arc's codes are one move and cubic pieces, so reversing the
            # vertices reverses the arc.
            arc = Path.arc(start + sweep, start)
            path = Path(arc.vertices[::-1], arc.codes)
        ends = [
            center
            + radius * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            for a in (start, start + sweep)
        ]
        if path is None:
            path = Path([ends[0], ends[0]])
        else:
            path = Path(path.vertices * radius + center, path.codes)
        self._path_original = path
        self._ends = (tuple(ends[0]), tuple(ends[1]))

    def _length_pt(self):
        if self.figure is None or self.axes is None:
            return None
        center, radius, _, sweep = self._arc
        a, b = self.get_transform().transform(
            np.array([center, (center[0] + radius, center[1])], dtype=float)
        )
        return (
            float(np.hypot(*(b - a)))
            * abs(math.radians(sweep))
            * 72.0
            / self.figure.dpi
        )


def angle_arc(
    center,
    start_angle,
    end_angle,
    radius,
    *,
    ax,
    head="end",
    label=None,
    label_scale=1.0,
    color=None,
    lw=None,
    head_size=1.0,
    label_kw=None,
    arrow_kw=None,
    zorder=6,
):
    r"""Draw an arc between two directions that can sweep without being redrawn.

    The arc about `center` runs from `start_angle` to `end_angle`, one patch
    along a circular path, so an arrowhead at an end is tangent to the arc
    there. `update` sets how much of the sweep is drawn and can move the end
    angle, so a turn can be shown opening from the reference ray to the
    measured one, which is what fixes a sign convention for the reader.

    Angles are in degrees, measured counterclockwise from +x in data
    coordinates, and the arc is a true circle only when the axes have equal
    aspect. A sweep that is negative (the end below the start) is clockwise;
    its size is not wrapped, so ``0 -> 350`` is a long way round and
    ``0 -> -10`` a short one the other way. A sweep may not exceed 360.

    Args:
        center: Vertex of the angle, ``(x, y)`` in data coordinates of `ax`.
        start_angle: Direction of the reference ray, degrees.
        end_angle: Direction of the measured ray, degrees.
        radius: Radius of the arc in data units.
        ax: The axes the arc is drawn on; it does not join autoscaling.
        head: ``"end"`` for an arrowhead where the sweep ends, ``"start"``,
            ``"both"`` or ``"none"``.
        label: Text at the middle of the drawn arc, on a backing box.
        label_scale: Distance of the label from `center` as a multiple of
            `radius`. At 1 it sits on the arc; a larger value moves it out.
        color: Arc and label color. None uses ``rcParams["text.color"]``.
        lw: Line width in points. None uses ``rcParams["lines.linewidth"]``.
        head_size: Head size as a multiple of the default.
        label_kw: Extra `Annotation` keywords for the label.
        arrow_kw: Extra `FancyArrowPatch` keywords, applied last.
        zorder: Drawing order of the arc; its label is just above.

    Returns:
        A `PlotResult` on `ax` with ``artists["arrow"]`` the patch and, with a
        label, ``artists["text"]``, and ``update(u=1.0, *, end_angle=None,
        start_angle=None, radius=None, center=None, level=1.0)``. `u`, from 0
        to 1 (values outside are clipped), is the fraction of the sweep
        drawn, from the start; the other arguments move the full arc and are
        kept for later calls, as `u` and `level` are not. `level` is the
        opacity as a fraction of the drawn one. The arc and its label are
        absent at ``u = 0``. The rest state is ``u = 1``.

    Raises:
        ValueError: If `head` is unknown, an argument is not finite, `radius`
            is not positive, or the sweep is larger than 360 degrees.

    Example::

        turn = ep.angle_arc((0, 0), 0.0, 40.0, 1.0, ax=ax, label=r"$\theta$")
        for u in ep.ease("smoothstep", np.linspace(0.0, 1.0, 20)):
            turn.update(u)
            rec.frame()
        turn.update(end_angle=70.0)  # the angle opens further
    """
    heads = {
        "end": ("-|>", 1),
        "start": ("<|-", 1),
        "both": ("<|-|>", 2),
        "none": ("-", 1),
    }
    if head not in heads:
        raise ValueError(f"head must be one of {sorted(heads)}, got {head!r}")
    style, count = heads[head]
    state = {
        "center": _point(center, "center"),
        "start": _finite(start_angle, "start_angle"),
        "end": _finite(end_angle, "end_angle"),
        "radius": _finite(radius, "radius"),
    }
    size = _finite(head_size, "head_size")
    mutation = HEAD_PER_MARKER * float(matplotlib.rcParams["lines.markersize"]) * size
    kw = {
        "arrowstyle": f"{style},head_length={HEAD_LENGTH},head_width={HEAD_WIDTH}"
        if head != "none"
        else "-",
        "mutation_scale": mutation,
        "color": matplotlib.rcParams["text.color"] if color is None else color,
        "linewidth": float(matplotlib.rcParams["lines.linewidth"])
        if lw is None
        else float(lw),
        "shrinkA": 0.0,
        "shrinkB": 0.0,
        "zorder": zorder,
        "clip_on": False,
        **(arrow_kw or {}),
    }
    patch = _ArcArrow(**kw)
    patch.heads = count
    ax.add_artist(patch)
    note = None
    if label is not None:
        note = _label_text(
            ax, label, {"color": kw["color"], **(label_kw or {})}, zorder + 0.1, True
        )
    scale = _finite(label_scale, "label_scale")
    base_alpha = patch.get_alpha()
    fade = None if note is None else _TextFade(note)

    def update(
        u=1.0, *, end_angle=None, start_angle=None, radius=None, center=None, level=1.0
    ):
        u = _check_u(u)
        level = _check_u(level)
        if end_angle is not None:
            state["end"] = _finite(end_angle, "end_angle")
        if start_angle is not None:
            state["start"] = _finite(start_angle, "start_angle")
        if radius is not None:
            state["radius"] = _finite(radius, "radius")
        if center is not None:
            state["center"] = _point(center, "center")
        if state["radius"] <= 0.0:
            raise ValueError(f"radius must be positive, got {state['radius']}")
        sweep = state["end"] - state["start"]
        if abs(sweep) > 360.0:
            raise ValueError(f"the sweep may not exceed 360 degrees, got {sweep}")
        drawn = u * sweep
        patch.set_arc(state["center"], state["radius"], state["start"], drawn)
        _set_level(patch, base_alpha, level)
        if note is not None:
            mid = math.radians(state["start"] + 0.5 * drawn)
            where = state["center"] + scale * state["radius"] * np.array(
                [math.cos(mid), math.sin(mid)]
            )
            note.xy = tuple(where)
            note.set_visible(u > 0.0 and level > 0.0)
            fade.set(level)

    update(1.0)
    artists = {"arrow": patch}
    if note is not None:
        artists["text"] = note
    return PlotResult(ax=ax, artists=artists, update=update)


def compass(
    center,
    length,
    *,
    ax,
    east,
    angle=0.0,
    labels=("N", "E"),
    label_pad=4.0,
    color=None,
    lw=None,
    head_size=0.8,
    label_kw=None,
    arrow_kw=None,
    zorder=6,
):
    """Draw north and east arrows from one point, turning together.

    Two arrows built on `arrow`, from `center`, each with a letter just
    beyond its tip. `update(angle=...)` turns the pair, as a roll turns a
    detector against the sky, and `u` grows them from the center.

    Which side east is on is the whole content of a compass, and a mirror of
    the sky looks exactly as plausible as the sky, so `east` has no default.
    ``east="left"`` is the sky as an observer looking up sees it: with north
    up, east is to the left, so east is a quarter turn counterclockwise from
    north. ``east="right"`` is the sky as a map or a detector chart draws it
    from outside, with east a quarter turn clockwise from north. A figure
    that takes its positions from a frame whose first projected axis is east
    with the line of sight toward the observer is the second kind, and a
    caller who draws it as the first has mirrored the sky.

    Args:
        center: Where the arrows start, ``(x, y)`` in data coordinates of
            `ax`. The axes should have equal aspect, so a quarter turn
            looks like one.
        length: Length of each arrow in data units.
        ax: The axes the compass is drawn on.
        east: ``"left"`` or ``"right"``: the side of north on which east
            lies. Required.
        angle: Direction of north in degrees counterclockwise from +y
            (up). 0 puts north up.
        labels: The letters at the north and east tips, or None for none.
        label_pad: Points between a tip and its letter.
        color: Color of arrows and letters. None uses ``rcParams["text.color"]``.
        lw: Line width in points. None uses ``rcParams["lines.linewidth"]``.
        head_size: Head size as a multiple of the default.
        label_kw: Extra `Annotation` keywords for the letters.
        arrow_kw: Extra `FancyArrowPatch` keywords for both arrows.
        zorder: Drawing order of the arrows; the letters are just above.

    Returns:
        A `PlotResult` on `ax` with ``artists["arrow"]`` the north and east
        patches in that order, ``artists["text"]`` the letters in the same
        order (with `labels`), and ``update(u=1.0, *, angle=None,
        center=None, length=None, level=1.0)``. `u`, from 0 to 1, is the
        fraction of each arrow drawn; `angle`, `center` and `length` move the
        full compass and are kept for later calls. `level` is the opacity as
        a fraction of the drawn one. The rest state is ``u = 1``.

    Raises:
        ValueError: If `east` is not ``"left"`` or ``"right"``, `length` is not
            positive, or `labels` is not a pair.
    """
    if east not in ("left", "right"):
        raise ValueError(f"east must be 'left' or 'right', got {east!r}")
    turn = 1.0 if east == "left" else -1.0
    state = {
        "center": _point(center, "center"),
        "length": _finite(length, "length"),
        "angle": _finite(angle, "angle"),
    }
    if state["length"] <= 0.0:
        raise ValueError(f"length must be positive, got {state['length']}")
    if labels is not None and len(labels) != 2:
        raise ValueError(f"labels must be a (north, east) pair, got {labels!r}")
    pad = _finite(label_pad, "label_pad")

    def tips():
        a = math.radians(state["angle"])
        north = np.array([-math.sin(a), math.cos(a)])
        east_vec = turn * np.array([-north[1], north[0]])
        tail = state["center"]
        return tail, (tail + state["length"] * north, tail + state["length"] * east_vec)

    tail, heads = tips()
    parts = []
    for tip in heads:
        part = arrow(
            tuple(tail),
            tuple(tip),
            ax=ax,
            lw=lw,
            color=color,
            head_size=head_size,
            arrow_kw=arrow_kw,
            zorder=zorder,
        )
        parts.append(part)
    notes = []
    ink = matplotlib.rcParams["text.color"] if color is None else color
    if labels is not None:
        for text in labels:
            notes.append(
                _label_text(
                    ax,
                    text,
                    {"color": ink, **(label_kw or {})},
                    zorder + 0.1,
                    boxed=False,
                )
            )
    fades = [_TextFade(note) for note in notes]

    def update(u=1.0, *, angle=None, center=None, length=None, level=1.0):
        u = _check_u(u)
        level = _check_u(level)
        if angle is not None:
            state["angle"] = _finite(angle, "angle")
        if center is not None:
            state["center"] = _point(center, "center")
        if length is not None:
            state["length"] = _finite(length, "length")
            if state["length"] <= 0.0:
                raise ValueError(f"length must be positive, got {state['length']}")
        tail, heads = tips()
        for part, tip, note, fade in zip(
            parts, heads, notes or [None] * 2, fades or [None] * 2, strict=True
        ):
            part.update(u, tail=tuple(tail), head=tuple(tip), level=level)
            if note is not None:
                away = _display_unit(ax, tail, tip)
                note.xy = tuple(tail + u * (tip - tail))
                note.xyann = tuple(pad * away)
                note.set_visible(u > 0.0 and level > 0.0)
                fade.set(level * u)

    update(1.0)
    artists = {"arrow": [p.artists["arrow"] for p in parts]}
    if notes:
        artists["text"] = notes
    return PlotResult(ax=ax, artists=artists, update=update)
