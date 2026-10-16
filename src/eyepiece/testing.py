"""Readability checks for a figure, or for a clip sampled at chosen times.

A figure can draw without error and still be unreadable: two labels that
overlap by a few pixels, a caption clipped by the frame edge, text too small
to read at the output size, a label written over a line of its own color.
Nothing in matplotlib raises for these, and a per-clip test that looks for
one of them goes stale as soon as the layout moves. This module is the shared
check, so a clip's test calls one function in place of a copied one.

`readability_faults` poses the figure at each sample time, draws it, and
returns a list of `ReadabilityFault` records, each naming the artist, the
time and the box in output pixels. `assert_readable` is the same check as a
test assertion that raises with a readable report.

Four kinds of fault are reported:

- ``"overlap"``: two visible text artists, or a text and the backing box of
  another, whose rendered extents intersect by more than `overlap_tol`
  pixels in both directions.
- ``"outside"``: visible text, or a mark, that the figure edge clips by more
  than `frame_tol` pixels.
- ``"small"``: text whose font size, in pixels at the output size, is under
  the minimum for the venue.
- ``"on-marks"``: text drawn over a data mark of the same color, so the
  glyphs cannot be told from the mark. A label with an opaque backing box is
  readable and is not reported. The leaders and links that `audit` already
  checks are not looked at here: `audit` covers a connector crossing data,
  and this check covers the words.

`motion_faults` is the same idea for the path between two states. A clip is
checked at u = 0 and u = 1 and at stills, and the defects that matter lie in
between: a final frame that jumps, a glyph that drifts, a word that tears
apart mid-move, two moving labels that cross, a label that leaves the frame.
It samples the clip at every frame, traces the window extent of every visible
text, patch, collection and line (glyphs drawn as paths included), keeps the
rendered frames, and reports seven kinds of fault: ``"jump"``, ``"dead"``,
``"drift"``, ``"tear"``, ``"overlap"``, ``"outside"`` and ``"spike"``.
`assert_motion` raises `MotionError` naming every fault.

`speech_rate` is the check on the voice rather than the picture: it reads the
word times of a `eyepiece.narration.Narration` and reports each line's words
per minute and the lines faster than a ceiling. `assert_speech_rate` raises
on them.

A clip is described by the figure and a callable `at` that poses it at a
time; the times to sample (beat ends, turn midpoints, cue times) are the
caller's.

Example::

    import eyepiece.testing as ept

    faults = ept.readability_faults(
        fig, at=lambda t: clip.pose(t), times=[0.5, 1.5, 2.0], venue="video"
    )
    ept.assert_readable(fig, at=clip.pose, times=cue_times)
"""

import itertools
import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from matplotlib.axis import Axis
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from eyepiece.attention import _data_colors, _default_avoid, _renderer

__all__ = [
    "KINDS",
    "MOTION_KINDS",
    "VENUES",
    "VENUE_FPS",
    "MotionError",
    "MotionFault",
    "ReadabilityError",
    "ReadabilityFault",
    "Span",
    "SpeechRate",
    "SpeechRateError",
    "assert_motion",
    "assert_readable",
    "assert_speech_rate",
    "caption_line",
    "motion_faults",
    "readability_faults",
    "speech_rate",
]

KINDS = ("overlap", "outside", "small", "on-marks")

# Smallest readable font size, in pixels at the output size. A 1080p video is
# watched from a distance, a phone screen is small, and a printed page at
# 300 dpi reads down to about 6 pt.
VENUES = {"video": 18.0, "phone": 24.0, "print": 25.0}

_SHOWN = 0.05  # opacity below which an artist counts as not drawn
_SAME_COLOR = 0.08  # largest per-channel RGB difference that counts as one color
_OPAQUE_BACKING = 0.8  # backing-box opacity that keeps a label readable


@dataclass(frozen=True, eq=False)
class ReadabilityFault:
    """One readability problem found at one sample time.

    Attributes:
        kind: One of `KINDS`.
        artist: The artist at fault: the text, or for ``"outside"`` the text
            or mark.
        other: The second artist of an ``"overlap"`` or ``"on-marks"`` fault;
            None for the others.
        time: The sample time, None when the figure was checked as drawn.
        box: ``(x0, y0, x1, y1)`` of the fault in output pixels, origin at
            the lower left of the figure: the overlap region, the clipped
            artist's extent, or the text extent.
        detail: A one-line description.
    """

    kind: str
    artist: object
    other: object
    time: object
    box: tuple
    detail: str

    def __str__(self):
        """One line: the time, the kind, the detail and the box."""
        when = "as drawn" if self.time is None else f"t={self.time:g}"
        x0, y0, x1, y1 = self.box
        return (
            f"[{when}] {self.kind}: {self.detail} "
            f"(box {x0:.0f},{y0:.0f} to {x1:.0f},{y1:.0f} px)"
        )


class ReadabilityError(AssertionError):
    """A figure or clip has readability faults.

    Attributes:
        faults: The `ReadabilityFault` list.
    """

    def __init__(self, faults):
        """Build the report from the faults."""
        self.faults = list(faults)
        lines = [f"{len(self.faults)} readability fault(s):"]
        lines += [f"  {fault}" for fault in self.faults]
        super().__init__("\n".join(lines))


def _name(artist):
    if isinstance(artist, Text):
        return f"text {artist.get_text()!r}"
    return type(artist).__name__


def _alpha(artist, color):
    alpha = artist.get_alpha()
    return (1.0 if alpha is None else float(alpha)) * to_rgba(color)[3]


def _walk(artist):
    """Visible artists below `artist`, in drawing order, tick labels in view only."""
    yield artist
    if isinstance(artist, Axis):
        if not artist.axes.axison:
            return
        low, high = sorted(artist.get_view_interval())
        children = [artist.label, artist.offsetText]
        for tick in artist.get_major_ticks() + artist.get_minor_ticks():
            if low <= tick.get_loc() <= high:
                children += [tick.label1, tick.label2]
    else:
        children = artist.get_children()
    for child in children:
        if child.get_visible():
            yield from _walk(child)


def _visible_texts(fig):
    out = []
    for artist in _walk(fig):
        if (
            isinstance(artist, Text)
            and artist.get_text().strip()
            and _alpha(artist, artist.get_color()) >= _SHOWN
        ):
            out.append(artist)
    return out


def _bbox(artist, renderer):
    box = artist.get_window_extent(renderer)
    if isinstance(artist, Text):
        patch = artist.get_bbox_patch()
        if patch is not None and patch.get_visible():
            box = Bbox.union([box, patch.get_window_extent(renderer)])
    return box


def _tuple(box):
    return (float(box.x0), float(box.y0), float(box.x1), float(box.y1))


def _finite(box):
    return np.all(np.isfinite([box.x0, box.y0, box.x1, box.y1]))


def _overlaps(texts, boxes, tol):
    found = []
    for i, first in enumerate(texts):
        for j in range(i + 1, len(texts)):
            a, b = boxes[i], boxes[j]
            width = min(a.x1, b.x1) - max(a.x0, b.x0)
            height = min(a.y1, b.y1) - max(a.y0, b.y0)
            if width > tol and height > tol:
                x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
                box = (x0, y0, x0 + width, y0 + height)
                detail = (
                    f"{_name(first)} and {_name(texts[j])} overlap by "
                    f"{width:.1f} x {height:.1f} px"
                )
                found.append(("overlap", first, texts[j], box, detail))
    return found


def _edge_overshoot(box, width, height):
    """Pixels past each figure edge, as {side: pixels}."""
    over = {
        "left": -box.x0,
        "right": box.x1 - width,
        "bottom": -box.y0,
        "top": box.y1 - height,
    }
    return over


def _marks(fig):
    """Axes marks that could be clipped, with their visible extent."""
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for child in ax.get_children():
            if child is ax.patch or isinstance(child, Spine) or not child.get_visible():
                continue
            if isinstance(child, (Line2D, Patch, Collection, AxesImage)):
                yield child


def _outside(fig, texts, boxes, renderer, tol):
    width, height = fig.bbox.width, fig.bbox.height
    found = []
    for text, box in zip(texts, boxes, strict=True):
        over = {k: v for k, v in _edge_overshoot(box, width, height).items() if v > tol}
        if over:
            sides = ", ".join(f"{k} by {v:.1f} px" for k, v in over.items())
            found.append(
                (
                    "outside",
                    text,
                    None,
                    _tuple(box),
                    f"{_name(text)} is clipped: {sides}",
                )
            )
    for mark in _marks(fig):
        box = mark.get_window_extent(renderer)
        if not _finite(box):
            continue
        clip = mark.get_clip_box() if mark.get_clip_on() else None
        if clip is not None:
            box = Bbox.intersection(box, clip)
            if box is None:
                continue
        over = {k: v for k, v in _edge_overshoot(box, width, height).items() if v > tol}
        if over:
            sides = ", ".join(f"{k} by {v:.1f} px" for k, v in over.items())
            found.append(
                (
                    "outside",
                    mark,
                    None,
                    _tuple(box),
                    f"{_name(mark)} is clipped: {sides}",
                )
            )
    return found


def _small(texts, boxes, dpi, minimum):
    found = []
    for text, box in zip(texts, boxes, strict=True):
        size = text.get_fontsize() * dpi / 72.0
        if size < minimum:
            detail = (
                f"{_name(text)} is {size:.1f} px tall, under the {minimum:g} px minimum"
            )
            found.append(("small", text, None, _tuple(box), detail))
    return found


# Text on marks: a mark is a list of parts. A part has a color and either an
# outline (display points the stroke passes through), a filled display path,
# or both, plus the radius in pixels the stroke or marker covers.


def _stroke(points, step=1.0):
    """`points` (N, 2) with a sample at least every `step` pixels along them."""
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 2:
        return points
    parts = [points[:1]]
    for p, q in itertools.pairwise(points):
        n = max(1, math.ceil(float(np.hypot(*(q - p))) / step))
        t = np.linspace(0.0, 1.0, n + 1)[1:, None]
        parts.append(p + t * (q - p))
    return np.vstack(parts)


def _parts(mark, dpi):
    """(rgba, outline points or None, filled Path or None, radius px) per part."""
    out = []
    if isinstance(mark, Line2D):
        xy = mark.get_transform().transform(mark.get_xydata())
        solid = mark.get_linestyle() not in ("None", "none", "", " ")
        radius = 0.5 * mark.get_linewidth() * dpi / 72.0
        marker = mark.get_marker() not in (None, "None", "none", "", " ")
        if marker:
            radius = max(radius, 0.5 * mark.get_markersize() * dpi / 72.0)
        pts = _stroke(xy) if solid else np.asarray(xy, dtype=float)
        out.append((to_rgba(mark.get_color()), pts, None, radius))
    elif isinstance(mark, Patch):
        path = mark.get_transform().transform_path(mark.get_path())
        filled = to_rgba(mark.get_facecolor())
        edge = to_rgba(mark.get_edgecolor())
        if filled[3] > 0.0 and mark.get_fill():
            out.append((filled, None, path, 0.0))
        if edge[3] > 0.0 and mark.get_linewidth() > 0.0:
            polys = path.to_polygons(closed_only=False)
            pts = np.vstack([_stroke(p) for p in polys]) if polys else np.empty((0, 2))
            out.append((edge, pts, None, 0.5 * mark.get_linewidth() * dpi / 72.0))
    elif isinstance(mark, PathCollection):
        offsets = mark.get_offset_transform().transform(mark.get_offsets())
        sizes = mark.get_sizes()
        radius = (
            0.5 * math.sqrt(float(np.max(sizes)) if len(sizes) else 0.0) * dpi / 72.0
        )
        colors = list(map(to_rgba, mark.get_facecolor())) or [(0, 0, 0, 0)]
        for rgba in colors:
            out.append((rgba, np.asarray(offsets, dtype=float), None, radius))
    elif isinstance(mark, Collection):
        transform = mark.get_transform()
        paths = [transform.transform_path(p) for p in mark.get_paths()]
        edges = list(map(to_rgba, mark.get_edgecolor()))
        faces = list(map(to_rgba, mark.get_facecolor()))
        for k, path in enumerate(paths):
            if not isinstance(mark, LineCollection) and faces:
                out.append((faces[k % len(faces)], None, path, 0.0))
            if edges:
                polys = path.to_polygons(closed_only=False)
                pts = (
                    np.vstack([_stroke(p) for p in polys])
                    if polys
                    else np.empty((0, 2))
                )
                width = mark.get_linewidth()
                lw = float(width[k % len(width)]) if len(width) else 0.0
                out.append((edges[k % len(edges)], pts, None, 0.5 * lw * dpi / 72.0))
    return out


def _same_color(a, b):
    return max(abs(a[i] - b[i]) for i in range(3)) <= _SAME_COLOR


def _hit(part, box):
    _, pts, path, radius = part
    if pts is not None and len(pts):
        inside = (
            (pts[:, 0] >= box.x0 - radius)
            & (pts[:, 0] <= box.x1 + radius)
            & (pts[:, 1] >= box.y0 - radius)
            & (pts[:, 1] <= box.y1 + radius)
        )
        if inside.any():
            return True
    if path is not None:
        xs = np.linspace(box.x0, box.x1, 7)
        ys = np.linspace(box.y0, box.y1, 3)
        grid = np.array([(x, y) for x in xs for y in ys])
        if path.contains_points(grid).any():
            return True
    return False


def _protected(text):
    patch = text.get_bbox_patch()
    if patch is None or not patch.get_visible():
        return False
    return (
        to_rgba(patch.get_facecolor())[3] * (patch.get_alpha() or 1.0)
        >= _OPAQUE_BACKING
    )


def _on_marks(fig, texts, boxes, dpi):
    cache = {}
    found = []
    for text, box in zip(texts, boxes, strict=True):
        if _protected(text):
            continue
        base = to_rgba(text.get_color())
        text_box = text.get_window_extent(_renderer(fig))
        for ax in fig.axes:
            if not ax.get_visible():
                continue
            for mark in _default_avoid(ax, set()):
                if not mark.get_visible() or not _data_colors(mark):
                    continue
                if mark not in cache:
                    cache[mark] = _parts(mark, dpi)
                for part in cache[mark]:
                    if part[0][3] < _SHOWN or not _same_color(part[0], base):
                        continue
                    if _hit(part, text_box):
                        kind = type(mark).__name__
                        detail = f"{_name(text)} is drawn over a {kind} of its color"
                        found.append(("on-marks", text, mark, _tuple(box), detail))
                        break
                else:
                    continue
                break
    return found


def _check_once(fig, dpi, minimum, overlap_tol, frame_tol, kinds):
    fig.canvas.draw()
    renderer = _renderer(fig)
    texts = [t for t in _visible_texts(fig) if _finite(t.get_window_extent(renderer))]
    boxes = [_bbox(t, renderer) for t in texts]
    found = []
    if "overlap" in kinds:
        found += _overlaps(texts, boxes, overlap_tol)
    if "outside" in kinds:
        found += _outside(fig, texts, boxes, renderer, frame_tol)
    if "small" in kinds:
        found += _small(texts, boxes, dpi, minimum)
    if "on-marks" in kinds:
        found += _on_marks(fig, texts, boxes, dpi)
    return found


def readability_faults(
    fig,
    *,
    at=None,
    times=None,
    venue="video",
    min_text_px=None,
    dpi=None,
    overlap_tol=2.0,
    frame_tol=1.0,
    kinds=KINDS,
):
    """Check a figure, or a clip sampled at chosen times, for readability faults.

    The figure is drawn once per sample time and the checks run on what is
    drawn, so the result describes the pixels, not the layout intent. The
    check is deterministic: the same figure and times give the same faults in
    the same order (by time, then kind, then drawing order). The figure is
    left posed at the last time, and at its original dpi.

    Args:
        fig: The figure to check.
        at: Callable ``at(t)`` that poses the figure at time `t` (a clip's
            update at a beat end, a turn midpoint or a cue time). None checks
            the figure as drawn.
        times: Sample times passed to `at`. None, with `at` given, is an
            error; with no `at`, the figure is checked once.
        venue: Key of `VENUES` setting the minimum text height.
        min_text_px: Minimum font size in pixels; overrides `venue`.
        dpi: Output dpi the clip is rendered at; None uses the figure's.
            Extents and tolerances are in pixels at this dpi.
        overlap_tol: Largest overlap, in pixels, that is not reported. An
            overlap counts only when it exceeds this in both directions, so
            boxes that only touch are fine.
        frame_tol: Largest overshoot past the figure edge, in pixels, that is
            not reported.
        kinds: The kinds of fault to look for, a subset of `KINDS`.

    Returns:
        A list of `ReadabilityFault`; empty when the figure is readable.

    Raises:
        ValueError: If `venue` or a kind is unknown, or `at` and `times` are
            not given together.

    Example::

        faults = ept.readability_faults(fig, at=pose, times=[1.0, 2.5])
    """
    kinds = tuple(kinds)
    unknown = [k for k in kinds if k not in KINDS]
    if unknown:
        raise ValueError(f"unknown fault kinds {unknown}; known: {list(KINDS)}")
    if min_text_px is None:
        if venue not in VENUES:
            raise ValueError(f"unknown venue {venue!r}; known: {sorted(VENUES)}")
        min_text_px = VENUES[venue]
    if (at is None) != (times is None):
        raise ValueError("give `at` and `times` together, or neither")
    samples = [None] if at is None else list(times)
    old_dpi = fig.dpi
    if dpi is not None:
        fig.set_dpi(dpi)
    found = []
    try:
        for t in samples:
            if at is not None:
                at(t)
            for kind, artist, other, box, detail in _check_once(
                fig, fig.dpi, min_text_px, overlap_tol, frame_tol, kinds
            ):
                found.append(ReadabilityFault(kind, artist, other, t, box, detail))
    finally:
        if dpi is not None:
            fig.set_dpi(old_dpi)
    order = {kind: i for i, kind in enumerate(KINDS)}
    index = {t: i for i, t in enumerate(samples)}
    found.sort(key=lambda f: (index[f.time], order[f.kind]))
    return found


def assert_readable(fig, **kwargs):
    """Raise `ReadabilityError` with a readable report if the figure has faults.

    Takes the arguments of `readability_faults`.

    Args:
        fig: The figure to check.
        **kwargs: Passed to `readability_faults`.

    Raises:
        ReadabilityError: If any fault is found; its `faults` attribute holds
            the list, and its message lists each with time, kind and box.

    Example::

        ept.assert_readable(fig, at=pose, times=[1.0, 2.5], venue="video")
    """
    found = readability_faults(fig, **kwargs)
    if found:
        raise ReadabilityError(found)


def caption_line(
    fig,
    *,
    at=None,
    times=None,
    height=0.14,
    width=0.8,
    margin=0.01,
    step=0.01,
):
    """The WebVTT ``line:`` percentage that puts captions clear of the figure's text.

    The caption band is a strip of the picture, `width` wide and centered,
    `height` tall. The band starts at the bottom, where players put captions,
    and moves up in steps until no visible text of the figure touches it, at
    any of the sampled times. The result is the top edge of the band as a
    percentage of the picture height, the value `eyepiece.narration.write_vtt`
    takes as ``line``.

    Args:
        fig: The figure of the clip.
        at: Callable ``at(t)`` that poses the figure at time `t`, as for
            `readability_faults`. None uses the figure as drawn.
        times: Sample times passed to `at`; give them with `at`. Use the
            times the captions are on screen.
        height: Height of the caption band as a fraction of the figure.
            Two caption lines at a typical player size are about 0.14.
        width: Width of the band as a fraction of the figure, centered.
        margin: Clear space kept between the band and any text, as a
            fraction of the figure height.
        step: How far the band moves up per try, as a fraction of the
            figure height.

    Returns:
        The ``line:`` value, in percent from the top of the picture.

    Raises:
        ValueError: If `at` and `times` are not given together, or no
            position is clear. Then reserve a caption band in the layout
            instead and pass its percentage to ``write_vtt``.
    """
    if (at is None) != (times is None):
        raise ValueError("give `at` and `times` together, or neither")
    samples = [None] if at is None else list(times)
    spans = []
    for t in samples:
        if at is not None:
            at(t)
        fig.canvas.draw()
        renderer = _renderer(fig)
        for text in _visible_texts(fig):
            if _finite(text.get_window_extent(renderer)):
                box = _bbox(text, renderer)
                left, right = box.x0 / fig.bbox.width, box.x1 / fig.bbox.width
                if right > (1.0 - width) / 2.0 and left < (1.0 + width) / 2.0:
                    spans.append(
                        (1.0 - box.y1 / fig.bbox.height, 1.0 - box.y0 / fig.bbox.height)
                    )
    top = 1.0 - height - margin
    while top >= 0.0:
        if all(
            high + margin <= top or low - margin >= top + height for low, high in spans
        ):
            return round(100.0 * top, 2)
        top -= step
    raise ValueError(
        "no clear position for the captions; reserve a caption band in the layout"
    )


# Motion faults: the path between two states.

MOTION_KINDS = ("jump", "dead", "drift", "tear", "overlap", "outside", "spike")

# Frames per second a clip is sampled at when `times` is a `Span`.
VENUE_FPS = {"video": 30.0, "phone": 30.0, "print": 30.0}

_STEP_FLOOR = 0.01  # mean pixel change (of 255) under which a step is a hold
_PAIR_ISOLATION = 2.0  # a jump must also exceed its neighbouring steps by this
_REST_STEP_RATIO = 0.5  # share of `jump_ratio` that flags a step into rest
_OVERLAP_INK = 0.3  # overlap is a fault once both artists are this opaque
_HOP_RATIO = 0.5  # a hop reaches this fraction of the way out, relative to net travel
_KINK_RATIO = 6.0  # a kink is this many median speeds in one frame
_KINK_FLOOR_PX = 4.0


class Span(NamedTuple):
    """A stretch of clip time to check at every frame.

    Attributes:
        start: First time, in seconds.
        stop: Last time, in seconds; it is always sampled.
    """

    start: float
    stop: float


@dataclass(frozen=True, eq=False)
class MotionFault:
    """One motion problem found along a clip.

    Attributes:
        kind: One of `MOTION_KINDS`.
        artist: The artist at fault; None for ``"jump"`` and ``"dead"``.
        other: The second artist of an ``"overlap"``; None for the others.
        time: The first sample time at which the fault shows.
        box: ``(x0, y0, x1, y1)`` in output pixels, origin at the lower left
            of the figure: the changed region of a jump, the artist's box, or
            the overlap region.
        detail: A one-line description.
        value: The number the rule measured (a ratio for a jump, pixels for
            drift, tear, outside, overlap and spike), None for ``"dead"``.
    """

    kind: str
    artist: object
    other: object
    time: object
    box: tuple
    detail: str
    value: object = None

    def __str__(self):
        """One line: the time, the kind, the detail and the box."""
        x0, y0, x1, y1 = self.box
        return (
            f"[t={self.time:g}] {self.kind}: {self.detail} "
            f"(box {x0:.0f},{y0:.0f} to {x1:.0f},{y1:.0f} px)"
        )


class MotionError(AssertionError):
    """A clip has motion faults.

    Attributes:
        faults: The `MotionFault` list.
    """

    def __init__(self, faults):
        """Build the report from the faults."""
        self.faults = list(faults)
        lines = [f"{len(self.faults)} motion fault(s):"]
        lines += [f"  {fault}" for fault in self.faults]
        super().__init__("\n".join(lines))


def _ink(artist):
    """Opacity of the strongest ink the artist paints, 0 when it paints none."""
    base = artist.get_alpha()
    base = 1.0 if base is None else float(base)
    if isinstance(artist, Text):
        if not artist.get_text().strip():
            return 0.0
        return base * to_rgba(artist.get_color())[3]
    if isinstance(artist, Patch):
        face = base * to_rgba(artist.get_facecolor())[3] if artist.get_fill() else 0.0
        edge = (
            base * to_rgba(artist.get_edgecolor())[3]
            if artist.get_linewidth() > 0.0
            else 0.0
        )
        return max(face, edge)
    if isinstance(artist, Line2D):
        drawn = artist.get_linestyle() not in ("None", "none", "", " ")
        drawn = drawn or artist.get_marker() not in (None, "None", "none", "", " ")
        return base * to_rgba(artist.get_color())[3] if drawn else 0.0
    if isinstance(artist, Collection):
        top = 0.0
        for colors in (artist.get_facecolor(), artist.get_edgecolor()):
            if len(colors):
                top = max(top, float(np.max(np.asarray(colors)[:, 3])))
        return base * top
    return 0.0


def _traced(fig):
    """Artists whose position is traced: text, patches, collections and lines."""
    skip = {id(fig.patch)} | {id(ax.patch) for ax in fig.axes}
    for artist in _walk(fig):
        if id(artist) in skip or isinstance(artist, Spine):
            continue
        if isinstance(artist, (Text, Patch, Collection, Line2D)) and (
            _ink(artist) >= _SHOWN
        ):
            yield artist


def _trace_box(artist, renderer):
    try:
        box = (
            _bbox(artist, renderer)
            if isinstance(artist, Text)
            else (artist.get_window_extent(renderer))
        )
    except (ValueError, AttributeError, TypeError):
        return None
    if not _finite(box):
        return None
    clip = artist.get_clip_box() if artist.get_clip_on() else None
    if clip is not None and not _finite(clip):
        clip = None
    clip = None if clip is None else _tuple(clip)
    return _tuple(box), clip, _ink(artist)


def _declared(move, key, default=None):
    if isinstance(move, dict):
        return move.get(key, default)
    contract = getattr(move, "contract", move)
    value = getattr(contract, key, default)
    return default if value is None else value


def _canvas_overshoot(box, width, height):
    x0, y0, x1, y1 = box
    return {"left": -x0, "right": x1 - width, "bottom": -y0, "top": y1 - height}


def _clip_overshoot(box, clip):
    x0, y0, x1, y1 = box
    c0, d0, c1, d1 = clip
    return {"left": c0 - x0, "right": x1 - c1, "bottom": d0 - y0, "top": y1 - d1}


def _intersect(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return None if x1 < x0 or y1 < y0 else (x0, y0, x1, y1)


def _edges_move(a, b):
    return max(abs(p - q) for p, q in zip(a, b, strict=True))


def _center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


class _Trace:
    """What a sampled clip leaves behind: times, frames and artist boxes."""

    def __init__(self, times):
        self.times = list(times)
        self.frames = []
        self.boxes = {}  # artist -> {sample index: (box, clip)}
        self.order = []  # artists in order of first appearance
        self.size = (0.0, 0.0)
        self.ticks = set()  # tick labels, which trade places as limits move

    def add(self, k, fig, renderer):
        fig.canvas.draw()
        self.frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        self.size = (float(fig.bbox.width), float(fig.bbox.height))
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                self.ticks.add(axis.offsetText)
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    self.ticks.update((tick.label1, tick.label2))
        for artist in _traced(fig):
            got = _trace_box(artist, renderer)
            if got is None:
                continue
            if artist not in self.boxes:
                self.boxes[artist] = {}
                self.order.append(artist)
            self.boxes[artist][k] = got

    def box(self, artist, k):
        got = self.boxes.get(artist, {}).get(k)
        return None if got is None else got[0]

    def moving(self, artist, k, tol):
        """True when the artist's box changes by more than `tol` next to sample k."""
        here = self.box(artist, k)
        if here is None:
            return False
        for j in (k - 1, k + 1):
            there = self.box(artist, j)
            if there is not None and _edges_move(here, there) > tol:
                return True
        return False

    def label(self, artist):
        if isinstance(artist, Text):
            return _name(artist)
        index = [a for a in self.order if type(a) is type(artist)].index(artist)
        tag = artist.get_gid() or artist.get_label()
        tag = "" if not tag or str(tag).startswith("_") else f" {tag!r}"
        x, y = self.first_box(artist)[:2]
        return f"{type(artist).__name__}{tag} #{index} at ({x:.0f}, {y:.0f})"

    def first_box(self, artist):
        k = min(self.boxes[artist])
        return self.boxes[artist][k][0]


def _steps(frames):
    """Mean absolute pixel change between consecutive frames."""
    out = []
    before = frames[0].astype(np.int16)
    for frame in frames[1:]:
        after = frame.astype(np.int16)
        out.append(float(np.abs(after - before).mean()))
        before = after
    return out


def _changed(before, after):
    """Box, in output pixels from the lower left, of what changed between frames."""
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    rows, cols = np.nonzero(diff)
    if not len(rows):
        return None
    return (
        float(cols.min()),
        float(diff.shape[0] - 1 - rows.max()),
        float(cols.max() + 1),
        float(diff.shape[0] - rows.min()),
    )


def _check_jump(trace, jump_ratio):
    size = _steps(trace.frames)
    moving = [v for v in size if v > _STEP_FLOOR]
    if len(moving) < 3:
        return []
    median = float(np.median(moving))
    found = []
    last = len(size) - 1
    for i, value in enumerate(size):
        into_rest = (
            i == last
            and i > 0
            and value > _REST_STEP_RATIO * jump_ratio * median
            and value > size[i - 1]
        )
        if not into_rest:
            if value <= jump_ratio * median:
                continue
            near = [size[j] for j in (i - 1, i + 1) if 0 <= j <= last]
            if near and value <= _PAIR_ISOLATION * max(near):
                continue
        where = "into the rest state" if i == last else "in one frame"
        detail = (
            f"the picture changes {value:.2f} levels {where}, "
            f"{value / median:.1f} times the median step of {median:.2f}"
        )
        box = _changed(trace.frames[i], trace.frames[i + 1])
        box = box or (0.0, 0.0, *trace.size)
        found.append((i + 1, "jump", None, None, box, detail, value / median))
    return found


def _check_dead(trace, moves):
    found = []
    times = trace.times
    for move in moves:
        span = _declared(move, "span")
        lo, hi = (times[0], times[-1]) if span is None else span
        before = [k for k, t in enumerate(times) if t <= lo]
        after = [k for k, t in enumerate(times) if t >= hi]
        first = before[-1] if before else 0
        stop = after[0] if after else len(times) - 1
        if stop <= first:
            continue
        region = None
        artists = [a for a in (_declared(move, "artists") or ()) if a in trace.boxes]
        if artists:
            boxes = [b[0] for a in artists for b in trace.boxes[a].values()]
            region = (
                min(b[0] for b in boxes) - 2.0,
                min(b[1] for b in boxes) - 2.0,
                max(b[2] for b in boxes) + 2.0,
                max(b[3] for b in boxes) + 2.0,
            )
        ref = trace.frames[first].astype(np.int16)
        height = ref.shape[0]
        peak = 0
        for k in range(first + 1, stop + 1):
            diff = np.abs(trace.frames[k].astype(np.int16) - ref).max(axis=2)
            if region is not None:
                r0 = max(int(height - region[3]), 0)
                r1 = max(int(height - region[1]) + 1, 0)
                c0 = max(int(region[0]), 0)
                c1 = max(int(region[2]) + 1, 0)
                diff = diff[r0:r1, c0:c1]
            peak = max(peak, int(diff.max()) if diff.size else 0)
        if peak == 0:
            name = _declared(move, "name", "a move")
            detail = (
                f"{name!r} changes no pixel between t={times[first]:g} "
                f"and t={times[stop]:g}"
            )
            box = region or (0.0, 0.0, *trace.size)
            found.append((first, "dead", None, None, box, detail, None))
    return found


def _live(trace, artist, what):
    if artist not in trace.boxes:
        raise ValueError(f"the {what} artist {artist!r} is not visible at any sample")


def _check_drift(trace, rest, tol):
    found = []
    for artist in rest:
        _live(trace, artist, "rest")
        ks = sorted(trace.boxes[artist])
        home = trace.boxes[artist][ks[0]][0]
        worst, first = 0.0, None
        for k in ks:
            off = _edges_move(trace.boxes[artist][k][0], home)
            if off > tol and first is None:
                first = k
            worst = max(worst, off)
        if first is not None:
            detail = f"{trace.label(artist)} should stay still and moves {worst:.1f} px"
            found.append(
                (
                    first,
                    "drift",
                    artist,
                    None,
                    trace.boxes[artist][first][0],
                    detail,
                    worst,
                )
            )
    return found


def _check_tear(trace, groups, tol):
    found = []
    for group in groups:
        scales = False
        if isinstance(group, dict):
            scales = bool(group.get("scales", False))
            group = group["artists"]
        group = list(group)
        for artist in group:
            _live(trace, artist, "group")
        common = sorted(set.intersection(*(set(trace.boxes[a]) for a in group)))
        if len(group) < 2 or len(common) < 2:
            continue
        pairs = np.array(list(itertools.combinations(range(len(group)), 2)))

        def dists(k, group=group, pairs=pairs):
            c = np.array([_center(trace.boxes[a][k][0]) for a in group])
            return np.hypot(*(c[pairs[:, 0]] - c[pairs[:, 1]]).T)

        base = dists(common[0])
        worst, hit = 0.0, None
        for k in common[1:]:
            d = dists(k)
            s = 1.0
            live = base > 1e-9
            if scales and live.any():
                s = float(np.median(d[live] / base[live]))
            dev = np.abs(d - s * base)
            top = int(np.argmax(dev))
            if dev[top] > tol and hit is None:
                hit = (k, top)
            worst = max(worst, float(dev[top]))
        if hit is not None:
            k, top = hit
            a, b = (group[i] for i in pairs[top])
            detail = (
                f"{trace.label(a)} and {trace.label(b)} are in one group and "
                f"their distance changes by up to {worst:.1f} px"
            )
            found.append((k, "tear", a, b, trace.boxes[a][k][0], detail, worst))
    return found


def _check_overlap(trace, candidates, allowed, tol, move_tol):
    found = []
    runs = {}
    cand = [a for a in trace.order if a in candidates]
    for k in range(len(trace.times)):
        here = [
            a
            for a in cand
            if k in trace.boxes[a]
            and trace.boxes[a][k][2] >= _OVERLAP_INK
            and trace.moving(a, k, move_tol)
        ]
        for i, a in enumerate(here):
            for b in here[i + 1 :]:
                if frozenset((a, b)) in allowed:
                    continue
                x = _intersect(trace.box(a, k), trace.box(b, k))
                if x is None or x[2] - x[0] <= tol or x[3] - x[1] <= tol:
                    continue
                key = (id(a), id(b))
                area = (x[2] - x[0]) * (x[3] - x[1])
                if key not in runs:
                    runs[key] = [k, a, b, x, area, 0]
                run = runs[key]
                run[5] += 1
                if area > run[4]:
                    run[3], run[4] = x, area
    for k, a, b, x, _, count in runs.values():
        detail = (
            f"{trace.label(a)} and {trace.label(b)} overlap by "
            f"{x[2] - x[0]:.1f} x {x[3] - x[1]:.1f} px while moving, "
            f"at {count} sample(s)"
        )
        found.append((k, "overlap", a, b, x, detail, min(x[2] - x[0], x[3] - x[1])))
    return found


def _check_outside(trace, texts, tol):
    width, height = trace.size
    found = []
    for artist in trace.order:
        first, worst, sides, count = None, 0.0, {}, 0
        clips = {c for _, c, _ in trace.boxes[artist].values()}
        # A clip box that changes is a reveal mask (a write-on), not a cut.
        static_clip = len(clips) == 1
        for k, (box, clip, _) in sorted(trace.boxes[artist].items()):
            seen = box
            if clip is not None:
                seen = _intersect(box, clip)
                if seen is None:
                    continue
            over = {
                s: v
                for s, v in _canvas_overshoot(seen, width, height).items()
                if v > tol
            }
            if clip is not None and artist in texts and static_clip:
                cut = _clip_overshoot(box, clip)
                over.update({s: v for s, v in cut.items() if v > tol})
            if not over:
                continue
            count += 1
            if first is None:
                first = k
            for side, v in over.items():
                sides[side] = max(sides.get(side, 0.0), v)
            worst = max(worst, max(over.values()))
        if first is not None:
            text = ", ".join(f"{s} by {v:.1f} px" for s, v in sides.items())
            detail = f"{trace.label(artist)} is cut off: {text} (at {count} sample(s))"
            found.append(
                (
                    first,
                    "outside",
                    artist,
                    None,
                    trace.boxes[artist][first][0],
                    detail,
                    worst,
                )
            )
    return found


def _check_spike(trace, spike_px):
    found = []
    for artist in trace.order:
        if artist in trace.ticks:
            continue
        ks = sorted(trace.boxes[artist])
        runs, run = [], [ks[0]]
        for k in ks[1:]:
            if k == run[-1] + 1:
                run.append(k)
            else:
                runs.append(run)
                run = [k]
        runs.append(run)
        size = max(
            trace.first_box(artist)[2] - trace.first_box(artist)[0],
            trace.first_box(artist)[3] - trace.first_box(artist)[1],
        )
        for run in runs:
            if len(run) < 6:
                continue
            boxes = np.array([trace.boxes[artist][k][0] for k in run])
            extent = np.ptp(boxes[:, 2:] - boxes[:, :2], axis=0)
            if extent.max() > 1.0:  # it changes shape: a bar, a connector, a text
                continue
            c = np.array([_center(trace.boxes[artist][k][0]) for k in run])
            chord = c[-1] - c[0]
            length = float(np.hypot(*chord))
            rel = c - c[0]
            if length > 1e-9:
                t = np.clip(rel @ chord / length**2, 0.0, 1.0)
                off = np.hypot(*(rel - t[:, None] * chord).T)
            else:
                off = np.hypot(*rel.T)
            hop = float(off.max())
            if spike_px < hop <= max(size, spike_px) and hop > _HOP_RATIO * length:
                k = run[int(np.argmax(off))]
                detail = (
                    f"{trace.label(artist)} hops {hop:.1f} px off its path "
                    f"while travelling {length:.1f} px"
                )
                found.append(
                    (k, "spike", artist, None, trace.boxes[artist][k][0], detail, hop)
                )
                continue
            v = np.diff(c, axis=0)
            speed = np.hypot(*v.T)
            moving = speed[speed > 0.05]
            if len(moving) < 3:
                continue
            kink = np.hypot(*np.diff(v, axis=0).T)
            floor = max(_KINK_FLOOR_PX, _KINK_RATIO * float(np.median(moving)))
            top = int(np.argmax(kink))
            if kink[top] > floor:
                k = run[top + 1]
                detail = (
                    f"{trace.label(artist)} changes velocity by {kink[top]:.1f} px "
                    f"in one frame ({kink[top] / np.median(moving):.1f} median speeds)"
                )
                found.append(
                    (
                        k,
                        "spike",
                        artist,
                        None,
                        trace.boxes[artist][k][0],
                        detail,
                        float(kink[top]),
                    )
                )
    return found


def _sample_times(times, fps):
    if isinstance(times, Span):
        n = max(math.floor((times.stop - times.start) * fps + 1e-9), 0)
        out = [times.start + k / fps for k in range(n + 1)]
        if out[-1] < times.stop - 1e-9:
            out.append(float(times.stop))
        return out
    return list(times)


def motion_faults(
    fig,
    at,
    times,
    *,
    moves=None,
    rest=(),
    groups=(),
    may_overlap=(),
    tol_px=0.5,
    jump_ratio=4.0,
    overlap_tol=2.0,
    frame_tol=1.0,
    spike_px=2.0,
    spikes=False,
    kinds=None,
    venue=None,
    fps=None,
    dpi=None,
):
    """Check a clip's path between states for motion faults.

    The figure is posed and drawn at every sample time. At each sample the
    window extent of every visible text, patch, collection and line is
    recorded under the artist itself, so glyphs drawn as paths are traced as
    text is, and the rendered frame is kept for the pixel checks. The checks
    run on that record, so the result describes what was drawn. Sampling must
    be dense for the continuity checks to mean anything: pass a `Span` to use
    every frame at the venue frame rate, or list the frame times.

    Kinds of fault:

    - ``"jump"``: a step between consecutive frames, the one into the rest
      state included, that exceeds `jump_ratio` times the median non-zero
      step and stands out from the steps next to it. A step into the rest
      state is also reported when it is `jump_ratio / 2` times the median and
      larger than the step before it, because a move settles into rest.
    - ``"dead"``: a declared move whose frames show no pixel change over its
      span.
    - ``"drift"``: an artist of `rest` whose box moves by more than `tol_px`.
    - ``"tear"``: artists of a group whose pairwise distances change by more
      than `tol_px` (relative to the group's common scale factor when the
      group is declared to scale).
    - ``"overlap"``: two artists that both move, whose boxes intersect by
      more than `overlap_tol` pixels in both directions, neither pair being
      allowed. Texts are checked against texts; the artists a move declares
      are checked as well.
    - ``"outside"``: an artist that the canvas cuts, or a text (or declared
      artist) cut by the axes it is clipped to, by more than `frame_tol`
      pixels. An artist clipped to its axes may leave the axes: that is
      the clip working, and only text is reported for it.
    - ``"spike"``: an artist that hops off its straight path by more than
      `spike_px` while travelling less than it hops, or that changes
      velocity by several median speeds in one frame. A false alarm costs more
      than a miss here, so this kind is off unless `spikes` is True or it is
      named in `kinds`.

    Args:
        fig: The figure of the clip.
        at: Callable ``at(t)`` that poses the figure at time `t`.
        times: Sample times, or a `Span` to sample at every frame.
        moves: Declared moves, each a dict (or an object with a ``contract``
            of the same fields) with any of ``name``, ``span`` (start, stop),
            ``artists`` (those that may move), ``rest``, ``groups`` and
            ``may_overlap``. None declares nothing; ``jump``, ``outside``
            and the text ``overlap`` still run.
        rest: Artists that must not move.
        groups: Groups that move rigidly: each a list of artists, or a dict
            ``{"artists": [...], "scales": True}`` for a group that scales.
        may_overlap: Pairs of artists allowed to overlap while moving.
        tol_px: Motion, in pixels, that is not reported.
        jump_ratio: Largest step, as a multiple of the median step.
        overlap_tol: Largest overlap, in pixels, that is not reported.
        frame_tol: Largest cut by the canvas or clip, in pixels, not reported.
        spike_px: Smallest hop, in pixels, that is a spike.
        spikes: Include ``"spike"`` in the default kinds.
        kinds: The kinds to look for, a subset of `MOTION_KINDS`; None is
            every kind but ``"spike"``, or every kind with `spikes`.
        venue: Key of `VENUE_FPS` giving the frame rate of a `Span`; default
            ``"video"``.
        fps: Frame rate of a `Span`; overrides `venue`.
        dpi: Output dpi; None uses the figure's.

    Returns:
        A list of `MotionFault` ordered by time, then kind; empty when the
        clip is clean. The figure is left posed at the last time and at its
        original dpi.

    Raises:
        ValueError: If a kind or venue is unknown, or a declared rest or group
            artist is never visible.

    Example::

        span = ept.Span(0.0, 2.0)
        faults = ept.motion_faults(fig, at=clip.pose, times=span, rest=[title])
    """
    if kinds is None:
        kinds = (
            MOTION_KINDS if spikes else tuple(k for k in MOTION_KINDS if k != "spike")
        )
    kinds = tuple(kinds)
    unknown = [k for k in kinds if k not in MOTION_KINDS]
    if unknown:
        raise ValueError(f"unknown fault kinds {unknown}; known: {list(MOTION_KINDS)}")
    if fps is None:
        venue = "video" if venue is None else venue
        if venue not in VENUE_FPS:
            raise ValueError(f"unknown venue {venue!r}; known: {sorted(VENUE_FPS)}")
        fps = VENUE_FPS[venue]
    samples = _sample_times(times, fps)
    if len(samples) < 3:
        raise ValueError("a motion check needs at least three samples")
    old_dpi = fig.dpi
    if dpi is not None:
        fig.set_dpi(dpi)
    try:
        trace = _Trace(samples)
        for k, t in enumerate(samples):
            at(t)
            trace.add(k, fig, _renderer(fig))
    finally:
        if dpi is not None:
            fig.set_dpi(old_dpi)
    return _analyze(
        trace,
        kinds=kinds,
        moves=moves,
        rest=rest,
        groups=groups,
        may_overlap=may_overlap,
        tol_px=tol_px,
        jump_ratio=jump_ratio,
        overlap_tol=overlap_tol,
        frame_tol=frame_tol,
        spike_px=spike_px,
    )


def _analyze(
    trace,
    *,
    kinds,
    moves,
    rest,
    groups,
    may_overlap,
    tol_px,
    jump_ratio,
    overlap_tol,
    frame_tol,
    spike_px,
):
    """Run the checks of `kinds` on a finished `_Trace`."""
    moves = [] if moves is None else list(moves)
    rest = list(rest)
    groups = list(groups)
    allowed = {frozenset(p) for p in may_overlap}
    declared = set()
    for move in moves:
        rest += list(_declared(move, "rest") or ())
        groups += list(_declared(move, "groups") or ())
        declared |= set(_declared(move, "artists") or ())
        allowed |= {frozenset(p) for p in (_declared(move, "may_overlap") or ())}
    found = []
    if "jump" in kinds:
        found += _check_jump(trace, jump_ratio)
    if "dead" in kinds:
        found += _check_dead(trace, moves)
    if "drift" in kinds:
        found += _check_drift(trace, rest, tol_px)
    if "tear" in kinds:
        found += _check_tear(trace, groups, tol_px)
    texts = {a for a in trace.order if isinstance(a, Text)}
    if "overlap" in kinds:
        found += _check_overlap(trace, texts | declared, allowed, overlap_tol, tol_px)
    if "outside" in kinds:
        found += _check_outside(trace, texts | declared, frame_tol)
    if "spike" in kinds:
        found += _check_spike(trace, spike_px)
    order = {kind: i for i, kind in enumerate(MOTION_KINDS)}
    found.sort(key=lambda f: (f[0], order[f[1]]))
    return [
        MotionFault(kind, artist, other, trace.times[k], box, detail, value)
        for k, kind, artist, other, box, detail, value in found
    ]


def assert_motion(fig, at, times, **kwargs):
    """Raise `MotionError` with a readable report if the clip has motion faults.

    Takes the arguments of `motion_faults`.

    Args:
        fig: The figure of the clip.
        at: Callable ``at(t)`` that poses the figure at time `t`.
        times: Sample times, or a `Span`.
        **kwargs: Passed to `motion_faults`.

    Raises:
        MotionError: If any fault is found; its `faults` attribute holds the
            list and its message lists each with time, kind and box.

    Example::

        ept.assert_motion(fig, clip.pose, ept.Span(0.0, 2.0), rest=[title])
    """
    found = motion_faults(fig, at, times, **kwargs)
    if found:
        raise MotionError(found)


# Fastest narration pace that stays easy to follow while a figure also moves.
SPEECH_CEILING_WPM = 160.0


class SpeechRate(NamedTuple):
    """Speaking pace of each narrated line.

    Attributes:
        rates: Words per minute of each line, in script order, as a tuple of
            ``(line_key, wpm)``.
        fast: The ``(line_key, wpm)`` pairs above the ceiling.
        ceiling: The ceiling the lines were compared with, in words per minute.
    """

    rates: tuple
    fast: tuple
    ceiling: float


class SpeechRateError(AssertionError):
    """A narrated line is spoken faster than the ceiling.

    Attributes:
        result: The `SpeechRate`.
    """

    def __init__(self, result):
        """Build the report from the result."""
        self.result = result
        lines = [
            f"{len(result.fast)} line(s) faster than {result.ceiling:g} words/min:"
        ]
        lines += [f"  {key}: {wpm:.0f} words/min" for key, wpm in result.fast]
        super().__init__("\n".join(lines))


def speech_rate(narration, *, ceiling=SPEECH_CEILING_WPM):
    """Words per minute of each narrated line, and the lines above a ceiling.

    The rate of a line is its spoken words over the time from the start of its
    first word to the end of its last, from the cached word times. Padding
    before and after a line and the pause between lines do not count; a pause
    inside a line does, as a listener hears it. A written word with no letters
    or digits (a dash standing alone) is not a spoken word and is not counted.

    Args:
        narration: A `eyepiece.narration.Narration`, from ``load`` for the
            cached word times or ``estimated`` for a first layout.
        ceiling: Fastest allowed pace in words per minute. The default, 160,
            is the pace past which a viewer cannot follow a moving figure.

    Returns:
        A `SpeechRate` of per-line rates, the lines strictly above `ceiling`,
        and the ceiling. A line whose words span no time has an infinite rate.
    """
    rates = []
    for beat in narration.beats:
        words = [w for w in beat.words if any(ch.isalnum() for ch in w.text)]
        if not words:
            continue
        span = max(w.end_s for w in words) - min(w.start_s for w in words)
        rates.append((beat.key, 60.0 * len(words) / span if span > 0 else math.inf))
    fast = tuple(item for item in rates if item[1] > ceiling)
    return SpeechRate(tuple(rates), fast, float(ceiling))


def assert_speech_rate(narration, **kwargs):
    """Raise `SpeechRateError` if any line is spoken faster than the ceiling.

    Takes the arguments of `speech_rate`.

    Args:
        narration: A `eyepiece.narration.Narration`.
        **kwargs: Passed to `speech_rate` (``ceiling``).

    Returns:
        The `SpeechRate`, when every line is within the ceiling.

    Raises:
        SpeechRateError: If any line is above the ceiling; its `result`
            attribute holds the `SpeechRate`.
    """
    result = speech_rate(narration, **kwargs)
    if result.fast:
        raise SpeechRateError(result)
    return result
