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
    "VENUES",
    "ReadabilityError",
    "ReadabilityFault",
    "SpeechRate",
    "SpeechRateError",
    "assert_readable",
    "assert_speech_rate",
    "caption_line",
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
