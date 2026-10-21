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
rendered frames, and reports eight kinds of fault: ``"jump"``, ``"dead"``,
``"drift"``, ``"tear"``, ``"overlap"``, ``"outside"``, ``"spike"`` and
``"shape-class"``.
`assert_motion` raises `MotionError` naming every fault.

`grounding_faults` checks that colored equation terms name visible marks,
share their expected colors, and appear after their concrete referents.

`speech_rate` is the check on the voice rather than the picture: it reads the
word times of a `eyepiece.narration.Narration` and reports each line's words
per minute and the lines faster than a ceiling. `assert_speech_rate` raises
on them.

`timing_faults` checks narration pace, the heard gaps between lines and
undeclared silence. `assert_timing` raises with every timing finding.

A clip is described by the figure and a callable `at` that poses it at a
time; the times to sample (span ends, turn midpoints, cue times) are the
caller's.

Example::

    import eyepiece.testing as ept

    faults = ept.readability_faults(
        fig, at=lambda t: clip.pose(t), times=[0.5, 1.5, 2.0], venue="video"
    )
    ept.assert_readable(fig, at=clip.pose, times=cue_times)
"""

import importlib
import itertools
import math
import numbers
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from matplotlib import rcParams
from matplotlib.axis import Axis
from matplotlib.backend_bases import GraphicsContextBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import Collection, LineCollection, PathCollection
from matplotlib.colors import to_hex, to_rgb, to_rgba
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.spines import Spine
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from eyepiece._equation import _binding_rgb
from eyepiece._meaning import (
    MEANING_KINDS,
    MeaningError,
    MeaningFault,
    assert_meaning,
    meaning_faults,
    meaning_faults_chain,
    meaning_report,
)
from eyepiece._motif_checks import (
    MOTIF_KINDS,
    MotifError,
    MotifFault,
    assert_motifs,
    motif_faults,
    motif_margins,
)
from eyepiece._motion import frame_count
from eyepiece.attention import _data_colors, _default_avoid, _renderer

if TYPE_CHECKING:
    from eyepiece._render_pool import assert_frames_pure, frame_faults
    from eyepiece._render_poster import same_picture

__all__ = [
    "GROUNDING_KINDS",
    "KINDS",
    "MEANING_KINDS",
    "MOTIF_KINDS",
    "MOTION_KINDS",
    "TIMING_KINDS",
    "VENUES",
    "VENUE_FPS",
    "ClipError",
    "ClipReport",
    "GroundingError",
    "GroundingFault",
    "HaloError",
    "HaloFault",
    "MeaningError",
    "MeaningFault",
    "MotifError",
    "MotifFault",
    "MotionError",
    "MotionFault",
    "ReadabilityError",
    "ReadabilityFault",
    "SpeechRate",
    "SpeechRateError",
    "TimingError",
    "TimingFault",
    "Window",
    "assert_clip",
    "assert_frames_pure",
    "assert_grounded",
    "assert_meaning",
    "assert_motifs",
    "assert_motion",
    "assert_no_halos",
    "assert_readable",
    "assert_speech_rate",
    "assert_timing",
    "caption_line",
    "check_clip",
    "frame_faults",
    "grounding_faults",
    "halo_faults",
    "meaning_faults",
    "meaning_faults_chain",
    "meaning_report",
    "motif_faults",
    "motif_margins",
    "motion_faults",
    "motion_ink",
    "readability_faults",
    "review_sheet",
    "same_picture",
    "speech_rate",
    "timing_faults",
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


def _text_alpha(artist):
    """The greatest opacity of a text's fill or stroked outline."""
    alpha = _alpha(artist, artist.get_color())
    for effect in artist.get_path_effects():
        if isinstance(effect, withStroke):
            # Reproduce the graphics context that Text.draw passes to Stroke,
            # including explicit opacity overrides on either one.
            gc = GraphicsContextBase()
            gc.set_foreground(to_rgba(artist.get_color()), isRGBA=True)
            gc.set_alpha(artist.get_alpha())
            effect._update_gc(gc, effect._gc)
            if gc.get_linewidth() > 0:
                alpha = max(alpha, gc.get_rgb()[3])
    return alpha


def _visible_texts(fig):
    out = []
    for artist in _walk(fig):
        if (
            isinstance(artist, Text)
            and artist.get_text().strip()
            and _text_alpha(artist) >= _SHOWN
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
        if size < minimum and not getattr(text, "_eyepiece_furniture", False):
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
    return to_rgba(patch.get_facecolor())[3] >= _OPAQUE_BACKING


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
            update at a span end, a turn midpoint or a cue time). None checks
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


@dataclass(frozen=True, eq=False)
class HaloFault:
    """One text artist drawn with a stroked halo instead of a backing box.

    Attributes:
        artist: The text artist at fault.
        text: Its text when checked.
        detail: A description naming the artist and the backing-box convention.
        kind: Always ``"halo"``.
    """

    artist: object
    text: str
    detail: str
    kind: str = "halo"

    def __str__(self):
        """Name the haloed text and the convention it breaks."""
        return f"{self.kind}: {self.detail}"


class HaloError(AssertionError):
    """A figure has haloed text.

    Attributes:
        faults: The list of `HaloFault` records.
    """

    def __init__(self, faults):
        """Build the assertion message from every finding."""
        self.faults = list(faults)
        super().__init__(
            "\n".join([f"{len(self.faults)} halo fault(s):", *map(str, self.faults)])
        )


def halo_faults(fig) -> list[HaloFault]:
    """List visible text artists drawn with a stroked halo.

    Labels use a backing box, as `eyepiece.backing` and `eyepiece.callout`
    do. A `matplotlib.patheffects.withStroke` on text breaks that convention,
    including when the same text also has a box. Halos on lines and other
    marks are outside this check. Hidden, blank and transparent texts are
    skipped, as in `readability_faults`.

    Args:
        fig: The figure to check as drawn.

    Returns:
        One `HaloFault` per haloed text, in drawing order. Each keeps the
        artist and its text; the detail includes its gid or label when set.

    Example::

        faults = ept.halo_faults(fig)
    """
    _renderer(fig)
    found = []
    for artist in _visible_texts(fig):
        if not any(
            isinstance(effect, withStroke) for effect in artist.get_path_effects()
        ):
            continue
        tag = artist.get_gid() or artist.get_label()
        name = _name(artist) + (f" ({tag!r})" if tag else "")
        found.append(
            HaloFault(
                artist,
                artist.get_text(),
                f"{name} uses a stroked halo; use a backing box",
            )
        )
    return found


def assert_no_halos(fig) -> list[HaloFault]:
    """Raise `HaloError` if visible text has a stroked halo.

    Args:
        fig: The figure to check as drawn.

    Returns:
        An empty list when no haloed text is found.

    Raises:
        HaloError: If any halo is found; its `faults` attribute holds the list.

    Example::

        ept.assert_no_halos(fig)
    """
    found = halo_faults(fig)
    if found:
        raise HaloError(found)
    return found


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

MOTION_KINDS = (
    "jump",
    "dead",
    "drift",
    "tear",
    "overlap",
    "outside",
    "spike",
    "shape-class",
    "hop-rate",
)

# Frames per second a clip is sampled at when `times` is a `Window`.
VENUE_FPS = {"video": 30.0, "phone": 30.0, "print": 30.0}

_MAX_HOP_RATE = 8.0  # draws per second above which a hop group is a flicker
_INK_LEVEL = 2  # levels (of 255) a channel must change to count as ink moving
_STEP_FLOOR = 0.01  # mean pixel change (of 255) under which a step is a hold
_PAIR_ISOLATION = 2.0  # a jump must also exceed its neighbouring steps by this
_REST_STEP_RATIO = 0.5  # share of `jump_ratio` that flags a step into rest
_REST_GROWTH = 2.0  # ... when it is also this many times the step before it
_OVERLAP_INK = 0.3  # overlap is a fault once both artists are this opaque
_HOP_RATIO = 0.5  # a hop reaches this fraction of the way out, relative to net travel
_KINK_RATIO = 6.0  # a kink is this many median speeds in one frame
_KINK_FLOOR_PX = 4.0


class Window(NamedTuple):
    """A stretch of clip time to check at every frame.

    A window is what a check samples, not a timeline span: `eyepiece.Span`
    is the record a `eyepiece.Timeline` verb returns. `check_clip` takes a
    plan span by name where it would take a window.

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
        other: The second artist of an ``"overlap"`` or a ``"tear"``; None for
            the others.
        time: The first sample time at which the fault shows. For ``"spike"``
            it is the first sample past `spike_px` of the hop, or the sample at
            which the speed changes.
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


def _alpha_value(artist):
    """The artist's own alpha as one number, None when it has none."""
    alpha = artist.get_alpha()
    if alpha is None or not np.size(alpha):
        return None
    return float(np.max(np.asarray(alpha, dtype=float)))


def _ink(artist):
    """Opacity of the strongest ink the artist paints, 0 when it paints none.

    A patch or collection folds its alpha into its colors, so only text and
    lines multiply the artist alpha by the color alpha.
    """
    base = _alpha_value(artist)
    base = 1.0 if base is None else base
    if isinstance(artist, Text):
        if not artist.get_text().strip():
            return 0.0
        return base * to_rgba(artist.get_color())[3]
    if isinstance(artist, Patch):
        face = to_rgba(artist.get_facecolor())[3] if artist.get_fill() else 0.0
        edge = (
            to_rgba(artist.get_edgecolor())[3] if artist.get_linewidth() > 0.0 else 0.0
        )
        return max(face, edge)
    if isinstance(artist, Line2D):
        drawn = artist.get_linestyle() not in ("None", "none", "", " ")
        drawn = drawn or artist.get_marker() not in (None, "None", "none", "", " ")
        return base * to_rgba(artist.get_color())[3] if drawn else 0.0
    if isinstance(artist, AxesImage):
        return base
    if isinstance(artist, Collection):
        top = 0.0
        for colors in (artist.get_facecolor(), artist.get_edgecolor()):
            if len(colors):
                top = max(top, float(np.max(np.asarray(colors)[:, 3])))
        return top
    return 0.0


GROUNDING_KINDS = ("no-referent", "color-mismatch", "symbol-first")


@dataclass(frozen=True)
class GroundingFault:
    """One equation term's grounding problem at its first sample time.

    Attributes:
        kind: One of `GROUNDING_KINDS`.
        term: The term name within its equation.
        time_s: Sample time in seconds, or None for the current state.
        detail: A plain sentence describing the finding.
    """

    kind: str
    term: str
    time_s: float | None
    detail: str


class GroundingError(AssertionError):
    """An equation has grounding problems; `faults` holds their list."""

    def __init__(self, faults):
        """Build a report with each term and kind on its own line."""
        self.faults = list(faults)
        lines = [f"{len(self.faults)} grounding fault(s):"]
        lines += [f"  {f.term}: {f.kind}: {f.detail}" for f in self.faults]
        super().__init__("\n".join(lines))


def _grounding_colored(rgb, tol):
    """Whether RGB spends a hue outside text ink and the neutral ramp."""
    text = to_rgb(rcParams["text.color"])
    if max(rgb) - min(rgb) <= tol or rgb == text:
        return False
    back = np.asarray(to_rgb(rcParams["axes.facecolor"]))
    ramp = np.asarray(text) - back
    length_sq = float(ramp @ ramp)
    level = (
        np.clip(float((np.asarray(rgb) - back) @ ramp) / length_sq, 0, 1)
        if length_sq
        else 0
    )
    neutral = np.allclose(rgb, back + level * ramp, rtol=0, atol=tol)
    return not neutral


def _grounding_mark_shown(mark):
    """Whether a bound mark currently paints ink in a visible axes or figure."""
    return (
        mark.get_visible()
        and (mark.axes is None or mark.axes.get_visible())
        and mark.get_figure() is not None
        and _ink(mark) >= _SHOWN
    )


def grounding_faults(
    equations, *, at=None, times=None, tol=0.08
) -> list[GroundingFault]:
    """Check equation terms for missing referents, wrong colors and early symbols.

    A term is shown when any glyph is visible with effective opacity at least
    0.05. Its first glyph in `term._pieces` order decides its color. A bound
    mark is shown when it is visible, belongs to a figure, has a visible axes
    if it has an axes, and has effective ink opacity at least 0.05.

    Text-colored terms and neutral tones, the ramp from the axes background to
    the text color that `ep.neutral` draws, are never colored. A color within
    `tol` per channel of the nearest point on that ramp, or whose channel
    spread is at most `tol`, is neutral; other colors are colored.

    A shown, colored term reports ``"no-referent"`` if it is unbound or its
    mark is not shown. A shown, bound term reports ``"color-mismatch"`` if
    its color differs from the mark's color by more than `tol` per channel.
    With a motif, the term must match the motif's label ink, and a mark with
    a readable color must match the motif's mark color instead.

    With at least two samples, a bound term reports ``"symbol-first"`` if it
    is first shown before its mark is first shown, including a mark never
    shown. A term carried from its mark by a flight passes this ordering
    check, because its mark is on screen when the flight starts. Colors are
    resolved at check time. A pulse in flight tints colors, so sample rest
    times.

    Args:
        equations: One equation result or an iterable of equation results.
        at: Callable ``at(t)`` that poses the scene at time `t`. Supply with
            `times`, or omit both to check the current state once.
        times: Sample times in seconds, checked in sorted order.
        tol: Largest allowed per-channel RGB difference, also used for
            neutral tones and gray channel spread.

    Returns:
        A list of `GroundingFault` objects, each equation, term and kind
        reported only at its first sample time. Results are sorted by time
        (None for the current state first), then term name.

    Raises:
        ValueError: If only one of `at` and `times` is supplied.
    """
    if (at is None) != (times is None):
        raise ValueError("at and times must be supplied together")
    samples = [None] if times is None else sorted(float(t) for t in times)
    equations = [equations] if hasattr(equations, "bindings") else list(equations)
    terms = [
        (i, name, term, equation.bindings.get(name))
        for i, equation in enumerate(equations)
        for name, term in equation.artists["terms"].items()
    ]
    found = {}
    first_term = {}
    first_mark = {}

    def record(i, name, kind, time_s, detail):
        when = "in the current state" if time_s is None else f"at {time_s:g} s"
        found.setdefault(
            (i, name, kind),
            GroundingFault(
                kind, name, time_s, f"Term {name!r} in equation {i} {detail} {when}."
            ),
        )

    def differs(a, b):
        return any(abs(x - y) > tol for x, y in zip(a, b, strict=True))

    for k, time_s in enumerate(samples):
        if at is not None:
            at(time_s)
        for i, name, term, binding in terms:
            key = (i, name)
            mark_shown = binding is not None and _grounding_mark_shown(binding.mark)
            if mark_shown:
                first_mark.setdefault(key, k)
            glyphs = term._pieces
            shown = any(
                g.get_visible() and _alpha(g, g.get_facecolor()) >= _SHOWN
                for g in glyphs
            )
            if not shown:
                continue
            first_term.setdefault(key, k)
            rgb = to_rgb(glyphs[0].get_facecolor())
            colored = _grounding_colored(rgb, tol)
            if colored and not mark_shown:
                detail = (
                    "is colored but unbound"
                    if binding is None
                    else "names a mark that is not shown"
                )
                record(i, name, "no-referent", time_s, detail)
            if binding is None:
                continue
            motif = binding.motif
            expected = (
                to_rgb(motif.resolve_label())
                if motif is not None
                else _binding_rgb(binding.mark)
            )
            mismatches = []
            if differs(rgb, expected):
                mismatches.append(
                    f"has color {to_hex(rgb)} instead of {to_hex(expected)}"
                )
            if motif is not None:
                # A motif also permits marks that have no readable color.
                try:
                    mark_rgb = _binding_rgb(binding.mark)
                except ValueError:
                    mark_rgb = None
                if mark_rgb is not None:
                    expected_mark = to_rgb(motif.resolve())
                    if differs(mark_rgb, expected_mark):
                        label = binding.mark.get_label() or type(binding.mark).__name__
                        mismatches.append(
                            f"names mark {label!r} with color {to_hex(mark_rgb)} "
                            f"instead of {to_hex(expected_mark)}"
                        )
            if mismatches:
                record(i, name, "color-mismatch", time_s, " and ".join(mismatches))

    if len(samples) >= 2:
        for i, name, _, binding in terms:
            key = (i, name)
            if binding is not None and key in first_term:
                k = first_term[key]
                if k < first_mark.get(key, len(samples)):
                    record(
                        i,
                        name,
                        "symbol-first",
                        samples[k],
                        "is shown before its mark is first shown",
                    )
    return sorted(
        found.values(),
        key=lambda f: (
            f.time_s is not None,
            0 if f.time_s is None else f.time_s,
            f.term,
        ),
    )


def assert_grounded(equations, **kwargs) -> list:
    """Return [] when grounded, otherwise raise `GroundingError` with all faults.

    Apply the rules of `grounding_faults`: a term is shown when any glyph is
    visible with effective opacity at least 0.05; its first glyph decides
    its color. Text-colored terms and neutral tones, the ramp from the axes
    background to the text color that `ep.neutral` draws, are never colored.
    Neutral tones include colors within `tol` per channel of the ramp and
    grays whose channel spread is at most `tol`.

    A shown, colored term without a shown bound mark reports no-referent.
    A shown, bound term whose color differs from its expected mark color or
    motif label ink reports color-mismatch, as does a readable mark color
    that differs from its motif color. With at least two samples, a bound
    term first shown before its mark reports symbol-first, including a mark
    never shown.

    Args:
        equations: One equation result or an iterable of equation results.
        **kwargs: Passed to `grounding_faults`.

    Returns:
        An empty list when no grounding faults are found.

    Raises:
        GroundingError: If any fault is found; its `faults` attribute holds
            the list and its message names every term and kind.
        ValueError: If only one of `at` and `times` is supplied.
    """
    found = grounding_faults(equations, **kwargs)
    if found:
        raise GroundingError(found)
    return found


def _traced(fig, extra=()):
    """Artists whose position is traced: text, patches, collections and lines.

    `extra` adds artists of other kinds, such as the images a move declares.
    """
    skip = {id(fig.patch)} | {id(ax.patch) for ax in fig.axes}
    done = set()
    for artist in _walk(fig):
        if id(artist) in skip or isinstance(artist, Spine):
            continue
        if isinstance(artist, (Text, Patch, Collection, Line2D)) and (
            _ink(artist) >= _SHOWN
        ):
            done.add(id(artist))
            yield artist
    for artist in extra:
        if id(artist) not in done and artist.get_visible() and _ink(artist) >= _SHOWN:
            yield artist


def _collection_points(artist, dpi=72.0):
    """(display points, pad around each) that mark where a collection sits.

    Paths, offsets and marker sizes pair the way the renderer pairs them: the
    i-th element uses the i-th path and offset, each cycling when shorter.
    """
    offsets = np.ma.filled(np.ma.asarray(artist.get_offsets(), dtype=float), np.nan)
    shifts = artist.get_offset_transform().transform(offsets) if offsets.size else None
    if isinstance(artist, PathCollection) and shifts is not None:
        sizes = np.asarray(artist.get_sizes(), dtype=float)
        if sizes.size:
            pad = np.sqrt(np.resize(sizes, len(shifts))) * dpi / 144.0
        else:
            pad = np.zeros(len(shifts))
        return shifts, pad
    shifts = np.zeros((1, 2)) if shifts is None else shifts
    paths = [artist.get_transform().transform(p.vertices) for p in artist.get_paths()]
    if not paths:
        return np.empty((0, 2)), np.zeros(0)
    if len(shifts) == 1:
        points = np.concatenate(paths) + shifts[0]
    elif len(paths) == 1:
        points = (paths[0][None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    else:
        count = max(len(paths), len(shifts))
        points = np.concatenate(
            [paths[i % len(paths)] + shifts[i % len(shifts)] for i in range(count)]
        )
    return points, np.zeros(len(points))


def _collection_box(artist, dpi):
    """Window extent of a collection from its data, for those that report none."""
    pad = 0.0
    widths = np.atleast_1d(artist.get_linewidths())
    if widths.size:
        pad = float(np.max(widths)) * dpi / 144.0
    points, around = _collection_points(artist, dpi)
    keep = np.isfinite(points).all(axis=1)
    points, around = points[keep], around[keep] + pad
    if not len(points):
        return None
    return Bbox(
        [(points - around[:, None]).min(axis=0), (points + around[:, None]).max(axis=0)]
    )


def _polygon_centroid(path):
    """Area centroid of a transformed path, the vertex mean when it has no area."""
    total, weighted = 0.0, np.zeros(2)
    for poly in path.to_polygons(closed_only=False):
        if len(poly) < 3:
            continue
        x, y = poly[:, 0], poly[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cross = x * y1 - x1 * y
        area = cross.sum() / 2.0
        if abs(area) < 1e-12:
            continue
        total += area
        weighted += np.array([((x + x1) * cross).sum(), ((y + y1) * cross).sum()]) / 6.0
    if abs(total) > 1e-9:
        return weighted / total
    vertices = path.vertices[np.isfinite(path.vertices).all(axis=1)]
    return vertices.mean(axis=0)


def _ink_centroid(artist, box):
    """Where the artist's ink sits: unlike a box center, it turns with a rotation."""
    try:
        if isinstance(artist, Patch):
            # Patch.get_transform already includes the patch transform.
            path = artist.get_transform().transform_path(artist.get_path())
            return _polygon_centroid(path)
        if isinstance(artist, Line2D):
            xy = artist.get_transform().transform(artist.get_xydata())
            xy = xy[np.isfinite(xy).all(axis=1)]
            seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            if artist.get_linestyle() not in ("None", "none", "", " ") and (
                seg.sum() > 1e-9
            ):
                mid = (xy[1:] + xy[:-1]) / 2.0  # resampling a line keeps its centroid
                return (mid * seg[:, None]).sum(axis=0) / seg.sum()
            return xy.mean(axis=0)
        if isinstance(artist, Collection):
            xy = _collection_points(artist)[0]
            return xy[np.isfinite(xy).all(axis=1)].mean(axis=0)
    except (ValueError, AttributeError, TypeError, IndexError):
        pass
    return _center(box)


def _trace_box(artist, renderer):
    """(box, clip box, ink, ink centroid) of the artist now, None when unmeasurable."""
    try:
        box = (
            _bbox(artist, renderer)
            if isinstance(artist, Text)
            else artist.get_window_extent(renderer)
        )
    except (ValueError, AttributeError, TypeError):
        box = None
    if isinstance(artist, Collection) and (box is None or not _finite(box)):
        try:
            box = _collection_box(artist, renderer.dpi)
        except (ValueError, AttributeError, TypeError, IndexError):
            box = None
    if box is None or not _finite(box):
        return None
    clip = artist.get_clip_box() if artist.get_clip_on() else None
    if clip is not None and not _finite(clip):
        clip = None
    clip = None if clip is None else _tuple(clip)
    box = _tuple(box)
    return box, clip, _ink(artist), _ink_centroid(artist, box)


def _declared(move, key, default=None):
    """Field `key` of a declared move: a dict, or an object holding a contract."""
    contract = move if isinstance(move, dict) else getattr(move, "contract", move)
    if key == "artists":
        default = _declared(move, "moves", default)
    if isinstance(contract, dict):
        value = contract.get(key, default)
    else:
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


_WINDOW_S = 0.1  # an artist is moving when it travels over this much clip time


class _Trace:
    """What a sampled clip leaves behind: times, frames and artist boxes."""

    def __init__(self, times, keep_frames=True):
        self.times = list(times)
        self.keep_frames = keep_frames
        self.frames = []
        self.boxes = {}  # artist -> {sample index: (box, clip, ink, centroid)}
        self.order = []  # artists in order of first appearance
        self.size = (0.0, 0.0)
        self.ticks = set()  # tick labels, which trade places as limits move
        self.tick_text = {}  # tick label -> {sample index: its string}
        self.shapes = {}  # artist -> {sample index: its outlines, display pixels}

    def add_shapes(self, k, artists):
        """Record the drawn outline of each visible artist in `artists`."""
        for artist in artists:
            if artist.get_visible() and artist.get_alpha() != 0:
                outline = _display_outline(artist)
                if outline:
                    self.shapes.setdefault(artist, {})[k] = outline

    def add(self, k, fig, renderer, extra=()):
        fig.canvas.draw()
        if self.keep_frames:
            self.frames.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        self.size = (float(fig.bbox.width), float(fig.bbox.height))
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                self.ticks.add(axis.offsetText)
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    self.ticks.update((tick.label1, tick.label2))
        for artist in _traced(fig, extra):
            got = _trace_box(artist, renderer)
            if got is None:
                continue
            if artist not in self.boxes:
                self.boxes[artist] = {}
                self.order.append(artist)
            self.boxes[artist][k] = got
            if artist in self.ticks:
                self.tick_text.setdefault(artist, {})[k] = artist.get_text()

    def box(self, artist, k):
        got = self.boxes.get(artist, {}).get(k)
        return None if got is None else got[0]

    def seen(self, artist, k):
        """The box of the artist at sample k after its clip, None when invisible."""
        got = self.boxes.get(artist, {}).get(k)
        if got is None:
            return None
        return got[0] if got[1] is None else _intersect(got[0], got[1])

    def moving(self, artist, k, tol):
        """True when the artist would travel more than `tol` in `_WINDOW_S`.

        The test is a speed over elapsed clip time, so it gives the same
        answer at any frame rate.
        """
        here = self.box(artist, k)
        if here is None:
            return False
        for direction in (-1, 1):
            j = k + direction
            while 0 <= j < len(self.times):
                dt = abs(self.times[j] - self.times[k])
                if dt > _WINDOW_S + 1e-9 and abs(j - k) > 1:
                    break
                there = self.box(artist, j)
                if (
                    there is not None
                    and dt > 0
                    and _edges_move(here, there) * _WINDOW_S / dt > tol
                ):
                    return True
                j += direction
        return False

    def label(self, artist):
        if isinstance(artist, Text):
            return _name(artist)
        tag = artist.get_gid() or artist.get_label()
        tag = "" if not tag or str(tag).startswith("_") else f" {tag!r}"
        if artist not in self.boxes:
            return f"{type(artist).__name__}{tag}"
        index = [a for a in self.order if type(a) is type(artist)].index(artist)
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


def _changed(before, after, mask=None):
    """Box, in output pixels from the lower left, of what changed between frames.

    Pixels where `mask` is True are left out.
    """
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    if mask is not None:
        diff[mask] = 0
    rows, cols = np.nonzero(diff)
    if not len(rows):
        return None
    return (
        float(cols.min()),
        float(diff.shape[0] - 1 - rows.max()),
        float(cols.max() + 1),
        float(diff.shape[0] - rows.min()),
    )


def _check_teleports(trace, tol, jump_ratio):
    """Jumps of one artist, for a clip too still for the pixel baseline to exist."""
    found = []
    for artist in trace.order:
        if artist in trace.ticks and len(set(trace.tick_text[artist].values())) > 1:
            continue  # a tick label that changes its string is trading places
        ks = sorted(trace.boxes[artist])
        first = trace.first_box(artist)
        limit = max(
            jump_ratio * tol, 0.5 * max(first[2] - first[0], first[3] - first[1])
        )
        flagged = False
        steps = [
            _edges_move(trace.box(artist, a), trace.box(artist, b))
            if b == a + 1
            else 0.0
            for a, b in itertools.pairwise(ks)
        ]
        for n, (_, b) in enumerate(itertools.pairwise(ks)):
            step = steps[n]
            after = steps[n + 1] if n + 1 < len(steps) else None
            before = steps[n - 1] if n else after
            after = before if after is None else after
            # Steady travel has a neighbor as long as the step; a hop has none.
            hit = step > limit and min(before, after) < 0.5 * step
            if hit and not flagged:
                detail = f"{trace.label(artist)} jumps {step:.0f} px in one frame"
                found.append(
                    (
                        b,
                        "jump",
                        artist,
                        None,
                        trace.box(artist, b),
                        detail,
                        step / limit,
                    )
                )
            flagged = hit
    return found


def _draw_cuts(trace, drawn, step_ks, captions=()):
    """Masks of the draw group's boxes at each declared step where it changes.

    Returns:
        ``{k: mask}`` for each step ending at sample `k` where the pixels
        inside the group's boxes (at ``k - 1`` and `k`, padded) change; the
        mask is a boolean array of the frame's rows and columns.
    """
    cuts = {}
    if not drawn:
        return cuts
    height, width = trace.frames[0].shape[:2]
    # Resolve each artist's crops once for the group. Comparing these small
    # rectangles avoids gathering whole frames through a boolean mask.
    regions, caption_regions = {}, {}
    for artists, target in ((drawn, regions), (captions, caption_regions)):
        for artist in artists:
            for k in trace.boxes.get(artist, {}):
                x0, y0, x1, y1 = trace.box(artist, k)
                rows = slice(
                    max(int(height - y1 - 2), 0), max(int(height - y0 + 2) + 1, 0)
                )
                cols = slice(max(int(x0 - 2), 0), max(int(x1 + 2) + 1, 0))
                target.setdefault(k, []).append((rows, cols))
    for k in sorted(step_ks):
        if k < 1:
            continue
        crops = regions.get(k - 1, []) + regions.get(k, [])
        if not any(
            not np.array_equal(trace.frames[k - 1][crop], trace.frames[k][crop])
            for crop in crops
        ):
            continue
        mask = np.zeros((height, width), dtype=bool)
        # Captions share the mask at the draw's cuts, but their own arrival
        # or other continuous changes do not create cuts or count as hops.
        for crop in crops + caption_regions.get(k - 1, []) + caption_regions.get(k, []):
            mask[crop] = True
        cuts[k] = mask
    return cuts


def _check_jump(trace, jump_ratio, rest_times, tol, cuts=None, group=()):
    size = _steps(trace.frames)
    for k, mask in (cuts or {}).items():
        # Outside the draw group only: its own cut is by design.
        diff = np.abs(
            trace.frames[k].astype(np.int16) - trace.frames[k - 1].astype(np.int16)
        )
        diff[mask] = 0
        size[k - 1] = float(diff.mean())
    moving = [i for i, v in enumerate(size) if v > _STEP_FLOOR]
    if len(moving) < 3:
        return _check_teleports(trace, tol, jump_ratio)
    found = _jumps(trace, size, moving, jump_ratio, rest_times, tol, cuts or {})
    if cuts:
        # The mask blinds the pixel baseline inside the group's boxes, so an
        # artist outside the group that sits there is followed on its own.
        found += [
            f
            for f in _check_teleports(trace, tol, jump_ratio)
            if f[0] in cuts and f[2] not in group and _in_mask(trace, f[2], f[0], cuts)
        ]
    return found


def _in_mask(trace, artist, k, cuts):
    """Whether `artist` overlaps the draw group's mask at the cut ending at `k`."""
    mask = cuts[k]
    height = mask.shape[0]
    for j in (k - 1, k):
        box = trace.box(artist, j)
        if box is None:
            continue
        rows = slice(max(int(height - box[3]), 0), max(int(height - box[1]) + 1, 0))
        cols = slice(max(int(box[0]), 0), max(int(box[2]) + 1, 0))
        if mask[rows, cols].any():
            return True
    return False


def _jumps(trace, size, moving, jump_ratio, rest_times, tol, cuts=None):
    median = float(np.median([size[i] for i in moving]))
    last = len(size) - 1
    if rest_times:
        entries = set()
        for t in rest_times:
            k = next((k for k, s in enumerate(trace.times) if s >= t), None)
            if k is not None and k >= 1:
                entries.add(min(k - 1, last))
    if not rest_times or not entries:
        entries = {moving[-1]}
    found = []
    for i, value in enumerate(size):
        into_rest = (
            i in entries
            and i > 0
            and value > _REST_STEP_RATIO * jump_ratio * median
            and value > _REST_GROWTH * size[i - 1]
        )
        if not into_rest:
            if value <= jump_ratio * median:
                continue
            near = [size[j] for j in (i - 1, i + 1) if 0 <= j <= last]
            if near and value <= _PAIR_ISOLATION * max(near):
                continue
        # An 8-bit fade can hold, then change a whole region by one level.
        # Require excess beyond that per-channel rounding allowance.
        diff = np.abs(
            trace.frames[i + 1].astype(np.int16) - trace.frames[i].astype(np.int16)
        )
        if i + 1 in (cuts or {}):
            diff[cuts[i + 1]] = 0
        excess = float(np.where(diff > 1, diff, 0).mean())
        limit = jump_ratio * median * (_REST_STEP_RATIO if into_rest else 1.0)
        if excess <= limit:
            continue
        where = "into the rest state" if i in entries else "in one frame"
        detail = (
            f"the picture changes {value:.2f} levels {where}, "
            f"{value / median:.1f} times the median step of {median:.2f}"
        )
        box = _changed(trace.frames[i], trace.frames[i + 1], (cuts or {}).get(i + 1))
        box = box or (0.0, 0.0, *trace.size)
        found.append((i + 1, "jump", None, None, box, detail, value / median))
    return found


def _check_dead(trace, moves):
    found = []
    cache = {}  # moves that watch one region over one span are checked once
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
        name = _declared(move, "name", "a move")
        artists = list(_declared(move, "artists") or ())
        crop = (slice(None), slice(None))
        region = None
        if artists:
            seen = [a for a in artists if a in trace.boxes]
            if not seen:
                detail = f"{name!r} declares artists that are never visible"
                found.append(
                    (first, "dead", None, None, (0.0, 0.0, *trace.size), detail, None)
                )
                continue
            boxes = [b[0] for a in seen for b in trace.boxes[a].values()]
            region = (
                min(b[0] for b in boxes) - 2.0,
                min(b[1] for b in boxes) - 2.0,
                max(b[2] for b in boxes) + 2.0,
                max(b[3] for b in boxes) + 2.0,
            )
            height = trace.frames[first].shape[0]
            crop = (
                slice(
                    max(int(height - region[3]), 0), max(int(height - region[1]) + 1, 0)
                ),
                slice(max(int(region[0]), 0), max(int(region[2]) + 1, 0)),
            )
        key = (crop[0].start, crop[0].stop, crop[1].start, crop[1].stop, first, stop)
        if key not in cache:
            ref = trace.frames[first][crop].astype(np.int16)
            cache[key] = False
            for k in range(first + 1, stop + 1):
                now = trace.frames[k][crop].astype(np.int16)
                if now.size and np.abs(now - ref).max() > 0:
                    cache[key] = True
                    break
        if not cache[key]:
            detail = (
                f"{name!r} changes no pixel between t={times[first]:g} "
                f"and t={times[stop]:g}"
            )
            box = region or (0.0, 0.0, *trace.size)
            found.append((first, "dead", None, None, box, detail, None))
    return found


def _live(trace, artist, what, lo, hi):
    own = sorted(trace.boxes.get(artist, {}))
    if not own:
        raise ValueError(f"the {what} artist {artist!r} is not visible at any sample")
    return [k for k in own if lo <= trace.times[k] <= hi]


def _check_drift(trace, rest, tol):
    found = []
    for artist, lo, hi in rest:
        ks = _live(trace, artist, "rest", lo, hi)
        if not ks:
            continue  # the declaration lies outside the sampled clip
        home = trace.boxes[artist][ks[0]]
        worst, first = 0.0, None
        for k in ks:
            here = trace.boxes[artist][k]
            off = max(
                _edges_move(here[0], home[0]), float(np.hypot(*(here[3] - home[3])))
            )
            if off > tol and first is None:
                first = k
            worst = max(worst, off)
        if first is not None:
            detail = f"{trace.label(artist)} should stay still and moves {worst:.1f} px"
            found.append(
                (first, "drift", artist, None, trace.box(artist, first), detail, worst)
            )
    return found


def _check_tear(trace, groups, tol):
    found = []
    for group, lo, hi in groups:
        scales = False
        if isinstance(group, dict):
            scales = bool(group.get("scales", False))
            group = group["artists"]
        group = list(group)
        if len(group) < 2:
            continue
        common = None
        for artist in group:
            ks = set(_live(trace, artist, "group", lo, hi))
            common = ks if common is None else common & ks
        common = sorted(common)
        if len(common) < 2:
            continue
        pairs = np.array(list(itertools.combinations(range(len(group)), 2)))

        def dists(k, group=group, pairs=pairs):
            c = np.array([trace.boxes[a][k][3] for a in group])
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
            found.append((k, "tear", a, b, trace.box(a, k), detail, worst))
    return found


def _display_outline(artist):
    """The drawn outline of a patch or line as closed runs of display pixels."""
    from matplotlib.path import Path

    if isinstance(artist, (Patch, Line2D)):
        path = artist.get_path()
        move = artist.get_transform()  # a patch's already holds its own
    else:
        raise TypeError(
            f"{type(artist).__name__} cannot be declared rectilinear; "
            "give a patch or a line"
        )
    runs, run = [], []
    for vertices, code in path.iter_segments(transform=move, simplify=False):
        if code == Path.CLOSEPOLY:
            if run:
                runs.append(np.array(run))
            run = []
            continue
        if code == Path.MOVETO and run:
            runs.append(np.array(run))
            run = []
        run.append(vertices[-2:])
    if run:
        runs.append(np.array(run))
    return [r for r in runs if len(r) >= 2]


_SHAPE_EPS = 1e-3  # pixels; repeats and collinear points, far below any tolerance


def _corners(run):
    """The corners of a run, dropping repeated and collinear points."""
    run = np.asarray(run, dtype=float)
    keep = [run[0]]
    for point in run[1:]:
        if np.abs(point - keep[-1]).max() > _SHAPE_EPS:
            keep.append(point)
    if len(keep) > 1 and np.abs(keep[0] - keep[-1]).max() <= _SHAPE_EPS:
        keep.pop()
    out = []
    n = len(keep)
    for i in range(n):
        before, here, after = keep[i - 1], keep[i], keep[(i + 1) % n]
        # A point on the straight line between its neighbours is not a corner.
        cross = abs(
            (here[0] - before[0]) * (after[1] - here[1])
            - (here[1] - before[1]) * (after[0] - here[0])
        )
        reach = max(np.hypot(*(after - before)), 1e-9)
        if cross / reach > _SHAPE_EPS:
            out.append(here)
    return out, keep


def _slant(run):
    """Largest pixel offset of a run from an axis-aligned rectangle.

    A run with no corner at all is a straight segment, a rectangle with no
    thickness: it is off axis by the smaller of its two extents.
    """
    corners, points = _corners(run)
    if not corners:
        span = np.ptp(np.asarray(points), axis=0)
        return float(min(span))
    if len(corners) != 4:
        return float("inf")
    worst = 0.0
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        worst = max(worst, min(abs(a[0] - b[0]), abs(a[1] - b[1])))
    return worst


def _check_shape_class(trace, pieces, tol):
    found = []
    for artist, lo, hi in pieces:
        samples = trace.shapes.get(artist, {})
        ks = [k for k in sorted(samples) if lo <= trace.times[k] <= hi]
        if not ks:
            raise ValueError(
                f"the rectilinear artist {artist!r} is not visible at any sample"
            )
        worst, first = 0.0, None
        for k in ks:
            off = max(_slant(run) for run in samples[k])
            if off > tol and first is None:
                first = k
            worst = max(worst, off)
        if first is not None:
            pts = np.vstack(samples[first])
            box = (*pts.min(axis=0), *pts.max(axis=0))
            size = (
                "is no longer a rectangle"
                if worst == float("inf")
                else f"slants by up to {worst:.1f} px"
            )
            detail = (
                f"{trace.label(artist)} should stay an axis-aligned rectangle "
                f"and {size}"
            )
            found.append((first, "shape-class", artist, None, box, detail, worst))
    return found


def _check_overlap(trace, candidates, allowed, tol, move_tol):
    found = []
    runs = {}
    # Cut moves can carry many captions, with an allowance for every pair.
    # Index once rather than scanning all pairs for each pair in every frame.
    allowances = {}
    for pair, lo, hi in allowed:
        allowances.setdefault(pair, []).append((lo, hi))
    cand = [a for a in trace.order if a in candidates]
    for k in range(len(trace.times)):
        here = [
            a
            for a in cand
            if k in trace.boxes[a]
            and trace.boxes[a][k][2] >= _OVERLAP_INK
            and trace.seen(a, k) is not None
            and trace.moving(a, k, move_tol)
        ]
        for i, a in enumerate(here):
            for b in here[i + 1 :]:
                pair = frozenset((a, b))
                t = trace.times[k]
                # A pair counts as moving when it travels within _WINDOW_S of
                # a sample, so an allowance over a span holds that far past
                # either end of it, where the span's own motion is seen.
                if any(
                    lo - _WINDOW_S <= t <= hi + _WINDOW_S
                    for lo, hi in allowances.get(pair, ())
                ):
                    continue
                x = _intersect(trace.seen(a, k), trace.seen(b, k))
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


def _is_reveal(trace, artist, tol):
    """True when the clip box only ever widens, and ends up uncovering the artist."""
    ks = sorted(trace.boxes[artist])
    clips = [trace.boxes[artist][k][1] for k in ks]
    removed = clips[-1] is None  # taking the clip off is the last step of a reveal
    while clips and clips[-1] is None:
        clips.pop()
    if (
        not clips
        or any(c is None for c in clips)
        or (len(set(clips)) < 2 and not removed)
    ):
        return False
    eps = 1e-6
    for a, b in itertools.pairwise(clips):
        if (
            b[0] > a[0] + eps
            or b[1] > a[1] + eps
            or b[2] < a[2] - eps
            or b[3] < a[3] - eps
        ):
            return False
    if removed:
        return True
    growth = (clips[0][0] - clips[-1][0]) + (clips[0][1] - clips[-1][1])
    growth += (clips[-1][2] - clips[0][2]) + (clips[-1][3] - clips[0][3])
    if growth <= tol:
        return False
    box = trace.boxes[artist][ks[-1]][0]
    return max(_clip_overshoot(box, clips[-1]).values()) <= tol


def _check_outside(trace, texts, tol):
    width, height = trace.size
    found = []
    for artist in trace.order:
        first, worst, sides, count = None, 0.0, {}, 0
        reveal = _is_reveal(trace, artist, tol)
        for k, (box, clip, _, _) in sorted(trace.boxes[artist].items()):
            seen = box
            if clip is not None:
                seen = _intersect(box, clip)
                if seen is None:
                    if artist in texts and not reveal:
                        over = {"clipped away": max(box[2] - box[0], box[3] - box[1])}
                        count += 1
                        first = k if first is None else first
                        worst = max(worst, over["clipped away"])
                        sides.update(over)
                    continue
            over = {
                s: v
                for s, v in _canvas_overshoot(seen, width, height).items()
                if v > tol
            }
            if clip is not None and artist in texts and not reveal:
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
            text = ", ".join(
                f"{s} by {v:.1f} px" if s != "clipped away" else s
                for s, v in sides.items()
            )
            detail = f"{trace.label(artist)} is cut off: {text} (at {count} sample(s))"
            found.append(
                (
                    first,
                    "outside",
                    artist,
                    None,
                    trace.box(artist, first),
                    detail,
                    worst,
                )
            )
    return found


def _spike_run(trace, artist, run, spike_px):
    """The spike fault of one unbroken stretch of an artist's samples, or None."""
    if artist in trace.ticks and len({trace.tick_text[artist][k] for k in run}) > 1:
        return None  # a tick label that changes its string is trading places
    boxes = np.array([trace.boxes[artist][k][0] for k in run])
    dims = boxes[:, 2:] - boxes[:, :2]
    # A dimension that grows or shrinks takes the center with it, so only the
    # other one can be followed; a change this small is not a shape change.
    free = np.ptp(dims, axis=0) <= spike_px
    if not free.any():
        return None
    c = ((boxes[:, :2] + boxes[:, 2:]) / 2.0)[:, free]
    size = float(dims[0].max())
    chord = c[-1] - c[0]
    length = float(np.linalg.norm(chord))
    rel = c - c[0]
    if length > 1e-9:
        t = np.clip(rel @ chord / length**2, 0.0, 1.0)
        off = np.linalg.norm(rel - t[:, None] * chord, axis=1)
    else:
        off = np.linalg.norm(rel, axis=1)
    hop = float(off.max())
    if hop > spike_px and hop > _HOP_RATIO * length:
        isolated = int((off > _HOP_RATIO * hop).sum()) <= 2
        if hop <= max(size, spike_px) or isolated:
            k = run[int(np.argmax(off > spike_px))]
            detail = (
                f"{trace.label(artist)} hops {hop:.1f} px off its path "
                f"while travelling {length:.1f} px"
            )
            return (k, "spike", artist, None, trace.box(artist, k), detail, hop)
    # Velocity is a change of position over elapsed time, shown per median frame.
    dt = np.diff([trace.times[k] for k in run])
    if (dt <= 0).any():
        return None
    v = np.diff(c, axis=0) / dt[:, None] * float(np.median(dt))
    speed = np.linalg.norm(v, axis=1)
    moving = speed[speed > 0.05]
    if len(moving) < 3:
        return None
    kink = np.linalg.norm(np.diff(v, axis=0), axis=1)
    floor = max(_KINK_FLOOR_PX, _KINK_RATIO * float(np.median(moving)))
    top = int(np.argmax(kink))
    if kink[top] > floor:
        k = run[int(np.argmax(kink > floor)) + 1]
        detail = (
            f"{trace.label(artist)} changes velocity by {kink[top]:.1f} px "
            f"in one frame ({kink[top] / np.median(moving):.1f} median speeds)"
        )
        return (
            k,
            "spike",
            artist,
            None,
            trace.box(artist, k),
            detail,
            float(kink[top]),
        )
    return None


def _check_spike(trace, spike_px):
    found = []
    for artist in trace.order:
        ks = sorted(trace.boxes[artist])
        runs, run = [], [ks[0]]
        for k in ks[1:]:
            if k == run[-1] + 1:
                run.append(k)
            else:
                runs.append(run)
                run = [k]
        runs.append(run)
        for run in runs:
            if len(run) < 6:
                continue
            got = _spike_run(trace, artist, run, spike_px)
            if got is not None:
                found.append(got)
    return found


def _sample_times(times, fps):
    if isinstance(times, Window):
        n = max(math.floor((times.stop - times.start) * fps + 1e-9), 0)
        out = [times.start + k / fps for k in range(n + 1)]
        if abs(out[-1] - times.stop) <= 1e-9:
            out[-1] = float(times.stop)
        elif out[-1] < times.stop:
            out.append(float(times.stop))
        return out
    return list(times)


def _sample_trace(fig, at, samples, *, dpi, keep_frames, extra=(), shaped=()):
    """Pose and draw the figure at each sample and return the `_Trace`."""
    old_dpi, old_canvas = fig.dpi, fig.canvas
    if dpi is not None:
        fig.set_dpi(dpi)
    if not hasattr(old_canvas, "buffer_rgba"):
        FigureCanvasAgg(fig)  # a canvas that cannot hand back pixels
    try:
        trace = _Trace(samples, keep_frames=keep_frames)
        for k, t in enumerate(samples):
            at(t)
            trace.add(k, fig, _renderer(fig), extra)
            if shaped:
                trace.add_shapes(k, shaped)
    finally:
        fig.set_canvas(old_canvas)
        if dpi is not None:
            fig.set_dpi(old_dpi)
    return trace


def motion_ink(fig, at, times, driven, *, venue=None, fps=None, dpi=None):
    """The share of the clip's changed pixels that belong to the driven artists.

    Each frame is compared with the one before it. A pixel counts as changed
    when any channel moves by more than `_INK_LEVEL` levels, and as driven when
    it lies within two pixels of the box of a driven artist, at either frame.
    The ratio is the driven changed pixels over all changed pixels, summed
    over the clip, so a frame that changes more weighs more. It says how much
    of what moves is the quantity the clip is about, and how much is
    furniture, arrivals and swaps. It is reported beside the motion checks
    and is never a pass or a fail.

    An artist is driven when the draw function sets it from the knob the clip
    is about: the dot that travels, the bar that grows, the readout that
    counts. Name them at the call site; the plan holds no artists.

    Args:
        fig: The figure of the clip.
        at: Callable ``at(t)`` that poses the figure at time `t`.
        times: Sample times, or a `Window` to sample at every frame.
        driven: The artists the knob drives.
        venue: Key of `VENUE_FPS` giving the frame rate of a `Window`.
        fps: Frame rate of a `Window`; overrides `venue`.
        dpi: Output dpi; None uses the figure's.

    Returns:
        A float in [0, 1], or None when no pixel changes.

    Raises:
        ValueError: If a driven artist is never visible, or there are fewer
            than two samples.
    """
    driven = list(driven)
    if fps is None:
        venue = "video" if venue is None else venue
        if venue not in VENUE_FPS:
            raise ValueError(f"unknown venue {venue!r}; known: {sorted(VENUE_FPS)}")
        fps = VENUE_FPS[venue]
    samples = _sample_times(times, fps)
    if len(samples) < 2:
        raise ValueError("the motion ink needs at least two samples")
    trace = _sample_trace(fig, at, samples, dpi=dpi, keep_frames=True)
    for artist in driven:
        if not trace.boxes.get(artist):
            raise ValueError(f"driven artist {artist!r} is not visible at any sample")
    changed = driving = 0
    for k in range(1, len(samples)):
        before = trace.frames[k - 1].astype(np.int16)
        after = trace.frames[k].astype(np.int16)
        moved = np.abs(after - before).max(axis=2) > _INK_LEVEL
        if not moved.any():
            continue
        mask = _boxes_mask(trace, driven, k, moved.shape)
        changed += int(moved.sum())
        driving += int((moved & mask).sum())
    return driving / changed if changed else None


def _boxes_mask(trace, artists, k, shape):
    """Boolean mask of the padded boxes of `artists` at samples ``k - 1`` and `k`."""
    height = shape[0]
    mask = np.zeros(shape, dtype=bool)
    for artist in artists:
        for j in (k - 1, k):
            box = trace.box(artist, j)
            if box is None:
                continue
            x0, y0, x1, y1 = box[0] - 2.0, box[1] - 2.0, box[2] + 2.0, box[3] + 2.0
            rows = slice(max(int(height - y1), 0), max(int(height - y0) + 1, 0))
            cols = slice(max(int(x0), 0), max(int(x1) + 1, 0))
            mask[rows, cols] = True
    return mask


def motion_faults(
    fig,
    at,
    times,
    *,
    moves=None,
    steps=(),
    draws=(),
    hops=(),
    rest=(),
    groups=(),
    may_overlap=(),
    rectilinear=(),
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
    be dense for the continuity checks to mean anything: pass a `Window` to use
    every frame at the venue frame rate, or list the frame times.

    Kinds of fault:

    - ``"jump"``: a step between consecutive frames that exceeds `jump_ratio`
      times the median non-zero step and stands out from the steps next to it.
      One 8-bit level per changed channel is allowed for fade quantization.
      The step into the rest state (the end of a declared move, else the last
      step that changes anything) is also reported when it is `jump_ratio / 2`
      times the median and twice the step before it, because a move settles
      into rest. A clip that is too still to have a median (fewer than three
      changing steps) reports an artist that moves further in one frame than
      `jump_ratio * tol_px` and half its own size.
    - ``"dead"``: a declared move whose frames show no pixel change over its
      span, within the region of its declared artists; declared artists that
      never show are dead too.
    - ``"drift"``: an artist of `rest` whose box or ink centroid moves by more
      than `tol_px`.
    - ``"tear"``: artists of a group whose pairwise distances between ink
      centroids change by more than `tol_px` (relative to the group's common
      scale factor when the group is declared to scale). A rigid rotation does
      not tear.
    - ``"overlap"``: two artists that both travel more than `tol_px` within a
      tenth of a second, whose visible (clipped) boxes intersect by more than
      `overlap_tol` pixels in both directions, neither pair being allowed.
      Texts are checked against texts; the artists a move declares are checked
      as well; an artist must be at least 0.3 opaque.
    - ``"outside"``: an artist that the canvas cuts, or a text (or declared
      artist) cut by its clip box or clipped away entirely, by more than
      `frame_tol` pixels. A clip box that only widens, and ends uncovering
      the artist, is a reveal, not a cut. An artist clipped to its axes may
      leave the axes: that is the clip working, and only text is reported.
    - ``"shape-class"``: a patch or line declared ``rectilinear`` (a fraction
      bar, a rule, a box) whose drawn outline stops being an axis-aligned
      rectangle: a corner count other than four, or a side slanted by more
      than `tol_px`. A rotated rectangle is not axis-aligned and is reported. A
      straight line is a rectangle with no thickness: it passes while it is
      horizontal or vertical and is reported when it tilts by more than
      `tol_px`.
    - ``"hop-rate"``: a hop group (see `hops`) drawn faster than 8 draws per
      second, or faster than its declared rate.
    - ``"spike"``: an artist that hops off its straight path by more than
      `spike_px` while travelling less than it hops, that makes a one-frame
      excursion, or that changes velocity (position over elapsed time) by
      several median speeds in one frame. A dimension that changes shape is
      not followed, and tick labels that change their text are skipped. A
      false alarm costs more than a miss here, so this kind is off unless
      `spikes` is True or it is named in `kinds`.

    Args:
        fig: The figure of the clip.
        at: Callable ``at(t)`` that poses the figure at time `t`.
        times: Sample times, or a `Window` to sample at every frame.
        moves: Declared moves, each a dict (or an object whose ``contract`` is
            a dict or has the same fields) with any of ``name``, ``span``
            (start, stop), ``artists`` (or ``moves``, those that may move),
            ``rest``, ``groups`` and ``may_overlap``; the last three hold only
            over the span (``may_overlap`` also a tenth of a second past
            either end, where the span's motion first and last counts as
            moving). None declares nothing; ``jump``, ``outside`` and the
            text ``overlap`` still run.
        steps: Sample times in seconds at which discontinuities are intended.
            A ``jump`` is omitted only when its destination sample time
            matches a declared time within an absolute tolerance of 1e-9 s.
            This includes steps into rest; all other kinds are unchanged.
            With a draw group declared, a step where the group changes
            excuses only the group (see `draws`).
        draws: The draw group: artists that change in a cut at the declared
            steps by design, such as the draws of a run of exchangeable
            draws (a declared move's ``draws`` adds to it). At a step where
            the group changes, the jump check looks at the picture outside
            the group's boxes (before and after the step, padded by two
            pixels): the group's own cut is excused, and any other artist
            that cuts at the same step is still reported, by the pixel
            baseline or, in a clip too still to have one, as an artist that
            jumps. Artists of the group may also overlap one another. Every
            other check holds for the group as for any artist. Text artists
            declared on the same move join its draw group as captions. A step where
            the group does not change (a scene cut) still excuses the whole
            picture. A change inside a group's box at its cut is not seen.
        hops: Hop groups: artists that redraw in a cut at any frame by design,
            such as exchangeable draws, each a dict ``{"artists": [...],
            "rate": draws_per_second}`` (a declared move's ``hops`` adds to
            them). The group's cuts are excused wherever they fall, with no
            declared step, and its artists are not reported as spikes. Text
            artists declared on the same move join its hop groups as captions
            at the hop artists' cuts, without counting as redraws themselves.
            The picture outside the group's boxes keeps every check, as does
            every other artist. The group is reported as a ``"hop-rate"``
            fault when the declared rate, or the rate at which the clip
            actually redraws it, is above 8 draws per second, or the clip
            redraws it faster than declared. Every excessive one-second
            window starting at an observed cut is reported.
        rest: Artists that must not move.
        groups: Groups that move rigidly: each a list of artists, or a dict
            ``{"artists": [...], "scales": True}`` for a group that scales.
        may_overlap: Pairs of artists allowed to overlap while moving.
        rectilinear: Patches or lines (a fraction bar, a rule, a box) that must
            stay axis-aligned rectangles; a declared move's ``rectilinear``
            holds over its span.
        tol_px: Motion, in pixels, that is not reported.
        jump_ratio: Largest step, as a multiple of the median step.
        overlap_tol: Largest overlap, in pixels, that is not reported.
        frame_tol: Largest cut by the canvas or clip, in pixels, not reported.
        spike_px: Smallest hop, in pixels, that is a spike.
        spikes: Include ``"spike"`` in the default kinds.
        kinds: The kinds to look for, a subset of `MOTION_KINDS`; None is
            every kind but ``"spike"``, or every kind with `spikes`.
        venue: Key of `VENUE_FPS` giving the frame rate of a `Window`; default
            ``"video"``.
        fps: Frame rate of a `Window`; overrides `venue`.
        dpi: Output dpi; None uses the figure's.

    Returns:
        A list of `MotionFault` ordered by time, then kind; empty when the
        clip is clean. The figure is left posed at the last time and at its
        original dpi.

    Raises:
        ValueError: If a kind or venue is unknown, a declared rest or group
            artist is never visible, or a hop group has no positive rate.

    Example::

        span = ept.Window(0.0, 2.0)
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
    _hop_groups(hops, moves)
    if fps is None:
        venue = "video" if venue is None else venue
        if venue not in VENUE_FPS:
            raise ValueError(f"unknown venue {venue!r}; known: {sorted(VENUE_FPS)}")
        fps = VENUE_FPS[venue]
    samples = _sample_times(times, fps)
    if len(samples) < 3:
        raise ValueError("a motion check needs at least three samples")
    extra = [
        a
        for move in moves or ()
        for a in (
            *(_declared(move, "artists") or ()),
            *(_declared(move, "rest") or ()),
        )
        if isinstance(a, AxesImage)
    ]
    shaped = [
        *rectilinear,
        *(a for move in moves or () for a in _declared(move, "rectilinear") or ()),
    ]
    trace = _sample_trace(
        fig,
        at,
        samples,
        dpi=dpi,
        keep_frames=bool({"jump", "dead", "hop-rate"} & set(kinds)),
        extra=extra,
        shaped=shaped if "shape-class" in kinds else (),
    )
    return _analyze(
        trace,
        kinds=kinds,
        moves=moves,
        steps=steps,
        draws=draws,
        hops=hops,
        rest=rest,
        groups=groups,
        may_overlap=may_overlap,
        rectilinear=rectilinear,
        tol_px=tol_px,
        jump_ratio=jump_ratio,
        overlap_tol=overlap_tol,
        frame_tol=frame_tol,
        spike_px=spike_px,
    )


def _cut_captions(move):
    """Text artists declared on the same move belong to its cut group."""
    return tuple(a for a in _declared(move, "artists") or () if isinstance(a, Text))


def _hop_groups(hops, moves):
    """The declared hop groups as ``[(artists, rate, captions)]``, checked.

    Raises:
        ValueError: If a group names no artists or no positive rate.
    """
    found = []
    declared = [
        *((h, ()) for h in hops or ()),
        *(
            (h, _cut_captions(m))
            for m in moves or ()
            for h in _declared(m, "hops") or ()
        ),
    ]
    for hop, captions in declared:
        artists = tuple(hop.get("artists") or ())
        rate = hop.get("rate")
        if not artists:
            raise ValueError("a hop group needs its artists")
        if not isinstance(rate, numbers.Real) or isinstance(rate, bool) or rate <= 0:
            raise ValueError(
                f"a hop group needs a positive draw rate in draws per second, "
                f"got {rate!r}"
            )
        found.append((artists, float(rate), captions))
    return found


def _check_hop_rate(trace, hop_groups, hop_cuts):
    """Hop groups drawn faster than the limit or than they declare.

    `hop_cuts` holds, per group, the sample indices at which its pixels change.
    """
    found = []
    for (artists, rate, _), ks in zip(hop_groups, hop_cuts, strict=True):
        times = [trace.times[k] for k in ks]
        end = 0
        windows = []
        for i, t in enumerate(times):
            while end < len(times) and times[end] < t + 1.0 - 1e-9:
                end += 1
            measured = float(end - i)
            if measured > _MAX_HOP_RATE or measured > rate + 1.0:
                windows.append((ks[i], measured))
        # An excessive declaration is a fault even without observed cuts.
        if not windows and rate > _MAX_HOP_RATE:
            windows = [(ks[0] if ks else 0, 0.0)]
        label = trace.label(artists[0])
        for first, measured in windows:
            if rate > _MAX_HOP_RATE:
                detail = (
                    f"{label} is declared at {rate:g} draws per second, "
                    f"above {_MAX_HOP_RATE:g}"
                )
            elif measured > _MAX_HOP_RATE:
                detail = (
                    f"{label} is redrawn {measured:g} times in a second, "
                    f"above {_MAX_HOP_RATE:g}"
                )
            else:
                detail = (
                    f"{label} is redrawn {measured:g} times in a second, "
                    f"declared {rate:g}"
                )
            box = trace.box(artists[0], first) if ks else None
            found.append(
                (
                    first,
                    "hop-rate",
                    artists[0],
                    None,
                    box or (0.0, 0.0, *trace.size),
                    detail,
                    max(rate, measured),
                )
            )
    return found


def _analyze(
    trace,
    *,
    kinds,
    moves,
    steps=(),
    draws=(),
    hops=(),
    rest,
    groups,
    may_overlap,
    rectilinear,
    tol_px,
    jump_ratio,
    overlap_tol,
    frame_tol,
    spike_px,
):
    """Run the checks of `kinds` on a finished `_Trace`."""
    moves = [] if moves is None else list(moves)
    everywhere = (-math.inf, math.inf)
    rests = [(a, *everywhere) for a in rest]
    group_list = [(g, *everywhere) for g in groups]
    allowed = [(frozenset(p), *everywhere) for p in may_overlap]
    pieces = [(a, *everywhere) for a in rectilinear]
    declared = set()
    group = set(draws)
    rest_times = []
    for move in moves:
        span = _declared(move, "span")
        lo, hi = everywhere if span is None else span
        if span is not None:
            rest_times.append(hi)
        rests += [(a, lo, hi) for a in _declared(move, "rest") or ()]
        group_list += [(g, lo, hi) for g in _declared(move, "groups") or ()]
        pieces += [(a, lo, hi) for a in _declared(move, "rectilinear") or ()]
        declared |= set(_declared(move, "artists") or ())
        allowed += [
            (frozenset(p), lo, hi) for p in _declared(move, "may_overlap") or ()
        ]
        own_draws = _declared(move, "draws") or ()
        group |= set(own_draws)
        if own_draws:
            group.update(_cut_captions(move))
    hop_groups = _hop_groups(hops, moves)
    hopped = {a for artists, _, captions in hop_groups for a in (*artists, *captions)}
    group |= hopped
    drawn = [a for a in trace.order if a in group]
    allowed += [(frozenset(p), *everywhere) for p in itertools.combinations(drawn, 2)]
    step_ks = {
        k
        for k, s in enumerate(trace.times)
        if any(math.isclose(s, t, rel_tol=0.0, abs_tol=1e-9) for t in steps)
    }
    cuts = (
        _draw_cuts(trace, [a for a in drawn if a not in hopped], step_ks)
        if drawn and trace.frames
        else {}
    )
    # A hop group cuts wherever it changes, and each cut is excused like a step.
    hop_cuts = []
    every = range(1, len(trace.times))
    for artists, _, captions in hop_groups if trace.frames else ():
        own = _draw_cuts(
            trace, [a for a in trace.order if a in artists], every, captions
        )
        hop_cuts.append(sorted(own))
        for k, mask in own.items():
            cuts[k] = cuts[k] | mask if k in cuts else mask
    hop_ks = {k for ks in hop_cuts for k in ks}
    found = []
    if "jump" in kinds:
        found += _check_jump(trace, jump_ratio, rest_times, tol_px, cuts, group)
    if "dead" in kinds:
        found += _check_dead(trace, moves)
    if "drift" in kinds:
        found += _check_drift(trace, rests, tol_px)
    if "tear" in kinds:
        found += _check_tear(trace, group_list, tol_px)
    texts = {a for a in trace.order if isinstance(a, Text)}
    if "overlap" in kinds:
        found += _check_overlap(trace, texts | declared, allowed, overlap_tol, tol_px)
    if "outside" in kinds:
        found += _check_outside(trace, texts | declared, frame_tol)
    if "spike" in kinds:
        found += [f for f in _check_spike(trace, spike_px) if f[2] not in hopped]
    if "hop-rate" in kinds:
        found += _check_hop_rate(
            trace,
            hop_groups,
            hop_cuts or [[] for _ in hop_groups],
        )
    if "shape-class" in kinds:
        found += _check_shape_class(trace, pieces, tol_px)

    # A declared step excuses its jump; at a cut of the draw group it excuses
    # only the group, whose boxes the pixel baseline already left out there.
    def excused(f):
        if f[1] != "jump":
            return False
        if f[2] in hopped and f[0] in hop_ks:
            return True  # a hop group's own cut, wherever it falls
        if f[0] not in step_ks:
            return False
        return not (f[0] in cuts and (f[2] is None or f[2] not in group))

    found = [f for f in found if not excused(f)]
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
        times: Sample times, or a `Window`.
        **kwargs: Passed to `motion_faults`.

    Raises:
        MotionError: If any fault is found; its `faults` attribute holds the
            list and its message lists each with time, kind and box.

    Example::

        ept.assert_motion(fig, clip.pose, ept.Window(0.0, 2.0), rest=[title])
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


TIMING_KINDS = (
    "speech-rate",
    "line-gap",
    "still-speech",
    "number-late",
    "number-contradicts",
    "late-first-cue",
    "early-last-cue",
    "undeclared-silence",
    "multiple-hero",
    "speech-in-hero",
)


@dataclass(frozen=True)
class TimingFault:
    """One timing problem in a laid-out narration.

    Attributes:
        kind: One of `TIMING_KINDS`.
        message: What was found, with times in seconds.
        key: Key of the line involved, or None.
        start_s: Start on the clip clock, in seconds, or None.
        end_s: End on the clip clock, in seconds, or None.
    """

    kind: str
    message: str
    key: str | None = None
    start_s: float | None = None
    end_s: float | None = None


class TimingError(AssertionError):
    """A narration has timing problems.

    Attributes:
        faults: The list of `TimingFault` findings, one per message line.
    """

    def __init__(self, faults):
        """Build a report listing every finding."""
        self.faults = faults
        lines = [f"{len(faults)} timing fault(s):"]
        lines += [f"  {f.kind}: {f.message}" for f in faults]
        super().__init__("\n".join(lines))


def _timing_speech_rate(narration, ceiling, short_ceiling, short_words):
    """Check whole-word pace, allowing noise in the rate of a short line."""
    rates = dict(speech_rate(narration).rates)
    found = []
    for beat in narration.beats:
        if beat.key not in rates:
            continue
        count = sum(any(ch.isalnum() for ch in w.text) for w in beat.words)
        limit = short_ceiling if count <= short_words else ceiling
        wpm = rates[beat.key]
        rounded = round(wpm) if math.isfinite(wpm) else wpm
        if rounded > limit:
            start, end = beat.words[0].start_s, beat.words[-1].end_s
            found.append(
                TimingFault(
                    "speech-rate",
                    f"line {beat.key!r}: {count} words at {rounded:g} words/min "
                    f"exceeds the {limit:g} words/min ceiling "
                    f"from {start:g} to {end:g} s",
                    beat.key,
                    start,
                    end,
                )
            )
    return found


def _timing_line_gap(narration, min_gap_s):
    """Check heard silence between spoken lines, so sentences stay separate."""
    from eyepiece.narration import Pause

    spoken = [
        b
        for b in narration.beats
        if b.words and not isinstance(narration.script.line(b.key), Pause)
    ]
    found = []
    for first, second in itertools.pairwise(spoken):
        start, end = first.words[-1].end_s, second.words[0].start_s
        gap = end - start
        if gap < min_gap_s:
            found.append(
                TimingFault(
                    "line-gap",
                    f"line {second.key!r}: {gap:g} s gap from {start:g} to "
                    f"{end:g} s is below the {min_gap_s:g} s floor",
                    second.key,
                    start,
                    end,
                )
            )
    return found


def _timing_silence(narration, plan):
    """Keep the silence and cue findings, so silent stretches are declared."""
    from eyepiece.narration import check_silence

    return [
        TimingFault(f.code, f.message, f.key, f.start_s, f.end_s)
        for f in check_silence(narration, plan)
    ]


def _source_frame(source, t):
    """Read one RGB uint8 frame at t, using the last video frame past its end."""
    if not isinstance(source, (str, os.PathLike)) and not callable(source):
        raise TypeError("source must be a video path or a callable source(t)")
    if callable(source) and not isinstance(source, (str, os.PathLike)):
        frame = np.asarray(source(t))
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("source(t) must return an (H, W, 3) or (H, W, 4) image")
        frame = frame[:, :, :3]
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.rint(frame * 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            raise TypeError("source(t) must return uint8 or floating-point pixels")
        return frame

    import io
    import subprocess
    import tempfile

    from PIL import Image

    from eyepiece._ffmpeg import resolve_ffmpeg

    command = [resolve_ffmpeg(), "-v", "error", "-nostdin"]
    result = subprocess.run(
        [
            *command, "-ss", str(t),
            "-i", os.fspath(source), "-map", "0:v:0", "-frames:v", "1",
            "-c:v", "png", "-f", "image2pipe", "pipe:1",
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip
    if not result.stdout:
        # A seek after the final frame can succeed without producing an image.
        # Overwrite a single PNG while decoding the tail to retain its last frame.
        with tempfile.TemporaryDirectory() as directory:
            last = Path(directory) / "last.png"
            # Sparse streams or a longer audio track can leave the tail empty.
            for seek in (["-sseof", "-1"], []):
                subprocess.run(
                    [
                        *command, *seek, "-i", os.fspath(source), "-map", "0:v:0",
                        "-vsync", "0", "-c:v", "png",
                        "-update", "1", "-y", str(last),
                    ],
                    check=True,
                    capture_output=True,
                )  # fmt: skip
                if last.exists():
                    with Image.open(last) as image:
                        return np.array(image.convert("RGB"))
        raise RuntimeError("video contains no decodable frames")
    with Image.open(io.BytesIO(result.stdout)) as image:
        return np.array(image.convert("RGB"))


def _source_frames(source, *, seconds, fps):
    """Yield (time, RGB uint8 frame) pairs from a callable or video path."""
    if not isinstance(source, (str, os.PathLike)) and not callable(source):
        raise TypeError("source must be a video path or a callable source(t)")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not isinstance(source, (str, os.PathLike)):
        for k in itertools.count():
            t = k / fps
            if t >= seconds:
                break
            yield t, _source_frame(source, t)
        return

    import subprocess
    import tempfile

    from eyepiece._ffmpeg import resolve_ffmpeg

    args = [resolve_ffmpeg(), "-v", "error", "-nostdin", "-i", os.fspath(source)]
    args += ["-map", "0:v:0"]
    height, width = _source_frame(source, 0.0).shape[:2]
    if width > 1280:
        height = max(2, round(height * 1280 / width / 2) * 2)
        width = 1280
    args += [
        "-vf", f"fps={fps},scale={width}:{height}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]  # fmt: skip
    # A file for stderr avoids blocking the frame pipe on decoder diagnostics.
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=errors) as process:
            try:
                size = height * width * 3
                for k in itertools.count():
                    data = process.stdout.read(size)
                    if not data:
                        break
                    if len(data) != size:
                        raise RuntimeError("ffmpeg returned an incomplete RGB frame")
                    yield (
                        k / fps,
                        np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3),
                    )
                if process.wait():
                    errors.seek(0)
                    raise subprocess.CalledProcessError(
                        process.returncode, args, stderr=errors.read()
                    )
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()


def _sheet_events(narration, plan, events):
    """Collect and merge tile labels without rounding the sampled time."""
    if events is None:
        if narration is None and plan is None:
            raise ValueError("provide events, narration or plan")
        events = []
        if narration is not None:
            events.extend((c.at_s, f"{c.kind}:{c.target}") for c in narration.cues())
            events.extend(
                (max(b.start_s, b.end_s - 0.001), "end") for b in narration.beats
            )
        if plan is not None:
            events.extend(
                (e.frame / plan.fps, e.name)
                for e in plan.events()
                if e.what in ("mid", "end") and e.kind not in ("hold", "line")
            )
    merged = {}
    for t, label in sorted(events, key=lambda event: event[0]):
        ms = round(t * 1000)
        if ms not in merged:
            merged[ms] = (t, [])
        merged[ms][1].append(str(label))
    if not merged:
        raise ValueError("events must not be empty")
    return [(t, ", ".join(labels)) for t, labels in merged.values()]


def _sheet_audio(audio):
    """Read PCM or numeric samples and average channels before normalizing."""
    if isinstance(audio, (str, os.PathLike)):
        import wave

        with wave.open(os.fspath(audio), "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
                raise ValueError("audio must be a 16-bit PCM WAV")
            rate = wav.getframerate()
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            samples = samples.reshape(-1, wav.getnchannels())
    else:
        samples, rate = audio
    samples = np.asarray(samples, dtype=float)
    if samples.ndim not in (1, 2):
        raise ValueError("audio samples must be one- or two-dimensional")
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("audio rate must be finite and positive")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    peak = np.max(np.abs(samples), initial=0)
    return samples / peak if peak else samples, rate


def _sheet_waveform(ax, audio, start, end, times, pixels):
    """Draw the linearly interpolated signal's min/max envelope and event ticks."""
    samples, rate = audio
    if start == end:
        start, end = start - 0.5, end + 0.5
    edges = np.linspace(start, end, max(2, int(pixels)) + 1)
    indices = np.clip(np.ceil(edges * rate), 0, len(samples)).astype(int)
    values = (
        np.interp(edges * rate, np.arange(len(samples)), samples, left=0, right=0)
        if len(samples)
        else np.zeros_like(edges)
    )
    low = np.minimum(values[:-1], values[1:])
    high = np.maximum(values[:-1], values[1:])
    for i, (a, b) in enumerate(itertools.pairwise(indices)):
        if b > a:
            low[i] = min(low[i], samples[a:b].min())
            high[i] = max(high[i], samples[a:b].max())
    centers = (edges[:-1] + edges[1:]) / 2
    ax.vlines(centers, low, high, color="#34495e", linewidth=0.8)
    ax.axhline(0, color="#999999", linewidth=0.5)
    for t in times:
        ax.axvline(t, color="#b04b31", linewidth=0.8)
    ax.set(xlim=(start, end), ylim=(-1.05, 1.05), yticks=[])
    ax.set_xlabel("Time (s)", fontsize=9, color="#222222")
    ax.tick_params(axis="x", labelsize=8, colors="#222222")
    for spine in ax.spines.values():
        spine.set_visible(False)


def review_sheet(
    path,
    source,
    *,
    narration=None,
    plan=None,
    events=None,
    audio=None,
    columns=6,
    width=320,
) -> Path:
    """A contact sheet shows a clip's frames at the moments that matter.

    A review needs no hand-picked preview list.

    Events default to every narration cue and each beat end minus 1 ms
    (clamped to its start), plus plan mid and end frames excluding hold and
    line spans. Times agreeing to the millisecond share a tile; a later cue
    sheet can pass its own ``events`` as ``(seconds, label)`` pairs.

    Args:
        path: Output PNG path. Missing parent directories are created.
        source: Video path decoded by ffmpeg, or a callable taking seconds
            and returning an (H, W, 3) or (H, W, 4) image as uint8 or floats
            in [0, 1]. Alpha is dropped. Times at or past the last decodable
            video frame use that frame. Images are area-averaged to the tile
            width before drawing, preserving aspect ratio to the nearest pixel.
        narration: Optional laid-out `eyepiece.narration.Narration`. Each beat
            gets a row with a header and active captions. Without narration,
            rows contain time ranges and tiles.
        plan: Optional `eyepiece.Plan` supplying default mid and end events.
        events: Optional iterable of (seconds, label) pairs overriding default
            events. Tiles are sorted by time and merged within a millisecond.
        audio: Optional 16-bit PCM WAV path or (samples, rate) pair, with rate
            in Hz and a one-dimensional or (samples, channels) numeric array.
            Channels are averaged and normalized before drawing a waveform
            and event ticks for each row's time range.
        columns: Maximum tiles per row before wrapping, default 6.
        width: Tile width in pixels, default 320.

    Returns:
        The output PNG path as a `pathlib.Path`.

    Raises:
        ValueError: If no events can be built, events are empty, or dimensions
            are not positive integers; if a callable image has an invalid
            shape; or if audio has an invalid shape, rate or WAV format.
        TypeError: If source is neither a video path nor a callable, or
            callable pixels are neither uint8 nor floating-point.
        RuntimeError: If ffmpeg is unavailable or a video has no decodable frames.
        subprocess.CalledProcessError: If ffmpeg cannot decode the video.
    """
    import textwrap
    from bisect import bisect_right

    from matplotlib.figure import Figure
    from PIL import Image

    from eyepiece.narration._mux import caption_at, captions

    if any(not isinstance(n, numbers.Integral) or n <= 0 for n in (columns, width)):
        raise ValueError("columns and width must be positive integers")
    events = _sheet_events(narration, plan, events)
    caption_cues = captions(narration) if narration is not None else []
    sound = _sheet_audio(audio) if audio is not None else None
    gap, margin, line_height = 16, 16, 16
    sheet_width = 2 * margin + columns * width + (columns - 1) * gap
    chars = max(1, int(width / 7.5))
    tiles = []
    for t, labels in events:
        frame = _source_frame(source, t)
        height = max(1, round(width * frame.shape[0] / frame.shape[1]))
        frame = np.array(
            Image.fromarray(frame).resize((width, height), Image.Resampling.BOX)
        )
        label = textwrap.fill(f"{t:.2f} s  {labels}", chars)
        caption = textwrap.fill(caption_at(caption_cues, t), chars)
        tiles.append(
            {
                "t": t,
                "frame": frame,
                "height": height,
                "label": label,
                "caption": caption,
                "label_height": line_height * len(label.splitlines()),
                "caption_height": line_height * len(caption.splitlines()),
            }
        )
    if narration is not None and narration.beats:
        rows = [dict(row, tiles=[]) for row in narration.audit()]
        starts = [beat.start_s for beat in narration.beats]
        for tile in tiles:
            rows[max(0, bisect_right(starts, tile["t"]) - 1)]["tiles"].append(tile)
    else:
        rows = [
            {
                "start_s": group[0]["t"],
                "end_s": group[-1]["t"],
                "tiles": group,
            }
            for i in range(0, len(tiles), columns)
            if (group := tiles[i : i + columns])
        ]
    # Lay out in pixels first so long captions and wrapped tiles reserve space.
    y = margin
    for row in rows:
        row["top"] = y
        y += 24
        row["bands"] = []
        for i in range(0, len(row["tiles"]), columns):
            band = row["tiles"][i : i + columns]
            label_height = max(tile["label_height"] for tile in band)
            image_height = max(tile["height"] for tile in band)
            caption_height = max(tile["caption_height"] for tile in band)
            row["bands"].append((y, label_height, image_height, band))
            y += label_height + image_height + caption_height + 12
        if sound is not None:
            row["wave_top"] = y
            y += 100
        y += gap
    sheet_height = math.ceil(y)
    fig = Figure(
        figsize=(sheet_width / 100, sheet_height / 100), dpi=100, facecolor="white"
    )
    canvas = FigureCanvasAgg(fig)

    def text(x, top, value, fontsize=9):
        fig.text(
            x / sheet_width,
            1 - top / sheet_height,
            value,
            ha="left",
            va="top",
            fontsize=fontsize,
            color="#222222",
            fontfamily="DejaVu Sans Mono",
            linespacing=1.28,
            usetex=False,
            parse_math=False,
        )

    def axes(x, top, w, h):
        return fig.add_axes(
            [
                x / sheet_width,
                1 - (top + h) / sheet_height,
                w / sheet_width,
                h / sheet_height,
            ],
            facecolor="white",
        )

    for row in rows:
        header = f"{row['start_s']:.2f} - {row['end_s']:.2f} s"
        if "key" in row:
            header = f"{row['key']}  {header}  {row['text']}"
        header = textwrap.shorten(
            header, max(4, int((sheet_width - 2 * margin) / 8.4)), placeholder="..."
        )
        text(margin, row["top"], header, fontsize=10)
        for top, label_height, image_height, band in row["bands"]:
            for col, tile in enumerate(band):
                x = margin + col * (width + gap)
                text(x, top, tile["label"])
                ax = axes(x, top + label_height, width, tile["height"])
                ax.imshow(tile["frame"], interpolation="nearest", aspect="auto")
                ax.set_axis_off()
                text(x, top + label_height + image_height + 4, tile["caption"])
        if sound is not None:
            ax = axes(margin, row["wave_top"], sheet_width - 2 * margin, 60)
            _sheet_waveform(
                ax,
                sound,
                row["start_s"],
                row["end_s"],
                [tile["t"] for tile in row["tiles"]],
                sheet_width - 2 * margin,
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_png(path)
    return path


def _timing_still_run(words, start, end, still_s, arrivals=()):
    """Find the longest speech group clipped to one still run."""
    if any(start <= t < end for t in arrivals):
        return []
    groups = []
    for word_start, word_end, line_index, key in words:
        if word_end <= start or word_start >= end:
            continue
        a, b = max(start, word_start), min(end, word_end)
        if not groups or a - groups[-1][1] >= 2.0:
            groups.append([a, b, {}])
        group = groups[-1]
        group[1] = max(group[1], b)
        group[2][line_index] = key
    if not groups:
        return []
    a, b, lines = max(groups, key=lambda g: g[1] - g[0])
    speech = b - a
    if speech <= still_s:
        return []
    keys = [lines[i] for i in sorted(lines)]
    return [
        TimingFault(
            "still-speech",
            f"still screen {start:.2f} to {end:.2f} s ({end - start:.1f} s) "
            f"under {speech:.1f} s of speech ({', '.join(keys)})",
            keys[0],
            start,
            end,
        )
    ]


def _timing_still(narration, source, plan, still_s, still_change, ignore_bottom, fps):
    """Compare each sample with the first frame of its still run."""
    words = sorted(
        (w.start_s, w.end_s, i, beat.key)
        for i, beat in enumerate(narration.beats)
        for w in beat.words
    )
    # The plan has arrival times but no artist geometry. Cue-linked arrivals
    # count independently of their pixel area, including the timeline's lead.
    cues = list(narration.cues())
    arrivals = [
        span.start_s
        for span in getattr(plan, "spans", {}).values()
        if span.kind in {"enter", "text", "caption", "write", "place", "set", "readout"}
        or (span.kind == "hold" and span.writes)
        if any(
            cue.line == span.section and -1e-9 <= cue.at_s - span.start_s <= 0.1 + 1e-9
            for cue in cues
        )
    ]
    seconds = plan.seconds if plan is not None else narration.total_s
    reference = None
    start = last = 0.0
    found = []
    for t, frame in _source_frames(source, seconds=seconds, fps=fps):
        height, width = frame.shape[:2]
        kept = frame[: int(height * (1 - ignore_bottom))].astype(np.int16)
        if reference is None:
            reference, start = kept, t
        else:
            changed = np.count_nonzero(np.any(np.abs(kept - reference) > 12, axis=2))
            if changed >= still_change * height * width:
                found += _timing_still_run(words, start, t, still_s, arrivals)
                reference, start = kept, t
        last = t
    if reference is not None:
        found += _timing_still_run(words, start, last + 1 / fps, still_s, arrivals)
    return found


_READOUT_NUMBER = re.compile(r"(?<![\d.])(?<!\d,)\d+(?:,\d{3})*(?:\.\d+)?(?!\d)")


_READOUT_SCIENTIFIC = re.compile(
    r"(?P<mantissa>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?:\\times|[xX]|\u00d7)\s*10\s*\^\s*"
    r"(?:\{(?P<braced>[+-]?\d+)\}|(?P<bare>[+-]?\d+))"
)


def _timing_numbers(narration, readouts, at, plan, settle_tol_s, fps):
    """Check readout text and ink against the number on the speech clock."""
    from eyepiece.narration._speech import _bare, _value_end, _word_key

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if plan is not None:
        settle_tol_s = max(settle_tol_s, 1 / plan.fps)
    found = []
    for word_key, artist in readouts.items():
        key, index = _word_key(word_key)
        beat = narration.beat(key)
        words = narration.script.line(key).words
        try:
            spoken, number_end = _value_end(words, index)
        except ValueError as error:
            raise ValueError(f"line {key!r} word {index}: {error}") from error
        phrase = " ".join(_bare(w) for w in words[index:number_end])
        spoken_scientific = "times ten to the" in phrase
        start, end = beat.words[index].start_s, beat.words[-1].end_s
        judged = start + settle_tol_s
        times = [judged]
        for k in itertools.count():
            t = start + k / fps
            if t >= end:
                break
            times.append(t)
        times.append(end)
        samples = {}
        for t in times:
            at(t)
            text = artist.get_text()
            ink = _ink(artist) if artist.get_visible() else 0.0
            token = _READOUT_NUMBER.search(text)
            value = float(token.group().replace(",", "")) if token is not None else None
            science = _READOUT_SCIENTIFIC.search(text)
            shown_scientific = science is not None and (
                token is None or science.start() <= token.start()
            )
            if shown_scientific:
                power = science["braced"] or science["bare"]
                value = float(f"{science['mantissa']}e{power}")
            matches = value is not None and math.isclose(
                value,
                spoken,
                rel_tol=1e-9 if spoken_scientific or shown_scientific else 0,
                abs_tol=0 if spoken_scientific or shown_scientific else 1e-9,
            )
            samples[t] = (text, ink, value is not None, matches)
        threshold = max(_SHOWN, 0.9 * max(s[1] for s in samples.values()))
        landing = next(
            (
                t
                for t in sorted(samples)
                if t >= start and samples[t][1] >= threshold and samples[t][3]
            ),
            None,
        )
        label = f"line {key!r} says {words[index]!r} ({spoken:g})"
        if samples[judged][1] < threshold or not samples[judged][3]:
            text, ink, _, _ = samples[start]
            shown = repr(text) if ink >= _SHOWN else "nothing"
            detail = (
                "does not show it during the line"
                if landing is None
                else f"lands at {landing:.2f} s, {landing - start:.2f} s late"
            )
            found.append(
                TimingFault(
                    "number-late",
                    f"{label} at {start:.2f} s but the readout shows {shown}; {detail}",
                    key,
                    start,
                    end if landing is None else landing,
                )
            )
        if landing is not None:
            for t in sorted(samples):
                text, ink, has_number, matches = samples[t]
                if landing < t <= end and ink >= _SHOWN and has_number and not matches:
                    found.append(
                        TimingFault(
                            "number-contradicts",
                            f"{label} but at {t:.2f} s the readout shows {text!r}, "
                            f"contradicting the spoken number {spoken:g}",
                            key,
                            t,
                        )
                    )
                    break
    return found


def timing_faults(
    narration,
    *,
    plan=None,
    source=None,
    at=None,
    readouts=None,
    ceiling=160.0,
    short_ceiling=166.0,
    short_words=10,
    min_gap_s=0.4,
    still_s=2.0,
    still_change=0.0016,
    ignore_bottom=0.0,
    fps=10.0,
    settle_tol_s=0.0,
) -> list[TimingFault]:
    """Check narration pace, gaps, silence, still screens and shown numbers.

    Speech rates come from `speech_rate` and are compared in whole words per
    minute: 160 wpm is the pace past which a viewer cannot follow a moving
    figure. Lines of at most `short_words` use `short_ceiling`, because a
    short line's rate is noisy. Set both ceilings equal to remove the allowance.

    Consecutive spoken lines need at least `min_gap_s` of silence between
    their words, because 0.4 s is the least pause that separates two sentences.
    Audio tails count as silence; `Pause` beats and lines without words are
    skipped when choosing the spoken pair.

    `eyepiece.narration.check_silence` runs with its defaults to check cue
    timing, undeclared silence and hero spans, so the screen and voice stay
    together and long silent stretches have a declared purpose.

    With `source`, frames are compared with the first frame of each still run.
    The bottom `ignore_bottom` fraction of rows is ignored. A pixel changes if
    any RGB channel differs by more than 12 out of 255. A run continues while
    fewer than `still_change` times the full frame's height times width pixels
    change; the first frame at or above that count ends the run at its time
    and starts the next. The final run ends one frame after the last sample.
    Comparing with the run's first frame lets slow drift accumulate.
    With `plan`, a declared arrival starting in the run on a narration cue
    (including the timeline's 0.1 s lead) makes the run non-still regardless
    of its area. Arrival kinds are enter, text, caption, write, place, set
    and readout; a hold that writes knobs also counts. Runs without such an
    arrival keep the pixel threshold.

    Every overlapping word is clipped to the run, and the words are ordered by time.
    A silence of at least 2.0 s from the latest word end to the next word
    start splits speech into groups. Speech is the longest group's span,
    from its first word start to its last word end, including shorter gaps.
    A run is reported only when that speech is strictly over `still_s`. The
    finding names the group's lines in clip order and is keyed by the first line.

    A 19 to 28 pt word changes about 900 to 2300 pixels of a 1280 x 720
    frame. This catches a screen where nothing arrives, not a small swell;
    a swell is the motion checks' job.

    With `readouts`, each number's artist is posed at its word start plus
    `settle_tol_s`, and at every 1 / `fps` from the word start through the
    last word's end, including the end itself. A readout's value is the first
    whole number in its text; numbers after it, such as a denominator or a
    year, are context. Readouts are written with their value first. Thousands
    commas and decimal parts are accepted. A whole numeric token is not
    preceded by a digit, a decimal point or a digit-comma, and is not followed
    by a digit. A readout shows its spoken number when its value matches with
    absolute tolerance 1e-9 and its ink is at least 0.05 and at least 90 percent
    of its largest ink over those poses. Invisible artists have no ink.
    A number that misses the judged time is reported as `number-late`.
    After landing and through the speech end, `number-contradicts` reports
    the first sample with ink at least 0.05 and a first number that differs
    from the spoken value beyond that tolerance. Scientific notation from
    `eyepiece.narration.scientific` is read as a whole quantity. If either
    the speech or the display uses scientific notation, matching uses a
    relative tolerance of 1e-9 with no absolute tolerance. Text with no number is not
    a contradiction. The figure is left posed at the last sample.

    Args:
        narration: A laid-out `eyepiece.narration.Narration`.
        plan: A `eyepiece.Plan` whose hero annotations declare silent spans;
            its `seconds` gives the clip length for a callable source.
        source: Optional video path (str or os.PathLike), decoded by ffmpeg
            through an RGB pipe until the stream ends, scaled to 1280 pixels
            wide when wider, keeping aspect with an even height, else native
            size. Or a callable source(t) returning an (H, W, 3) or (H, W, 4)
            array: uint8 or float in [0, 1], scaled by 255 and rounded. Alpha
            is dropped. Sample at t = k / fps, starting at zero, while t is
            below plan.seconds or, without a plan, narration.total_s.
        at: Callable that poses the figure at a clip time; required with `readouts`.
        readouts: Mapping from a number word, ``(line_key, word_index)`` or
            ``"line_key:word_index"``, to the `matplotlib.text.Text` showing it.
        ceiling: Fastest whole-word pace of a longer line, in words/min.
        short_ceiling: Fastest whole-word pace of a short line, in words/min.
        short_words: Largest spoken word count considered a short line.
        min_gap_s: Least heard silence between spoken lines, in seconds.
        still_s: Maximum speech span over one still run, default 2.0 seconds.
        still_change: Changed pixel fraction ending a run, default 0.0016
            (0.16 percent of the full frame, including ignored rows).
        ignore_bottom: Bottom fraction of rows to ignore, default 0.0; keep
            rows from zero up to int(H * (1 - ignore_bottom)).
        fps: Sampling rate, default 10 frames per second. Source frames start
            at zero; readout scans start at the number word.
        settle_tol_s: Seconds after the word start by which its readout must
            land, default zero. With `plan`, floored at one frame, 1 / plan.fps.

    Returns:
        A list of `TimingFault`, sorted by start time with None last and
        original rule order retained for equal times. Findings are not raised.

    Raises:
        TypeError: If `source` is neither a video path nor a callable.
        ValueError: If `readouts` lacks `at`, a word does not start a number,
            or a sampling rate is not finite and positive.
        KeyError: If a readout names a line the narration lacks.
    """
    if readouts is not None and at is None:
        raise ValueError("readouts requires at(t) to pose the figure")
    found = _timing_speech_rate(narration, ceiling, short_ceiling, short_words)
    found += _timing_line_gap(narration, min_gap_s)
    found += _timing_silence(narration, plan)
    if source is not None:
        found += _timing_still(
            narration, source, plan, still_s, still_change, ignore_bottom, fps
        )
    if readouts is not None:
        found += _timing_numbers(narration, readouts, at, plan, settle_tol_s, fps)
    return sorted(found, key=lambda f: (f.start_s is None, f.start_s or 0.0))


def assert_timing(narration, **kwargs) -> list[TimingFault]:
    """Raise `TimingError` if narration pace, gaps or silence break a rule.

    Args:
        narration: A laid-out `eyepiece.narration.Narration`.
        **kwargs: Passed to `timing_faults`.

    Returns:
        An empty list when the narration has no timing findings.

    Raises:
        TimingError: If any finding is returned; its `faults` attribute holds
            the list and its message lists every finding on its own line.
    """
    found = timing_faults(narration, **kwargs)
    if found:
        raise TimingError(found)
    return found


@dataclass(frozen=True)
class ClipReport:
    """Findings from checking a plan-driven clip.

    Attributes:
        readability: `ReadabilityFault` list, keeping the first occurrence
            of each kind for each artist or artist pair across review frames.
        meaning: `MeaningFault` list, empty when no meaning keys were supplied.
            Warnings and advisories count as findings too.
        motion: `MotionFault` list from sampling every frame.
        plan_faults: The plan's `eyepiece.Fault` list, including info findings,
            which are kept here but do not count toward `faults` or `ok`.
        unread_knobs: List of knob names never read by the draw function.
            Constant knobs that were read do not count as unread.
        speech: List of ``(line_key, wpm)`` pairs above the speech ceiling.
        motion_ink: The share of changed pixels that belong to the artists
            named in ``driven`` (see `motion_ink`), or None when none were
            named or nothing changes. It is reported, never a fault.
    """

    readability: list
    meaning: list
    motion: list
    plan_faults: list
    unread_knobs: list
    speech: list
    motion_ink: float | None = None

    @property
    def faults(self):
        """A flat list of unique ``(kind, detail)`` pairs from all checks.

        Visual findings use their own kind, plan findings use their code,
        and the remaining kinds are ``"unread-knob"`` and ``"speech-rate"``.
        Plan findings of severity ``"info"`` (a pulse skipped in a silent
        clip, a span that starts slightly late) are not faults.
        """
        entries = [
            (f.kind, f.detail) for f in (*self.readability, *self.meaning, *self.motion)
        ]
        entries += [
            (f.code, f.message) for f in self.plan_faults if f.severity != "info"
        ]
        entries += [("unread-knob", name) for name in self.unread_knobs]
        entries += [
            ("speech-rate", f"{key}: {wpm:g} words/min") for key, wpm in self.speech
        ]
        return list(dict.fromkeys(entries))

    @property
    def ok(self):
        """Whether every check returned no findings."""
        return not self.faults

    def __str__(self):
        """Summarize the clip, naming every finding by kind."""
        found = self.faults
        ink = (
            []
            if self.motion_ink is None
            else [f"motion ink: {self.motion_ink:.2f} of changed pixels are driven"]
        )
        if not found:
            return "\n".join(["clip is clean", *ink])
        lines = [f"{len(found)} clip fault(s):"]
        lines += [f"  {kind}: {detail}" for kind, detail in found]
        return "\n".join(lines + ink)


class ClipError(AssertionError):
    """A plan-driven clip has findings.

    Attributes:
        report: The `ClipReport`.
        faults: Its flat list of ``(kind, detail)`` pairs.
    """

    def __init__(self, report):
        """Build the assertion message from the report."""
        self.report = report
        self.faults = report.faults
        super().__init__(str(report))


def _clip_moves(moves, plan):
    """Translate artist-to-span mappings and named spans in move dicts."""

    def span(value):
        if isinstance(value, str):
            resolved = plan._span(value)
            return Window(float(resolved.start_s), float(resolved.end_s))
        return value

    if isinstance(moves, dict):
        return [
            {"name": str(value), "artists": [artist], "span": span(value)}
            for artist, value in moves.items()
        ]
    return [
        {**move, "span": span(move["span"])}
        if isinstance(move, dict) and "span" in move
        else move
        for move in moves or ()
    ]


def check_clip(
    fig,
    draw,
    plan,
    *,
    venue="video",
    keys=None,
    motifs=None,
    moves=None,
    steps=None,
    hops=None,
    driven=None,
) -> ClipReport:
    """Re-render a plan-driven clip and collect its checks without writing files.

    The draw signature is ``draw(k)``, with just the integer frame index,
    as in a caller-owned loop around `eyepiece.record` with a plan. The
    figure and plan are captured by the callable. This differs from
    `eyepiece.animate`, whose callable is ``draw(fig, ctx)``.

    Frame k is posed at k / plan.fps, up to plan.n_frames - 1, with no extra
    endpoint frame. Readability and meaning use `plan.review_frames`, which
    includes cue frames and their preceding frames. Motion uses every frame.
    Draw must set a frame's state from its index, since checks revisit frames.
    The figure ends on the last frame. Knob reads start clean and the prior
    read record is restored even if drawing or a check raises.

    Args:
        fig: The figure of the clip.
        draw: Callable ``draw(k)`` that poses the figure at frame index k.
        plan: A `eyepiece.Plan`, resolved at the clip's frame rate.
        venue: Readability venue, a key of `VENUES`.
        keys: Meaning gid-prefix mapping. An explicit mapping, even an empty
            one, overrides `motifs`. With neither, meaning checks are skipped.
        motifs: A `eyepiece.MotifSet` supplying `meaning_keys()`.
        moves: Mapping from artists to plan span names or `Window`
            values, or an iterable of declarations accepted by `motion_faults`.
            A declaration dict's ``span`` may also be a plan span name.
        steps: Additional intended discontinuities as integer frame indices
            or plan span names, which declare the span's first frame (k0).
            The plan's `step_frames()` are always declared, including when
            this is None or empty. Frames become times k / plan.fps for
            `motion_faults`, suppressing only jumps ending at those samples
            within 1e-9 s; other kinds are unchanged. A move whose contract
            names ``draws`` (a draw group) is excused at those steps only for
            its group, as in `motion_faults`.
        hops: Hop groups, each ``{"artists": [...], "rate": draws_per_second}``,
            as in `motion_faults`; a declared move's ``hops`` adds to them.
        driven: The artists the clip's knob drives, for the report's
            ``motion_ink`` (see `motion_ink`). None leaves it None.

    Returns:
        A frozen `ClipReport` with the findings from each check.

    Raises:
        ValueError: If the venue is unknown or the plan has fewer than three
            frames, as required by `motion_faults`.
        KeyError: If a move or step names an unknown plan span.
        ValueError: If a step is neither a span name nor a frame index in
            the plan.
    """
    declared = _clip_moves(moves, plan)
    step_frames = set(plan.step_frames())
    if isinstance(steps, str):
        steps = [steps]
    for step in steps or ():
        if isinstance(step, str):
            step_frames.add(plan._span(step).k0)
        elif (
            isinstance(step, numbers.Integral)
            and not isinstance(step, bool)
            and 0 <= step < plan.n_frames
        ):
            step_frames.add(int(step))
        else:
            raise ValueError(
                f"a step is a span name or a frame index in [0, {plan.n_frames}), "
                f"got {step!r}"
            )
    step_times = [k / plan.fps for k in sorted(step_frames)]
    if keys is None and motifs is not None:
        keys = motifs.meaning_keys()
    review_times = [k / plan.fps for k in plan.review_frames()]
    times = [k / plan.fps for k in range(plan.n_frames)]

    def at(t):
        draw(frame_count(t, plan.fps))

    saved_reads = set(plan._read)
    plan._read.clear()
    try:
        readability = readability_faults(fig, at=at, times=review_times, venue=venue)
        # A persistent collision is one finding, rather than one per still.
        seen = set()
        unique = []
        for fault in readability:
            ident = (fault.kind, fault.artist, fault.other)
            if ident not in seen:
                seen.add(ident)
                unique.append(fault)
        meaning = (
            meaning_faults(fig, at=at, times=review_times, keys=keys)
            if keys is not None
            else []
        )
        motion = motion_faults(
            fig,
            at=at,
            times=times,
            moves=declared,
            steps=step_times,
            hops=hops or (),
            fps=plan.fps,
        )
        unread, _ = plan.unused_knobs()
        speech = (
            [] if plan.narration is None else list(speech_rate(plan.narration).fast)
        )
        ink = (
            None if driven is None else motion_ink(fig, at, times, driven, fps=plan.fps)
        )
        return ClipReport(
            unique, meaning, motion, list(plan.faults), list(unread), speech, ink
        )
    finally:
        plan._read.clear()
        plan._read.update(saved_reads)


def assert_clip(fig, draw, plan, **kwargs) -> ClipReport:
    """Check a plan-driven clip, raising `ClipError` for any finding.

    Args:
        fig: The figure of the clip.
        draw: Callable ``draw(k)`` taking the integer frame index.
        plan: A `eyepiece.Plan`.
        **kwargs: Passed to `check_clip`.

    Returns:
        The `ClipReport` when the clip is clean.

    Raises:
        ClipError: If any check has findings. Its message names every kind
            and its `report` attribute holds the complete `ClipReport`.
    """
    report = check_clip(fig, draw, plan, **kwargs)
    if not report.ok:
        raise ClipError(report)
    return report


_LAZY_EXPORTS = {
    "assert_frames_pure": "eyepiece._render_pool",
    "frame_faults": "eyepiece._render_pool",
    "same_picture": "eyepiece._render_poster",
}
_DEPRECATED = {"Span": "Window"}


def __getattr__(name):
    """Load shared render checks or resolve a deprecated name."""
    if name in _LAZY_EXPORTS:
        value = getattr(importlib.import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    if name in _DEPRECATED:
        new = _DEPRECATED[name]
        warnings.warn(
            f"eyepiece.testing.{name} is deprecated; use eyepiece.testing.{new}, "
            "the same class under a name that does not clash with eyepiece.Span",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include the lazy render checks in namespace discovery."""
    return sorted(set(globals()) | set(__all__))
