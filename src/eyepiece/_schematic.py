"""Miniature optical-train "you are here" rail.

`rail` draws a small diagram of an optical train from a plain element list
-- a beam envelope that pinches at focal planes and opens at pupil planes,
a lens after every plane but the last, and a glyph at each plane -- with
chosen planes picked out in the accent color. It is meant to sit beside a
physics panel (a field display, a PSF) so a figure never leaves the reader
guessing which plane they are looking at. It fills its own axes by default,
or draws in the caller's data coordinates beside other artists.

`schematic` is a thin preset wrapper over `rail` for the two trains that
come up constantly, an imager and a Lyot coronagraph, with hand-tuned
plane positions.

The glyph names are this library's own generic vocabulary. They describe
what to draw, not what a simulation library calls its objects, so nothing
here has to track another package's class names.
"""

from itertools import pairwise
from typing import NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle

from eyepiece import _style
from eyepiece._result import PlotResult

# Glyph name -> whether the beam is wide there (a pupil plane) or pinched
# (an image plane). The envelope reads straight off this mapping.
GLYPHS = {
    "source": False,
    "pupil": True,
    "focal": False,
    "mask": True,
    "apodizer": True,
    "dm": True,
    "fpm": False,
    "phase_mask": False,
    "lyot": True,
    "detector": False,
}

_BAR_GLYPHS = ("pupil", "lyot", "mask")

# What the optics do across the gap between two consecutive planes.
_GAPS = ("fourier", "relay", "none")

# Where a "fourier" gap draws its lens.
_FOURIER_LENS = ("after", "middle")

# Each preset: the (label, glyph) planes and their hand-tuned x positions.
_PRESETS = {
    "imager": (
        (("Pupil", "pupil"), ("Focal", "focal")),
        (0.30, 0.88),
    ),
    "coronagraph": (
        (("Pupil", "pupil"), ("FPM", "fpm"), ("Lyot", "lyot"), ("Focal", "focal")),
        (0.10, 0.36, 0.63, 0.92),
    ),
}


# Coordinate systems a rail can be laid out in.
_COORDS = ("axes", "data")

# The axes-coordinate layout every size in this module is written in: the
# optical-axis height, the collimated beam's half-width, and the half-width
# at a focus, as fractions of the axes.
_AXIS_Y = 0.52
_WIDE = 0.20
_TIGHT = 0.018

# In data coordinates, horizontal sizes (glyph widths, a lens drawn just
# after its plane, the cap) are this many vertical size units, so that under
# an equal data aspect a glyph keeps about the proportions it has on a
# typical rail axes, which is several times wider than it is tall.
_DATA_X_PER_Y = 2.0

# How far the beam runs before the first plane and past the last one in
# data coordinates, in units of the collimated half-width.
_DATA_MARGIN = 1.5


class _Frame(NamedTuple):
    """Where the rail's layout units land on the axes.

    Every size in this module is written in the axes-coordinate layout and
    multiplied by `ux` (along the optical axis) or `uy` (across it). In axes
    coordinates both units are exactly 1.0, so every product is the bare
    layout constant and the geometry is the axes layout itself.

    Attributes:
        y0: Optical-axis height.
        ux: Size unit along the optical axis.
        uy: Size unit across the optical axis.
        wide: Collimated beam half-width, `_WIDE * uy` held exact.
    """

    y0: float
    ux: float
    uy: float
    wide: float


def _frame(coords, axis_y, beam_half):
    """The `_Frame` for a rail laid out in `coords`."""
    if coords == "axes":
        return _Frame(_AXIS_Y, 1.0, 1.0, _WIDE)
    uy = beam_half / _WIDE
    return _Frame(axis_y, _DATA_X_PER_Y * uy, uy, beam_half)


def _default_positions(n):
    """Evenly spaced x positions for `n` planes, leaving room for labels."""
    if n == 1:
        return [0.5]
    return list(np.linspace(0.10, 0.90, n))


def _plane_keys(names, keys, labels, what):
    """Lower-cased plane keys for `names`, a label or a sequence of labels.

    Raises:
        ValueError: If `names` is neither, or any entry is not one of the
            planes' labels. The message starts with `what`.
    """
    if names is None:
        return set()
    entries = [names] if isinstance(names, str) else names
    try:
        entries = list(entries)
    except TypeError:
        raise ValueError(f"unknown {what} {names!r}; known planes: {labels}") from None
    for entry in entries:
        if not isinstance(entry, str) or entry.lower() not in keys:
            raise ValueError(f"unknown {what} {entry!r}; known planes: {labels}")
    return {entry.lower() for entry in entries}


def _set_hatch_color(patch, color):
    """Set a patch's hatch color; matplotlib before 3.10 has no setter."""
    setter = getattr(patch, "set_hatchcolor", None)
    if setter is not None:
        setter(color)
    else:
        patch._hatch_color = to_rgba(color)
        patch.stale = True


def _paint(artist, color):
    """Recolor one glyph artist's ink, keeping its alpha and hatch."""
    if isinstance(artist, Line2D):
        artist.set_color(color)
    elif artist.get_hatch():
        artist.set_edgecolor(color)
        _set_hatch_color(artist, color)
    else:
        artist.set_facecolor(color)


def _draw_glyph(ax, glyph, x, frame, color):
    """Draw one element glyph centered on the optical axis at `x`.

    Args:
        ax: Axes to draw into.
        glyph: A name from `GLYPHS`.
        x: Glyph center, in the rail's coordinates.
        frame: The rail's `_Frame`: the axis height and the size units.
        color: Color for the glyph's ink.

    Returns:
        The artists drawn, in drawing order.

    Raises:
        ValueError: If `glyph` has no drawing here. Reaching this means a
            name was added to `GLYPHS` without a branch below; failing
            loudly beats silently drawing an empty plane.
    """
    y0, ux, uy = frame.y0, frame.ux, frame.uy
    drawn = []
    if glyph == "source":
        star = 2.3 * matplotlib.rcParams["lines.markersize"]
        drawn += ax.plot(
            [x], [y0], marker="*", ms=star, ls="none", color=color, zorder=5
        )
    elif glyph in _BAR_GLYPHS:
        for low in (y0 - 0.25 * uy, y0 + 0.13 * uy):
            bar = Rectangle(
                (x - 0.006 * ux, low),
                0.012 * ux,
                0.12 * uy,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
            drawn.append(ax.add_patch(bar))
    elif glyph == "apodizer":
        plate = Rectangle(
            (x - 0.006 * ux, y0 - 0.24 * uy),
            0.012 * ux,
            0.48 * uy,
            facecolor=color,
            alpha=0.45,
            edgecolor="none",
            zorder=5,
        )
        drawn.append(ax.add_patch(plate))
    elif glyph == "dm":
        # An opaque plate with a rippled face toward the incoming beam.
        plate = Rectangle(
            (x - 0.005 * ux, y0 - 0.24 * uy),
            0.010 * ux,
            0.48 * uy,
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
        drawn.append(ax.add_patch(plate))
        ys = np.linspace(y0 - 0.24 * uy, y0 + 0.24 * uy, 80)
        drawn += ax.plot(
            x - 0.010 * ux + 0.003 * ux * np.sin(ys * 7.0 * np.pi / (0.24 * uy)),
            ys,
            color=color,
            lw=0.8,
            zorder=5,
        )
    elif glyph == "phase_mask":
        # A clear plate spanning the focus: it shifts phase, it blocks nothing.
        plate = Rectangle(
            (x - 0.006 * ux, y0 - 0.15 * uy),
            0.012 * ux,
            0.30 * uy,
            facecolor=color,
            alpha=0.7,
            edgecolor="none",
            zorder=5,
        )
        drawn.append(ax.add_patch(plate))
    elif glyph == "fpm":
        diamond = Polygon(
            [
                (x, y0 - 0.14 * uy),
                (x + 0.016 * ux, y0),
                (x, y0 + 0.14 * uy),
                (x - 0.016 * ux, y0),
            ],
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
        drawn.append(ax.add_patch(diamond))
    elif glyph == "focal":
        for sign in (-1.0, 1.0):
            wedge = Polygon(
                [
                    (x, y0),
                    (x - 0.020 * ux, y0 + sign * 0.13 * uy),
                    (x + 0.020 * ux, y0 + sign * 0.13 * uy),
                ],
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
            drawn.append(ax.add_patch(wedge))
    elif glyph == "detector":
        # A Patch captures rcParams["hatch.color"] at construction, and that
        # rcParam only defaults to the edge color from matplotlib 3.11. On an
        # older supported version, or under a style that pins it, the hatch
        # would come out black on a dark ground, so it is set explicitly for
        # the length of the construction.
        with matplotlib.rc_context({"hatch.color": color}):
            box = Rectangle(
                (x - 0.016 * ux, y0 - 0.10 * uy),
                0.032 * ux,
                0.20 * uy,
                facecolor="none",
                edgecolor=color,
                lw=1.4,
                hatch="///",
                zorder=5,
            )
            drawn.append(ax.add_patch(box))
    else:
        raise ValueError(f"glyph {glyph!r} is in GLYPHS but has no drawing")
    return drawn


def rail(
    planes,
    *,
    ax=None,
    positions=None,
    highlight=None,
    accent=None,
    cap=None,
    gaps=None,
    fourier_lens="after",
    stops=None,
    colors=None,
    beam_color=None,
    coords="axes",
    axis_y=None,
    beam_half=None,
    span=None,
):
    """Draw a miniature optical-train rail with chosen planes highlighted.

    Each plane contributes a marker, a label, and a glyph drawn on a beam
    envelope that opens at the pupil-like glyphs (`"pupil"`, `"lyot"`,
    `"mask"`, `"apodizer"`, `"dm"`) and pinches at the image-like ones
    (`"source"`, `"focal"`, `"fpm"`, `"phase_mask"`, `"detector"`). A lens
    is drawn just
    after every plane but the last: a pupil-plane lens forms the next
    focal plane, a lens after a focal-plane mask re-collimates to the next
    pupil. A rail that ends on a `"focal"` plane is capped with a small
    detector block by default, so the train ends somewhere; see `cap` to
    force that block on or off. `gaps` replaces the one-lens default per
    gap, for a train that relays a pupil or runs a collimated beam between
    two planes; `fourier_lens` moves each Fourier lens to its gap's middle;
    and `stops` narrows the beam at a stop that passes less than all of it.

    By default the rail fills its own axes: positions are axes fractions,
    the limits are set to the unit square, and the axis is turned off.
    `coords="data"` instead draws the train in the caller's data
    coordinates, among other artists on a shared axes, at the optical-axis
    height `axis_y` with a collimated half-width of `beam_half`.

    The glyphs are::

        source     a star marker
        pupil      two bars, top and bottom, clipping the beam edges
        lyot       the same two bars (a Lyot stop is a pupil stop)
        mask       the same two bars (any other pupil-plane stop)
        apodizer   one translucent bar spanning the whole beam
        dm         an opaque plate with a rippled face (a deformable mirror,
                   drawn unfolded)
        fpm        a diamond on the optical axis
        phase_mask a clear plate spanning the focus (a focal-plane phase
                   mask, such as a vortex)
        focal      a bowtie, the beam waist pinching to a point
        detector   a hatched, unfilled box

    Every artist the rail draws carries a gid, `rail/<label>/<part>`, where
    `<label>` is the plane's label exactly as given. The parts are::

        marker     the plane's marker line (also in artists["lines"])
        label      the plane's label text (also in artists["text"])
        glyph      every artist of the plane's glyph (a bar pair, the dm's
                   plate and ripple, and the bowtie's two wedges each share
                   one gid)
        lens       each lens in the gap that follows the plane (two for a
                   relay; also in artists["ellipse"])
        cap        the detector block past the last plane

    and the beam, which belongs to no plane, uses the label `beam`:
    `rail/beam/fill` (the envelope, also artists["fill"]), `rail/beam/edge`
    (its two edge lines), and `rail/beam/axis` (the dotted optical axis).
    Glyphs are not returned under an artist key because one glyph mixes
    lines and patches; find them by gid, for instance
    `[a for a in ax.get_children() if a.get_gid() == "rail/FPM/glyph"]`.

    Args:
        planes: Sequence of `(label, glyph)` pairs in optical order.
            `label` is the display text; `glyph` is a name from `GLYPHS`.
            At least one plane is required.
        ax: Axes to draw into. None creates a new figure and axes.
        positions: Sequence of x positions, one per plane, in
            non-decreasing order: axes fractions, or data x values when
            `coords` is `"data"`. None spaces the planes evenly across the
            rail, which needs axes coordinates.
        highlight: A plane's label, or a sequence of labels, matched
            case-insensitively, whose planes are drawn lit: marker, label,
            and (unless the plane is in `colors`) glyph in the accent color.
            None or an empty sequence leaves every plane neutral. Anything
            that is not one of `planes`' labels raises `ValueError` rather
            than silently matching nothing. Labels are not required to be
            unique, and a label that matches several planes lights every
            one.
        accent: Highlight color override; None uses `_style.color(1)`.
        cap: Whether to close the beam with a detector block just past the
            last plane. True always draws it, False never does, and None
            (the default) draws it only when the last plane's glyph is
            `"focal"` -- a rail that ends on its own `"detector"` plane
            already terminates, and one that ends on a pupil is a train
            still in progress.
        gaps: What the optics do between each pair of consecutive planes,
            one entry per gap (`len(planes) - 1`). `"fourier"` is one lens
            just after the first plane, taking a pupil to its focal plane or
            back; `"relay"` is a lens pair at the gap's quarter points with
            an intermediate focus (between two pupil-like planes) or
            collimated beam (between two image-like planes) at its middle,
            re-imaging a plane onto the next one of the same kind; `"none"`
            is free space with no lens, as for a stop placed in a collimated
            beam. None makes every gap `"fourier"`.
        fourier_lens: Where every `"fourier"` gap draws its lens. `"after"`
            (the default) is just after the gap's first plane. `"middle"`
            is the gap's midpoint, as in a train drawn to scale: the beam
            stays collimated on the pupil side of the lens and focuses (or
            opens from a focus) on the other.
        stops: Mapping from a plane's label, matched case-insensitively as
            `highlight` is, to the fraction of the beam's half-width that
            plane passes, in (0, 1]. The beam steps down to that fraction at
            the plane and stays narrower downstream, so an undersized Lyot
            stop reads as the stop that trims the beam. Only a pupil-like
            plane can take a stop, since the beam is already pinched at an
            image-like one. None leaves the beam at full width.
        colors: Mapping from a plane's label, matched case-insensitively,
            to its glyph color, so each element can carry its own role
            color. A plane named here keeps that glyph color when lit; the
            highlight then shows on its marker and label only. Planes not
            named keep the neutral glyph tone (the accent when lit). None
            colors no glyph.
        beam_color: Color of the beam envelope's fill and edges. None uses
            the neutral beam tone.
        coords: `"axes"` (the default) lays the rail out in axes fractions
            on the unit square, sets those limits, and turns the axis off.
            `"data"` places it in data coordinates and leaves the limits
            and the axis alone: `positions` are data x values, the beam is
            centered on `axis_y` with collimated half-width `beam_half`, and
            every glyph, lens, and label offset scales with `beam_half`.
            Sizes along the optical axis are in the same units as
            `beam_half`, so the glyphs keep their shapes under an equal data
            aspect (`ax.set_aspect("equal")`).
        axis_y: Optical-axis height in data coordinates. Only for
            `coords="data"`, where None means 0.0.
        beam_half: Collimated beam half-width in data coordinates, > 0.
            Only for `coords="data"`, where None means 1.0.
        span: `(start, end)` x extent of the beam and the optical axis,
            which must bracket `positions`. None runs from 0.02 to 0.99 (or
            further, to reach the outer planes) in axes coordinates, and
            from 1.5 `beam_half` before the first plane to 1.5 `beam_half`
            after the last in data coordinates.

    Returns:
        A `PlotResult` with artists `"fill"` (the beam-envelope
        `PolyCollection`), `"lines"` (the list of per-plane marker `Line2D`
        artists, drawn together on one axes), `"text"` (the list of
        per-plane label `Text` artists), the last two in plane order, and,
        when any lens is drawn, `"ellipse"` (the list of lens `Ellipse`
        patches, in optical order). Its `update(highlight=None)` relights
        the rail in place, as if it had been drawn with that `highlight`:
        it restyles the existing markers, labels, and uncolored glyphs and
        adds no artist, so an animation can move the highlight per frame
        without clearing the axes. It validates `highlight` as `rail` does
        and keeps the tones and accent of the first draw.

    Raises:
        ValueError: If `planes` is empty, a glyph is not in `GLYPHS`,
            `positions` is the wrong length or decreases (or is missing in
            data coordinates), `highlight` or `colors` names a label that is
            not one of the planes', `gaps` is the wrong length or names an
            unknown gap, `fourier_lens` is not `"after"` or `"middle"`,
            `stops` names an unknown or image-like plane or a fraction
            outside (0, 1], `coords` is not `"axes"` or `"data"`, `axis_y`
            or `beam_half` is given in axes coordinates, `beam_half` is not
            positive, or `span` does not bracket `positions`.

    Note:
        Every tone but the accent and the caller's colors is neutral
        scenery resolved from the active rcParams at call time, so the rail
        reads on a light or a dark background rather than fixing one gray
        for both.

    Example::

        rail([("Pupil", "pupil"), ("FPM", "fpm")], highlight="FPM")
        rail(
            [("Pupil", "pupil"), ("FPM", "fpm"), ("Lyot", "lyot")],
            ax=ax,
            coords="data",
            positions=(1.2, 4.0, 6.8),
            beam_half=1.0,
            colors={"Pupil": "tab:blue"},
            highlight=["Pupil", "Lyot"],
        )
    """
    planes = [(label, glyph) for label, glyph in planes]
    if not planes:
        raise ValueError("rail needs at least one plane")
    for _, glyph in planes:
        if glyph not in GLYPHS:
            raise ValueError(f"unknown glyph: {glyph!r}; known: {sorted(GLYPHS)}")

    labels = [label for label, _ in planes]
    keys = [label.lower() for label in labels]
    lit = _plane_keys(highlight, keys, labels, "highlight")

    if coords not in _COORDS:
        raise ValueError(f"unknown coords: {coords!r}; known: {list(_COORDS)}")
    if coords == "axes":
        if axis_y is not None or beam_half is not None:
            raise ValueError('axis_y and beam_half need coords="data"')
    else:
        axis_y = 0.0 if axis_y is None else float(axis_y)
        beam_half = 1.0 if beam_half is None else float(beam_half)
        if not beam_half > 0.0:
            raise ValueError(f"beam_half must be positive, got {beam_half}")
        if positions is None:
            raise ValueError('positions are required with coords="data"')

    if positions is None:
        positions = _default_positions(len(planes))
    else:
        positions = [float(x) for x in positions]
        if len(positions) != len(planes):
            raise ValueError(
                f"positions has {len(positions)} entries for {len(planes)} planes"
            )
        if any(b < a for a, b in pairwise(positions)):
            raise ValueError(f"positions must be non-decreasing, got {positions}")

    if gaps is None:
        gaps = ["fourier"] * (len(planes) - 1)
    else:
        gaps = list(gaps)
        if len(gaps) != len(planes) - 1:
            raise ValueError(
                f"gaps has {len(gaps)} entries for {len(planes) - 1} gaps "
                f"between {len(planes)} planes"
            )
        for gap in gaps:
            if gap not in _GAPS:
                raise ValueError(f"unknown gap: {gap!r}; known: {list(_GAPS)}")

    if fourier_lens not in _FOURIER_LENS:
        raise ValueError(
            f"unknown fourier_lens: {fourier_lens!r}; known: {list(_FOURIER_LENS)}"
        )

    passed = {}
    for label, fraction in (stops or {}).items():
        key = label.lower() if isinstance(label, str) else None
        if key not in keys:
            raise ValueError(f"unknown stop plane {label!r}; known planes: {labels}")
        fraction = float(fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"stop fraction for {label!r} must be in (0, 1], got {fraction}"
            )
        passed[key] = fraction
    for key, (label, glyph) in zip(keys, planes, strict=True):
        if key in passed and not GLYPHS[glyph]:
            raise ValueError(
                f"stop plane {label!r} has the image-like glyph {glyph!r}; "
                "a stop narrows the beam only at a pupil-like plane"
            )

    role = {}
    for label, color in (colors or {}).items():
        key = label.lower() if isinstance(label, str) else None
        if key not in keys:
            raise ValueError(f"unknown color plane {label!r}; known planes: {labels}")
        role[key] = color

    frame = _frame(coords, axis_y, beam_half)
    y0, ux, uy = frame.y0, frame.ux, frame.uy

    if span is None and coords == "axes":
        left = min(0.02, positions[0])
        right = max(0.99, positions[-1])
    elif span is None:
        left = positions[0] - _DATA_MARGIN * beam_half
        right = positions[-1] + _DATA_MARGIN * beam_half
    else:
        left, right = (float(x) for x in span)
        if not left <= positions[0] <= positions[-1] <= right:
            raise ValueError(
                f"span {span} must bracket the plane positions "
                f"{positions[0]} to {positions[-1]}"
            )

    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    accent_color = _style.color(1, accent)
    if coords == "axes":
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    wide, tight = frame.wide, _TIGHT * uy
    heights = [wide if GLYPHS[glyph] else tight for _, glyph in planes]

    # Envelope control points and lens positions. A relay gap bends the
    # envelope at its lenses and at the intermediate plane between them, and
    # a mid-gap Fourier lens holds the collimated width up to the lens;
    # every other gap interpolates straight from one plane to the next.
    # `full` is the collimated half-width, which a stop narrows for the rest
    # of the train; a stop's step sits a hair past its plane so the control
    # points stay strictly ordered. `owners` holds the label of the plane
    # each lens follows.
    full = wide
    steps = []
    xs, hs = [left], [heights[0]]
    lenses = []
    owners = []
    rows = zip(positions, keys, planes, strict=True)
    for i, (xp, key, (label, glyph)) in enumerate(rows):
        xs.append(xp)
        hs.append(full if GLYPHS[glyph] else tight)
        if key in passed and passed[key] < 1.0:
            full *= passed[key]
            steps += [xp, xp + 1e-6 * ux]
            xs.append(xp + 1e-6 * ux)
            hs.append(full)
        if i == len(planes) - 1:
            break
        gap, x_next = gaps[i], positions[i + 1]
        if gap == "fourier" and fourier_lens == "after":
            lenses.append(xp + 0.035 * ux)
        elif gap == "fourier":
            lenses.append(0.5 * (xp + x_next))
            xs.append(0.5 * (xp + x_next))
            hs.append(full)
        elif gap == "relay":
            quarter = 0.25 * (x_next - xp)
            middle = tight if GLYPHS[glyph] else full
            xs += [xp + quarter, xp + 2 * quarter, x_next - quarter]
            hs += [full, middle, full]
            lenses += [xp + quarter, x_next - quarter]
        owners += [label] * (len(lenses) - len(owners))
    xs.append(right)
    hs.append(hs[-1])
    beam = _style.neutral(0.45) if beam_color is None else beam_color
    faint = _style.neutral(0.25)
    xf = np.linspace(left, right, 400)
    if steps:
        xf = np.union1d(xf, steps)
    hf = np.interp(xf, xs, hs)
    fill = ax.fill_between(xf, y0 - hf, y0 + hf, color=beam, alpha=0.20, lw=0)
    fill.set_gid("rail/beam/fill")
    (upper,) = ax.plot(xf, y0 + hf, color=beam, lw=0.7)
    (lower,) = ax.plot(xf, y0 - hf, color=beam, lw=0.7)
    upper.set_gid("rail/beam/edge")
    lower.set_gid("rail/beam/edge")
    (axis_line,) = ax.plot([left, right], [y0, y0], color=faint, lw=0.6, ls=":")
    axis_line.set_gid("rail/beam/axis")

    ellipses = []
    for xm, owner in zip(lenses, owners, strict=True):
        hm = float(np.interp(xm, xs, hs))
        lens = Ellipse(
            (xm, y0),
            0.026 * ux,
            2 * max(hm, 0.06 * uy) * 0.95,
            facecolor=faint,
            edgecolor=_style.neutral(0.7),
            lw=0.7,
            zorder=3,
        )
        lens.set_gid(f"rail/{owner}/lens")
        ellipses.append(ax.add_patch(lens))

    if cap is None:
        cap = planes[-1][1] == "focal"
    if cap:
        block = Rectangle(
            (positions[-1] + 0.015 * ux, y0 - 0.055 * uy),
            0.035 * ux,
            0.11 * uy,
            facecolor=_style.neutral(0.8),
            edgecolor="none",
            zorder=3,
        )
        block.set_gid(f"rail/{labels[-1]}/cap")
        ax.add_patch(block)

    plain = _style.neutral(0.55)
    glyph_tone = _style.neutral(0.65)
    lines = []
    texts = []
    glyph_parts = []
    for (label, glyph), key, xp in zip(planes, keys, positions, strict=True):
        on = key in lit
        color = accent_color if on else plain
        hp = max(wide if GLYPHS[glyph] else tight, 0.13 * uy)
        (line,) = ax.plot(
            [xp, xp],
            [y0 - hp - 0.06 * uy, y0 + hp + 0.06 * uy],
            color=color,
            lw=2.0 if on else 1.0,
            ls="-" if on else "--",
            zorder=4,
        )
        line.set_gid(f"rail/{label}/marker")
        ink = role.get(key, accent_color if on else glyph_tone)
        parts = _draw_glyph(ax, glyph, xp, frame, ink)
        for part in parts:
            part.set_gid(f"rail/{label}/glyph")
        text = ax.text(
            xp,
            y0 + hp + 0.10 * uy,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold" if on else "normal",
        )
        text.set_gid(f"rail/{label}/label")
        lines.append(line)
        texts.append(text)
        glyph_parts.append(parts)

    def update(highlight=None):
        now_lit = _plane_keys(highlight, keys, labels, "highlight")
        styled = zip(keys, lines, texts, glyph_parts, strict=True)
        for key, line, text, parts in styled:
            on = key in now_lit
            color = accent_color if on else plain
            line.set_color(color)
            line.set_linewidth(2.0 if on else 1.0)
            line.set_linestyle("-" if on else "--")
            text.set_color(color)
            text.set_fontweight("bold" if on else "normal")
            if key not in role:
                for part in parts:
                    _paint(part, accent_color if on else glyph_tone)

    artists = {"fill": fill, "lines": lines, "text": texts}
    if ellipses:
        artists["ellipse"] = ellipses
    return PlotResult(ax=ax, artists=artists, update=update)


def schematic(kind, *, ax=None, highlight=None, accent=None):
    """Draw one of the preset optical-train rails.

    A thin wrapper over `rail` for the two trains that come up constantly,
    with hand-tuned plane positions. Build the `(label, glyph)` list
    yourself and call `rail` for anything else.

    Args:
        kind: `"imager"` (pupil -> focal) or `"coronagraph"` (pupil ->
            focal-plane mask -> Lyot pupil -> focal).
        ax: Axes to draw into. None creates a new figure and axes.
        highlight: Plane label, or sequence of labels, to draw in the
            accent color, matched case-insensitively (`"pupil"`, `"focal"`,
            and for `"coronagraph"` also `"fpm"`, `"lyot"`). None leaves
            every plane in the neutral color. Anything that is not one of `kind`'s
            plane labels raises `ValueError` rather than silently matching
            nothing.
        accent: Highlight color override; None uses `_style.color(1)`.

    Returns:
        A `PlotResult` exactly as `rail` returns it.

    Raises:
        ValueError: If `kind` is not a known train, or `highlight` is not
            one of that train's plane labels.
    """
    if kind not in _PRESETS:
        raise ValueError(f"unknown schematic kind: {kind!r}; known: {sorted(_PRESETS)}")
    train, positions = _PRESETS[kind]
    return rail(train, ax=ax, positions=positions, highlight=highlight, accent=accent)
