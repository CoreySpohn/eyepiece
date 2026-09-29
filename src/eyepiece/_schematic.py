"""Miniature optical-train "you are here" rail.

`rail` draws a small diagram of an optical train from a plain element list
-- a beam envelope that pinches at focal planes and opens at pupil planes,
a lens after every plane but the last, and a glyph at each plane -- with
one plane picked out in the accent color. It is meant to sit beside a
physics panel (a field display, a PSF) so a figure never leaves the reader
guessing which plane they are looking at.

`schematic` is a thin preset wrapper over `rail` for the two trains that
come up constantly, an imager and a Lyot coronagraph, with hand-tuned
plane positions.

The glyph names are this library's own generic vocabulary. They describe
what to draw, not what a simulation library calls its objects, so nothing
here has to track another package's class names.
"""

from itertools import pairwise

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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


def _default_positions(n):
    """Evenly spaced x positions for `n` planes, leaving room for labels."""
    if n == 1:
        return [0.5]
    return list(np.linspace(0.10, 0.90, n))


def _draw_glyph(ax, glyph, x, y0, color):
    """Draw one element glyph centered on `(x, y0)` in axes coordinates.

    Args:
        ax: Axes to draw into.
        glyph: A name from `GLYPHS`.
        x: Glyph center, in axes coordinates.
        y0: Optical-axis height, in axes coordinates.
        color: Color for the glyph's ink.

    Raises:
        ValueError: If `glyph` has no drawing here. Reaching this means a
            name was added to `GLYPHS` without a branch below; failing
            loudly beats silently drawing an empty plane.
    """
    if glyph == "source":
        star = 2.3 * matplotlib.rcParams["lines.markersize"]
        ax.plot([x], [y0], marker="*", ms=star, ls="none", color=color, zorder=5)
    elif glyph in _BAR_GLYPHS:
        for low in (y0 - 0.25, y0 + 0.13):
            ax.add_patch(
                Rectangle(
                    (x - 0.006, low),
                    0.012,
                    0.12,
                    facecolor=color,
                    edgecolor="none",
                    zorder=5,
                )
            )
    elif glyph == "apodizer":
        ax.add_patch(
            Rectangle(
                (x - 0.006, y0 - 0.24),
                0.012,
                0.48,
                facecolor=color,
                alpha=0.45,
                edgecolor="none",
                zorder=5,
            )
        )
    elif glyph == "dm":
        # An opaque plate with a rippled face toward the incoming beam.
        ax.add_patch(
            Rectangle(
                (x - 0.005, y0 - 0.24),
                0.010,
                0.48,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )
        ys = np.linspace(y0 - 0.24, y0 + 0.24, 80)
        ax.plot(
            x - 0.010 + 0.003 * np.sin(ys * 7.0 * np.pi / 0.24),
            ys,
            color=color,
            lw=0.8,
            zorder=5,
        )
    elif glyph == "phase_mask":
        # A clear plate spanning the focus: it shifts phase, it blocks nothing.
        ax.add_patch(
            Rectangle(
                (x - 0.006, y0 - 0.15),
                0.012,
                0.30,
                facecolor=color,
                alpha=0.7,
                edgecolor="none",
                zorder=5,
            )
        )
    elif glyph == "fpm":
        ax.add_patch(
            Polygon(
                [(x, y0 - 0.14), (x + 0.016, y0), (x, y0 + 0.14), (x - 0.016, y0)],
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )
    elif glyph == "focal":
        for sign in (-1.0, 1.0):
            ax.add_patch(
                Polygon(
                    [
                        (x, y0),
                        (x - 0.020, y0 + sign * 0.13),
                        (x + 0.020, y0 + sign * 0.13),
                    ],
                    facecolor=color,
                    edgecolor="none",
                    zorder=5,
                )
            )
    elif glyph == "detector":
        # A Patch captures rcParams["hatch.color"] at construction, and that
        # rcParam only defaults to the edge color from matplotlib 3.11. On an
        # older supported version, or under a style that pins it, the hatch
        # would come out black on a dark ground, so it is set explicitly for
        # the length of the construction.
        with matplotlib.rc_context({"hatch.color": color}):
            ax.add_patch(
                Rectangle(
                    (x - 0.016, y0 - 0.10),
                    0.032,
                    0.20,
                    facecolor="none",
                    edgecolor=color,
                    lw=1.4,
                    hatch="///",
                    zorder=5,
                )
            )
    else:
        raise ValueError(f"glyph {glyph!r} is in GLYPHS but has no drawing")


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
):
    """Draw a miniature optical-train rail with one plane highlighted.

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

    Args:
        planes: Sequence of `(label, glyph)` pairs in optical order.
            `label` is the display text; `glyph` is a name from `GLYPHS`.
            At least one plane is required.
        ax: Axes to draw into. None creates a new figure and axes.
        positions: Sequence of x positions in axes coordinates, one per
            plane, in non-decreasing order. None spaces the planes evenly
            across the rail.
        highlight: A plane's label, matched case-insensitively, to draw in
            the accent color. None leaves every plane in the neutral color.
            Anything that is not one of `planes`' labels raises
            `ValueError` rather than silently matching nothing. Labels are
            not required to be unique, and a `highlight` that matches
            several of them lights every one.
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

    Returns:
        A `PlotResult` with artists `"fill"` (the beam-envelope
        `PolyCollection`), `"lines"` (the list of per-plane marker `Line2D`
        artists, drawn together on one axes), and `"text"` (the list of
        per-plane label `Text` artists), the last two in plane order.

    Raises:
        ValueError: If `planes` is empty, a glyph is not in `GLYPHS`,
            `positions` is the wrong length or decreases, `highlight` is
            not one of the planes' labels, `gaps` is the wrong length or
            names an unknown gap, `fourier_lens` is not `"after"` or
            `"middle"`, or `stops` names an unknown or image-like plane or
            a fraction outside (0, 1].

    Note:
        Every tone but the accent is neutral scenery resolved from the
        active rcParams at call time, so the rail reads on a light or a
        dark background rather than fixing one gray for both.

    Example::

        rail([("Pupil", "pupil"), ("FPM", "fpm")], highlight="FPM")
    """
    planes = [(label, glyph) for label, glyph in planes]
    if not planes:
        raise ValueError("rail needs at least one plane")
    for _, glyph in planes:
        if glyph not in GLYPHS:
            raise ValueError(f"unknown glyph: {glyph!r}; known: {sorted(GLYPHS)}")

    labels = [label for label, _ in planes]
    keys = [label.lower() for label in labels]
    if highlight is not None and (
        not isinstance(highlight, str) or highlight.lower() not in keys
    ):
        raise ValueError(f"unknown highlight {highlight!r}; known planes: {labels}")

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

    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    accent_color = _style.color(1, accent)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y0, wide, tight = 0.52, 0.20, 0.018
    heights = [wide if GLYPHS[glyph] else tight for _, glyph in planes]

    left = min(0.02, positions[0])
    right = max(0.99, positions[-1])
    # Envelope control points and lens positions. A relay gap bends the
    # envelope at its lenses and at the intermediate plane between them, and
    # a mid-gap Fourier lens holds the collimated width up to the lens;
    # every other gap interpolates straight from one plane to the next.
    # `full` is the collimated half-width, which a stop narrows for the rest
    # of the train; a stop's step sits a hair past its plane so the control
    # points stay strictly ordered.
    full = wide
    steps = []
    xs, hs = [left], [heights[0]]
    lenses = []
    rows = zip(positions, keys, planes, strict=True)
    for i, (xp, key, (_, glyph)) in enumerate(rows):
        xs.append(xp)
        hs.append(full if GLYPHS[glyph] else tight)
        if key in passed and passed[key] < 1.0:
            full *= passed[key]
            steps += [xp, xp + 1e-6]
            xs.append(xp + 1e-6)
            hs.append(full)
        if i == len(planes) - 1:
            break
        gap, x_next = gaps[i], positions[i + 1]
        if gap == "fourier" and fourier_lens == "after":
            lenses.append(xp + 0.035)
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
    xs.append(right)
    hs.append(hs[-1])
    beam = _style.neutral(0.45)
    faint = _style.neutral(0.25)
    xf = np.linspace(left, right, 400)
    if steps:
        xf = np.union1d(xf, steps)
    hf = np.interp(xf, xs, hs)
    fill = ax.fill_between(xf, y0 - hf, y0 + hf, color=beam, alpha=0.20, lw=0)
    ax.plot(xf, y0 + hf, color=beam, lw=0.7)
    ax.plot(xf, y0 - hf, color=beam, lw=0.7)
    ax.plot([left, right], [y0, y0], color=faint, lw=0.6, ls=":")

    for xm in lenses:
        hm = float(np.interp(xm, xs, hs))
        ax.add_patch(
            Ellipse(
                (xm, y0),
                0.026,
                2 * max(hm, 0.06) * 0.95,
                facecolor=faint,
                edgecolor=_style.neutral(0.7),
                lw=0.7,
                zorder=3,
            )
        )

    if cap is None:
        cap = planes[-1][1] == "focal"
    if cap:
        ax.add_patch(
            Rectangle(
                (positions[-1] + 0.015, y0 - 0.055),
                0.035,
                0.11,
                facecolor=_style.neutral(0.8),
                edgecolor="none",
                zorder=3,
            )
        )

    highlight_key = highlight.lower() if highlight is not None else None
    plain = _style.neutral(0.55)
    glyph_tone = _style.neutral(0.65)
    lines = []
    texts = []
    for (label, glyph), key, xp in zip(planes, keys, positions, strict=True):
        on = highlight_key is not None and key == highlight_key
        color = accent_color if on else plain
        hp = max(wide if GLYPHS[glyph] else tight, 0.13)
        (line,) = ax.plot(
            [xp, xp],
            [y0 - hp - 0.06, y0 + hp + 0.06],
            color=color,
            lw=2.0 if on else 1.0,
            ls="-" if on else "--",
            zorder=4,
        )
        _draw_glyph(ax, glyph, xp, y0, accent_color if on else glyph_tone)
        text = ax.text(
            xp,
            y0 + hp + 0.10,
            label,
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold" if on else "normal",
        )
        lines.append(line)
        texts.append(text)

    return PlotResult(ax=ax, artists={"fill": fill, "lines": lines, "text": texts})


def schematic(kind, *, ax=None, highlight=None, accent=None):
    """Draw one of the preset optical-train rails.

    A thin wrapper over `rail` for the two trains that come up constantly,
    with hand-tuned plane positions. Build the `(label, glyph)` list
    yourself and call `rail` for anything else.

    Args:
        kind: `"imager"` (pupil -> focal) or `"coronagraph"` (pupil ->
            focal-plane mask -> Lyot pupil -> focal).
        ax: Axes to draw into. None creates a new figure and axes.
        highlight: Plane label to draw in the accent color, matched
            case-insensitively (`"pupil"`, `"focal"`, and for
            `"coronagraph"` also `"fpm"`, `"lyot"`). None leaves every
            plane in the neutral color. Anything that is not one of `kind`'s
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
