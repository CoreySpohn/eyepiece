"""Miniature optical-train "you are here" rail.

`schematic` draws a small diagram of an optical train -- a beam envelope
that pinches at focal planes and opens at pupil planes, a lens after every
plane but the last, and a detector capping the final image -- with one
plane picked out in the accent color. It is meant to sit beside a physics
panel (a field display, a PSF) so a figure never leaves the reader
guessing which plane they are looking at.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle

from eyepiece import _style
from eyepiece._result import PlotResult

# Each train: an ordered tuple of (key, display label, x position, is_pupil).
_TRAINS = {
    "imager": (
        ("pupil", "Pupil", 0.30, True),
        ("focal", "Focal", 0.88, False),
    ),
    "coronagraph": (
        ("pupil", "Pupil", 0.10, True),
        ("fpm", "FPM", 0.36, False),
        ("lyot", "Lyot", 0.63, True),
        ("focal", "Focal", 0.92, False),
    ),
}


def schematic(kind, *, ax=None, highlight=None, accent=None):
    """Draw a miniature optical-train rail with one plane highlighted.

    Args:
        kind: `"imager"` (pupil -> focal) or `"coronagraph"` (pupil ->
            focal-plane mask -> Lyot pupil -> focal).
        ax: Axes to draw into. None creates a new figure and axes.
        highlight: Plane key to draw in the accent color, matched
            case-insensitively (`"pupil"`, `"focal"`, and for
            `"coronagraph"` also `"fpm"`, `"lyot"`). None leaves every
            plane in the neutral color.
        accent: Highlight color override; None uses `_style.color(1)`.

    Returns:
        A `PlotResult` with artists `"fill"` (the beam-envelope
        `PolyCollection`), `"line"` (list of per-plane marker `Line2D`s),
        and `"text"` (list of per-plane label `Text`s), the last two in
        plane order.

    Raises:
        ValueError: If `kind` is not a known train.
    """
    if kind not in _TRAINS:
        raise ValueError(f"unknown schematic kind: {kind!r}; known: {sorted(_TRAINS)}")
    planes = _TRAINS[kind]

    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    accent_color = _style.color(1, accent)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y0, wide, tight = 0.52, 0.20, 0.018

    xs = [0.02] + [p[2] for p in planes] + [0.99]
    hs = (
        [wide if planes[0][3] else tight]
        + [wide if p[3] else tight for p in planes]
        + [tight if not planes[-1][3] else wide]
    )
    xf = np.linspace(0.02, 0.99, 400)
    hf = np.interp(xf, xs, hs)
    fill = ax.fill_between(xf, y0 - hf, y0 + hf, color="0.55", alpha=0.20, lw=0)
    ax.plot(xf, y0 + hf, color="0.55", lw=0.7)
    ax.plot(xf, y0 - hf, color="0.55", lw=0.7)
    ax.plot([0.02, 0.99], [y0, y0], color="0.75", lw=0.6, ls=":")

    # A lens just after every plane but the last: a pupil-plane lens forms
    # the next focal plane, a lens after a focal-plane mask re-collimates to
    # the next pupil.
    for _, _, xp, _ in planes[:-1]:
        xm = xp + 0.035
        hm = float(np.interp(xm, xs, hs))
        ax.add_patch(
            Ellipse(
                (xm, y0),
                0.026,
                2 * max(hm, 0.06) * 0.95,
                facecolor="0.75",
                edgecolor="0.3",
                lw=0.7,
                zorder=3,
            )
        )

    # The detector, at the final image plane.
    ax.add_patch(
        Rectangle(
            (planes[-1][2] + 0.015, y0 - 0.055),
            0.035,
            0.11,
            facecolor="0.2",
            edgecolor="none",
            zorder=3,
        )
    )

    highlight_key = highlight.lower() if highlight is not None else None
    lines = []
    texts = []
    for key, label, xp, is_pupil in planes:
        on = highlight_key is not None and key == highlight_key
        color = accent_color if on else "0.45"
        hp = max(wide if is_pupil else tight, 0.13)
        (line,) = ax.plot(
            [xp, xp],
            [y0 - hp - 0.06, y0 + hp + 0.06],
            color=color,
            lw=2.0 if on else 1.0,
            ls="-" if on else "--",
            zorder=4,
        )
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

    return PlotResult(ax=ax, artists={"fill": fill, "line": lines, "text": texts})
