"""Fixed panel furniture for a relay between clips."""

import math

import matplotlib
from matplotlib.transforms import ScaledTranslation

from eyepiece import _style
from eyepiece._result import MotionContract, PlotResult


def carried_tag(
    ax, text, *, corner="upper left", pad_pt=6.0, fontsize=None, level=0.55,
):  # fmt: skip
    """Place a small italic relay note in a panel's top corner.

    The note is neutral furniture: it fades with an axes Stage, stays fixed
    during label lifts, and declares its geometry at rest. Meaning checks
    do not treat it as data. Readability checks allow its small type while
    retaining overlap, on-mark, and clipping checks.

    Args:
        ax: The panel that carries the note.
        text: The note, such as "(from clip 1)".
        corner: "upper left" or "upper right".
        pad_pt: Padding from the top and side of the axes, in points.
        fontsize: Type size in points; None is 75% of the current font size.
        level: Neutral tone in [0, 1], resolved at call time.

    Returns:
        A PlotResult with ``artists["text"]`` and a rest MotionContract.

    Example::

        tag = ep.carried_tag(ax, "(from clip 1)")
        stage = ep.Stage({"panel": ax})
    """
    if corner not in ("upper left", "upper right"):
        raise ValueError("carried_tag corner is 'upper left' or 'upper right'")
    pad_pt = float(pad_pt)
    fontsize = (
        float(matplotlib.rcParams["font.size"]) * 0.75
        if fontsize is None
        else float(fontsize)
    )
    if not math.isfinite(pad_pt) or pad_pt < 0:
        raise ValueError("carried_tag pad_pt must be finite and nonnegative")
    if not math.isfinite(fontsize) or fontsize <= 0:
        raise ValueError("carried_tag fontsize must be finite and positive")
    if not 0 <= float(level) <= 1:
        raise ValueError("carried_tag level must be in [0, 1]")
    right = corner == "upper right"
    offset = ScaledTranslation(
        (-pad_pt if right else pad_pt) / 72,
        -pad_pt / 72,
        ax.get_figure(root=True).dpi_scale_trans,
    )
    note = ax.text(
        1 if right else 0,
        1,
        str(text),
        transform=ax.transAxes + offset,
        ha="right" if right else "left",
        va="top",
        fontsize=fontsize,
        fontstyle="italic",
        color=_style.neutral(level),
    )
    note.set_in_layout(False)
    note._eyepiece_furniture = True
    return PlotResult(
        ax=ax, artists={"text": note}, contract=MotionContract(rest=(note,))
    )
