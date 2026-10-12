"""Scalar-colored point clouds with colorbars inside caller-owned geometry."""

import matplotlib.pyplot as plt
import numpy as np

from eyepiece import _style
from eyepiece._result import PlotResult


def scatter_values(
    x,
    y,
    values,
    *,
    ax=None,
    cmap=None,
    vmin=None,
    vmax=None,
    cbar_label=None,
    colorbar=True,
    scatter_kw=None,
    cbar_kw=None,
):
    """Color points by scalar values and attach an inset colorbar.

    Args:
        x: One-dimensional x coordinates.
        y: One-dimensional y coordinates, shaped like x.
        values: One-dimensional scalar colors, shaped like x.
        ax: Single Axes, or None to create a figure.
        cmap: hwostyle role, registered colormap name, or Colormap. None
            uses "probability". The "statistic" role falls back to PuOr
            only when the active style does not define it.
        vmin: Lower color limit; None uses the data minimum.
        vmax: Upper color limit; None uses the data maximum.
        cbar_label: Optional colorbar label.
        colorbar: Whether to draw a colorbar to the right of ax.
        scatter_kw: Overrides passed to ax.scatter.
        cbar_kw: Overrides passed to fig.colorbar. Its cax is owned here.

    Returns:
        PlotResult with "scatter" and, if requested, "cbar". The colorbar
        Axes is in insets[0], at [1.02, 0, 0.04, 1] in axes coordinates.

    Example::

        import eyepiece as ep
        result = ep.scatter_values([0, 1], [1, 2], [0.2, 0.8], cmap="probability")
    """
    x, y, values = (np.asarray(v) for v in (x, y, values))
    if x.ndim != 1 or y.shape != x.shape or values.shape != x.shape:
        raise ValueError(
            "expected x, y, values shape (N,); "
            f"received {x.shape}, {y.shape}, {values.shape}"
        )
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    options = {
        "c": values,
        "cmap": _style.cmap("probability", cmap),
        "vmin": vmin,
        "vmax": vmax,
    }
    options.update(scatter_kw or {})
    scatter = ax.scatter(x, y, **options)
    artists = {"scatter": scatter}
    insets = ()
    if colorbar:
        options = {"label": cbar_label}
        options.update(cbar_kw or {})
        if "ax" in options or "cax" in options:
            raise ValueError("cbar_kw cannot replace the owned inset axes")
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        artists["cbar"] = ax.figure.colorbar(scatter, cax=cax, **options)
        insets = (cax,)
    return PlotResult(ax, artists, insets=insets)
