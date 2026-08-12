"""Radial-profile line plot, contrast curve, and a hwoutils-backed convenience.

`plot_radial` takes precomputed `(r, values)` arrays -- this library never
recomputes a profile itself; the computation belongs to `hwoutils.radial`
(or whichever library produced the arrays), not to a plotting primitive.
`radial_profile_plot`, at the bottom of this module, is the one function
here that reaches outside that rule: it calls `hwoutils.radial.radial_profile`
directly, under the `eyepiece[hwo]` extra, then hands the result to
`plot_radial` unchanged.

`plot_contrast_curve` is the same idea with the annotations that make a
contrast curve readable: shaded IWA/OWA exclusion regions and labeled
reference floor curves. Both are axes-level scenery, not per-call data, so
calling `plot_contrast_curve` twice on the SAME axes -- the normal way to
compare two curves -- draws each call's own curve every time but the
IWA/OWA shading, labels, and floor curves only once. See the "Annotations
draw once per axes" note on `plot_contrast_curve` for the mechanism.
"""

import matplotlib.pyplot as plt
import numpy as np

from eyepiece import _style
from eyepiece._result import PlotResult


def plot_radial(r, values, *, ax=None, label=None, color=None, log=False, line_kw=None):
    """Draw a profile of precomputed values against radial separation.

    Args:
        r: 1D array-like of separations.
        values: 1D array-like of profile values, same length as `r`.
        ax: Axes to draw into. None creates a new figure and axes.
        label: Optional legend label for the curve.
        color: Line color override; None uses `_style.color(0)`.
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to `ax.plot`, applied last.

    Returns:
        A `PlotResult` with artist `"line"` (the `Line2D`).
    """
    r_arr = np.asarray(r, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    lkw = {"color": _style.color(0, color), "label": label, **(line_kw or {})}
    (line,) = ax.plot(r_arr, v_arr, **lkw)

    if log:
        ax.set_yscale("log")

    return PlotResult(ax=ax, artists={"line": line})


_ANNOTATIONS_DRAWN_ATTR = "_eyepiece_contrast_annotations_drawn"


def plot_contrast_curve(
    r,
    contrast,
    *,
    ax=None,
    iwa=None,
    owa=None,
    floors=None,
    label=None,
    color=None,
    log=True,
    line_kw=None,
    span_kw=None,
    floor_kw=None,
):
    """Draw a contrast-vs-separation curve, annotated for readability.

    Two kinds of artist live on this axes: the contrast curve itself,
    drawn fresh on every call, and the working-angle/floor annotations,
    which are axes-level scenery rather than per-call data.

    Annotations draw once per axes: comparing two contrast curves is the
    normal way to use this function -- call it twice with the same `ax`
    and a different `contrast`. A second call's IWA/OWA shading and floor
    curves would exactly duplicate the first call's, so this function
    stamps a private marker attribute (named by the module-level
    `_ANNOTATIONS_DRAWN_ATTR` constant) directly onto `ax` the first time
    it draws the annotations, and every later call that is handed that
    SAME `ax` finds the attribute already set and skips the annotation
    block entirely -- only the curve line is drawn again. The state lives
    on the axes object itself, not in a global registry keyed by id() or
    similar, so it is visible on the object a reader already has in hand
    (`getattr(ax, "_eyepiece_contrast_annotations_drawn", False)`) and it
    can never leak between different axes or figures, or across process
    lifetimes.

    Args:
        r: 1D array-like of separations.
        contrast: 1D array-like of contrast values, same length as `r`.
        ax: Axes to draw into. None creates a new figure and axes.
        iwa: Inner working angle, in the same units as `r`. None omits the
            marker. Shaded from the axes' current left x-limit to `iwa`.
        owa: Outer working angle, in the same units as `r`. None omits the
            marker. Shaded from `owa` to the axes' current right x-limit.
        floors: Optional iterable of `(r, y, label)` reference-floor
            curves, each drawn as its own dashed line beneath the main
            curve (in draw order, not z-order -- see `floor_kw` to
            override the z-order if that distinction matters).
        label: Optional legend label for the main curve.
        color: Main curve color override; None uses `_style.color(0)`.
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to `ax.plot` for the main curve,
            applied last.
        span_kw: Extra kwargs passed to `ax.axvspan` for the IWA/OWA
            shading, applied last.
        floor_kw: Extra kwargs passed to `ax.plot` for every floor curve,
            applied last.

    Returns:
        A `PlotResult`. `artists["line"]` is the main curve's `Line2D`,
        drawn on every call. `artists["lines"]` (only on the first call on
        a given `ax`, and only when `floors` is given) is the list of
        floor curves' `Line2D`, in `floors` order. `artists["fill"]`
        (only on the first call, one entry per marker actually given) is
        the list of `axvspan` shading patches, in `[iwa, owa]` order.
        `artists["text"]` (same first-call-only condition) is the
        matching list of "IWA"/"OWA" label `Text` artists.
    """
    r_arr = np.asarray(r, dtype=float)
    c_arr = np.asarray(contrast, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    lkw = {"color": _style.color(0, color), "label": label, **(line_kw or {})}
    (line,) = ax.plot(r_arr, c_arr, **lkw)
    artists = {"line": line}

    if not getattr(ax, _ANNOTATIONS_DRAWN_ATTR, False):
        skw = {
            "color": _style.neutral(0.2),
            "alpha": 0.5,
            "zorder": 0,
            **(span_kw or {}),
        }
        fills = []
        texts = []
        if iwa is not None:
            left, _right = ax.get_xlim()
            fills.append(ax.axvspan(left, iwa, **skw))
            texts.append(
                ax.text(
                    iwa,
                    1.02,
                    "IWA",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
            )
        if owa is not None:
            _left, right = ax.get_xlim()
            fills.append(ax.axvspan(owa, right, **skw))
            texts.append(
                ax.text(
                    owa,
                    1.02,
                    "OWA",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                )
            )
        if floors is not None:
            lines = []
            for i, (f_r, f_y, f_label) in enumerate(floors):
                fkw = {
                    "color": _style.color(i + 1),
                    "ls": "--",
                    "label": f_label,
                    **(floor_kw or {}),
                }
                (fl,) = ax.plot(f_r, f_y, **fkw)
                lines.append(fl)
            if lines:
                artists["lines"] = lines
        if fills:
            artists["fill"] = fills
        if texts:
            artists["text"] = texts
        setattr(ax, _ANNOTATIONS_DRAWN_ATTR, True)

    if log:
        ax.set_yscale("log")

    return PlotResult(ax=ax, artists=artists)


def radial_profile_plot(
    image,
    pixscale_lod,
    *,
    center=None,
    nbins=None,
    ax=None,
    label=None,
    color=None,
    log=False,
    line_kw=None,
):
    """Compute a radial profile of `image` via hwoutils, then plot it.

    Computation is `hwoutils.radial.radial_profile`, never reimplemented
    here. That function's parameter is named `pixel_scale_arcsec`, but its
    own docstring describes it as a generic pixels-to-physical-units
    factor and names lambda/D-per-pixel as a use case, so passing
    `pixscale_lod` through it is correct despite the upstream name.

    Args:
        image: 2D array-like image to profile.
        pixscale_lod: Pixel scale in lambda/D per pixel, forwarded to
            `hwoutils.radial.radial_profile`'s `pixel_scale_arcsec`
            argument.
        center: Center coordinates `(cy, cx)`. None uses the geometric
            center.
        nbins: Number of radial bins. None uses `hwoutils`' own default.
        ax: Axes to draw into. None creates a new figure and axes.
        label: Optional legend label for the curve.
        color: Line color override; None uses `_style.color(0)`.
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to `ax.plot`, applied last.

    Returns:
        A `PlotResult` with artist `"line"` (the `Line2D`); see
        `plot_radial`.

    Raises:
        ImportError: If `hwoutils` is not installed. Install the
            `eyepiece[hwo]` extra to enable this function.
    """
    try:
        from hwoutils import radial
    except ImportError:
        raise ImportError(
            "radial_profile_plot needs hwoutils: pip install eyepiece[hwo]"
        ) from None

    separations, profile = radial.radial_profile(
        np.asarray(image), pixscale_lod, center=center, nbins=nbins
    )
    return plot_radial(
        separations,
        profile,
        ax=ax,
        label=label,
        color=color,
        log=log,
        line_kw=line_kw,
    )
