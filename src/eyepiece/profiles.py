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

Colors cycle per axes: since two calls on one axes is the normal usage,
every curve this module draws without an explicit `color` takes the next
color of the active palette, counted per axes, so a second call is
distinguishable from the first instead of repeating `color(0)`. The
counter is shared by `plot_radial`, `plot_contrast_curve`, and the floor
curves, and lives in the same per-axes state dict as the annotation
tracking (`ax._eyepiece_profile_state`, named by the module-level
`_STATE_ATTR` constant). An explicit `color` wins and does not advance the
counter -- a caller who names every color should not shift the palette for
the calls that did not.

Axis labels: `plot_radial` and `plot_contrast_curve` are unit-agnostic --
`r` may be lambda/D, arcsec, AU, or pixels, and neither function can know
which, so both only take an optional `xlabel`/`ylabel` and never guess one.
`radial_profile_plot` DOES know its units (it takes `pixscale_lod`
explicitly), so it defaults the x label to a lambda/D separation label,
still overridable by an explicit `xlabel`.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from eyepiece import _style
from eyepiece._result import PlotResult

_STATE_ATTR = "_eyepiece_profile_state"


def _state(ax):
    """The per-axes state dict for `ax`, created on first use.

    Holds `"annotations"` (drawn-annotation artists, keyed by marker kind)
    and `"color_index"` (the palette position the next unnamed color takes).
    """
    state = getattr(ax, _STATE_ATTR, None)
    if state is None:
        state = {"annotations": {}, "color_index": 0}
        setattr(ax, _STATE_ATTR, state)
    return state


def _next_color(ax, override=None):
    """Next palette color for `ax`, advancing its per-axes color index.

    An `override` wins and leaves the index alone; see the module docstring.
    """
    if override is not None:
        return override
    state = _state(ax)
    color = _style.color(state["color_index"])
    state["color_index"] += 1
    return color


def plot_radial(
    r,
    values,
    *,
    ax=None,
    label=None,
    color=None,
    log=False,
    line_kw=None,
    xlabel=None,
    ylabel=None,
):
    """Draw a profile of precomputed values against radial separation.

    Args:
        r: 1D array-like of separations.
        values: 1D array-like of profile values, same length as `r`.
        ax: Axes to draw into. None creates a new figure and axes.
        label: Optional legend label for the curve.
        color: Line color override; None takes the next color of the axes'
            palette cycle, so a second call on the same axes is drawn in a
            different color (see the module docstring).
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to `ax.plot`, applied last.
        xlabel: Optional x-axis label. None sets no label -- `r`'s units
            (lambda/D, arcsec, AU, pixels, ...) are the caller's to know,
            not this function's to guess.
        ylabel: Optional y-axis label. None sets no label, for the same
            reason.

    Returns:
        A `PlotResult` with artist `"line"` (the `Line2D`).
    """
    r_arr = np.asarray(r, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    lkw = {"color": _next_color(ax, color), "label": label, **(line_kw or {})}
    (line,) = ax.plot(r_arr, v_arr, **lkw)

    if log:
        ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return PlotResult(ax=ax, artists={"line": line})


def _still_attached(artists, ax):
    """Whether every artist in `artists` is non-empty and still part of `ax`.

    `Axes.clear()` detaches every child artist it removes -- each one's
    `.axes` becomes `None` -- without touching arbitrary Python attributes
    set on the Axes object itself (verified directly: the cached state
    dict this module stamps onto `ax` survives a `clear()` call, but the
    artists it references do not). Checking `.axes` here, rather than
    trusting the cached state alone, is what makes a post-`clear()` call
    redraw instead of silently staying poisoned: the axes itself is the
    source of truth, and the state dict is only a cache of it.
    """
    return bool(artists) and all(a.axes is ax for a in artists)


_SPAN_OUTER = 1e9

# IWA/OWA labels sit just INSIDE the top of the axes. Above it (the obvious
# place) is exactly where `set_title` draws, so any titled contrast curve --
# which is most of them -- collided with its own annotations.
_LABEL_Y = 0.98


def _span(ax, inner, outer_sign, span_kw):
    """Shade from `inner` to the axes edge, permanently.

    The shading marks a region the instrument cannot see, so it has to
    reach the edge of the axes whatever the x limits turn out to be -- and
    they are not known here: autoscaling runs at draw time, and a second
    curve drawn later legitimately widens them. Reading `ax.get_xlim()`
    and shading between those numbers would freeze the annotation at the
    limits of the moment, leaving unshaded axes on the excluded side.

    So the region is a `Rectangle` whose y runs 0 to 1 in axes coordinates
    (`ax.get_xaxis_transform()`) and whose x runs from `inner` to
    `_SPAN_OUTER` past it, in data coordinates, on the side `outer_sign`
    picks (-1 for an inner working angle, +1 for an outer one). It is
    added with `add_artist` rather than `add_patch`, which keeps it out of
    the data limits: the x range of a contrast curve should be set by the
    data, not by an annotation reaching a billion units off-screen. The
    axes clips it to its own box on every draw, so it always reaches the
    edge and never overshoots it. `add_artist` still sets the artist's
    `.axes`, so `_still_attached` reads it exactly as it reads a patch.

    Args:
        ax: Axes to draw into.
        inner: Data x coordinate of the edge facing the visible region.
        outer_sign: -1 to extend left of `inner`, +1 to extend right.
        span_kw: Extra `Rectangle` kwargs, applied last.

    Returns:
        The `Rectangle`.
    """
    kw = {"color": _style.neutral(0.2), "alpha": 0.5, "zorder": 0, **(span_kw or {})}
    width = outer_sign * _SPAN_OUTER
    x0 = inner if width > 0 else inner + width
    rect = Rectangle(
        (x0, 0.0), abs(width), 1.0, transform=ax.get_xaxis_transform(), **kw
    )
    ax.add_artist(rect)
    return rect


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
    xlabel=None,
    ylabel=None,
):
    """Draw a contrast-vs-separation curve, annotated for readability.

    Two kinds of artist live on this axes: the contrast curve itself,
    drawn fresh on every call, and the working-angle/floor annotations,
    which are axes-level scenery rather than per-call data.

    Annotations draw once per axes: comparing two contrast curves is the
    normal way to use this function -- call it twice with the same `ax`
    and a different `contrast`. A second call that repeats the same
    `iwa`/`owa`/`floors` would exactly duplicate the first call's
    annotations, so this function tracks, per `ax` and per marker KIND
    (`"iwa"`, `"owa"`, `"floors"`, tracked independently of each other),
    the artists it already drew for that kind. That record lives under
    `"annotations"` in the per-axes state dict stamped directly onto `ax`
    (named by the module-level `_STATE_ATTR` constant), so it is visible on
    the object a reader already has in hand (`ax._eyepiece_profile_state`)
    and can never leak between different axes or figures.

    A kind is skipped only when it was both drawn before AND its artists
    are still attached to `ax` (see `_still_attached`); this is what keeps
    the mechanism honest against two failure modes a plain "has this
    function run before" flag would get wrong:

    - A first call that supplies no markers must not poison a later call
      on the same `ax` that does: since each kind is gated independently,
      an `iwa` requested for the first time on call 2 is not blocked by
      call 1 never having drawn it.
    - `ax.clear()` removes the drawn artists from `ax` but does not touch
      this module's cached state dict (it is a plain attribute, not
      matplotlib state) -- so a call after `clear()` would otherwise find
      stale "already drawn" state pointing at artists no longer on the
      axes. Checking `.axes is ax` on the cached artists before trusting
      them catches exactly this: on a cleared axes the check fails and the
      annotation redraws.

    Args:
        r: 1D array-like of separations.
        contrast: 1D array-like of contrast values, same length as `r`.
        ax: Axes to draw into. None creates a new figure and axes.
        iwa: Inner working angle, in the same units as `r`. None omits the
            marker. Everything left of `iwa` is shaded, out to the left
            edge of the axes however the limits later change (see `_span`).
        owa: Outer working angle, in the same units as `r`. None omits the
            marker. Everything right of `owa` is shaded, out to the right
            edge of the axes.
        floors: Optional iterable of `(r, y, label)` reference-floor
            curves -- a fundamental limit typically sitting below the main
            curve in value (photon noise, speckle residuals, and the
            like) -- each drawn as its own dashed line, in the next
            palette colors after the main curve's.
        label: Optional legend label for the main curve.
        color: Main curve color override; None takes the next color of the
            axes' palette cycle, so a second call on the same axes is drawn
            in a different color (see the module docstring).
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to `ax.plot` for the main curve,
            applied last.
        span_kw: Extra kwargs passed to the IWA/OWA shading `Rectangle`,
            applied last.
        floor_kw: Extra kwargs passed to `ax.plot` for every floor curve,
            applied last.
        xlabel: Optional x-axis label. None sets no label -- `r`'s units
            are the caller's to know, not this function's to guess.
        ylabel: Optional y-axis label. None sets no label, for the same
            reason.

    Returns:
        A `PlotResult`. `artists["line"]` is the main curve's `Line2D`,
        drawn on every call. `artists["lines"]` is the list of floor
        curves' `Line2D`, in `floors` order -- present only on the call
        that actually draws them (see above). `artists["fill"]` is the
        list of shading regions actually drawn THIS call, in `[iwa, owa]`
        order when both are new this call -- each is a `Rectangle` patch
        rather than the `fill_between`-style `PolyCollection` the `"fill"`
        key's docstring names as the usual case, which `ARTIST_KEYS`
        allows: it is a convention, not an enforced schema, and
        `Rectangle` is the closest documented key for a shaded region.
        `artists["text"]` (same this-call-only condition) is the matching
        list of "IWA"/"OWA" label `Text` artists. Any of
        `"lines"`/`"fill"`/`"text"` is absent from a call that draws
        nothing new for that kind.
    """
    r_arr = np.asarray(r, dtype=float)
    c_arr = np.asarray(contrast, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    lkw = {"color": _next_color(ax, color), "label": label, **(line_kw or {})}
    (line,) = ax.plot(r_arr, c_arr, **lkw)
    artists = {"line": line}

    drawn = _state(ax)["annotations"]

    if iwa is not None and not _still_attached(drawn.get("iwa"), ax):
        fill = _span(ax, iwa, -1, span_kw)
        text = ax.text(
            iwa,
            _LABEL_Y,
            "IWA",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )
        drawn["iwa"] = [fill, text]
        artists.setdefault("fill", []).append(fill)
        artists.setdefault("text", []).append(text)

    if owa is not None and not _still_attached(drawn.get("owa"), ax):
        fill = _span(ax, owa, 1, span_kw)
        text = ax.text(
            owa,
            _LABEL_Y,
            "OWA",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )
        drawn["owa"] = [fill, text]
        artists.setdefault("fill", []).append(fill)
        artists.setdefault("text", []).append(text)

    if floors is not None and not _still_attached(drawn.get("floors"), ax):
        lines = []
        for f_r, f_y, f_label in floors:
            fkw = {
                "color": _next_color(ax),
                "ls": "--",
                "label": f_label,
                **(floor_kw or {}),
            }
            (fl,) = ax.plot(f_r, f_y, **fkw)
            lines.append(fl)
        drawn["floors"] = lines
        if lines:
            artists["lines"] = lines

    if log:
        ax.set_yscale("log")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return PlotResult(ax=ax, artists=artists)


_LOD_SEPARATION_XLABEL = r"$r$ [$\lambda/D$]"


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
    xlabel=None,
    ylabel=None,
):
    """Compute a radial profile of `image` via hwoutils, then plot it.

    Computation is `hwoutils.radial.radial_profile`, never reimplemented
    here. That function's parameter is named `pixel_scale_arcsec`, but its
    own docstring describes it as a generic pixels-to-physical-units
    factor and names lambda/D-per-pixel as a use case, so passing
    `pixscale_lod` through it is correct despite the upstream name.

    Unlike `plot_radial`, this function DOES know its units -- the caller
    hands it a lambda/D pixel scale explicitly -- so it defaults the x
    label to a lambda/D separation label in LaTeX mathtext, matching
    `layout.py`'s mathtext style (`layout.label_lod` labels a 2D image's
    x/y spatial axes the same way; this is the equivalent for a 1D
    separation axis, which `label_lod` does not fit). An explicit `xlabel`
    still overrides it.

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
        color: Line color override, forwarded to `plot_radial`; None takes
            the next color of the axes' palette cycle.
        log: Whether to set a log y-scale.
        line_kw: Extra kwargs passed to `ax.plot`, applied last.
        xlabel: Optional x-axis label override. None uses the default
            lambda/D separation label described above.
        ylabel: Optional y-axis label. None sets no label -- the profile's
            value units are the caller's to know, not this function's to
            guess.

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
    resolved_xlabel = _LOD_SEPARATION_XLABEL if xlabel is None else xlabel
    return plot_radial(
        separations,
        profile,
        ax=ax,
        label=label,
        color=color,
        log=log,
        line_kw=line_kw,
        xlabel=resolved_xlabel,
        ylabel=ylabel,
    )
