"""Core image primitives: log-scaled, diverging, and side-by-side comparison.

`imshow_log` is the most-repeated figure idiom this library exists to
replace: a coronagraph PSF or contrast map spans many decades of dynamic
range and a raw pixel can be exactly zero, which breaks `LogNorm` outright.
Every function here clips the data to a floor BEFORE building the norm, so
zero-valued pixels never propagate into a `LogNorm` construction.

All three functions route colormaps through `_style.cmap` (never a
hardcoded colormap name) and default `imshow` to `interpolation="nearest"`,
per the house rule that interpolating simulated detector data misrepresents
the pixels.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from eyepiece import _style
from eyepiece._result import MosaicResult, PlotResult


def imshow_log(
    image,
    *,
    ax=None,
    extent=None,
    floor=1e-20,
    vmin=None,
    vmax=None,
    cmap=None,
    colorbar=True,
    cbar_label=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a log-scaled image, clipped to a floor so zeros do not break LogNorm.

    The floor is applied to the data BEFORE the norm is built, so a
    zero-valued or negative pixel is silently lifted to `floor` rather than
    raising inside `LogNorm`. Returned `.update` re-applies the same floor
    and calls `set_data` on the existing `AxesImage`, never creating a new
    artist. A later `.update(new_image)` call with values outside the norm
    built from the FIRST image is not an error: those pixels render clipped
    to the colormap's end colors and the norm itself is not rescaled. Call
    `imshow_log` again (or build the norm from the full data range up front
    via `vmin`/`vmax`) if the range is expected to change.

    Args:
        image: 2D array-like of intensities.
        ax: Axes to draw into. None creates a new figure and axes.
        extent: `(left, right, bottom, top)` passed to `imshow`.
        floor: Minimum value the data is clipped to before norm/display.
        vmin: Norm lower bound. None uses `max(data.min(), floor)`.
        vmax: Norm upper bound. None uses `data.max()`.
        cmap: Colormap override; None uses the semantic "intensity" cmap.
        colorbar: Whether to attach a colorbar.
        cbar_label: Label for the colorbar.
        imshow_kw: Extra kwargs passed to `ax.imshow`, applied last.
        cbar_kw: Extra kwargs passed to `fig.colorbar`.

    Returns:
        A `PlotResult` with artists `"image"` (and `"cbar"` if drawn) and an
        `.update(new_image)` callable.
    """
    img = np.asarray(image, dtype=float)
    data = np.clip(img, floor, None)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    lo = max(float(np.nanmin(data)), floor) if vmin is None else vmin
    hi = float(np.nanmax(data)) if vmax is None else vmax
    norm = LogNorm(vmin=lo, vmax=hi)

    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    im = ax.imshow(
        data, norm=norm, cmap=_style.cmap("intensity", cmap), extent=extent, **kw
    )
    artists = {"image": im}

    if colorbar:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = ax.figure.colorbar(im, cax=cax, label=cbar_label, **(cbar_kw or {}))
        artists["cbar"] = cb

    def update(new_image):
        im.set_data(np.clip(np.asarray(new_image, dtype=float), floor, None))

    return PlotResult(ax=ax, artists=artists, update=update)


def _diverging_draw(
    ax, data, norm, resolved_cmap, extent, colorbar, cbar_label, imshow_kw, cbar_kw
):
    """Draw one image + inset colorbar under an already-built diverging norm.

    Shared by `imshow_diverging` and `triptych`'s ratio panel, which builds
    a norm centered on 1 rather than 0 but otherwise draws identically.

    Returns:
        An `(AxesImage, Colorbar | None)` tuple; the colorbar is None when
        `colorbar` is False.
    """
    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    im = ax.imshow(data, norm=norm, cmap=resolved_cmap, extent=extent, **kw)
    cb = None
    if colorbar:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = ax.figure.colorbar(im, cax=cax, label=cbar_label, **(cbar_kw or {}))
    return im, cb


def imshow_diverging(
    image,
    *,
    ax=None,
    extent=None,
    vlim=None,
    cmap=None,
    colorbar=True,
    cbar_label=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw an image on a symmetric linear norm centered on zero.

    Args:
        image: 2D array-like, typically a signed residual or difference map.
        ax: Axes to draw into. None creates a new figure and axes.
        extent: `(left, right, bottom, top)` passed to `imshow`.
        vlim: Symmetric norm bound; the norm spans `(-vlim, vlim)`. None
            uses `max(abs(data.min()), abs(data.max()))`.
        cmap: Colormap override; None uses the semantic "residual" cmap.
        colorbar: Whether to attach a colorbar.
        cbar_label: Label for the colorbar.
        imshow_kw: Extra kwargs passed to `ax.imshow`, applied last.
        cbar_kw: Extra kwargs passed to `fig.colorbar`.

    Returns:
        A `PlotResult` with artists `"image"` (and `"cbar"` if drawn). No
        `.update` -- the norm has no floor or other stateful transform to
        reapply.
    """
    data = np.asarray(image, dtype=float)
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    if vlim is None:
        vlim = max(abs(float(np.nanmin(data))), abs(float(np.nanmax(data))))
    norm = Normalize(vmin=-vlim, vmax=vlim)

    im, cb = _diverging_draw(
        ax,
        data,
        norm,
        _style.cmap("residual", cmap),
        extent,
        colorbar,
        cbar_label,
        imshow_kw,
        cbar_kw,
    )
    artists = {"image": im}
    if cb is not None:
        artists["cbar"] = cb

    return PlotResult(ax=ax, artists=artists)


def _shared_norm(images, kind, floor, vmin=None, vmax=None):
    """Build one norm from the min/max across all images in `images`.

    `vmin`/`vmax` pin either end instead of deriving it from the data;
    either may be given alone to pin one end while the other is still
    derived. For `kind="diverging"` the (pinned or derived) lo/hi are
    folded into the single symmetric `vlim` the diverging norm always
    uses -- a pin still yields a symmetric norm, never an asymmetric one.
    """
    lo = min(float(np.nanmin(img)) for img in images) if vmin is None else vmin
    hi = max(float(np.nanmax(img)) for img in images) if vmax is None else vmax
    if kind == "log":
        lo = max(lo, floor)
        return LogNorm(vmin=lo, vmax=hi)
    if kind == "diverging":
        vlim = max(abs(lo), abs(hi))
        return Normalize(vmin=-vlim, vmax=vlim)
    if kind == "linear":
        return Normalize(vmin=lo, vmax=hi)
    raise ValueError(f"unknown norm kind: {kind!r}")


def compare_row(
    images,
    titles=None,
    *,
    axes=None,
    norm="log",
    floor=1e-20,
    extent=None,
    cmap=None,
    cbar_label=None,
    vmin=None,
    vmax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a row of images sharing one norm and one colorbar.

    A single norm object is built from the min/max across ALL images and
    passed, by identity, to every panel's `imshow` -- panels are directly
    comparable, not merely rescaled to look alike. `norm="log"` first
    clips every image to `floor` so a zero pixel cannot break the shared
    `LogNorm`.

    Args:
        images: Sequence of 2D array-likes, one per panel. At least one
            image is required.
        titles: Optional sequence of per-panel titles, same length as
            `images`.
        axes: Axes to draw into, one per panel: either a sequence of Axes
            or a single bare Axes (only valid for a one-image call), both
            normalized to a 1D array via `numpy.atleast_1d`. None creates a
            new figure with `len(images)` panels in a row.
        norm: `"log"`, `"linear"`, or `"diverging"` -- which shared norm to
            build.
        floor: Clip floor used when `norm="log"`.
        extent: `(left, right, bottom, top)` passed to every panel's
            `imshow`.
        cmap: Colormap override; None uses the semantic "intensity" cmap
            (or "residual" when `norm="diverging"`).
        cbar_label: Label for the shared colorbar.
        vmin: Pins the shared norm's lower bound instead of deriving it
            from the data across all panels. May be given alone to pin
            only the lower bound while the upper bound is still derived.
            For `norm="diverging"` this still yields a symmetric norm
            (see `_shared_norm`), never an asymmetric one.
        vmax: Pins the shared norm's upper bound instead of deriving it
            from the data. May be given alone.
        imshow_kw: Extra kwargs passed to each panel's `ax.imshow`, applied
            last.
        cbar_kw: Extra kwargs passed to the shared colorbar's
            `fig.colorbar`, applied last.

    Returns:
        A `MosaicResult` whose `axes` is always a 1D array of length
        `len(images)` (never a 2D block, regardless of panel count), with
        `artists["image"]` a list of `AxesImage`, one per panel, and
        `artists["cbar"]` the single shared colorbar.

    Raises:
        ValueError: If `images` is empty.
    """
    if len(images) == 0:
        raise ValueError("compare_row needs at least one image")

    images = [np.asarray(img, dtype=float) for img in images]
    if norm == "log":
        images = [np.clip(img, floor, None) for img in images]

    created = axes is None
    if created:
        fig, panel_axes = plt.subplots(
            1, len(images), layout="constrained", squeeze=False
        )
        axes = panel_axes[0]
    else:
        axes = np.atleast_1d(axes)
        fig = axes[0].figure

    shared_norm = _shared_norm(images, norm, floor, vmin=vmin, vmax=vmax)
    semantic_cmap = "residual" if norm == "diverging" else "intensity"
    resolved_cmap = _style.cmap(semantic_cmap, cmap)

    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    ims = []
    for i, (ax, img) in enumerate(zip(axes, images, strict=True)):
        im = ax.imshow(img, norm=shared_norm, cmap=resolved_cmap, extent=extent, **kw)
        if titles is not None:
            ax.set_title(titles[i])
        ims.append(im)

    artists = {"image": ims}
    if created:
        cb = fig.colorbar(ims[-1], ax=axes, label=cbar_label, **(cbar_kw or {}))
    else:
        cax = axes[-1].inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = fig.colorbar(ims[-1], cax=cax, label=cbar_label, **(cbar_kw or {}))
    artists["cbar"] = cb

    return MosaicResult(axes=axes, artists=artists)


def _decade(values):
    """Power-of-ten exponent of a panel's peak, for a scale annotation.

    Args:
        values: Array-like of real values.

    Returns:
        `floor(log10(max(abs(values))))` as an int, or 0 when the peak is
        zero or non-finite.
    """
    peak = float(np.max(np.abs(values)))
    if not np.isfinite(peak) or peak == 0.0:
        return 0
    return int(np.floor(np.log10(peak)))


def _decade_panel(ax, data, name, cmap, extent, peak_all, zero_ratio, signed):
    """Draw one auto-decade-scaled panel with a machine-zero-aware title.

    Rescales `data` by its own power-of-ten peak so a field with tiny
    values (e.g. the residual imaginary part of a real, symmetric pupil,
    which sits near float64 machine zero) stays readable, and annotates the
    title with the scale factor applied. A panel that is identically zero
    is labeled as such rather than rescaled by an undefined factor, and a
    panel far below the field-wide peak (but not exactly zero) is flagged
    as machine noise rather than shown as if it were real structure.

    Args:
        ax: Axes to draw into.
        data: 2D array-like for this panel.
        name: Panel name used in the title.
        cmap: Resolved Colormap for this panel.
        extent: `(left, right, bottom, top)` passed to `imshow`.
        peak_all: The field-wide peak amplitude, for the machine-zero test.
        zero_ratio: A panel peak below `zero_ratio * peak_all` (and not
            exactly zero) is labeled machine zero.
        signed: Whether the panel data can be negative (Real/Imaginary) or
            is non-negative by construction (Amplitude).

    Returns:
        A `(AxesImage, Colorbar, Text)` tuple.
    """
    panel_peak = float(np.max(np.abs(data)))
    if panel_peak == 0.0:
        vmin = -1.0 if signed else 0.0
        im = ax.imshow(
            np.zeros_like(data),
            cmap=cmap,
            vmin=vmin,
            vmax=1.0,
            extent=extent,
            interpolation="nearest",
        )
        title = ax.set_title(f"{name}\n= 0 exactly")
    else:
        exp = _decade(data)
        scale = 10.0**exp
        lim = panel_peak / scale
        vmin = -lim if signed else 0.0
        im = ax.imshow(
            data / scale,
            cmap=cmap,
            vmin=vmin,
            vmax=lim,
            extent=extent,
            interpolation="nearest",
        )
        title_str = rf"{name}  [$\times 10^{{{exp}}}$]"
        if panel_peak < zero_ratio * peak_all:
            title_str += "\n<- machine zero"
        title = ax.set_title(title_str)

    cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    cb = ax.figure.colorbar(im, cax=cax)
    return im, cb, title


def show_field(
    field,
    *,
    fig=None,
    axes=None,
    extent=None,
    label=None,
    mask=None,
    amp_cmap=None,
    signed_cmap=None,
    phase_cmap=None,
):
    """Draw a complex field as Real / Imaginary over Amplitude / Phase.

    The four panels are laid out as::

        [[ Real,      Imaginary ],
         [ Amplitude, Phase     ]]

    The Real, Imaginary, and Amplitude panels are each auto-decade-scaled
    to their own peak (see `_decade_panel`), so a field with tiny values --
    the focal-plane imaginary part of a real, symmetric pupil comes out
    around 1e-16, which is float64 machine zero and an exact theorem being
    confirmed, not a small number -- stays readable instead of looking like
    a blank panel.

    Phase is undefined where there is no light, so it is masked rather than
    rendered as numerical noise: `mask` (a pupil, say) marks where the field
    has support, defaulting to wherever the amplitude clears a small
    fraction of its peak.

    Args:
        field: 2D array-like of complex values.
        fig: A Figure or SubFigure to build the 2x2 panel block inside.
            None (with `axes` also None) creates a new figure. Ignored when
            `axes` is given.
        axes: A `(2, 2)` array of Axes to draw into. None creates the grid
            from `fig` (or a new figure when `fig` is also None).
        extent: `(left, right, bottom, top)` passed to every panel's
            `imshow`.
        label: Optional bold panel-block label placed above the Real panel.
        mask: Boolean array-like, same shape as `field`, marking where
            phase is defined. None derives it from the amplitude.
        amp_cmap: Colormap override for the Amplitude panel; None uses the
            semantic "intensity" cmap.
        signed_cmap: Colormap override for the Real/Imaginary panels; None
            uses the semantic "residual" cmap.
        phase_cmap: Colormap override for the Phase panel; None uses the
            semantic "phase" cmap.

    Returns:
        A `MosaicResult` whose `axes` is the `(2, 2)` grid above, with
        `artists["image"]`, `artists["cbar"]`, and `artists["title"]` each
        a list of four, in `[Real, Imaginary, Amplitude, Phase]` order.
    """
    zero_ratio = 1e-10
    field = np.asarray(field)
    amp = np.abs(field)
    peak = float(amp.max())
    if mask is None:
        mask = amp > max(peak * 1e-8, np.finfo(float).tiny)
    mask = np.asarray(mask).astype(bool)

    if axes is not None:
        axes = np.asarray(axes)
    elif fig is not None:
        axes = np.asarray(fig.subplots(2, 2))
    else:
        _, axes = plt.subplots(2, 2, layout="constrained")

    resolved_signed_cmap = _style.cmap("residual", signed_cmap)
    resolved_amp_cmap = _style.cmap("intensity", amp_cmap)
    resolved_phase_cmap = _style.cmap("phase", phase_cmap).with_extremes(
        bad=axes[1, 1].get_facecolor()
    )

    ims = []
    cbs = []
    titles = []

    for ax, data, name in (
        (axes[0, 0], field.real, "Real"),
        (axes[0, 1], field.imag, "Imaginary"),
    ):
        im, cb, title = _decade_panel(
            ax, data, name, resolved_signed_cmap, extent, peak, zero_ratio, True
        )
        ims.append(im)
        cbs.append(cb)
        titles.append(title)

    im, cb, title = _decade_panel(
        axes[1, 0], amp, "Amplitude", resolved_amp_cmap, extent, peak, zero_ratio, False
    )
    ims.append(im)
    cbs.append(cb)
    titles.append(title)

    phase = np.ma.masked_where(~mask, np.angle(field))
    im = axes[1, 1].imshow(
        phase,
        cmap=resolved_phase_cmap,
        vmin=-np.pi,
        vmax=np.pi,
        extent=extent,
        interpolation="nearest",
    )
    title = axes[1, 1].set_title("Phase")
    cax = axes[1, 1].inset_axes([1.02, 0.0, 0.04, 1.0])
    cb = axes[1, 1].figure.colorbar(im, cax=cax, label="rad")
    ims.append(im)
    cbs.append(cb)
    titles.append(title)

    for ax in axes.ravel():
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])

    if label:
        axes[0, 0].text(
            -0.18,
            1.28,
            label,
            transform=axes[0, 0].transAxes,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    artists = {"image": ims, "cbar": cbs, "title": titles}
    return MosaicResult(axes=axes, artists=artists)


_RATIO_CLIP_PERCENTILE = 99.0


def triptych(
    a,
    b,
    *,
    mode="ratio",
    a_b_norm="log",
    titles=None,
    axes=None,
    ratio_clip=None,
    extent=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw A, B, and a panel comparing them, side by side.

    A and B are drawn with `compare_row` under one shared norm and one
    shared colorbar (`a_b_norm`, `"log"` or `"linear"`), so the two are
    directly comparable rather than merely rescaled to look alike. The
    third, comparison panel is independent of that shared norm and depends
    on `mode`::

        mode="ratio":    b / a, on a diverging norm centered on 1 (not 0).
        mode="residual": b - a, on the symmetric-about-0 norm
                          `imshow_diverging` already builds; reused
                          directly rather than reimplemented.

    Ratio orientation: `b / a` reads "how does B compare to A", matching
    the left-to-right A, B, comparison layout -- a ratio above 1 means B
    exceeds A at that pixel.

    Division-by-zero guard: a raw `b / a` is undefined wherever `a` is
    exactly zero (+-inf where b is nonzero, nan where b is also zero).
    Both are replaced before display rather than left to render
    undefined: nan (0/0) becomes 1.0, "no change", since neither value
    carries information about the other; +-inf becomes the panel's own
    clip bound (1 +/- clip, see below), so it renders fully saturated at
    the diverging colormap's extreme rather than raising or breaking the
    norm.

    Tight diverging clip: a raw ratio panel is often dominated by a
    handful of pixels where `a` is tiny, which would blow the norm out to
    a range where the interesting structure near 1.0 is invisible. The
    default clip is symmetric about 1: `clip` is the `_RATIO_CLIP_PERCENTILE`
    (99th) percentile of `abs(ratio - 1)` over the finite ratio values, so
    the norm spans `[1 - clip, 1 + clip]` -- wide enough to show the bulk
    of the panel without letting a handful of outliers wash out real
    structure. Pass `ratio_clip` to override this rule with a fixed value.

    Args:
        a: 2D array-like, the first panel (the reference/"before").
        b: 2D array-like, the second panel (the comparison/"after"), same
            shape as `a`.
        mode: `"ratio"` or `"residual"`; see above.
        a_b_norm: `"log"` or `"linear"` -- the norm A and B share, passed
            through to `compare_row`'s `norm`. Independent of the
            comparison panel's own norm. `compare_row` also takes
            `"diverging"`, but a triptych does not: the comparison panel is
            already the diverging one, and rejecting the value now leaves
            room to accept it later, which the reverse would not.
        titles: Optional length-3 sequence of panel titles. None uses
            `("A", "B", "B / A")` for `mode="ratio"` or
            `("A", "B", "B - A")` for `mode="residual"`.
        axes: Length-3 sequence of Axes to draw into. None creates a new
            figure via `plt.subplots(1, 3, layout="constrained")`.
        ratio_clip: Fixed clip value for the ratio panel's norm, which
            then spans `[1 - ratio_clip, 1 + ratio_clip]`. None derives it
            from the data (see above). Ignored when `mode="residual"`.
        extent: `(left, right, bottom, top)` passed to all three panels'
            `imshow`, so a triptych can carry axis units the way every
            other image primitive here can.
        imshow_kw: Extra kwargs passed to all three panels' `ax.imshow`,
            applied last, exactly as in `compare_row`.
        cbar_kw: Extra kwargs passed to both colorbars' `fig.colorbar`,
            applied last, exactly as in `compare_row`.

    Returns:
        A `MosaicResult` whose `axes` is the length-3 array of panels
        (A, B, comparison), with `artists["image"]` the list of three
        `AxesImage` in that order and `artists["cbar"]` the list of two
        `Colorbar`: the A/B shared one and the comparison panel's own.

    Raises:
        ValueError: If `mode` is not `"ratio"` or `"residual"`, if
            `a_b_norm` is not `"log"` or `"linear"`, or if `axes` or
            `titles` is given with anything other than three entries. Every
            check runs before anything is drawn.
    """
    if mode not in ("ratio", "residual"):
        raise ValueError(f"unknown mode: {mode!r}")
    if a_b_norm not in ("log", "linear"):
        raise ValueError(f"unknown a_b_norm: {a_b_norm!r}; use 'log' or 'linear'")
    if axes is not None:
        axes = np.atleast_1d(axes)
        if axes.size != 3:
            raise ValueError(f"triptych needs 3 axes, got {axes.size}")
    if titles is not None and len(titles) != 3:
        raise ValueError(f"triptych needs 3 titles, got {len(titles)}")

    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if titles is None:
        comparison_title = "B / A" if mode == "ratio" else "B - A"
        titles = ("A", "B", comparison_title)

    if axes is None:
        _, axes = plt.subplots(1, 3, layout="constrained")

    ab_result = compare_row(
        [a_arr, b_arr],
        titles=list(titles[:2]),
        axes=axes[:2],
        norm=a_b_norm,
        extent=extent,
        imshow_kw=imshow_kw,
        cbar_kw=cbar_kw,
    )

    if mode == "ratio":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = b_arr / a_arr
        finite = ratio[np.isfinite(ratio)]
        if ratio_clip is not None:
            clip = float(ratio_clip)
        elif finite.size:
            clip = float(np.percentile(np.abs(finite - 1.0), _RATIO_CLIP_PERCENTILE))
        else:
            clip = 1.0
        clip = max(clip, 1e-6)
        safe_ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0 + clip, neginf=1.0 - clip)
        cmp_norm = Normalize(vmin=1.0 - clip, vmax=1.0 + clip)
        cmp_im, cmp_cb = _diverging_draw(
            axes[2],
            safe_ratio,
            cmp_norm,
            _style.cmap("residual"),
            extent,
            True,
            None,
            imshow_kw,
            cbar_kw,
        )
    else:
        cmp_result = imshow_diverging(
            b_arr - a_arr,
            ax=axes[2],
            extent=extent,
            imshow_kw=imshow_kw,
            cbar_kw=cbar_kw,
        )
        cmp_im = cmp_result.artists["image"]
        cmp_cb = cmp_result.artists["cbar"]

    axes[2].set_title(titles[2])

    artists = {
        "image": [*ab_result.artists["image"], cmp_im],
        "cbar": [ab_result.artists["cbar"], cmp_cb],
    }
    return MosaicResult(axes=axes, artists=artists)
