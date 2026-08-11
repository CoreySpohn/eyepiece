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

    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    im = ax.imshow(
        data, norm=norm, cmap=_style.cmap("residual", cmap), extent=extent, **kw
    )
    artists = {"image": im}

    if colorbar:
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = ax.figure.colorbar(im, cax=cax, label=cbar_label, **(cbar_kw or {}))
        artists["cbar"] = cb

    return PlotResult(ax=ax, artists=artists)


def _shared_norm(images, kind, floor):
    """Build one norm from the min/max across all images in `images`."""
    lo = min(float(np.nanmin(img)) for img in images)
    hi = max(float(np.nanmax(img)) for img in images)
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

    shared_norm = _shared_norm(images, norm, floor)
    semantic_cmap = "residual" if norm == "diverging" else "intensity"
    resolved_cmap = _style.cmap(semantic_cmap, cmap)

    ims = []
    for i, (ax, img) in enumerate(zip(axes, images, strict=True)):
        im = ax.imshow(
            img,
            norm=shared_norm,
            cmap=resolved_cmap,
            extent=extent,
            interpolation="nearest",
            origin="lower",
        )
        if titles is not None:
            ax.set_title(titles[i])
        ims.append(im)

    artists = {"image": ims}
    if created:
        cb = fig.colorbar(ims[-1], ax=axes, label=cbar_label)
    else:
        cax = axes[-1].inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = fig.colorbar(ims[-1], cax=cax, label=cbar_label)
    artists["cbar"] = cb

    return MosaicResult(axes=axes, artists=artists)
