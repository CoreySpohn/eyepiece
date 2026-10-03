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

`kymograph` stacks a cut through an image against a second variable (a
time, a wavelength) as a log image on the pixel centers given. The overlay
annotations, `overlay_circle`, `overlay_line`, and `ruler`, draw scenery that
stays legible over bright and dark pixels alike, and `bracket` groups a run
of things along the horizontal axis under one label.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from eyepiece import _style
from eyepiece._phasor import (
    HEAD_LENGTH,
    HEAD_PER_MARKER,
    HEAD_WIDTH,
    _Arrow,
    _outward_alignment,
)
from eyepiece._result import MosaicResult, PlotResult


def _hide_index_ticks(axes, extent):
    """Drop the tick labels when the axes carry no physical coordinates.

    Without an `extent`, an image's axes are raw array indices, which are
    almost never what the reader is meant to measure -- they add a frame of
    numbers to a picture whose units live in the colorbar. `show_field` has
    applied this rule since it was ported; every image primitive here
    applies it too, so the whole module answers "no extent" the same way.

    Pass an `extent` to get real coordinates and their ticks back.
    """
    if extent is not None:
        return
    for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel():
        ax.set_xticks([])
        ax.set_yticks([])


def display_limits(image, *, low=1.0, high=99.5, low_scale=1.0, positive=False):
    """Compute `(vmin, vmax)` display bounds from the data itself.

    Deriving display bounds from percentiles rather than from the raw
    min/max is what keeps one hot pixel or one dead pixel from setting the
    whole scale, and it is hand-rolled constantly. The parts that are easy
    to get wrong, and that every hand-rolled copy solves slightly
    differently, are the ones this owns: dropping non-finite samples,
    dropping non-positive ones before a log scale, and still returning an
    ordered, finite pair when nothing survives that filtering.

    The bounds come back as plain floats for a primitive's `vmin`/`vmax`,
    rather than as a norm object, so they compose with every image
    primitive here and with a hand-built norm equally well.

    Args:
        image: Array-like of values to derive the bounds from. Shape is
            irrelevant; the data is flattened.
        low: Percentile in [0, 100] for the lower bound.
        high: Percentile in [0, 100] for the upper bound.
        low_scale: Factor applied to the lower bound after the percentile.
            `low=50.0, low_scale=0.5` puts the floor half a median below the
            data, which is what a rate map whose median IS its background
            needs: a floor referenced to the peak instead washes the
            structure out.
        positive: Whether to drop values <= 0 before computing. Required
            before a log scale, where a zero or negative sample would
            otherwise drag the floor onto it.

    Returns:
        An ordered `(vmin, vmax)` tuple of plain floats. Data with nothing
        left to measure -- all non-finite, or nothing positive under
        `positive=True` -- yields `(0.0, 1.0)` rather than raising, and
        constant data is widened around its value, so the result is always
        usable as a norm's bounds.

    Raises:
        ValueError: If `low` is not below `high`.
    """
    if low >= high:
        raise ValueError(f"low must be below high; got low={low}, high={high}")
    data = np.asarray(image, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if positive:
        data = data[data > 0.0]
    if data.size == 0:
        return (0.0, 1.0)

    vmin = float(np.percentile(data, low)) * float(low_scale)
    vmax = float(np.percentile(data, high))
    if not vmax > vmin:
        # constant data, or a low_scale that lifted the floor past the
        # ceiling: still owe the caller something a norm will accept
        span = abs(vmax) * 0.5 or 0.5
        vmin, vmax = vmax - span, vmax + span
    return (vmin, vmax)


def _centers_extent(centers, extent, shape, origin):
    """The pixel-edge extent of an image sampled at 1D pixel centers.

    `centers` is `(x, y)`: the horizontal coordinate of each column and the
    vertical coordinate of each row, in row order. Each axis must be evenly
    spaced, because `imshow` draws one uniform grid; an uneven axis belongs
    in `pcolormesh`. The extent runs half a step past the first and last
    center, and with `origin="upper"` its bottom and top swap, so row 0
    sits at `y[0]` either way.

    Returns:
        `extent` unchanged when `centers` is None, else the extent tuple.

    Raises:
        ValueError: If both are given, an axis is not 1D, holds fewer than
            two centers, does not match the image's shape, or is unevenly
            spaced.
    """
    if centers is None:
        return extent
    if extent is not None:
        raise ValueError("pass extent or centers, not both")
    if len(centers) != 2:
        raise ValueError("centers must be (x, y): column centers, row centers")
    n_rows, n_cols = shape[0], shape[1]
    edges = []
    for name, values, count in (("x", centers[0], n_cols), ("y", centers[1], n_rows)):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or values.size < 2:
            raise ValueError(f"centers {name} must be 1D with at least two entries")
        if values.size != count:
            raise ValueError(
                f"centers {name} has {values.size} entries for {count} pixels"
            )
        steps = np.diff(values)
        if not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0):
            raise ValueError(
                f"centers {name} must be evenly spaced for imshow; use "
                "pcolormesh for an uneven grid"
            )
        half = 0.5 * float(steps[0])
        edges.append((float(values[0]) - half, float(values[-1]) + half))
    (left, right), (bottom, top) = edges
    if origin == "upper":
        bottom, top = top, bottom
    return (left, right, bottom, top)


def imshow_log(
    image,
    *,
    ax=None,
    extent=None,
    centers=None,
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
        centers: `(x, y)` 1D arrays of the column and row pixel centers,
            evenly spaced, from which the pixel-edge extent is built (half
            a step past the outer centers). An alternative to `extent`, for
            data sampled on known coordinates, such as a cut against time
            or wavelength.
        floor: Minimum value the data is clipped to before norm/display.
        vmin: Norm lower bound. None uses `max(data.min(), floor)`.
        vmax: Norm upper bound. None uses `data.max()`.
        cmap: Colormap override; None uses the semantic "intensity" cmap.
        colorbar: Whether to attach a colorbar. True hangs it in an inset
            just outside `ax`, which keeps a handed-in axes from stealing
            space from its siblings under a layout engine. Use `"figure"`
            for a figure-level colorbar, which takes its room from the axes
            and so stays on-canvas on a figure with no layout engine -- the
            right choice for a plain `plt.subplots` figure, and in recorded
            animations, where a clipped label is permanent.
        cbar_label: Label for the colorbar.
        imshow_kw: Extra kwargs passed to `ax.imshow`, applied last.
        cbar_kw: Extra kwargs passed to `fig.colorbar`.

    Returns:
        A `PlotResult` with artists `"image"` (and `"cbar"` if drawn) and an
        `.update(new_image)` callable.

    Raises:
        ValueError: If `colorbar` is not a known mode, or `centers` is given
            with `extent`, or does not match the image (see `centers`).
    """
    _check_colorbar(colorbar)
    img = np.asarray(image, dtype=float)
    data = np.clip(img, floor, None)
    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    extent = _centers_extent(centers, extent, data.shape, kw["origin"])
    created = ax is None
    if created:
        _, ax = plt.subplots(layout="constrained")

    lo = max(float(np.nanmin(data)), floor) if vmin is None else vmin
    hi = float(np.nanmax(data)) if vmax is None else vmax
    norm = LogNorm(vmin=lo, vmax=hi)

    im = ax.imshow(
        data, norm=norm, cmap=_style.cmap("intensity", cmap), extent=extent, **kw
    )
    artists = {"image": im}
    _hide_index_ticks(ax, extent)

    if colorbar:
        artists["cbar"] = _attach_colorbar(ax, im, colorbar, cbar_label, cbar_kw)

    def update(new_image):
        im.set_data(np.clip(np.asarray(new_image, dtype=float), floor, None))

    return PlotResult(ax=ax, artists=artists, update=update)


def _check_colorbar(colorbar):
    """Validate a `colorbar` argument, naming the modes on failure."""
    if colorbar not in (True, False, "figure"):
        raise ValueError(
            f"colorbar must be True (inset), False, or 'figure'; got {colorbar!r}"
        )


def _attach_colorbar(ax, im, mode, cbar_label, cbar_kw):
    """Attach one colorbar to `ax` in the mode the caller asked for.

    `True` hangs it in an inset just outside `ax`. A layout engine does see
    that inset -- it is a child axes, and `get_tightbbox` unions
    `child_axes` -- so under constrained or tight layout the room is
    reserved either way. On a figure with NO layout engine, which is what
    plain `plt.subplots` gives, nothing reserves anything and the inset's
    label runs off the canvas; in a recorded animation that is silent and
    permanent, since every frame carries the clipped label. `"figure"`
    attaches a figure-level colorbar, which takes its room from the axes
    itself and so stays on-canvas with no layout engine at all.
    """
    if mode == "figure":
        return ax.figure.colorbar(im, ax=ax, label=cbar_label, **(cbar_kw or {}))
    cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
    return ax.figure.colorbar(im, cax=cax, label=cbar_label, **(cbar_kw or {}))


def _diverging_draw(
    ax, data, norm, resolved_cmap, extent, colorbar, cbar_label, imshow_kw, cbar_kw
):
    """Draw one image + its colorbar under an already-built diverging norm.

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
        cb = _attach_colorbar(ax, im, colorbar, cbar_label, cbar_kw)
    return im, cb


def imshow_diverging(
    image,
    *,
    ax=None,
    extent=None,
    centers=None,
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
        centers: `(x, y)` 1D arrays of the column and row pixel centers,
            evenly spaced, from which the pixel-edge extent is built, as in
            `imshow_log`. An alternative to `extent`.
        vlim: Symmetric norm bound; the norm spans `(-vlim, vlim)`. None
            uses `max(abs(data.min()), abs(data.max()))`.
        cmap: Colormap override; None uses the semantic "residual" cmap.
        colorbar: Whether to attach a colorbar. True hangs it in an inset
            just outside `ax`, which keeps a handed-in axes from stealing
            space from its siblings under a layout engine. Use `"figure"`
            for a figure-level colorbar, which takes its room from the axes
            and so stays on-canvas on a figure with no layout engine -- the
            right choice for a plain `plt.subplots` figure, and in recorded
            animations, where a clipped label is permanent.
        cbar_label: Label for the colorbar.
        imshow_kw: Extra kwargs passed to `ax.imshow`, applied last.
        cbar_kw: Extra kwargs passed to `fig.colorbar`.

    Returns:
        A `PlotResult` with artists `"image"` (and `"cbar"` if drawn) and an
        `.update(new_image)` that redraws new data under the first draw's
        symmetric norm, which is never refitted, so an animated residual
        keeps one zero and one scale across frames.

    Raises:
        ValueError: If `colorbar` is not a known mode, or `centers` is given
            with `extent`, or does not match the image.
    """
    _check_colorbar(colorbar)
    data = np.asarray(image, dtype=float)
    origin = (imshow_kw or {}).get("origin", "lower")
    extent = _centers_extent(centers, extent, data.shape, origin)
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
    _hide_index_ticks(ax, extent)

    def update(new_image):
        # the symmetric norm is deliberately NOT refitted: an animated
        # residual or OPD map has to keep one scale across frames, or the
        # colors stop meaning the same thing from frame to frame
        im.set_data(np.asarray(new_image, dtype=float))

    return PlotResult(ax=ax, artists=artists, update=update)


def _row_figsize(image, extent, panel_size, n_panels):
    """Figure size in inches for a row of `n_panels` drawn like `image`.

    `imshow` takes its drawn aspect from `extent` when there is one and from
    the array shape otherwise, so the figure has to be sized off whichever
    one `imshow` will actually use -- sizing off the shape alone puts an
    extent-carrying call straight back into a row of thin panels stranded in
    a tall figure. The ratio is clamped because a long strip or a tall stack
    would otherwise size the figure to a sliver or to a hundred inches, both
    worse than the fixed default this replaced.
    """
    if extent is not None:
        left, right, bottom, top = extent
        span_x, span_y = abs(right - left), abs(top - bottom)
        raw_aspect = span_y / span_x if span_x else 1.0
    else:
        shape = np.shape(image)
        raw_aspect = shape[0] / shape[1] if len(shape) >= 2 and shape[1] else 1.0
    aspect = min(max(raw_aspect, 0.4), 2.5)
    return (panel_size * n_panels, panel_size * aspect)


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
    panel_size=3.2,
    cax=None,
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
        panel_size: Size in inches of one panel along the row, used only for
            a figure this function creates: the width scales with the panel
            count and the height follows the first image's aspect, so a row
            of k square images stays square instead of shrinking into
            matplotlib's fixed default figure. Ignored when `axes` is given.
        cax: Axes to draw the shared colorbar in, for a caller who placed
            the colorbar slot themselves (beside a hand-built mosaic, or
            spanning several rows). None attaches it by the rule above: a
            figure-level colorbar on a figure this function created, an
            inset beside the last panel otherwise.
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
        # matplotlib's default figsize does not know how many panels it is
        # about to hold, so k square images land as k small squares stranded
        # in a tall figure beside a full-height colorbar. An owned figure
        # grows with the panel count instead, keeping each panel near the
        # image's own aspect; constrained layout fits the colorbar inside
        # that width. A caller who hands in axes owns the figure and its
        # size, so this applies only to the figure this function creates.
        fig, panel_axes = plt.subplots(
            1,
            len(images),
            figsize=_row_figsize(images[0], extent, panel_size, len(images)),
            layout="constrained",
            squeeze=False,
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
    _hide_index_ticks(axes, extent)

    artists = {"image": ims}
    if cax is not None:
        cb = fig.colorbar(ims[-1], cax=cax, label=cbar_label, **(cbar_kw or {}))
    elif created:
        cb = fig.colorbar(ims[-1], ax=axes, label=cbar_label, **(cbar_kw or {}))
    else:
        cax = axes[-1].inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = fig.colorbar(ims[-1], cax=cax, label=cbar_label, **(cbar_kw or {}))
    artists["cbar"] = cb

    return MosaicResult(axes=axes, artists=artists)


def _grid_cells(images, titles):
    """Validate a 2D image layout and return it as rows of arrays or None.

    Rows may differ in length; a short row is padded with empty cells so the
    layout is a rectangle of ``(n_rows, n_cols)``.
    """
    rows = [list(row) for row in images]
    if not rows or all(cell is None for row in rows for cell in row):
        raise ValueError("compare_grid needs at least one image")
    n_cols = max(len(row) for row in rows)
    cells = [
        [None if cell is None else np.asarray(cell, dtype=float) for cell in row]
        + [None] * (n_cols - len(row))
        for row in rows
    ]
    if titles is not None:
        title_rows = [list(row) for row in titles]
        if len(title_rows) != len(rows) or any(
            len(t) != len(r) for t, r in zip(title_rows, rows, strict=True)
        ):
            raise ValueError("compare_grid titles must match the shape of images")
    return cells, n_cols


def compare_grid(
    images,
    titles=None,
    *,
    axes=None,
    norm="linear",
    floor=1e-20,
    extent=None,
    cmap=None,
    cbar_label=None,
    vmin=None,
    vmax=None,
    panel_size=1.6,
    cax=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a 2D layout of images sharing one norm and one colorbar.

    The two-dimensional counterpart of `compare_row`. `images` is a sequence
    of rows, and a cell may be None, which leaves that slot empty with its
    axes switched off. Empty cells let a caller lay images out on any
    pattern a rectangular grid can hold (a triangle indexed by two integers,
    a sparse design matrix, a staircase), while every drawn panel still
    shares the one norm object, so equal colors mean equal values anywhere
    in the grid.

    Args:
        images: Sequence of rows; each row a sequence of 2D array-likes or
            None. Rows may have different lengths, and short rows are
            padded with empty cells on the right.
        titles: Optional nested sequence matching `images` row by row; an
            entry is ignored where its image is None.
        axes: A 2D array of Axes of shape `(n_rows, n_cols)` to draw into.
            None creates a new figure.
        norm: `"linear"`, `"log"`, or `"diverging"` -- which shared norm to
            build, as in `compare_row`.
        floor: Clip floor used when `norm="log"`.
        extent: `(left, right, bottom, top)` passed to every panel's
            `imshow`.
        cmap: Colormap override; None uses the semantic "intensity" cmap
            (or "residual" when `norm="diverging"`).
        cbar_label: Label for the shared colorbar.
        vmin: Pins the shared norm's lower bound (symmetric for
            `norm="diverging"`).
        vmax: Pins the shared norm's upper bound.
        panel_size: Size in inches of one square cell, used only for a
            figure this function creates.
        cax: Axes to draw the shared colorbar in, for a caller who placed
            the colorbar slot themselves (for example one tall slot beside
            every row). None attaches it as a figure-level colorbar on a
            figure this function created, and as an inset beside the last
            cell otherwise.
        imshow_kw: Extra kwargs passed to each panel's `ax.imshow`, applied
            last.
        cbar_kw: Extra kwargs passed to the shared colorbar's
            `fig.colorbar`, applied last.

    Returns:
        A `MosaicResult` whose `axes` is a 2D array of shape
        `(n_rows, n_cols)`, with `artists["image"]` a nested list of the
        same shape holding an `AxesImage` per drawn cell and None per empty
        cell, and `artists["cbar"]` the single shared colorbar.

    Raises:
        ValueError: If no cell holds an image, if `titles` does not match
            the shape of `images`, or if `axes` has the wrong shape.
    """
    cells, n_cols = _grid_cells(images, titles)
    n_rows = len(cells)
    if norm == "log":
        cells = [
            [None if c is None else np.clip(c, floor, None) for c in row]
            for row in cells
        ]

    created = axes is None
    if created:
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(panel_size * n_cols + 1.0, panel_size * n_rows),
            layout="constrained",
            squeeze=False,
        )
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != (n_rows, n_cols):
            raise ValueError(
                f"compare_grid: expected axes shape ({n_rows}, {n_cols}), "
                f"got {axes.shape}"
            )
        fig = axes[0, 0].figure

    drawn = [c for row in cells for c in row if c is not None]
    shared_norm = _shared_norm(drawn, norm, floor, vmin=vmin, vmax=vmax)
    semantic_cmap = "residual" if norm == "diverging" else "intensity"
    resolved_cmap = _style.cmap(semantic_cmap, cmap)

    kw = {"interpolation": "nearest", "origin": "lower", **(imshow_kw or {})}
    ims = []
    last = None
    for i, row in enumerate(cells):
        row_ims = []
        for j, img in enumerate(row):
            ax = axes[i, j]
            if img is None:
                ax.set_axis_off()
                row_ims.append(None)
                continue
            im = ax.imshow(
                img, norm=shared_norm, cmap=resolved_cmap, extent=extent, **kw
            )
            if titles is not None and j < len(titles[i]):
                ax.set_title(titles[i][j])
            _hide_index_ticks(ax, extent)
            row_ims.append(im)
            last = im
        ims.append(row_ims)

    if cax is not None:
        cb = fig.colorbar(last, cax=cax, label=cbar_label, **(cbar_kw or {}))
    elif created:
        cb = fig.colorbar(last, ax=axes, label=cbar_label, **(cbar_kw or {}))
    else:
        cax = axes[-1, -1].inset_axes([1.02, 0.0, 0.04, 1.0])
        cb = fig.colorbar(last, cax=cax, label=cbar_label, **(cbar_kw or {}))

    return MosaicResult(axes=axes, artists={"image": ims, "cbar": cb})


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
    panel_size=3.2,
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
            figure sized to hold three panels; see `panel_size`.
        ratio_clip: Fixed clip value for the ratio panel's norm, which
            then spans `[1 - ratio_clip, 1 + ratio_clip]`. None derives it
            from the data (see above). Ignored when `mode="residual"`.
        extent: `(left, right, bottom, top)` passed to all three panels'
            `imshow`, so a triptych can carry axis units the way every
            other image primitive here can.
        panel_size: Size in inches of one panel, used only for a figure this
            function creates: the width holds three of them and the height
            follows the drawn aspect, so three square panels stay square
            instead of shrinking into matplotlib's fixed default figure.
            Ignored when `axes` is given.
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
        _, axes = plt.subplots(
            1,
            3,
            figsize=_row_figsize(a, extent, panel_size, 3),
            layout="constrained",
        )

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
    _hide_index_ticks(axes, extent)

    artists = {
        "image": [*ab_result.artists["image"], cmp_im],
        "cbar": [ab_result.artists["cbar"], cmp_cb],
    }
    return MosaicResult(axes=axes, artists=artists)


# How far a label sits outward from the point it labels, in points.
_LABEL_OFFSET_PT = 3.0


def _light_and_dark():
    """The style's background and text tones, ordered lighter first."""
    face = to_rgb(matplotlib.rcParams["axes.facecolor"])
    ink = to_rgb(matplotlib.rcParams["text.color"])

    def luminance(c):
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    return (face, ink) if luminance(face) >= luminance(ink) else (ink, face)


def overlay_circle(
    ax,
    center,
    radius,
    *,
    color=None,
    underlay=True,
    ls=None,
    label=None,
    label_angle=45.0,
    circle_kw=None,
    underlay_kw=None,
    text_kw=None,
):
    """Draw a dashed circle over an image, legible on bright and dark pixels.

    An aperture, a dark ring, or a working angle drawn over an image crosses
    pixels from the darkest to the brightest end of the colormap, so no one
    color reads everywhere along it. The circle is therefore a light dash
    over a thin, solid, dark underlay: where the pixels are bright the dark
    underlay outlines it, and where they are dark the light dash does. The
    two tones are the style's background and text colors, ordered by
    lightness, so the pair holds in either mode.

    The circle is added without touching the data limits, so drawing a
    circle larger than the view neither rescales the axes nor moves an
    image's extent.

    Args:
        ax: Axes to draw on, in its data coordinates.
        center: `(x, y)` of the center, in data units.
        radius: Radius in data units.
        color: Color of the dash. None uses the lighter of the style's
            background and text colors.
        underlay: Whether to draw the solid dark underlay. True uses the
            darker of the style's background and text colors; a color draws
            it in that color; False omits it.
        ls: Line style of the circle. None draws the default dash; `"-"`
            draws a solid light line over the underlay, as for the rim of a
            pupil where a dashed circle marks a stop drawn inside it. The
            underlay stays solid either way.
        label: Text written on the circle at `label_angle`, on a backing box
            in the background color, just outside the circle and aligned
            away from its center. None writes nothing.
        label_angle: Where the label sits, in degrees counterclockwise from
            the +x direction about `center`. Ignored without `label`.
        circle_kw: Extra kwargs for the dashed `Circle` (for example `lw`,
            `ls`, `zorder`, or `gid`), applied last. The underlay follows the
            dash's line width and z-order and stays solid.
        underlay_kw: Extra kwargs for the underlay `Circle`, applied last
            over its defaults: a line width 1 pt wider than the dash's, a
            solid line style, and the dash's z-order. Pass `lw` to set the
            underlay's width outright, or `gid` to tag it alongside the dash.
            Ignored when `underlay` is False.
        text_kw: Extra kwargs for the label's `Text` (for example
            `fontsize`, `color`, or `bbox=None` for bare text), applied
            last. Ignored without `label`.

    Returns:
        A `PlotResult` whose `artists["ellipse"]` is the list of `Circle`
        patches in draw order: the underlay, when drawn, then the dash, so
        `artists["ellipse"][-1]` is always the dashed circle. With `label`,
        `artists["text"]` is the label, an `Annotation` anchored on the
        circle and offset a few points outward. Like any annotation of a
        data point, it is hidden while that point is outside the view.

    Raises:
        ValueError: If `radius` is not positive.

    Example::

        result = ep.imshow_log(psf, extent=extent)
        ep.overlay_circle(result.ax, (3.0, 0.0), 0.7)
        ep.overlay_circle(result.ax, (0.0, 0.0), 0.5, ls="-", label="rim",
                          label_angle=40.0)
    """
    radius = float(radius)
    if not radius > 0.0:
        raise ValueError(f"overlay_circle radius must be positive, got {radius}")
    center = (float(center[0]), float(center[1]))
    light, dark = _light_and_dark()
    kw = {
        "fill": False,
        "edgecolor": light if color is None else color,
        "lw": 1.0,
        "ls": (0, (3, 2)) if ls is None else ls,
        "zorder": 4,
        **(circle_kw or {}),
    }
    dash = Circle(center, radius, **kw)
    patches = []
    if underlay is not False and underlay is not None:
        under = Circle(
            center,
            radius,
            fill=False,
            edgecolor=dark if underlay is True else underlay,
            lw=dash.get_linewidth() + 1.0,
            ls="-",
            zorder=dash.get_zorder(),
        )
        under.set(**(underlay_kw or {}))
        patches.append(ax.add_artist(under))
    patches.append(ax.add_artist(dash))
    artists = {"ellipse": patches}
    if label is not None:
        artists["text"] = _circle_label(
            ax, center, radius, label_angle, label, dash.get_zorder() + 1, text_kw
        )
    return PlotResult(ax=ax, artists=artists)


def _circle_label(ax, center, radius, angle_deg, text, zorder, text_kw):
    """An annotation anchored on a circle at `angle_deg`, offset outward."""
    angle = np.deg2rad(float(angle_deg))
    point = (center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle))
    ha, va = _outward_alignment(angle)
    kw = {
        "color": matplotlib.rcParams["text.color"],
        "fontsize": "small",
        "ha": ha,
        "va": va,
        "zorder": zorder,
        "bbox": _style.backing(),
        **(text_kw or {}),
    }
    offset = (_LABEL_OFFSET_PT * np.cos(angle), _LABEL_OFFSET_PT * np.sin(angle))
    return ax.annotate(text, point, xytext=offset, textcoords="offset points", **kw)


# Where a ruler's label sits: (alignment, unit offset direction).
_RULER_SIDES = {
    "above": (("center", "bottom"), (0.0, 1.0)),
    "below": (("center", "top"), (0.0, -1.0)),
    "left": (("right", "center"), (-1.0, 0.0)),
    "right": (("left", "center"), (1.0, 0.0)),
}


class _DoubleArrow(_Arrow):
    """A head-capped arrow with a head at each end."""

    heads = 2


def ruler(
    ax,
    p0,
    p1,
    text=None,
    *,
    side="above",
    backing=True,
    color=None,
    offset_pt=3.0,
    head_scale=1.0,
    arrow_kw=None,
    line_kw=None,
    text_kw=None,
):
    """Draw a dimension arrow between two points, with its label on a box.

    The arrow is double headed, running from `p0` to `p1` in the data
    coordinates of `ax`, and its label sits at the midpoint, a few points to
    one `side`. Over an image the arrow crosses bright and dark pixels, so by
    default it is drawn in the text color over a wider line in the background
    color, and the label sits on a backing box in the background color; that
    pair reads on either end of a colormap in either mode. On a plain plot,
    where the background is already behind it, `backing=False` drops both.

    Heads are sized in points like a phasor's and grow with
    `rcParams["lines.markersize"]`. On a ruler shorter on screen than its two
    heads, each head shrinks to half its length, and a ruler of zero length
    draws nothing but its label. The ruler is added without touching the
    data limits.

    Args:
        ax: Axes to draw on.
        p0: `(x, y)` of one end, in data units.
        p1: `(x, y)` of the other end, in data units.
        text: The label. None draws the arrow alone.
        side: Where the label sits relative to the midpoint on screen:
            `"above"`, `"below"`, `"left"`, or `"right"`.
        backing: Draw the background-colored line under the arrow and the
            backing box under the label.
        color: Color of the arrow. None uses `rcParams["text.color"]`.
        offset_pt: Distance from the midpoint to the label, in points.
        head_scale: Head size relative to a full-size phasor head. 0 draws
            a plain line with no heads.
        arrow_kw: Extra kwargs for the `FancyArrowPatch` (for example `lw`,
            `zorder`, or `gid`), applied last.
        line_kw: Extra kwargs for the backing `Line2D`, applied last over
            its defaults: the background color, 3.2 times the arrow's line
            width, butt caps, and the arrow's z-order. Ignored when
            `backing` is False.
        text_kw: Extra kwargs for the label's `Text` (for example
            `fontsize` or `color`), applied last.

    Returns:
        A `PlotResult` with artists `"arrow"` (the `FancyArrowPatch`),
        `"lines"` (a list holding the backing `Line2D`, when `backing`),
        and `"text"` (the label, an `Annotation` of the midpoint, when
        `text` is given), and an `update(p0=None, p1=None, text=None)` that
        moves the same artists to new ends and relabels them, keeping every
        style; an argument of None keeps the last value. A label added by
        `update` to a ruler drawn without one raises.

    Raises:
        ValueError: If `side` is not one of the four sides, or `update`
            receives `text` for a ruler drawn without a label.

    Example::

        image = ep.imshow_log(psf, extent=extent)
        ep.ruler(image.ax, (-1.22, -3.0), (1.22, -3.0), "2.44 lambda/D")
    """
    if side not in _RULER_SIDES:
        raise ValueError(f"unknown ruler side: {side!r}; known: {list(_RULER_SIDES)}")
    rc = matplotlib.rcParams
    state = {"p0": _point(p0), "p1": _point(p1)}
    lw = 0.8 * float(rc["lines.linewidth"])
    head = float(head_scale)
    style = f"<|-|>,head_length={HEAD_LENGTH},head_width={HEAD_WIDTH}"
    akw = {
        "arrowstyle": style if head > 0.0 else "-",
        "mutation_scale": HEAD_PER_MARKER * float(rc["lines.markersize"]) * head,
        "color": rc["text.color"] if color is None else color,
        "lw": lw,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
        "zorder": 5,
        **(arrow_kw or {}),
    }
    arrow = _DoubleArrow(state["p0"], state["p1"], **akw)
    artists = {}
    under = None
    if backing:
        lkw = {
            "color": rc["axes.facecolor"],
            "lw": 3.2 * arrow.get_linewidth(),
            "solid_capstyle": "butt",
            "zorder": arrow.get_zorder(),
            **(line_kw or {}),
        }
        under = Line2D(*_line_data(state), **lkw)
        ax.add_artist(under)
        artists["lines"] = [under]
    ax.add_artist(arrow)
    artists["arrow"] = arrow

    label = None
    if text is not None:
        (ha, va), (ux, uy) = _RULER_SIDES[side]
        tkw = {
            "color": rc["text.color"],
            "fontsize": "small",
            "ha": ha,
            "va": va,
            "zorder": arrow.get_zorder() + 1,
            **({"bbox": _style.backing()} if backing else {}),
            **(text_kw or {}),
        }
        offset = (ux * float(offset_pt), uy * float(offset_pt))
        label = ax.annotate(
            text, _midpoint(state), xytext=offset, textcoords="offset points", **tkw
        )
        artists["text"] = label

    def update(p0=None, p1=None, text=None):
        if text is not None and label is None:
            raise ValueError("ruler update got text for a ruler drawn without a label")
        if p0 is not None:
            state["p0"] = _point(p0)
        if p1 is not None:
            state["p1"] = _point(p1)
        arrow.set_positions(state["p0"], state["p1"])
        if under is not None:
            under.set_data(*_line_data(state))
        if label is not None:
            label.xy = _midpoint(state)
            if text is not None:
                label.set_text(text)

    return PlotResult(ax=ax, artists=artists, update=update)


def _point(p):
    return (float(p[0]), float(p[1]))


def _line_data(state):
    (x0, y0), (x1, y1) = state["p0"], state["p1"]
    return [x0, x1], [y0, y1]


def _midpoint(state):
    (x0, y0), (x1, y1) = state["p0"], state["p1"]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def kymograph(
    values,
    x,
    y,
    *,
    ax=None,
    floor=1e-20,
    vmin=None,
    vmax=None,
    cmap=None,
    colorbar=True,
    cbar_label=None,
    imshow_kw=None,
    cbar_kw=None,
):
    """Draw a cut through an image stacked against a second variable.

    Each row of `values` is the brightness along one cut (a line through a
    star and a planet, say) at one value of a second variable: a time, a
    wavelength, a roll angle. Stacked, a feature fixed on the sky stands as
    a vertical stripe, and one that moves with the second variable leans,
    so the eye reads at once which features belong to the sky and which to
    the instrument. The image is log scaled like `imshow_log`, from the
    pixel centers given, with no interpolation, and on an `"auto"` aspect,
    since the two axes carry different units.

    Draw guides over it, such as the path a feature should follow, with
    `overlay_line`, which stays legible over the bright and the dark pixels.

    Args:
        values: 2D array-like indexed `[y, x]`: one row per value of the
            second variable, one column per sample along the cut.
        x: 1D, evenly spaced positions of the samples along the cut (the
            column centers), drawn on the horizontal axis.
        y: 1D, evenly spaced values of the second variable (the row
            centers), drawn on the vertical axis.
        ax: Axes to draw into. None creates a new figure and axes.
        floor: Minimum value the data is clipped to before norm and
            display, as in `imshow_log`.
        vmin: Norm lower bound. None uses `max(data.min(), floor)`.
        vmax: Norm upper bound. None uses `data.max()`.
        cmap: Colormap override; None uses the semantic "intensity" cmap.
        colorbar: True (an inset beside `ax`), `"figure"`, or False, as in
            `imshow_log`.
        cbar_label: Label for the colorbar.
        imshow_kw: Extra kwargs passed to `ax.imshow`, applied last over
            the `"auto"` aspect.
        cbar_kw: Extra kwargs passed to `fig.colorbar`.

    Returns:
        The `PlotResult` of `imshow_log`: artists `"image"` (and `"cbar"`
        if drawn) and an `.update(new_values)` that redraws the same image
        under the first draw's floor and norm.

    Raises:
        ValueError: If `x` or `y` is not 1D, holds fewer than two entries,
            does not match `values`, or is unevenly spaced.

    Example::

        res = ep.kymograph(cuts, x_lod, t_hours, vmin=1e-9, vmax=1e-6)
        ep.overlay_line(res.ax, [r_planet, r_planet], [t_hours[0], t_hours[-1]])
    """
    return imshow_log(
        values,
        ax=ax,
        centers=(x, y),
        floor=floor,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        colorbar=colorbar,
        cbar_label=cbar_label,
        imshow_kw={"aspect": "auto", **(imshow_kw or {})},
        cbar_kw=cbar_kw,
    )


def _along(xs, ys, fraction):
    """The point `fraction` of the way along a polyline, by length."""
    if xs.size == 1:
        return float(xs[0]), float(ys[0])
    lengths = np.hypot(np.diff(xs), np.diff(ys))
    run = np.concatenate([[0.0], np.cumsum(lengths)])
    if run[-1] == 0.0:
        return float(xs[0]), float(ys[0])
    target = float(np.clip(fraction, 0.0, 1.0)) * run[-1]
    return float(np.interp(target, run, xs)), float(np.interp(target, run, ys))


def _path(x, y):
    xs = np.asarray(x, dtype=float).ravel()
    ys = np.asarray(y, dtype=float).ravel()
    if xs.size != ys.size or xs.size == 0:
        raise ValueError(
            f"overlay_line needs matching, nonempty x and y; got {xs.size} and "
            f"{ys.size} points"
        )
    return xs, ys


def overlay_line(
    ax,
    x,
    y,
    *,
    color=None,
    underlay=True,
    ls=None,
    label=None,
    label_at=0.5,
    line_kw=None,
    underlay_kw=None,
    text_kw=None,
):
    """Draw a dashed path over an image, legible on bright and dark pixels.

    The line counterpart of `overlay_circle`, for a cut through an image, a
    guide over a kymograph, or an arc of fixed radius: a light dash over a
    thin, solid, dark underlay, the two tones being the style's background
    and text colors ordered by lightness, so where the pixels are bright the
    underlay outlines the path and where they are dark the dash does. The
    path is added without touching the data limits.

    Args:
        ax: Axes to draw on, in its data coordinates.
        x: The path's x coordinates, in data units: two points for a
            straight line, more for a polyline or an arc.
        y: The path's y coordinates, one per x.
        color: Color of the dash. None uses the lighter of the style's
            background and text colors.
        underlay: Whether to draw the solid dark underlay. True uses the
            darker of the two tones; a color draws it in that color; False
            omits it.
        ls: Line style of the dash. None draws the dash `overlay_circle`
            draws; `":"` a dotted guide, `"-"` a solid line. The underlay
            stays solid either way.
        label: Text written on the path at `label_at`, centered on a backing
            box in the background color. None writes nothing.
        label_at: Where the label sits, as a fraction of the path's length
            in data units from its first point (0) to its last (1).
        line_kw: Extra kwargs for the dash's `Line2D` (for example `lw`,
            `zorder`, or `gid`), applied last.
        underlay_kw: Extra kwargs for the underlay's `Line2D`, applied last
            over its defaults: a line width 1 pt wider than the dash's, a
            solid line style, and the dash's z-order. Ignored when
            `underlay` is False.
        text_kw: Extra kwargs for the label's `Text` (for example `ha`,
            `va`, `fontsize`, or `bbox=None` for bare text), applied last.

    Returns:
        A `PlotResult` whose `artists["lines"]` holds the `Line2D` artists
        in draw order: the underlay, when drawn, then the dash, so
        `artists["lines"][-1]` is always the dash. With `label`,
        `artists["text"]` is the label. Its `update(x, y)` moves the same
        lines, and the label to the same fraction of the new path, keeping
        every style; the new path may have a different number of points.

    Raises:
        ValueError: If `x` and `y` differ in length or are empty, here or in
            `update`.

    Example::

        res = ep.imshow_log(image, extent=extent)
        ep.overlay_line(res.ax, [-6.0, 6.0], [-2.0, 2.0], label="cut",
                        label_at=1.0)
        phi = np.linspace(0.0, np.pi / 2, 90)
        ep.overlay_line(res.ax, 4.0 * np.cos(phi), 4.0 * np.sin(phi), ls=":")
    """
    xs, ys = _path(x, y)
    light, dark = _light_and_dark()
    kw = {
        "color": light if color is None else color,
        "lw": 1.0,
        "ls": (0, (3, 2)) if ls is None else ls,
        "zorder": 4,
        **(line_kw or {}),
    }
    dash = Line2D(xs, ys, **kw)
    lines = []
    if underlay is not False and underlay is not None:
        under = Line2D(
            xs,
            ys,
            color=dark if underlay is True else underlay,
            lw=dash.get_linewidth() + 1.0,
            ls="-",
            zorder=dash.get_zorder(),
        )
        under.set(**(underlay_kw or {}))
        lines.append(ax.add_artist(under))
    lines.append(ax.add_artist(dash))
    artists = {"lines": lines}
    text = None
    if label is not None:
        tkw = {
            "color": matplotlib.rcParams["text.color"],
            "fontsize": "small",
            "ha": "center",
            "va": "center",
            "zorder": dash.get_zorder() + 1,
            "bbox": _style.backing(),
            **(text_kw or {}),
        }
        text = ax.text(*_along(xs, ys, label_at), label, **tkw)
        artists["text"] = text

    def update(new_x, new_y):
        nx, ny = _path(new_x, new_y)
        for line in lines:
            line.set_data(nx, ny)
        if text is not None:
            text.set_position(_along(nx, ny, label_at))

    return PlotResult(ax=ax, artists=artists, update=update)


# Where a bracket's label sits and which way its ticks point.
_BRACKET_SIDES = {
    "above": ("bottom", 1.0),
    "below": ("top", -1.0),
}


def bracket(
    ax,
    x0,
    x1,
    y,
    text=None,
    *,
    depth,
    side="above",
    color=None,
    offset_pt=2.0,
    line_kw=None,
    text_kw=None,
):
    """Draw a square bracket spanning `x0` to `x1`, labeled on its far side.

    A bracket groups a run of things along the horizontal axis (the
    elements of a train upstream of a mask, the frames of one exposure)
    under one label: a bar at height `y` with a tick at each end pointing
    toward the things grouped, and the label centered on the bar's other
    side. It is drawn in the text color as scenery and added without
    touching the data limits.

    Args:
        ax: Axes to draw on, in its data coordinates.
        x0: Left end of the bar, in data units.
        x1: Right end of the bar, in data units.
        y: Height of the bar, in data units.
        text: The label. None draws the bracket alone.
        depth: Length of the end ticks, in data units, > 0.
        side: Where the label sits: `"above"` the bar, with the ticks
            pointing down toward what is grouped beneath it, or `"below"`,
            with the ticks pointing up.
        color: Color of the bracket. None uses `rcParams["text.color"]`.
        offset_pt: Gap between the bar and the label, in points.
        line_kw: Extra kwargs for the bracket's `Line2D` (for example `lw`,
            `zorder`, or `gid`), applied last.
        text_kw: Extra kwargs for the label's `Text` (for example
            `fontsize`, `color`, or a backing `bbox`), applied last.

    Returns:
        A `PlotResult` with artists `"line"` (the bracket, one `Line2D`
        through both ticks and the bar) and `"text"` (the label, an
        `Annotation` of the bar's midpoint, when `text` is given). There is
        no `update`.

    Raises:
        ValueError: If `side` is not `"above"` or `"below"`, or `depth` is
            not positive.

    Example::

        ep.bracket(ax, 0.9, 17.0, 2.75, "upstream of the mask", depth=0.25)
    """
    if side not in _BRACKET_SIDES:
        raise ValueError(
            f"unknown bracket side: {side!r}; known: {list(_BRACKET_SIDES)}"
        )
    depth = float(depth)
    if not depth > 0.0:
        raise ValueError(f"bracket depth must be positive, got {depth}")
    rc = matplotlib.rcParams
    va, sign = _BRACKET_SIDES[side]
    x0, x1, y = float(x0), float(x1), float(y)
    tip = y - sign * depth
    lkw = {
        "color": rc["text.color"] if color is None else color,
        "lw": 0.8 * float(rc["lines.linewidth"]),
        "zorder": 3,
        **(line_kw or {}),
    }
    line = ax.add_artist(Line2D([x0, x0, x1, x1], [tip, y, y, tip], **lkw))
    artists = {"line": line}
    if text is not None:
        tkw = {
            "color": rc["text.color"] if color is None else color,
            "fontsize": "small",
            "ha": "center",
            "va": va,
            "zorder": line.get_zorder(),
            **(text_kw or {}),
        }
        artists["text"] = ax.annotate(
            text,
            (0.5 * (x0 + x1), y),
            xytext=(0.0, sign * float(offset_pt)),
            textcoords="offset points",
            **tkw,
        )
    return PlotResult(ax=ax, artists=artists)
