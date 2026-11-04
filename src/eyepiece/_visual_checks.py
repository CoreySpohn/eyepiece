"""Agg data coverage for image occlusion and visibly empty axes."""

from copy import copy

import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.collections import QuadMesh
from matplotlib.image import AxesImage
from matplotlib.transforms import Bbox

_IMAGE_TYPES = (AxesImage, QuadMesh)


def _raster(fig, artists):
    """Draw just these artists, retaining their transforms and clipping."""
    renderer = RendererAgg(int(fig.bbox.width), int(fig.bbox.height), fig.dpi)
    renderer.clear()
    for artist in sorted(artists, key=lambda a: a.get_zorder()):
        stale = artist.stale
        try:
            artist.draw(renderer)
        finally:
            artist._stale = stale
    return np.asarray(renderer.buffer_rgba()).copy()


def _data_image(image):
    """An independent image with invalid samples transparent, even with bad ink."""
    data = np.ma.masked_invalid(image.get_array())
    invalid = np.ma.getmaskarray(data)
    if data.ndim == 3:
        invalid = invalid.any(axis=-1)
    rgba = image.to_rgba(data, alpha=image.get_alpha()).copy()
    if isinstance(image, AxesImage) and data.ndim == 3:
        # to_rgba passes RGB(A) through; AxesImage applies its opacity later.
        alpha = image.get_alpha()
        if alpha is not None:
            if np.ndim(alpha):
                rgba[..., 3] = alpha
            else:
                rgba[..., 3] *= alpha
    rgba[..., 3][invalid] = 0
    clone = copy(image)
    # Alpha is already in RGBA, including array-valued image opacity.
    clone._alpha = None
    if isinstance(image, AxesImage):
        clone.set_data(rgba)
    else:
        clone.set_array(rgba)
        clone.set_edgecolor("none")
    return clone


def _region(box, shape, *, inset=0):
    """Pixel centers inside a display box, as raster row/column slices."""
    height, width = shape[:2]
    x0 = min(width, max(0, int(np.ceil(box.x0 + inset - 0.5))))
    x1 = max(0, min(width, int(np.ceil(box.x1 - inset - 0.5))))
    y0 = min(height, max(0, int(np.ceil(box.y0 + inset - 0.5))))
    y1 = max(0, min(height, int(np.ceil(box.y1 - inset - 0.5))))
    return slice(height - y1, height - y0), slice(x0, x1)


def _clip_box(artist, box):
    if artist.get_clip_on():
        clip = artist.get_clip_box()
        if clip is not None:
            box = Bbox.intersection(box, clip)
    return box


def _background(fig, ax):
    """Axes facecolor composited over the figure, on a white output page."""
    figure = np.asarray(fig.get_facecolor())
    background = figure[:3] * figure[3] + (1 - figure[3])
    if ax.axison and ax.patch.get_visible():
        face = np.asarray(ax.get_facecolor())
        background = face[:3] * face[3] + background * (1 - face[3])
    return background


def _rgb(pixels, background):
    rgba = pixels.astype(float) / 255
    return rgba[..., :3] * rgba[..., 3:] + background * (1 - rgba[..., 3:])


def _visual_findings(
    fig,
    texts,
    boxes,
    marks,
    *,
    kinds,
    overlap_tol,
    min_ink_fraction,
    min_image_range,
    ink_tol,
):
    """Return readability tuples; geometry must already have been drawn."""
    found = []
    images = [mark for mark in marks if isinstance(mark, _IMAGE_TYPES)]
    rasters = {}
    for image in images:
        # This raster represents valid data, rather than a colormap's bad color.
        pixels = _raster(fig, [_data_image(image)])
        rasters[image] = pixels
        if "on-image" not in kinds:
            continue
        ax = image.axes
        for text, box in zip(texts, boxes, strict=True):
            clipped = _clip_box(text, box)
            if clipped is None:
                continue
            clipped = Bbox.intersection(clipped, ax.bbox)
            if clipped is None or min(clipped.width, clipped.height) <= overlap_tol:
                continue
            rows, cols = _region(clipped, pixels.shape)
            occupied = pixels[rows, cols, 3] >= 13  # 5 percent visible opacity
            if not occupied.any():
                continue
            # A text clip path can exclude data within its bounding rectangle.
            path = text.get_clip_path() if text.get_clip_on() else None
            if path is not None:
                ys, xs = np.nonzero(occupied)
                points = np.column_stack(
                    (xs + cols.start + 0.5, pixels.shape[0] - rows.start - ys - 0.5)
                )
                if not path.get_fully_transformed_path().contains_points(points).any():
                    continue
            found.append(
                (
                    "on-image",
                    text,
                    image,
                    tuple(map(float, clipped.extents)),
                    f"text {text.get_text()!r} hides valid {type(image).__name__} data",
                )
            )
    if "empty-panel" not in kinds:
        return found
    for ax in fig.axes:
        own = [mark for mark in marks if mark.axes is ax]
        if not ax.get_visible() or not own:
            continue
        pixels = _raster(fig, own)
        rows, cols = _region(ax.bbox, pixels.shape, inset=1)
        pixels = pixels[rows, cols]
        if not pixels.size:
            continue
        background = _background(fig, ax)
        ink = np.max(np.abs(_rgb(pixels, background) - background), axis=-1)
        fraction = float(np.mean(ink > ink_tol))
        detail = None
        if fraction < min_ink_fraction:
            detail = (
                f"panel ink fraction {fraction:.4%} is below {min_ink_fraction:.4%}"
            )
        else:
            # Measure the displayed data alone: text and axes decoration must
            # not make a uniform image pass. Edges with partial pixel coverage
            # are excluded from the range to avoid antialiasing false negatives.
            image_pixels = [rasters[m][rows, cols] for m in own if m in rasters]
            if image_pixels:
                covered = np.zeros(pixels.shape[:2], dtype=bool)
                for painted in image_pixels:
                    alpha = painted[..., 3]
                    covered |= (alpha > 0) & (alpha == alpha.max())
                # Use the composed display: an opaque uniform layer can hide
                # all the variation of an image underneath it.
                colors = _rgb(pixels, background)[covered]
                span = float(np.ptp(colors, axis=0).max()) if colors.size else 0
                if span < min_image_range:
                    detail = (
                        f"panel image range {span:.4f} is below {min_image_range:g}"
                    )
        if detail is not None:
            found.append(
                ("empty-panel", ax, None, tuple(map(float, ax.bbox.extents)), detail)
            )
    return found
