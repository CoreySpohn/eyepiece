"""Morphs: an image drawn as movable pixel quads, and linked brushing.

A change of representation, an image becoming a profile or a cloud of
values, is easiest to follow when every pixel visibly keeps its identity on
the way. `quad_image` draws an image as one quadrilateral per pixel in a
single `PolyCollection`, so each frame can move every pixel anywhere with
one `update`, keeping its color; `pixel_quads` builds the quads from a grid
of corners, which a caller moves through any mapping it likes (unrolling
rings around a star into columns, for example).

`brush` is how two panels show that they hold the same objects without an
arrow drawn across them: it veils each panel in its background color and
redraws a chosen subset of the objects above the veil in every panel at
once, so a ring of pixels lights in the image, in its unrolled column, and
in the cloud of its values together.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Rectangle

from eyepiece import _style
from eyepiece._motion import pixel_quads
from eyepiece._result import MosaicResult, PlotResult


def _quads(quads, n, what="quads"):
    q = np.asarray(quads, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (4, 2) or q.shape[0] != n:
        raise ValueError(f"{what} must be ({n}, 4, 2), got {q.shape}")
    return q


def _background(ax):
    """An axes' own background color, or its figure's where it is transparent."""
    face = to_rgba(ax.get_facecolor())
    if face[3] > 0.0:
        return face
    fig_face = to_rgba(ax.figure.get_facecolor())
    return fig_face if fig_face[3] > 0.0 else to_rgba("white")


def _default_norm(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("values has no finite entry to scale a colormap by")
    return Normalize(vmin=float(finite.min()), vmax=float(finite.max()))


def quad_image(values, corners, *, ax=None, keep=None, cmap=None, norm=None,
               collection_kw=None):  # fmt: skip
    """Draw an image as one movable quadrilateral per pixel.

    Each pixel is the quad of its four corners and is filled with its value's
    color; nothing is interpolated. Move the pixels by passing new quads to
    `update` (from `pixel_quads` on moved corners, or any ``(n, 4, 2)``
    array in the same order), and every pixel keeps its color and identity.
    Edges take the face color at a hairline width, so adjacent quads that are
    no longer axis aligned leave no background seams.

    Args:
        values: ``(ny, nx)`` pixel values; nonfinite values draw in the
            colormap's bad color.
        corners: ``(ny + 1, nx + 1, 2)`` corner positions in data units.
        ax: Axes to draw on. None creates and owns a new figure.
        keep: Optional ``(ny, nx)`` boolean mask of the pixels to draw.
        cmap: Colormap or registered name. None uses the intensity role at
            call time.
        norm: A matplotlib `Normalize` (such as `LogNorm`). None scales
            linearly over the finite drawn values.
        collection_kw: Extra keyword arguments for the `PolyCollection`,
            merged last.

    Returns:
        A `PlotResult` whose ``artists["collection"]`` is the
        `PolyCollection`, with ``update(quads=None, values=None)``: new
        ``(n, 4, 2)`` quads and/or ``(n,)`` values (or the full ``(ny, nx)``
        grid) for the drawn pixels, applied to the same artist under the
        first draw's norm, and never rescaling the axes.

    Raises:
        ValueError: If `corners` does not match `values`, `keep` does not
            match, or no drawn value is finite and no norm is given.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"values must be (ny, nx), got {values.shape}")
    ny, nx = values.shape
    if np.shape(corners) != (ny + 1, nx + 1, 2):
        raise ValueError(
            f"corners must be ({ny + 1}, {nx + 1}, 2) for {ny}x{nx} values, "
            f"got {np.shape(corners)}"
        )
    quads = pixel_quads(corners, keep)
    keep_mask = None if keep is None else np.asarray(keep, dtype=bool)
    drawn = values.ravel() if keep_mask is None else values[keep_mask]
    n = drawn.size
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    kw = {
        "cmap": _style.cmap("intensity", cmap),
        "norm": _default_norm(drawn) if norm is None else norm,
        "edgecolors": "face",
        "linewidths": 0.25,
        "antialiased": True,
    }
    kw.update(collection_kw or {})
    coll = PolyCollection(quads, array=np.ma.masked_invalid(drawn), **kw)
    ax.add_collection(coll, autolim=True)
    ax.autoscale_view()

    def update(quads=None, values=None):
        # Everything is validated before anything is applied, so a bad
        # update leaves the artist exactly as it was.
        new_quads = None if quads is None else _quads(quads, n)
        new_values = None
        if values is not None:
            v = np.asarray(values, dtype=float)
            if keep_mask is not None and v.shape == keep_mask.shape:
                v = v[keep_mask]
            elif v.shape == (ny, nx):
                v = v.ravel()
            if v.shape != (n,):
                raise ValueError(
                    f"values must be ({n},) or ({ny}, {nx}), got {v.shape}"
                )
            new_values = v
        if new_quads is not None:
            coll.set_verts(new_quads)
        if new_values is not None:
            coll.set_array(np.ma.masked_invalid(new_values))

    return PlotResult(ax=ax, artists={"collection": coll}, update=update)


def brush(axes, quads, values, *, cmap=None, norm=None, dim=0.72, color=None):
    """Light the same objects in several panels at once, dimming the rest.

    Each panel gets a veil in its background color and an overlay holding
    the brushed objects, both hidden until `update` names which objects to
    light. The overlay redraws those objects above the veil from each
    panel's own quads, so one object can sit at different places in
    different panels (a pixel in the image, in an unrolled column, in a
    cloud of values) and still light everywhere together. Marks drawn at
    zorder above 11 (reference lines, labels) stay above the veil.

    Args:
        axes: The panels, a sequence of axes already drawn.
        quads: One ``(n, 4, 2)`` array per panel, the objects' current
            shapes there, in the same object order.
        values: ``(n,)`` values that color the objects.
        cmap: Colormap or registered name. None uses the intensity role at
            call time.
        norm: A matplotlib `Normalize`. None scales linearly over the finite
            values.
        dim: Veil opacity while brushing.
        color: Veil color for every panel. None uses each panel's own
            background, or the figure's where the panel is transparent.

    Returns:
        A `MosaicResult` with ``artists["fill"]`` (the veils) and
        ``artists["collection"]`` (the overlays), one per panel, and
        ``update(mask, quads=None)``: light the objects where the ``(n,)``
        boolean `mask` is True, optionally at new per-panel quads, or clear
        the brush with ``mask=None``. A rejected update changes nothing.

    Raises:
        ValueError: If the panels, quads and values do not agree.
        TypeError: From `update`, if the mask is not boolean.
    """
    axes = list(np.ravel(np.asarray(axes, dtype=object)))
    values = np.asarray(values, dtype=float).ravel()
    n = values.size
    quads = list(quads)
    if len(quads) != len(axes):
        raise ValueError(
            f"brush needs one quads array per axes: {len(axes)} axes, "
            f"{len(quads)} quads arrays"
        )
    current = [_quads(q, n, f"quads for panel {i}") for i, q in enumerate(quads)]
    cmap = _style.cmap("intensity", cmap)
    norm = _default_norm(values) if norm is None else norm
    veils, overlays = [], []
    for ax in axes:
        face = _background(ax) if color is None else color
        veil = Rectangle(
            (0.0, 0.0), 1.0, 1.0, transform=ax.transAxes, facecolor=face,
            edgecolor="none", alpha=dim, zorder=10, visible=False,
        )  # fmt: skip
        ax.add_patch(veil)
        overlay = PolyCollection(
            np.zeros((0, 4, 2)), array=np.zeros(0), cmap=cmap, norm=norm,
            edgecolors="face", linewidths=0.25, zorder=11, visible=False,
        )  # fmt: skip
        ax.add_collection(overlay, autolim=False)
        veils.append(veil)
        overlays.append(overlay)

    def update(mask, quads=None):
        if quads is not None:
            new = list(quads)
            if len(new) != len(axes):
                raise ValueError("update needs one quads array per axes")
            new = [_quads(q, n, f"quads for panel {i}") for i, q in enumerate(new)]
        else:
            new = current
        if mask is None:
            for veil, overlay in zip(veils, overlays, strict=True):
                veil.set_visible(False)
                overlay.set_visible(False)
            current[:] = new
            return
        m = np.asarray(mask)
        if m.dtype != bool:
            raise TypeError(
                f"mask must be a boolean array, got {m.dtype}; an index array "
                "would silently light the wrong objects"
            )
        if m.shape != (n,):
            raise ValueError(f"mask must be ({n},), got {m.shape}")
        current[:] = new
        for veil, overlay, q in zip(veils, overlays, current, strict=True):
            overlay.set_verts(q[m])
            overlay.set_array(np.ma.masked_invalid(values[m]))
            veil.set_visible(True)
            overlay.set_visible(True)

    return MosaicResult(
        axes=np.array(axes, dtype=object),
        artists={"fill": veils, "collection": overlays},
        update=update,
    )
