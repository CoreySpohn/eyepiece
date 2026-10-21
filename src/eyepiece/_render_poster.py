"""Deterministic posters, stills, and PNG comparisons."""

import operator
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib._pylab_helpers import Gcf
from PIL import Image

import eyepiece
from eyepiece._render_pool import (
    _close_new_figures,
    _default_style,
    _figure,
    _frame_source,
    _raster,
)


def render_poster(
    build, frames, index, path, *, dpi, frame_at=None, setup=None, rc=None
) -> Path:
    """Atomically save a worker's frame pixels after even-size adjustment as a PNG.

    Layout is frozen after drawing frame zero, exactly as for a video worker.
    The PNG contains RGBA pixels without metadata. Figure, style, and rc state
    are restored even when drawing or writing fails.

    Args:
        build: Callable returning (fig, draw), as for render_frames.
        frames: Frame item sequence or integer frame count.
        index: Zero-based frame index before any video holds.
        path: Destination PNG; parent directories are created as needed.
        dpi: Rasterization dots per inch.
        frame_at: Optional index-to-item callable for an integer frame count.
        setup: Callable run before applying rc and building the figure.
        rc: Matplotlib overrides held open during building and drawing.

    Returns:
        The destination Path.

    Raises:
        ValueError: For a non-PNG suffix, invalid frame source, or nonpositive dpi.
        IndexError: If index lies outside the frame source.
    """
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("render_poster requires a .png path")
    count = _frame_source(frames, frame_at)
    index = operator.index(index)
    if not 0 <= index < count:
        raise IndexError(f"poster frame {index} outside [0, {count})")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    with _figure(build, frames, dpi, frame_at, setup, rc) as state:
        pixels = _raster(state, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        Image.fromarray(pixels).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def render_still(
    build, path, *, dpi=None, setup=None, rc=None, metadata=None, **savefig_kwargs
) -> Path:
    """Build and save a deterministic still through eyepiece.save_fig.

    Default style state, setup, and rc are applied in the same order as for
    workers. The built figure is closed and the parent's state restored on exit.

    Args:
        build: Callable returning a matplotlib Figure.
        path: Destination PNG, PDF, or SVG; parent directories are created.
        dpi: PNG resolution override; ignored for vector formats.
        setup: Callable run before applying rc and building the figure.
        rc: Matplotlib overrides held open during building and saving.
        metadata: Optional format-specific metadata passed to save_fig.
        **savefig_kwargs: Extra savefig options passed to save_fig, overriding
            style defaults. Use bbox_inches=None to preserve the figure size.

    Returns:
        The saved Path returned by save_fig.
    """
    path = Path(path)
    with ExitStack() as stack:
        stack.enter_context(matplotlib.rc_context())
        stack.enter_context(_default_style())
        stack.enter_context(matplotlib.rc_context(matplotlib.rcParamsDefault))
        stack.callback(_close_new_figures, set(Gcf.get_all_fig_managers()))
        if setup is not None:
            setup()
        stack.enter_context(matplotlib.rc_context(rc))
        fig = build()
        stack.callback(plt.close, fig)
        overrides = {}
        if dpi is not None and path.suffix.lower() in ("", ".png"):
            overrides["dpi"] = dpi
        if metadata is not None:
            overrides["metadata"] = metadata
        overrides.update(savefig_kwargs)
        return eyepiece.save_fig(fig, path.name, dir=path.parent, **overrides)


def same_picture(a, b, *, tol=2, ignore_bottom_px=0) -> bool:
    """Compare PNG RGBA levels, excluding an optional bottom band.

    Images without alpha are treated as fully opaque.

    Args:
        a: First PNG path.
        b: Second PNG path.
        tol: Maximum allowed absolute difference in any RGBA channel, inclusive.
        ignore_bottom_px: Number of bottom rows excluded from comparison.

    Returns:
        Whether shapes match and every compared RGBA difference is within tol.

    Raises:
        ValueError: If either argument is negative or the bottom band excludes
            the entire image.
    """
    if tol < 0 or ignore_bottom_px < 0:
        raise ValueError("tol and ignore_bottom_px must be nonnegative")
    with Image.open(a) as image:
        first = np.array(image.convert("RGBA"))
    with Image.open(b) as image:
        second = np.array(image.convert("RGBA"))
    if ignore_bottom_px >= min(first.shape[0], second.shape[0]):
        raise ValueError("ignore_bottom_px must be less than the image height")
    if first.shape != second.shape:
        return False
    rows = first.shape[0] - ignore_bottom_px
    difference = first[:rows].astype(np.int16) - second[:rows].astype(np.int16)
    return bool(np.max(np.abs(difference)) <= tol)


# Keep public pickle references stable through the facade.
render_poster.__module__ = "eyepiece.render"
render_still.__module__ = "eyepiece.render"
same_picture.__module__ = "eyepiece.render"
