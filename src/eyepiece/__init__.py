"""Plotting and animation primitives for astronomical imaging simulations.

eyepiece provides a small set of pure functions for rendering, comparing,
and animating simulated images (point spread functions, coronagraph
detection maps, complex optical fields, and the like). Every public name
lives at the top level of this package; the submodules that implement them
(``images``, ``layout``, ``output``, ``anim``) are internal organization
and are not part of the public API.

Example::

    import eyepiece as ep

    result = ep.imshow_log(image)
    ep.save_fig(result.fig, "psf")

The library never imports a simulation library and never mutates
matplotlib's global state at import time; style is applied explicitly by
the caller (see ``eyepiece._style``), not captured when this package loads.
"""

from eyepiece._result import MosaicResult, PlotResult
from eyepiece._version import __version__
from eyepiece.anim import animate, record
from eyepiece.images import compare_row, imshow_diverging, imshow_log, show_field
from eyepiece.layout import (
    extent_arcsec,
    extent_au,
    extent_lod,
    extent_lod_from_pixels,
    label_arcsec,
    label_au,
    label_lod,
)
from eyepiece.output import save_fig
from eyepiece.stats import corner, corner_overlay, cov_ellipse, hist_vs_pdf

__all__ = [
    "MosaicResult",
    "PlotResult",
    "__version__",
    "animate",
    "compare_row",
    "corner",
    "corner_overlay",
    "cov_ellipse",
    "extent_arcsec",
    "extent_au",
    "extent_lod",
    "extent_lod_from_pixels",
    "hist_vs_pdf",
    "imshow_diverging",
    "imshow_log",
    "label_arcsec",
    "label_au",
    "label_lod",
    "record",
    "save_fig",
    "show_field",
]
