"""Plotting and animation primitives for astronomical imaging simulations.

eyepiece provides a small set of pure functions for rendering, comparing,
and animating simulated images (point spread functions, coronagraph
detection maps, complex optical fields, and the like). Every public name
lives at the top level of this package; the submodules that implement them
(``images``, ``layout``, ``output``, ``anim``, ``scene``, ``stats``,
``schematic``, ``profiles``) are internal organization and are not part of
the public API. The vocabularies a caller reads against are exported alongside the
functions: ``ARTIST_KEYS``, the key set a result's ``artists`` dict draws
from; ``PRESETS``, the measured fps/dpi pairs for the usual animation
destinations; and ``GLYPHS``, the optical-element names ``rail`` draws.

Example::

    import eyepiece as ep

    result = ep.imshow_log(image)
    ep.save_fig(result.fig, "psf")

The library never imports a simulation library and never mutates
matplotlib's global state at import time; style is applied explicitly by
the caller (see ``eyepiece._style``), not captured when this package loads.
"""

from eyepiece._result import ARTIST_KEYS, MosaicResult, PlotResult
from eyepiece._version import __version__
from eyepiece.anim import PRESETS, animate, record
from eyepiece.images import (
    compare_row,
    imshow_diverging,
    imshow_log,
    show_field,
    triptych,
)
from eyepiece.layout import (
    Frame,
    SourceStyles,
    extent_arcsec,
    extent_au,
    extent_lod,
    extent_lod_from_pixels,
    label_arcsec,
    label_au,
    label_lod,
)
from eyepiece.output import save_fig
from eyepiece.profiles import plot_contrast_curve, plot_radial, radial_profile_plot
from eyepiece.scene import fading_track, sky_fan, trail
from eyepiece.schematic import GLYPHS, rail, schematic
from eyepiece.stats import corner, corner_overlay, cov_ellipse, hist_vs_pdf

__all__ = [
    "ARTIST_KEYS",
    "GLYPHS",
    "PRESETS",
    "Frame",
    "MosaicResult",
    "PlotResult",
    "SourceStyles",
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
    "fading_track",
    "hist_vs_pdf",
    "imshow_diverging",
    "imshow_log",
    "label_arcsec",
    "label_au",
    "label_lod",
    "plot_contrast_curve",
    "plot_radial",
    "radial_profile_plot",
    "rail",
    "record",
    "save_fig",
    "schematic",
    "show_field",
    "sky_fan",
    "trail",
    "triptych",
]
