"""Still and updatable data displays, layouts, and marks.

Example::

    import eyepiece as ep

    result = ep.plot.imshow_log(image)

Exports load on first access and preserve their implementation objects.
"""

from typing import TYPE_CHECKING

from eyepiece import _EXPORT_TARGETS, _lazy_lookup

if TYPE_CHECKING:
    from eyepiece._diagnostics import (
        calibration,
        strip,
    )
    from eyepiece._hop import (
        hop,
    )
    from eyepiece._labels import (
        spread_labels,
    )
    from eyepiece._phasor import (
        phase_ring,
        phasor,
    )
    from eyepiece._result import (
        ARTIST_KEYS,
        MosaicResult,
        PlotResult,
    )
    from eyepiece._scatter import (
        scatter_values,
    )
    from eyepiece._schematic import (
        GLYPHS,
        rail,
        rail_panels,
        schematic,
    )
    from eyepiece._weighted import (
        weighted_draws,
    )
    from eyepiece.emphasis import (
        step_list,
    )
    from eyepiece.images import (
        bracket,
        compare_grid,
        compare_row,
        display_limits,
        imshow_diverging,
        imshow_log,
        kymograph,
        overlay_circle,
        overlay_line,
        ruler,
        show_field,
        triptych,
    )
    from eyepiece.insets import (
        curve_insets,
    )
    from eyepiece.layout import (
        Frame,
        extent_arcsec,
        extent_au,
        extent_lod,
        extent_lod_from_pixels,
        label_arcsec,
        label_au,
        label_lod,
    )
    from eyepiece.marks import (
        log_zero_floor,
        ref_line,
        shade_intervals,
    )
    from eyepiece.profiles import (
        plot_contrast_curve,
        plot_radial,
        plot_radial_image,
    )
    from eyepiece.scene import (
        fading_track,
        sky_fan,
        trail,
    )
    from eyepiece.stats import (
        convergence,
        corner,
        corner_overlay,
        cov_ellipse,
        hist_fill,
        hist_vs_pdf,
        interval_points,
        signed_trace,
    )

__all__ = [
    "ARTIST_KEYS",
    "GLYPHS",
    "Frame",
    "MosaicResult",
    "PlotResult",
    "bracket",
    "calibration",
    "compare_grid",
    "compare_row",
    "convergence",
    "corner",
    "corner_overlay",
    "cov_ellipse",
    "curve_insets",
    "display_limits",
    "extent_arcsec",
    "extent_au",
    "extent_lod",
    "extent_lod_from_pixels",
    "fading_track",
    "hist_fill",
    "hist_vs_pdf",
    "hop",
    "imshow_diverging",
    "imshow_log",
    "interval_points",
    "kymograph",
    "label_arcsec",
    "label_au",
    "label_lod",
    "log_zero_floor",
    "overlay_circle",
    "overlay_line",
    "phase_ring",
    "phasor",
    "plot_contrast_curve",
    "plot_radial",
    "plot_radial_image",
    "rail",
    "rail_panels",
    "ref_line",
    "ruler",
    "scatter_values",
    "schematic",
    "shade_intervals",
    "show_field",
    "signed_trace",
    "sky_fan",
    "spread_labels",
    "step_list",
    "strip",
    "trail",
    "triptych",
    "weighted_draws",
]

_EXPORTS = {name: _EXPORT_TARGETS[name] for name in __all__}


def __getattr__(name):
    """Load one public task export on first access."""
    return _lazy_lookup(name, _EXPORTS, globals())


def __dir__():
    """Include lazy task exports in discovery."""
    return sorted(set(globals()) | set(__all__))
