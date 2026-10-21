"""Moves, attention, and camera geometry for explanatory clips.

Example::

    import eyepiece as ep

    reveal = ep.motion.head_draw(curve)

Exports load on first access and preserve their implementation objects.
"""

from typing import TYPE_CHECKING

from eyepiece import _EXPORT_TARGETS, _lazy_lookup

if TYPE_CHECKING:
    from eyepiece._arrows import (
        angle_arc,
        arrow,
        compass,
    )
    from eyepiece._equation import (
        equation,
        transform,
    )
    from eyepiece._motion import (
        pixel_quads,
        view_limits,
        zoom_path,
    )
    from eyepiece._photons import (
        ApertureCounts,
        Arrivals,
        aperture_counts,
        arrivals,
        exposure_text,
        photons,
    )
    from eyepiece._pixel_morph import (
        brush,
        quad_image,
    )
    from eyepiece._relay import (
        carried_tag,
    )
    from eyepiece._result import (
        MotionContract,
    )
    from eyepiece._warp import (
        warp,
        warp_grid,
    )
    from eyepiece._zoom import (
        follow_zoom,
    )
    from eyepiece.attention import (
        Anchor,
        Stage,
        build_axes,
        callout,
        circumscribe,
        click,
        collapse,
        draw_on,
        flight,
        head_draw,
        indicate,
        leader,
        link,
        morph,
        readout,
        ruler_readout,
        term_row,
        underline,
        words,
        write,
    )
    from eyepiece.camera import (
        PageCamera,
        overview_box,
        spotlight,
    )
    from eyepiece.emphasis import (
        blend,
        capture,
        fade,
    )

__all__ = [
    "Anchor",
    "ApertureCounts",
    "Arrivals",
    "MotionContract",
    "PageCamera",
    "Stage",
    "angle_arc",
    "aperture_counts",
    "arrivals",
    "arrow",
    "blend",
    "brush",
    "build_axes",
    "callout",
    "capture",
    "carried_tag",
    "circumscribe",
    "click",
    "collapse",
    "compass",
    "draw_on",
    "equation",
    "exposure_text",
    "fade",
    "flight",
    "follow_zoom",
    "head_draw",
    "indicate",
    "leader",
    "link",
    "morph",
    "overview_box",
    "photons",
    "pixel_quads",
    "quad_image",
    "readout",
    "ruler_readout",
    "spotlight",
    "term_row",
    "transform",
    "underline",
    "view_limits",
    "warp",
    "warp_grid",
    "words",
    "write",
    "zoom_path",
]

_EXPORTS = {name: _EXPORT_TARGETS[name] for name in __all__}


def __getattr__(name):
    """Load one public task export on first access."""
    return _lazy_lookup(name, _EXPORTS, globals())


def __dir__():
    """Include lazy task exports in discovery."""
    return sorted(set(globals()) | set(__all__))
