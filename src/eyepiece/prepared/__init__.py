"""Pure numerical prepared-view records.

Library viz modules (physicaloptix, orbix, ...) build these records once
from scientific data; ``eyepiece.mpl`` and ``eyepiece.manim`` render them.
This package imports only the standard library and NumPy, so importing it
never loads matplotlib or a simulation library.
"""

from eyepiece.prepared._display import (
    map_rgba,
    normalize_values,
    resolve_bounds,
    weight_opacity,
)
from eyepiece.prepared._sequence import (
    ArrayChannel,
    Clock,
    PathWindow,
    Sample,
    Sequence,
)
from eyepiece.prepared._views import (
    AxisSpec,
    CurveView,
    ImageView,
    Label,
    Mark,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    Scale,
    TrackView,
    View,
    find_element,
    replace_elements,
)

__all__ = [
    "ArrayChannel",
    "AxisSpec",
    "Clock",
    "CurveView",
    "ImageView",
    "Label",
    "Mark",
    "PanelGroup",
    "Path",
    "PathWindow",
    "Points",
    "ReferenceLine",
    "Region",
    "Sample",
    "Scale",
    "Sequence",
    "TrackView",
    "View",
    "find_element",
    "map_rgba",
    "normalize_values",
    "replace_elements",
    "resolve_bounds",
    "weight_opacity",
]
