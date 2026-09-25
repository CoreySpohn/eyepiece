"""Pure numerical prepared-view records.

Library viz modules (physicaloptix, orbix, ...) build these records once
from scientific data; ``eyepiece.mpl`` and ``eyepiece.manim`` render them.
This package imports only the standard library and NumPy, so importing it
never loads matplotlib or a simulation library.
"""

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
    "AxisSpec",
    "CurveView",
    "ImageView",
    "Label",
    "Mark",
    "PanelGroup",
    "Path",
    "Points",
    "ReferenceLine",
    "Region",
    "Scale",
    "TrackView",
    "View",
    "find_element",
    "replace_elements",
]
