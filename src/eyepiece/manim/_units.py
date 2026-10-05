"""Point sizes at the delivered frame, converted to Manim's scene sizes."""

import dataclasses
import math

# Manim's default frame is 8 scene units tall.
_FRAME_HEIGHT = 8.0
# A Manim `font_size` of 72 sets a one-unit em.
_FONT_SIZE_PER_UNIT = 72.0
# Cairo draws a Manim `stroke_width` of 1 as 0.01 scene units.
_UNITS_PER_STROKE_WIDTH = 0.01
_PT_PER_INCH = 72.0


@dataclasses.dataclass(frozen=True)
class Units:
    """How many points of the delivered picture one Manim scene unit spans.

    Manim measures lengths in scene units, text in `font_size`, and lines in
    `stroke_width`. A figure is designed in points at the size it is shown,
    so a scene that must match a still (or a minimum legible size) needs one
    scale between the two. `Units` holds that scale and converts both ways.

    The default, 72 points per unit, is Manim's own font-size points:
    `font_size(pt) == pt`, the convention a `RenderProfile` passed to
    `render` or `animate` follows. `Units.for_frame` derives the scale from
    the delivered pixel size and resolution instead.

    Attributes:
        pt_per_unit: Points at the delivered size per scene unit; positive
            and finite.

    Raises:
        ValueError: If `pt_per_unit` is not positive and finite.

    Example:
        >>> units = Units.for_frame(pixel_height=1080, dpi=135.0)
        >>> units.pt_per_unit
        72.0
        >>> units.stroke_width(2.16)
        3.0
    """

    pt_per_unit: float = 72.0

    def __post_init__(self):
        value = float(self.pt_per_unit)
        if not (math.isfinite(value) and value > 0):
            raise ValueError(
                f"pt_per_unit must be positive and finite, got {self.pt_per_unit!r}"
            )
        object.__setattr__(self, "pt_per_unit", value)

    @classmethod
    def for_frame(cls, pixel_height, dpi, frame_height=_FRAME_HEIGHT):
        """Units for a frame shown `pixel_height` pixels tall at `dpi`.

        Args:
            pixel_height: Height of the rendered frame in pixels.
            dpi: Pixels per inch at the delivered size.
            frame_height: Scene units across the frame height (Manim's
                default, 8).

        Returns:
            A `Units`.
        """
        return cls(pixel_height / frame_height * _PT_PER_INCH / dpi)

    def units(self, pt):
        """Scene units for a length of `pt` points."""
        return pt / self.pt_per_unit

    def pt(self, units):
        """Points for a length of `units` scene units."""
        return units * self.pt_per_unit

    def font_size(self, pt):
        """Manim `font_size` that draws text with a `pt`-point em."""
        return _FONT_SIZE_PER_UNIT * self.units(pt)

    def stroke_width(self, pt):
        """Manim `stroke_width` that draws a line `pt` points wide."""
        return self.units(pt) / _UNITS_PER_STROKE_WIDTH

    def stroke_pt(self, stroke_width):
        """Points wide of a line drawn with Manim `stroke_width`."""
        return self.pt(stroke_width * _UNITS_PER_STROKE_WIDTH)
