"""Manim point-size conversions: one scale, both directions."""

import pytest

pytest.importorskip("manim")


def test_default_is_manim_font_size_points():
    from eyepiece.manim import Units

    units = Units()
    assert units.pt_per_unit == 72.0
    assert units.font_size(24.0) == pytest.approx(24.0)
    assert units.units(72.0) == pytest.approx(1.0)


def test_for_frame_derives_points_per_unit():
    from eyepiece.manim import Units

    # 1080 pixels over 8 units at 135 dpi: 135 px/unit, so 72 pt/unit.
    assert Units.for_frame(1080, 135.0).pt_per_unit == pytest.approx(72.0)
    # Same frame shown at 270 dpi: each unit spans half as many points.
    assert Units.for_frame(1080, 270.0).pt_per_unit == pytest.approx(36.0)


def test_stroke_conversions_invert():
    from eyepiece.manim import Units

    units = Units()
    # Cairo draws stroke_width 3 as 0.03 units, 2.16 pt at 72 pt/unit.
    assert units.stroke_pt(3.0) == pytest.approx(2.16)
    assert units.stroke_width(2.16) == pytest.approx(3.0)
    assert units.pt(units.units(5.0)) == pytest.approx(5.0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_rejects_nonpositive_or_nonfinite_scale(bad):
    from eyepiece.manim import Units

    with pytest.raises(ValueError, match="pt_per_unit"):
        Units(bad)
