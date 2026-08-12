"""Optical-train rail smoke + highlight."""

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse, Polygon, Rectangle

import eyepiece
from eyepiece.schematic import GLYPHS, rail, schematic

# A single-plane rail draws four Line2D artists before any glyph does:
# the two beam-envelope edges, the dotted optical axis, and the plane
# marker. Anything past that came from the glyph.
BASELINE_LINES = 4

# What each glyph is expected to add on top of that baseline, as
# (extra Line2D, Rectangle patches, Polygon patches). Measured with
# cap=False so the trailing detector block cannot mask a glyph.
GLYPH_SIGNATURES = {
    "source": (1, 0, 0),
    "pupil": (0, 2, 0),
    "lyot": (0, 2, 0),
    "mask": (0, 2, 0),
    "apodizer": (0, 1, 0),
    "fpm": (0, 0, 1),
    "focal": (0, 0, 2),
    "detector": (0, 1, 0),
}


def test_schematic_draws_and_highlights():
    res = schematic("imager", highlight="focal")
    assert res.ax.patches or res.ax.lines
    lines = res.artists["lines"]
    # "imager" is (pupil, focal); only "focal" is highlighted, so its
    # marker color must differ from the unhighlighted pupil plane's.
    assert lines[0].get_color() != lines[1].get_color()
    plt.close(res.fig)


def test_schematic_unknown_highlight_raises():
    with pytest.raises(ValueError, match="highlight"):
        schematic("imager", highlight="nope")


def test_schematic_non_string_highlight_raises_value_error():
    with pytest.raises(ValueError, match="highlight"):
        schematic("imager", highlight=3)


def _envelope_facecolor():
    res = schematic("coronagraph")
    facecolor = tuple(np.ravel(res.artists["fill"].get_facecolor()))
    plt.close(res.fig)
    return facecolor


def test_schematic_neutrals_follow_the_mode():
    hwostyle.use("dark")
    dark_envelope = _envelope_facecolor()
    with hwostyle.light():
        light_envelope = _envelope_facecolor()
    assert dark_envelope != light_envelope


def test_rail_acceptance_gate():
    r = eyepiece.rail([("PP", "pupil"), ("FP", "focal")])
    assert r.ax is not None
    plt.close(r.fig)
    for g in ("source", "apodizer", "fpm"):
        res = eyepiece.rail([("s", g)])
        assert res.ax is not None
        plt.close(res.fig)


def test_rail_is_exported_at_top_level():
    assert eyepiece.rail is rail


def test_glyph_vocabulary_is_exported_at_top_level():
    assert eyepiece.GLYPHS is GLYPHS
    assert "GLYPHS" in eyepiece.__all__


def _hatch_color(patch):
    for name in ("get_hatchcolor", "get_hatch_color"):
        getter = getattr(patch, name, None)
        if getter is not None:
            return to_rgba(getter())
    return to_rgba(patch._hatch_color)


def test_detector_hatch_follows_the_glyph_ink_not_the_rcparam():
    hwostyle.use("dark")
    with matplotlib.rc_context({"hatch.color": "black"}):
        res = rail([("Det", "detector")], cap=False)
        box = next(p for p in res.ax.patches if isinstance(p, Rectangle))
        hatch = _hatch_color(box)
        edge = to_rgba(box.get_edgecolor())
        plt.close(res.fig)
    assert hatch == edge
    assert hatch != to_rgba("black")


def test_source_star_size_follows_rcparams():
    def star_size():
        res = rail([("Star", "source")], cap=False)
        marker = next(ln for ln in res.ax.lines if ln.get_marker() == "*")
        size = marker.get_markersize()
        plt.close(res.fig)
        return size

    with matplotlib.rc_context({"lines.markersize": 6.0}):
        small = star_size()
    with matplotlib.rc_context({"lines.markersize": 12.0}):
        big = star_size()
    assert big == pytest.approx(2 * small)


def test_every_glyph_in_the_vocabulary_has_a_signature():
    assert set(GLYPH_SIGNATURES) == set(GLYPHS)


@pytest.mark.parametrize("glyph", sorted(GLYPH_SIGNATURES))
def test_glyph_draws_its_own_artists(glyph):
    res = rail([("Plane", glyph)], cap=False)
    extra_lines = len(res.ax.lines) - BASELINE_LINES
    rects = sum(isinstance(p, Rectangle) for p in res.ax.patches)
    polys = sum(isinstance(p, Polygon) for p in res.ax.patches)
    plt.close(res.fig)
    assert (extra_lines, rects, polys) == GLYPH_SIGNATURES[glyph]


def test_rail_draws_every_glyph_together():
    planes = [(g.upper(), g) for g in sorted(GLYPHS)]
    res = rail(planes, cap=False)
    expected_lines = sum(sig[0] for sig in GLYPH_SIGNATURES.values())
    expected_rects = sum(sig[1] for sig in GLYPH_SIGNATURES.values())
    expected_polys = sum(sig[2] for sig in GLYPH_SIGNATURES.values())
    # Three envelope lines, one marker per plane, plus the glyph lines.
    assert len(res.ax.lines) == 3 + len(planes) + expected_lines
    assert sum(isinstance(p, Rectangle) for p in res.ax.patches) == expected_rects
    assert sum(isinstance(p, Polygon) for p in res.ax.patches) == expected_polys
    # One lens after every plane but the last.
    assert sum(isinstance(p, Ellipse) for p in res.ax.patches) == len(planes) - 1
    assert len(res.artists["lines"]) == len(planes)
    assert len(res.artists["text"]) == len(planes)
    plt.close(res.fig)


def _cap_rectangles(**kwargs):
    res = rail([("PP", "pupil"), ("FP", "focal")], **kwargs)
    rects = sum(isinstance(p, Rectangle) for p in res.ax.patches)
    plt.close(res.fig)
    return rects


def test_cap_defaults_to_capping_a_focal_terminated_rail():
    # The pupil glyph contributes two bars; the cap is the third rectangle.
    assert _cap_rectangles() == 3


def test_cap_false_leaves_a_focal_rail_uncapped():
    assert _cap_rectangles(cap=False) == 2


def test_cap_true_caps_a_rail_that_would_not_be_capped():
    uncapped = rail([("PP", "pupil"), ("LS", "lyot")])
    forced = rail([("PP", "pupil"), ("LS", "lyot")], cap=True)
    uncapped_rects = sum(isinstance(p, Rectangle) for p in uncapped.ax.patches)
    forced_rects = sum(isinstance(p, Rectangle) for p in forced.ax.patches)
    plt.close(uncapped.fig)
    plt.close(forced.fig)
    assert forced_rects == uncapped_rects + 1


def test_cap_default_ignores_a_detector_that_is_not_last():
    # A detector early in the train must not suppress the cap on a rail
    # that still ends at a focal plane.
    res = rail([("Det", "detector"), ("FP", "focal")])
    plt.close(res.fig)
    detector_and_cap = 2
    assert sum(isinstance(p, Rectangle) for p in res.ax.patches) == detector_and_cap


def test_cap_default_leaves_a_detector_terminated_rail_alone():
    res = rail([("PP", "pupil"), ("Det", "detector")])
    rects = sum(isinstance(p, Rectangle) for p in res.ax.patches)
    plt.close(res.fig)
    # Two pupil bars plus the detector box, and no cap on top of it.
    assert rects == 3


def test_unknown_glyph_raises_naming_the_vocabulary():
    with pytest.raises(ValueError, match="glyph") as excinfo:
        rail([("Plane", "wormhole")])
    message = str(excinfo.value)
    for name in GLYPHS:
        assert name in message


def test_rail_unknown_highlight_raises():
    with pytest.raises(ValueError, match="highlight"):
        rail([("PP", "pupil"), ("FP", "focal")], highlight="nope")


def test_rail_non_string_highlight_raises():
    with pytest.raises(ValueError, match="highlight"):
        rail([("PP", "pupil"), ("FP", "focal")], highlight=1)


def test_rail_highlight_matches_the_label_case_insensitively():
    res = rail([("PP", "pupil"), ("FP", "focal")], highlight="fp")
    lines = res.artists["lines"]
    assert lines[0].get_color() != lines[1].get_color()
    plt.close(res.fig)


def test_rail_highlight_changes_that_planes_color():
    plain = rail([("PP", "pupil"), ("FP", "focal")])
    plain_color = plain.artists["lines"][1].get_color()
    plt.close(plain.fig)
    lit = rail([("PP", "pupil"), ("FP", "focal")], highlight="FP")
    lit_color = lit.artists["lines"][1].get_color()
    plt.close(lit.fig)
    assert plain_color != lit_color


def test_rail_accent_override_is_honored():
    res = rail([("PP", "pupil"), ("FP", "focal")], highlight="FP", accent="#123456")
    assert res.artists["lines"][1].get_color() == "#123456"
    plt.close(res.fig)


def test_rail_honors_explicit_positions():
    res = rail([("PP", "pupil"), ("FP", "focal")], positions=[0.2, 0.7])
    xs = [float(line.get_xdata()[0]) for line in res.artists["lines"]]
    assert xs == pytest.approx([0.2, 0.7])
    plt.close(res.fig)


def test_rail_position_count_must_match_plane_count():
    with pytest.raises(ValueError, match="positions"):
        rail([("PP", "pupil"), ("FP", "focal")], positions=[0.5])


def test_single_plane_rail_works():
    res = rail([("Only", "pupil")])
    assert len(res.artists["lines"]) == 1
    x = float(res.artists["lines"][0].get_xdata()[0])
    assert 0.0 < x < 1.0
    plt.close(res.fig)


def test_rail_needs_at_least_one_plane():
    with pytest.raises(ValueError, match="plane"):
        rail([])


def test_rail_draws_into_a_given_ax():
    fig, ax = plt.subplots()
    res = rail([("PP", "pupil"), ("FP", "focal")], ax=ax)
    assert res.ax is ax
    plt.close(fig)


def test_rail_artist_keys_are_in_the_vocabulary():
    res = rail([("PP", "pupil"), ("FP", "focal")])
    assert set(res.artists) <= set(eyepiece.ARTIST_KEYS)
    plt.close(res.fig)


def _rail_envelope_facecolor():
    res = rail([("PP", "pupil"), ("FP", "focal")])
    facecolor = tuple(np.ravel(res.artists["fill"].get_facecolor()))
    plt.close(res.fig)
    return facecolor


def test_rail_envelope_neutral_follows_the_mode():
    hwostyle.use("dark")
    dark = _rail_envelope_facecolor()
    with hwostyle.light():
        light = _rail_envelope_facecolor()
    assert dark != light


def _pupil_bar_facecolor():
    res = rail([("PP", "pupil")])
    facecolor = res.ax.patches[0].get_facecolor()
    plt.close(res.fig)
    return facecolor


def test_rail_glyph_tone_follows_the_mode():
    hwostyle.use("dark")
    dark = _pupil_bar_facecolor()
    with hwostyle.light():
        light = _pupil_bar_facecolor()
    assert dark != light
