"""Optical-train rail smoke + highlight."""

import hwostyle
import matplotlib.pyplot as plt
import numpy as np
import pytest

import eyepiece
from eyepiece.schematic import GLYPHS, rail, schematic


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


@pytest.mark.parametrize("glyph", sorted(GLYPHS))
def test_every_glyph_renders(glyph):
    res = rail([("Plane", glyph)])
    assert res.ax.patches or res.ax.lines
    plt.close(res.fig)


def test_rail_draws_every_glyph_together():
    planes = [(g.upper(), g) for g in sorted(GLYPHS)]
    res = rail(planes)
    assert len(res.artists["lines"]) == len(planes)
    assert len(res.artists["text"]) == len(planes)
    plt.close(res.fig)


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


def _rail_glyph_facecolor():
    res = rail([("PP", "pupil"), ("FP", "focal")])
    facecolor = tuple(np.ravel(res.artists["fill"].get_facecolor()))
    plt.close(res.fig)
    return facecolor


def test_rail_neutrals_follow_the_mode():
    hwostyle.use("dark")
    dark = _rail_glyph_facecolor()
    with hwostyle.light():
        light = _rail_glyph_facecolor()
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
