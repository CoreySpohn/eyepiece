"""Optical-train rail smoke + highlight."""

import hwostyle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch, Polygon, Rectangle
from matplotlib.text import Text

import eyepiece
from eyepiece._schematic import GLYPHS, rail, schematic

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
    "dm": (1, 1, 0),
    "fpm": (0, 0, 1),
    "phase_mask": (0, 1, 0),
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


def _lenses(res):
    return sorted(p.center[0] for p in res.ax.patches if isinstance(p, Ellipse))


def _half_width_at(res, x):
    # The first line a rail draws is the upper beam edge.
    upper = res.ax.lines[0]
    return float(np.interp(x, upper.get_xdata(), upper.get_ydata())) - 0.52


def test_default_gaps_are_one_fourier_lens_each():
    planes = [("P", "pupil"), ("F", "fpm"), ("L", "lyot"), ("I", "focal")]
    default = rail(planes)
    explicit = rail(planes, gaps=["fourier"] * 3)
    assert len(_lenses(default)) == 3
    assert _lenses(default) == _lenses(explicit)
    np.testing.assert_allclose(
        default.ax.lines[0].get_ydata(), explicit.ax.lines[0].get_ydata()
    )
    plt.close(default.fig)
    plt.close(explicit.fig)


def test_pupil_relay_is_a_lens_pair_around_an_intermediate_focus():
    positions = (0.1, 0.5, 0.9)
    res = rail(
        [("P", "pupil"), ("DM", "pupil"), ("I", "focal")],
        positions=positions,
        gaps=["relay", "fourier"],
    )
    lenses = _lenses(res)
    assert lenses[:2] == pytest.approx([0.2, 0.4])
    assert _half_width_at(res, 0.3) < 0.05  # the relay's intermediate focus
    assert _half_width_at(res, 0.2) > 0.15  # collimated up to the first lens
    plt.close(res.fig)


def test_image_relay_opens_a_collimated_beam_between_two_foci():
    res = rail([("F1", "focal"), ("F2", "focal")], positions=(0.1, 0.9), gaps=["relay"])
    assert len(_lenses(res)) == 2
    assert _half_width_at(res, 0.5) > 0.15
    assert _half_width_at(res, 0.1) < 0.05
    plt.close(res.fig)


def test_none_gap_is_free_space_in_a_collimated_beam():
    res = rail(
        [("P", "pupil"), ("Stop", "mask"), ("I", "focal")],
        positions=(0.1, 0.4, 0.9),
        gaps=["none", "fourier"],
    )
    lenses = _lenses(res)
    assert len(lenses) == 1 and lenses[0] > 0.4
    assert _half_width_at(res, 0.25) == pytest.approx(_half_width_at(res, 0.1))
    plt.close(res.fig)


@pytest.mark.parametrize(
    ("gaps", "match"), [(["fourier"], "gaps has 1"), (["warp", "none"], "unknown gap")]
)
def test_bad_gaps_raise(gaps, match):
    with pytest.raises(ValueError, match=match):
        rail([("P", "pupil"), ("F", "focal"), ("L", "lyot")], gaps=gaps)


_TRAIN = [
    ("P", "pupil"),
    ("DM", "dm"),
    ("F", "phase_mask"),
    ("L", "lyot"),
    ("I", "focal"),
]
_TRAIN_AT = (0.1, 0.3, 0.5, 0.7, 0.9)


def test_fourier_lens_after_is_the_default():
    default = rail(_TRAIN, positions=_TRAIN_AT)
    explicit = rail(_TRAIN, positions=_TRAIN_AT, fourier_lens="after")
    assert _lenses(default) == _lenses(explicit)
    np.testing.assert_array_equal(
        default.ax.lines[0].get_ydata(), explicit.ax.lines[0].get_ydata()
    )
    plt.close(default.fig)
    plt.close(explicit.fig)


def test_middle_fourier_lens_sits_mid_gap_with_the_beam_collimated_to_it():
    res = rail(
        _TRAIN,
        positions=_TRAIN_AT,
        gaps=["relay", "fourier", "fourier", "fourier"],
        fourier_lens="middle",
    )
    assert _lenses(res)[2:] == pytest.approx([0.4, 0.6, 0.8])
    # DM (pupil) -> lens: collimated; lens -> phase mask: converging to a focus.
    assert _half_width_at(res, 0.35) == pytest.approx(0.20)
    assert _half_width_at(res, 0.45) == pytest.approx(0.109, abs=0.003)
    assert _half_width_at(res, 0.5) == pytest.approx(0.018, abs=0.003)
    # Phase mask -> lens: opening from the focus; lens -> Lyot: collimated.
    assert _half_width_at(res, 0.65) == pytest.approx(0.20)
    plt.close(res.fig)


def test_a_stop_narrows_the_beam_from_its_plane_onward():
    res = rail(
        _TRAIN,
        positions=_TRAIN_AT,
        fourier_lens="middle",
        stops={"l": 0.8},
    )
    assert _half_width_at(res, 0.7 - 0.01) == pytest.approx(0.20)
    assert _half_width_at(res, 0.7 + 0.01) == pytest.approx(0.16)
    # The narrowed beam stays collimated to the last lens, then focuses.
    assert _half_width_at(res, 0.75) == pytest.approx(0.16)
    assert _half_width_at(res, 0.9) == pytest.approx(0.018, abs=0.003)
    # The step is drawn at the plane, not smeared across a sample.
    xs = res.ax.lines[0].get_xdata()
    assert np.any(np.isclose(xs, 0.7)) and np.any(np.isclose(xs, 0.7 + 1e-6))
    plt.close(res.fig)


def test_a_full_stop_leaves_the_beam_alone():
    plain = rail(_TRAIN, positions=_TRAIN_AT)
    full = rail(_TRAIN, positions=_TRAIN_AT, stops={"L": 1.0})
    np.testing.assert_array_equal(
        plain.ax.lines[0].get_ydata(), full.ax.lines[0].get_ydata()
    )
    plt.close(plain.fig)
    plt.close(full.fig)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fourier_lens": "before"}, "unknown fourier_lens"),
        ({"stops": {"Nowhere": 0.8}}, "unknown stop plane"),
        ({"stops": {"L": 0.0}}, r"must be in \(0, 1\]"),
        ({"stops": {"L": 1.2}}, r"must be in \(0, 1\]"),
        ({"stops": {"F": 0.8}}, "image-like glyph"),
    ],
)
def test_bad_fourier_lens_or_stops_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rail(_TRAIN, **kwargs)


# The axes-coordinate layout of the released rail, pinned as numbers: the
# optical axis sits at 0.52, the collimated beam is 0.20 either side of it
# and a focus 0.018, a pupil-like marker reaches 0.06 past the beam and an
# image-like one 0.06 past a 0.13 floor, labels sit 0.04 above the marker,
# and a lens follows its plane by 0.035.
_CORONAGRAPH_AT = (0.10, 0.36, 0.63, 0.92)


def test_default_coronagraph_geometry_matches_the_released_layout():
    res = schematic("coronagraph", highlight="lyot")
    markers = res.artists["lines"]
    assert [float(m.get_xdata()[0]) for m in markers] == pytest.approx(_CORONAGRAPH_AT)
    spans = [tuple(float(y) for y in m.get_ydata()) for m in markers]
    pupil_like, image_like = (0.26, 0.78), (0.33, 0.71)
    assert spans == [
        pytest.approx(pupil_like),
        pytest.approx(image_like),
        pytest.approx(pupil_like),
        pytest.approx(image_like),
    ]
    tops = [float(t.get_position()[1]) for t in res.artists["text"]]
    assert tops == pytest.approx([0.82, 0.75, 0.82, 0.75])
    # Collimated before the pupil; the drawn envelope is sampled, so a plane's
    # corner is matched to the sampling.
    assert _half_width_at(res, 0.05) == pytest.approx(0.20)
    for x, half in zip(_CORONAGRAPH_AT, (0.20, 0.018, 0.20, 0.018), strict=True):
        assert _half_width_at(res, x) == pytest.approx(half, abs=0.003)
    assert _lenses(res) == pytest.approx([0.135, 0.395, 0.665])
    assert res.ax.get_xlim() == (0.0, 1.0)
    assert res.ax.get_ylim() == (0.0, 1.0)
    assert not res.ax.axison
    plt.close(res.fig)


def _gid(res, gid):
    return [a for a in res.ax.get_children() if a.get_gid() == gid]


_FOUR = [("Pupil", "pupil"), ("Focal", "fpm"), ("Lyot", "lyot"), ("Image", "detector")]


def test_highlight_sequence_lights_exactly_the_named_planes():
    res = rail(_FOUR, highlight=["pupil", "IMAGE"], accent="#123456")
    widths = [line.get_linewidth() for line in res.artists["lines"]]
    weights = [text.get_fontweight() for text in res.artists["text"]]
    assert widths == [2.0, 1.0, 1.0, 2.0]
    assert weights == ["bold", "normal", "normal", "bold"]
    lit = to_rgba("#123456")
    assert to_rgba(_gid(res, "rail/Pupil/glyph")[0].get_facecolor()) == lit
    assert to_rgba(_gid(res, "rail/Lyot/glyph")[0].get_facecolor()) != lit
    plt.close(res.fig)


def test_empty_highlight_sequence_lights_nothing():
    plain = rail(_FOUR)
    empty = rail(_FOUR, highlight=[])
    assert [ln.get_color() for ln in plain.artists["lines"]] == [
        ln.get_color() for ln in empty.artists["lines"]
    ]
    plt.close(plain.fig)
    plt.close(empty.fig)


@pytest.mark.parametrize("highlight", [["Pupil", "nope"], ["Pupil", 3], 3.5])
def test_highlight_sequence_with_an_unknown_label_raises(highlight):
    with pytest.raises(ValueError, match="highlight"):
        rail(_FOUR, highlight=highlight)


def test_colors_reach_the_glyph_artists_by_gid():
    res = rail(_FOUR, colors={"lyot": "#00ff00", "Image": "#ff00ff"}, highlight="Lyot")
    bars = _gid(res, "rail/Lyot/glyph")
    assert len(bars) == 2
    # A colored glyph keeps its role color when its plane is lit.
    assert all(to_rgba(b.get_facecolor()) == to_rgba("#00ff00") for b in bars)
    (box,) = _gid(res, "rail/Image/glyph")
    assert to_rgba(box.get_edgecolor()) == to_rgba("#ff00ff")
    assert _hatch_color(box) == to_rgba("#ff00ff")
    # An uncolored glyph keeps the neutral tone of a plain rail.
    plain = rail(_FOUR)
    neutral = _gid(plain, "rail/Pupil/glyph")[0].get_facecolor()
    assert _gid(res, "rail/Pupil/glyph")[0].get_facecolor() == neutral
    plt.close(res.fig)
    plt.close(plain.fig)


def test_unknown_color_plane_raises():
    with pytest.raises(ValueError, match="unknown color plane"):
        rail(_FOUR, colors={"Nowhere": "red"})


def test_beam_color_reaches_the_envelope_and_its_edges():
    res = rail(_FOUR, beam_color="#abcdef")
    fill = to_rgba(np.ravel(res.artists["fill"].get_facecolor())[:3])
    assert fill == to_rgba("#abcdef")
    edges = _gid(res, "rail/beam/edge")
    assert len(edges) == 2
    assert all(to_rgba(e.get_color()) == to_rgba("#abcdef") for e in edges)
    plt.close(res.fig)


def test_every_rail_artist_carries_a_rail_gid():
    planes = [*_FOUR[:3], ("End", "focal")]
    res = rail(planes, gaps=["relay", "fourier", "fourier"], cap=True)
    frame = [res.ax.patch, *res.ax.spines.values()]
    drawn = [
        a
        for a in res.ax.get_children()
        if (isinstance(a, (Line2D, Patch, Collection)) and a not in frame)
        or (isinstance(a, Text) and a.get_text())
    ]
    gids = {a.get_gid() for a in drawn}
    assert all(gid is not None and gid.startswith("rail/") for gid in gids)
    assert {
        "rail/beam/fill",
        "rail/beam/edge",
        "rail/beam/axis",
        "rail/Pupil/lens",
        "rail/Focal/lens",
        "rail/Lyot/lens",
        "rail/End/cap",
        "rail/End/marker",
        "rail/End/label",
        "rail/End/glyph",
    } <= gids
    # The relay after the pupil draws two lenses, both owned by the pupil.
    assert len(_gid(res, "rail/Pupil/lens")) == 2
    plt.close(res.fig)


def test_lenses_are_returned_under_ellipse_in_optical_order():
    res = rail(_FOUR)
    ellipses = res.artists["ellipse"]
    assert [e.center[0] for e in ellipses] == _lenses(res)
    assert set(res.artists) <= set(eyepiece.ARTIST_KEYS)
    plt.close(res.fig)


def test_a_rail_without_lenses_omits_the_ellipse_key():
    res = rail([("Only", "pupil")])
    assert "ellipse" not in res.artists
    plt.close(res.fig)


def _styles(res):
    lines = [
        (ln.get_color(), ln.get_linewidth(), ln.get_linestyle())
        for ln in res.artists["lines"]
    ]
    texts = [(t.get_color(), t.get_fontweight()) for t in res.artists["text"]]
    glyphs = [
        to_rgba(a.get_color() if isinstance(a, Line2D) else a.get_facecolor())
        for label, _ in _FOUR
        for a in _gid(res, f"rail/{label}/glyph")
        if not (isinstance(a, Patch) and a.get_hatch())
    ]
    return lines, texts, glyphs


def test_update_relights_in_place_like_a_fresh_draw():
    res = rail(_FOUR, highlight="Pupil", colors={"Lyot": "#00ff00"})
    before = len(res.ax.get_children())
    res.update(highlight=["Focal", "Lyot"])
    assert len(res.ax.get_children()) == before
    fresh = rail(_FOUR, highlight=["Focal", "Lyot"], colors={"Lyot": "#00ff00"})
    assert _styles(res) == _styles(fresh)
    res.update()
    plain = rail(_FOUR, colors={"Lyot": "#00ff00"})
    assert _styles(res) == _styles(plain)
    plt.close(res.fig)
    plt.close(fresh.fig)
    plt.close(plain.fig)


def test_update_recolors_a_detector_hatch_with_its_edge():
    res = rail(_FOUR, accent="#123456")
    res.update(highlight="Image")
    (box,) = _gid(res, "rail/Image/glyph")
    assert to_rgba(box.get_edgecolor()) == to_rgba("#123456")
    assert _hatch_color(box) == to_rgba("#123456")
    plt.close(res.fig)


def test_update_rejects_an_unknown_label():
    res = rail(_FOUR)
    with pytest.raises(ValueError, match="highlight"):
        res.update(highlight="nope")
    plt.close(res.fig)


_DATA_AT = (1.2, 4.0, 6.8, 9.6)


def _data_rail(**kwargs):
    _, ax = plt.subplots()
    ax.set(xlim=(-0.3, 10.4), ylim=(-1.9, 2.0))
    ax.set_aspect("equal")
    options = {
        "coords": "data",
        "positions": _DATA_AT,
        "axis_y": 0.25,
        "beam_half": 1.0,
        "fourier_lens": "middle",
    }
    return rail(_FOUR, ax=ax, **{**options, **kwargs})


def test_data_coords_place_the_train_in_data_units():
    res = _data_rail()
    markers = [float(m.get_xdata()[0]) for m in res.artists["lines"]]
    assert markers == pytest.approx(_DATA_AT)
    lens_x = [e.center[0] for e in res.artists["ellipse"]]
    assert lens_x == pytest.approx([2.6, 5.4, 8.2])
    assert all(e.center[1] == pytest.approx(0.25) for e in res.artists["ellipse"])
    upper = _gid(res, "rail/beam/edge")[0]
    for x in (1.2, 2.0, 6.0, 6.8):  # collimated around both pupils
        top = float(np.interp(x, upper.get_xdata(), upper.get_ydata()))
        assert top - 0.25 == pytest.approx(1.0)
    plt.close(res.fig)


def test_data_coords_leave_the_callers_limits_and_axis_alone():
    res = _data_rail()
    assert res.ax.get_xlim() == pytest.approx((-0.3, 10.4))
    assert res.ax.get_ylim() == pytest.approx((-1.9, 2.0))
    assert res.ax.axison
    assert res.ax.get_aspect() == 1.0
    plt.close(res.fig)


def _bar_height(res):
    bar = _gid(res, "rail/Pupil/glyph")[0]
    return bar.get_height(), bar.get_width()


def test_data_coord_glyphs_scale_with_beam_half():
    small = _data_rail(beam_half=0.5)
    big = _data_rail(beam_half=1.5)
    (h_small, w_small), (h_big, w_big) = _bar_height(small), _bar_height(big)
    assert h_big / h_small == pytest.approx(3.0)
    assert w_big / w_small == pytest.approx(3.0)
    plt.close(small.fig)
    plt.close(big.fig)


def test_data_coord_span_sets_the_beam_extent():
    res = _data_rail(span=(0.0, 9.6))
    axis = _gid(res, "rail/beam/axis")[0]
    assert tuple(axis.get_xdata()) == pytest.approx((0.0, 9.6))
    plt.close(res.fig)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coords": "figure"}, "unknown coords"),
        ({"axis_y": 0.5}, "need coords"),
        ({"beam_half": 0.1}, "need coords"),
        ({"coords": "data", "positions": _DATA_AT, "beam_half": 0.0}, "positive"),
        ({"coords": "data"}, "positions are required"),
        ({"positions": (0.1, 0.4, 0.6, 0.9), "span": (0.2, 1.0)}, "bracket"),
    ],
)
def test_bad_coordinate_options_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rail(_FOUR, **kwargs)
