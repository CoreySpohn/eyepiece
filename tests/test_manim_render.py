"""Prepared views rendered as native Manim objects: parts, geometry, and updates."""

import numpy as np
import pytest

from eyepiece.prepared import (
    AxisSpec,
    ImageView,
    Label,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    Scale,
    TrackView,
    map_rgba,
    replace_elements,
    weight_opacity,
)
from eyepiece.style import SourceCast, snapshot_profile

manim = pytest.importorskip("manim")


@pytest.fixture(autouse=True)
def media_in_tmp(tmp_path):
    """Keep Manim's text-layout cache out of the working directory."""
    with manim.tempconfig({"media_dir": str(tmp_path / "media")}):
        yield


def _xy(point):
    return np.asarray(point, dtype=float)[:2]


def _frame_basis(result, view_id):
    """(origin, x_vec, y_vec) of a panel's live frame, from its corners."""
    frame = result.parts[f"{view_id}/frame"]
    return (
        _xy(frame.get_corner(manim.DL)),
        _xy(frame.get_corner(manim.DR) - frame.get_corner(manim.DL)),
        _xy(frame.get_corner(manim.UL) - frame.get_corner(manim.DL)),
    )


def _track(view_id="track", prefix=""):
    """A TrackView with every mark kind, RA increasing to the left."""
    axes = AxisSpec("RA", "Dec", (-2.0, 2.0), (-1.0, 1.0), x_reverse=True)
    angles = np.linspace(0.0, np.pi, 5)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    return TrackView(
        view_id,
        axes,
        marks=(
            Path(f"{prefix}path/0", circle, source_id="b", weight=1.0),
            Path(f"{prefix}path/1", circle * 0.5, source_id="c", weight=0.25),
            Points(
                f"{prefix}obs",
                np.array([[0.5, 0.25], [-0.5, -0.25]]),
                source_id="b",
                xerr=np.array([0.1, 0.2]),
                yerr=np.array([0.05, 0.15]),
            ),
            Region(f"{prefix}iwa", np.array([0.0, 0.0]), 0.3),
            Region(f"{prefix}owa", np.array([0.0, 0.0]), 1.8, inner_radius=1.5),
            ReferenceLine(f"{prefix}zero", "y", 0.0),
            Label(f"{prefix}clock", "0 d", (0.02, 0.95)),
            Label(f"{prefix}tag", "b", (0.5, 0.25), space="data"),
        ),
    )


def _image(x_reverse=False, marks=()):
    """An asymmetric (2, 3) image whose every pixel maps to a distinct color."""
    data = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    return ImageView(
        "image",
        data,
        AxisSpec("x", "y", (0.0, 3.0), (0.0, 2.0), x_reverse=x_reverse),
        Scale("linear", 0.0, 5.0),
        "signal",
        marks=marks,
    )


# --- The brief's contract ------------------------------------------------------


def test_update_preserves_hidden_part_after_transform(track_sequence):
    import eyepiece.manim as em

    result = em.render(track_sequence.frame(0))
    region = result.parts["iwa"]
    region.set_opacity(0)
    result.mobject.scale(0.7).shift([1, 0, 0])
    result.update(track_sequence.frame(2))
    assert result.parts["iwa"] is region
    assert region.get_fill_opacity() == 0
    assert region.get_stroke_opacity() == 0


# --- Structure -------------------------------------------------------------------


def test_render_rejects_anything_but_a_prepared_view():
    import eyepiece.manim as em

    with pytest.raises(TypeError, match="prepared view"):
        em.render(np.zeros((2, 2)))


def test_parts_name_every_element_and_the_root_mixes_raster_and_vector():
    import eyepiece.manim as em

    view = PanelGroup("pair", (_image(), _track()))
    result = em.render(view)
    assert isinstance(result.mobject, manim.Group)
    assert result.parts["pair"] is result.mobject
    assert isinstance(result.parts["image"], manim.ImageMobject)
    for key in (
        "path/0",
        "path/1",
        "obs",
        "obs/xerr",
        "obs/yerr",
        "iwa",
        "owa",
        "zero",
        "clock",
        "tag",
        "image/colorbar",
        "image/frame",
        "image/axes",
        "track/frame",
        "track/axes",
    ):
        assert isinstance(result.parts[key], manim.Mobject), key
        assert result.parts[key] in result.mobject.get_family(), key
    assert isinstance(result.parts["obs"], manim.VGroup)
    assert len(result.parts["obs"]) == 2


def test_row_direction_places_panels_left_to_right():
    import eyepiece.manim as em

    result = em.render(PanelGroup("pair", (_track("a", "a/"), _track("b", "b/"))))
    assert result.parts["a"].get_center()[0] < result.parts["b"].get_center()[0]
    column = em.render(
        PanelGroup("pair", (_track("a", "a/"), _track("b", "b/")), direction="column")
    )
    assert column.parts["a"].get_center()[1] > column.parts["b"].get_center()[1]


def test_derived_part_names_cannot_collide_with_element_ids():
    import eyepiece.manim as em

    view = TrackView(
        "track",
        AxisSpec("x", "y", (0, 1), (0, 1)),
        (Label("track/frame", "x", (0.1, 0.1)),),
    )
    with pytest.raises(ValueError, match="track/frame"):
        em.render(view)


# --- Raster orientation and alignment --------------------------------------------


def test_image_rows_start_at_the_bottom_and_fill_the_axis_limits():
    import eyepiece.manim as em

    profile = snapshot_profile()
    view = _image()
    result = em.render(view, profile=profile)
    image = result.parts["image"]
    rgba = map_rgba(view.data, valid=None, scale=view.scale, profile=profile)
    np.testing.assert_array_equal(image.pixel_array, rgba[::-1])
    origin, x_vec, y_vec = _frame_basis(result, "image")
    np.testing.assert_allclose(_xy(image.get_corner(manim.DL)), origin, atol=1e-9)
    np.testing.assert_allclose(
        _xy(image.get_corner(manim.UR)), origin + x_vec + y_vec, atol=1e-9
    )


def test_ra_reversal_flips_the_display_once_not_the_data():
    import eyepiece.manim as em

    profile = snapshot_profile()
    corner = Points("corner", np.array([[0.5, 0.5], [2.5, 0.5]]))
    view = _image(x_reverse=True, marks=(corner,))
    data_before = view.data.copy()
    result = em.render(view, profile=profile)
    rgba = map_rgba(view.data, valid=None, scale=view.scale, profile=profile)
    np.testing.assert_array_equal(result.parts["image"].pixel_array, rgba[::-1, ::-1])
    np.testing.assert_array_equal(view.data, data_before)
    # x = 0.5 (the center of data column 0) now sits in the rightmost
    # displayed pixel column, x = 2.5 in the leftmost.
    low_x, high_x = (_xy(m.get_center()) for m in result.parts["corner"])
    assert low_x[0] > high_x[0]
    image = result.parts["image"]
    pixel = image.width / 3
    assert low_x[0] == pytest.approx(image.get_right()[0] - pixel / 2)
    assert high_x[0] == pytest.approx(image.get_left()[0] + pixel / 2)


def test_observations_errors_and_iwa_share_one_data_mapping():
    import eyepiece.manim as em

    result = em.render(_track())
    iwa = result.parts["iwa"]
    center = _xy(iwa.get_center())
    scale = iwa.width / 0.6  # scene units per data unit (diameter 0.6)
    assert iwa.height / 0.6 == pytest.approx(scale)
    first, second = (_xy(m.get_center()) for m in result.parts["obs"])
    # RA increases to the left: data (+0.5, +0.25) is left of the origin.
    np.testing.assert_allclose(first, center + scale * np.array([-0.5, 0.25]))
    np.testing.assert_allclose(second, center + scale * np.array([0.5, -0.25]))
    xerr = result.parts["obs/xerr"][0]
    np.testing.assert_allclose(
        sorted(np.asarray(xerr.get_anchors())[:, 0]),
        sorted([first[0] - 0.1 * scale, first[0] + 0.1 * scale]),
    )
    yerr = result.parts["obs/yerr"][1]
    np.testing.assert_allclose(
        sorted(np.asarray(yerr.get_anchors())[:, 1]),
        sorted([second[1] - 0.15 * scale, second[1] + 0.15 * scale]),
    )


def test_moved_and_scaled_panels_keep_the_data_mapping(track_sequence):
    import eyepiece.manim as em

    result = em.render(track_sequence.frame(0))
    result.mobject.scale(0.5).shift([2.0, -1.0, 0.0])
    iwa = result.parts["iwa"]
    scale = iwa.width / 1.0
    result.update(track_sequence.frame(2))  # head at data (0, 1)
    head = _xy(result.parts["head"][0].get_center())
    np.testing.assert_allclose(
        head, _xy(iwa.get_center()) + np.array([0.0, scale]), atol=1e-9
    )
    result.update(track_sequence.frame(4))  # head at data (-1, 0): right side
    head = _xy(result.parts["head"][0].get_center())
    np.testing.assert_allclose(
        head, _xy(iwa.get_center()) + np.array([scale, 0.0]), atol=1e-9
    )
    # The marker shrank with the panel instead of snapping back to full size.
    unscaled = em.render(track_sequence.frame(0)).parts["head"][0].width
    assert result.parts["head"][0].width == pytest.approx(unscaled * 0.5)


# --- Appearance -----------------------------------------------------------------------


def test_invalid_pixels_use_the_bad_color():
    import eyepiece.manim as em

    profile = snapshot_profile()
    valid = np.array([[True, False, True], [True, True, True]])
    view = ImageView(
        "image",
        np.array([[0.0, 1.0, np.nan], [3.0, 4.0, 5.0]]),
        AxisSpec("x", "y", (0.0, 3.0), (0.0, 2.0)),
        Scale("linear", 0.0, 5.0),
        "signal",
        valid=valid,
    )
    pixels = em.render(view, profile=profile).parts["image"].pixel_array
    bad = np.asarray(profile.bad_rgba)
    # Data row 0 is the bottom pixel row.
    np.testing.assert_array_equal(pixels[-1, 1], bad)
    np.testing.assert_array_equal(pixels[-1, 2], bad)
    np.testing.assert_array_equal(pixels[-1, 0], profile.colormaps["intensity"][0])


def test_source_colors_follow_the_cast_and_path_opacity_the_weights():
    import eyepiece.manim as em

    profile = snapshot_profile()
    result = em.render(_track(), profile=profile)
    colors = profile.colors
    assert result.parts["path/0"].get_stroke_color().to_hex().lower() == (
        manim.ManimColor(colors[0]).to_hex().lower()
    )
    assert result.parts["path/1"].get_stroke_color().to_hex().lower() == (
        manim.ManimColor(colors[1]).to_hex().lower()
    )
    expected = weight_opacity([1.0, 0.25])
    assert result.parts["path/0"].get_stroke_opacity() == pytest.approx(expected[0])
    assert result.parts["path/1"].get_stroke_opacity() == pytest.approx(expected[1])
    swapped = em.render(_track(), cast=SourceCast(("c", "b")), profile=profile)
    assert swapped.parts["path/0"].get_stroke_color().to_hex().lower() == (
        manim.ManimColor(colors[1]).to_hex().lower()
    )
    with pytest.raises(ValueError, match="path/0"):
        em.render(_track(), cast=SourceCast(("c",)), profile=profile)


def test_a_lone_path_is_drawn_at_the_top_weight_opacity():
    import eyepiece.manim as em

    view = TrackView(
        "track",
        AxisSpec("x", "y", (0, 1), (0, 1)),
        (Path("p", np.array([[0.1, 0.1], [0.9, 0.9]])),),
    )
    assert em.render(view).parts["p"].get_stroke_opacity() == pytest.approx(0.80)


def test_unsupported_region_role_raises_before_drawing():
    import eyepiece.manim as em

    view = TrackView(
        "track",
        AxisSpec("x", "y", (0, 1), (0, 1)),
        (Region("ring", np.array([0.5, 0.5]), 0.2, role="planet"),),
    )
    with pytest.raises(ValueError, match="ring"):
        em.render(view)


def _label_points(text, profile, space="panel"):
    """Outline points of `text` laid out alone, shifted to a zero corner."""
    import eyepiece.manim as em

    view = TrackView(
        "t", AxisSpec("x", "y", (0, 1), (0, 1)), (Label("l", text, (0.5, 0.5), space),)
    )
    points = em.render(view, profile=profile).parts["l"].submobjects[0].points
    return points - points.min(axis=0)


def test_labels_use_the_profile_and_anchor_like_matplotlib(monkeypatch):
    import eyepiece.manim as em

    fonts = []
    original = manim.Text

    def text_spy(*args, **kwargs):
        fonts.append(kwargs.get("font"))
        return original(*args, **kwargs)

    profile = snapshot_profile()
    monkeypatch.setattr(manim, "Text", text_spy)
    result = em.render(_track(), profile=profile)
    monkeypatch.setattr(manim, "Text", original)
    assert fonts and set(fonts) == {profile.font_family}
    clock = result.parts["clock"]
    glyphs = clock.submobjects[0]
    assert isinstance(glyphs, manim.VMobject) and not glyphs.submobjects
    assert glyphs.get_fill_color().to_hex().lower() == (
        manim.ManimColor(profile.text_color).to_hex().lower()
    )
    origin, x_vec, y_vec = _frame_basis(result, "track")
    # Panel space: the top-left corner sits at the axes fraction (0.02, 0.95).
    np.testing.assert_allclose(
        _xy(clock.get_corner(manim.UL)), origin + 0.02 * x_vec + 0.95 * y_vec
    )
    # Data space: left edge at x = 0.5, baseline (below the glyph box
    # bottom only by descenders; "b" has none) at y = 0.25.
    tag = result.parts["tag"]
    iwa = result.parts["iwa"]
    scale = iwa.width / 0.6
    anchor = _xy(iwa.get_center()) + scale * np.array([-0.5, 0.25])
    assert tag.get_left()[0] == pytest.approx(anchor[0], abs=0.02)
    assert tag.get_bottom()[1] == pytest.approx(anchor[1], abs=0.02)


# --- Updates -------------------------------------------------------------------------


def test_result_keeps_the_profile_it_rendered_with(monkeypatch):
    import eyepiece.manim as em
    from eyepiece.manim import _render

    result = em.render(_track())
    used = result.profile

    def fail():
        raise AssertionError("update must not re-snapshot the profile")

    monkeypatch.setattr(_render, "snapshot_profile", fail)
    moved = replace_elements(_track(), {"clock": Label("clock", "5 d", (0.02, 0.95))})
    result.update(moved)
    assert result.profile is used


def test_update_moves_marks_in_place_and_rewrites_label_outlines():
    import eyepiece.manim as em

    template = _track()
    result = em.render(template)
    handles = dict(result.parts)
    clock = result.parts["clock"]
    old_glyphs = clock.submobjects[0]
    clock.set_opacity(0.4)
    moved = replace_elements(
        template,
        {
            "obs": Points(
                "obs",
                np.array([[0.0, 0.0], [1.0, 0.0]]),
                source_id="b",
                xerr=np.array([0.1, 0.2]),
                yerr=np.array([0.05, 0.15]),
            ),
            "clock": Label("clock", "3 d", (0.02, 0.95)),
            "path/0": Path(
                "path/0", template.marks[0].xy, source_id="b", visible=(1, 3)
            ),
        },
    )
    result.update(moved)
    for key, part in handles.items():
        assert result.parts[key] is part, key
    iwa_center = _xy(result.parts["iwa"].get_center())
    np.testing.assert_allclose(
        _xy(result.parts["obs"][0].get_center()), iwa_center, atol=1e-9
    )
    # The same glyph mobject is drawn with the new text's outline: a
    # scene that flattened the family at play time redraws it.
    glyphs = clock.submobjects[0]
    assert glyphs is old_glyphs and len(clock.submobjects) == 1
    new_outline = glyphs.points - glyphs.points.min(axis=0)
    np.testing.assert_allclose(new_outline, _label_points("3 d", result.profile))
    assert glyphs.get_fill_opacity() == pytest.approx(0.4)
    # Two visible vertices make one straight segment (one cubic curve).
    assert len(result.parts["path/0"].points) == 4


def test_update_rejects_topology_changes_naming_the_element():
    import eyepiece.manim as em

    template = _track()
    result = em.render(template)
    fewer = replace_elements(
        template, {"obs": Points("obs", np.zeros((3, 2)), source_id="b")}
    )
    with pytest.raises(ValueError, match="obs"):
        result.update(fewer)
    moved_axes = TrackView(
        "track",
        AxisSpec("RA", "Dec", (-3.0, 3.0), (-1.0, 1.0), x_reverse=True),
        template.marks,
    )
    with pytest.raises(ValueError, match="track"):
        result.update(moved_axes)


def test_update_rejects_image_shape_and_scale_changes(image_sequence):
    import eyepiece.manim as em

    template = image_sequence().frame(0)
    result = em.render(template)
    reshaped = ImageView("image", np.zeros((3, 3)), template.axes, template.scale, "s")
    with pytest.raises(ValueError, match="image"):
        result.update(reshaped)
    rescaled = ImageView(
        "image", template.data, template.axes, Scale("linear", 0, 40), "s"
    )
    with pytest.raises(ValueError, match="image"):
        result.update(rescaled)


def test_invalid_second_view_leaves_the_whole_group_unchanged():
    import eyepiece.manim as em

    group = PanelGroup("pair", (_track("a", "a/"), _track("b", "b/")))
    result = em.render(group)
    before = {
        mob: mob.points.copy()
        for key in ("a/obs", "b/obs", "a/path/0")
        for mob in result.parts[key].get_family()
    }
    glyphs_before = result.parts["a/clock"].submobjects[0].points.copy()
    changed = replace_elements(
        group,
        {
            "a/obs": Points(
                "a/obs",
                np.array([[9.0, 9.0], [8.0, 8.0]]),
                "b",
                xerr=np.array([0.1, 0.2]),
                yerr=np.array([0.05, 0.15]),
            ),
            "a/path/0": Path(
                "a/path/0", group.views[0].marks[0].xy, source_id="b", visible=(0, 2)
            ),
            "a/clock": Label("a/clock", "99 d", (0.02, 0.95)),
            "b/obs": Points("b/obs", np.zeros((5, 2)), "b"),
        },
    )
    with pytest.raises(ValueError, match="b/obs"):
        result.update(changed)
    for mob, points in before.items():
        np.testing.assert_array_equal(mob.points, points)
    np.testing.assert_array_equal(
        result.parts["a/clock"].submobjects[0].points, glyphs_before
    )


def test_update_reuses_the_mapped_image_while_the_sample_is_held(
    image_sequence, monkeypatch
):
    import eyepiece.manim as em
    from eyepiece.manim import _render

    calls = []
    original = _render.map_rgba

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_render, "map_rgba", spy)
    sequence = image_sequence()
    result = em.render(sequence.frame(0))
    image = result.parts["image"]
    buffer = image.pixel_array
    assert len(calls) == 1
    result.update(sequence.frame(0))
    result.update(sequence.at(0.5).view)
    assert len(calls) == 1
    result.update(sequence.frame(2))
    assert len(calls) == 2
    assert result.parts["image"] is image
    assert image.pixel_array is buffer  # the buffer is rewritten, not replaced


def test_consumer_image_opacity_survives_new_samples(image_sequence):
    import eyepiece.manim as em

    sequence = image_sequence()
    result = em.render(sequence.frame(0))
    image = result.parts["image"]
    image.set_opacity(0.5)
    result.update(sequence.frame(2))
    assert np.all(image.pixel_array[..., 3] == 127)


def test_nonfinite_coordinates_are_hidden_gaps_then_restored():
    import eyepiece.manim as em

    xy = np.array([[0.1, 0.1], [np.nan, 0.5], [0.9, 0.9], [np.inf, 0.2]])
    path_xy = np.array([[0.1, 0.1], [0.3, 0.3], [np.nan, 0.5], [0.7, 0.7], [0.9, 0.9]])
    view = TrackView(
        "track",
        AxisSpec("x", "y", (0, 1), (0, 1)),
        (
            Path("trail", path_xy),
            Points("obs", xy, yerr=np.array([0.05, np.nan, 0.05, np.inf])),
        ),
    )
    result = em.render(view)
    markers = result.parts["obs"]
    errors = result.parts["obs/yerr"]
    assert len(markers) == 4 and len(errors) == 4
    assert [m.has_points() for m in markers] == [True, False, True, False]
    assert [e.has_points() for e in errors] == [True, False, True, False]
    trail = result.parts["trail"]
    assert np.all(np.isfinite(trail.points))
    # Two finite runs of two vertices: two separate straight subpaths.
    assert len(trail.get_subpaths()) == 2
    finite = replace_elements(
        view,
        {
            "obs": Points(
                "obs",
                np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.2, 0.2]]),
                yerr=np.full(4, 0.05),
            )
        },
    )
    handles = list(markers)
    result.update(finite)
    assert list(result.parts["obs"]) == handles
    assert all(m.has_points() for m in result.parts["obs"])
    assert all(e.has_points() for e in errors)


def test_ticks_are_round_values_that_follow_the_ra_reversal():
    import eyepiece.manim as em

    result = em.render(_track())
    ticks, x_labels, y_labels = result.parts["track/axes"][:3]
    assert [t.original_text for t in x_labels] == ["-2", "-1", "0", "1", "2"]
    assert [t.original_text for t in y_labels] == ["-1.0", "-0.5", "0.0", "0.5", "1.0"]
    positions = [t.get_center()[0] for t in x_labels]
    assert positions == sorted(positions, reverse=True)  # RA increases left
    assert len(ticks) == len(x_labels) + len(y_labels)


def test_an_annulus_is_drawn_with_a_hole():
    import eyepiece.manim as em

    view = TrackView(
        "track",
        AxisSpec("x", "y", (-2.0, 2.0), (-2.0, 2.0)),
        (Region("owa", np.array([0.0, 0.0]), 1.8, inner_radius=1.5),),
    )
    result = em.render(view)
    camera = manim.Camera(pixel_width=320, pixel_height=180)
    camera.capture_mobject(result.mobject)
    pixels = camera.pixel_array
    ring = result.parts["owa"]
    per_unit = ring.width / 3.6

    def pixel(point):
        col = int((point[0] + camera.frame_width / 2) / camera.frame_width * 320)
        row = int((camera.frame_height / 2 - point[1]) / camera.frame_height * 180)
        return pixels[row, col, :3].astype(int)

    center = ring.get_center()
    background = pixel(center)
    filled = pixel(center + np.array([1.65 * per_unit, 0.0, 0.0]))
    np.testing.assert_array_equal(background, pixel(np.array([6.8, 3.8, 0.0])))
    assert np.any(filled != background)


def test_default_profile_keeps_the_talk_sizes_whatever_the_rc():
    import matplotlib

    import eyepiece.manim as em

    with matplotlib.rc_context({"font.size": 9.0, "lines.linewidth": 0.8}):
        result = em.render(_track())
    assert result.profile.text_size_pt == 24.0
    assert result.profile.stroke_width_pt == 1.5


def test_regions_on_images_are_outlines_but_track_regions_fill():
    import eyepiece.manim as em

    ring = Region("iwa", np.array([1.5, 1.0]), 0.5)
    on_image = em.render(_image(marks=(ring,))).parts["iwa"]
    assert on_image.get_fill_opacity() == 0.0
    assert on_image.get_stroke_opacity() == 1.0
    on_track = em.render(_track()).parts["iwa"]
    assert on_track.get_fill_opacity() == pytest.approx(0.2)
