"""Prepared views rendered through Matplotlib: handles, placement, and updates."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

from eyepiece.prepared import (
    AxisSpec,
    Clock,
    ImageView,
    Label,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    Scale,
    Sequence,
    TrackView,
    find_element,
    replace_elements,
    weight_opacity,
)
from eyepiece.style import SourceCast, snapshot_profile


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


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


def _pixel_rgba(fig, ax, x, y):
    """The rendered RGBA at data coordinate (x, y)."""
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    px, py = ax.transData.transform((x, y))
    row = buffer.shape[0] - round(py)
    return buffer[row, round(px)]


# --- Axes ownership -----------------------------------------------------------


def test_render_without_axes_owns_one_new_figure():
    import eyepiece.mpl as mpl

    before = set(plt.get_fignums())
    result = mpl.render(_track())
    assert len(set(plt.get_fignums()) - before) == 1
    assert result.ax is result.axes["track"]
    assert result.ax.figure is result.fig


def test_render_on_caller_axes_keeps_figure_and_slot():
    import eyepiece.mpl as mpl

    fig, (left, right) = plt.subplots(1, 2)
    positions = [a.get_position(original=True).bounds for a in (left, right)]
    before = set(plt.get_fignums())
    result = mpl.render(_track(), ax=right)
    assert set(plt.get_fignums()) == before
    assert result.fig is fig
    assert result.ax is right
    assert fig.get_layout_engine() is None
    fig.canvas.draw()
    after = [a.get_position(original=True).bounds for a in (left, right)]
    assert after == positions


def test_group_renders_into_caller_axes_in_leaf_order():
    import eyepiece.mpl as mpl

    group = PanelGroup("pair", (_track("a", "a/"), _track("b", "b/")))
    _, (left, right) = plt.subplots(1, 2)
    result = mpl.render(group, axes=[left, right])
    assert result.axes["a"] is left
    assert result.axes["b"] is right
    with pytest.raises(ValueError, match="pair"):
        result.ax  # noqa: B018


def test_group_owned_figure_follows_direction():
    import eyepiece.mpl as mpl

    group = PanelGroup(
        "pair", (_track("a", "a/"), _track("b", "b/")), direction="column"
    )
    result = mpl.render(group)
    top = result.axes["a"].get_position(original=True)
    bottom = result.axes["b"].get_position(original=True)
    assert top.y0 > bottom.y0


def test_axes_count_mismatch_raises_before_drawing():
    import eyepiece.mpl as mpl

    group = PanelGroup("pair", (_track("a", "a/"), _track("b", "b/")))
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="pair"):
        mpl.render(group, ax=ax)
    with pytest.raises(ValueError, match="pair"):
        mpl.render(group, axes=[ax])
    assert len(ax.lines) == 0 and len(ax.collections) == 0


# --- Image placement and the shared scale ------------------------------------


def _hot_pixel_view(x_reverse=False):
    data = np.zeros((2, 3))
    data[0, 2] = 20.0
    return ImageView(
        "image",
        data,
        AxisSpec("x", "y", (0, 3), (0, 2), x_reverse=x_reverse),
        Scale("linear", 0, 20),
        "signal",
    )


def test_image_extent_is_the_axis_limits_and_rows_start_at_bottom():
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    fig, ax = plt.subplots()
    result = mpl.render(_hot_pixel_view(), ax=ax, profile=profile)
    assert tuple(result.parts["image"].get_extent()) == (0, 3, 0, 2)
    hot = profile.colormaps["intensity"][255]
    cold = profile.colormaps["intensity"][0]
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 2.5, 0.5), hot)
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 0.5, 0.5), cold)
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 2.5, 1.5), cold)


def test_ra_reversal_flips_the_display_once_not_the_data():
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    fig, ax = plt.subplots()
    result = mpl.render(_hot_pixel_view(x_reverse=True), ax=ax, profile=profile)
    assert tuple(result.parts["image"].get_extent()) == (0, 3, 0, 2)
    assert ax.get_xlim() == (3.0, 0.0)
    fig.canvas.draw()
    left_px, _ = ax.transData.transform((2.5, 0.5))
    right_px, _ = ax.transData.transform((0.5, 0.5))
    assert left_px < right_px
    hot = profile.colormaps["intensity"][255]
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 2.5, 0.5), hot)


def test_invalid_pixels_use_the_bad_color_not_a_clipped_extreme():
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    data = np.array([[np.nan, 50.0, -5.0], [1.0, 2.0, 3.0]])
    view = ImageView(
        "image", data, AxisSpec("x", "y", (0, 3), (0, 2)), Scale("linear", 0, 20), "s"
    )
    fig, ax = plt.subplots()
    mpl.render(view, ax=ax, profile=profile)
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 0.5, 0.5), profile.bad_rgba)
    lut = profile.colormaps["intensity"]
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 1.5, 0.5), lut[255])
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 2.5, 0.5), lut[0])


@pytest.mark.parametrize(
    ("scale", "norm_type"),
    [
        (Scale("linear", 0.0, 20.0), "Normalize"),
        (Scale("symmetric", -5.0, 5.0, cmap_role="residual"), "Normalize"),
        (Scale("log", 1e-3, 1.0, floor=1e-4), "LogNorm"),
    ],
)
def test_colorbar_shares_the_scale_and_lut(scale, norm_type):
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    view = ImageView(
        "image",
        np.full((2, 3), 0.5),
        AxisSpec("x", "y", (0, 3), (0, 2)),
        scale,
        "contrast",
    )
    result = mpl.render(view, profile=profile)
    cbar = result.parts["image/colorbar"]
    assert type(cbar.mappable.norm).__name__ == norm_type
    assert (cbar.mappable.norm.vmin, cbar.mappable.norm.vmax) == (
        scale.vmin,
        scale.vmax,
    )
    assert isinstance(cbar.mappable.norm, LogNorm) == (scale.kind == "log")
    np.testing.assert_allclose(
        cbar.cmap.colors, profile.colormaps[scale.cmap_role] / 255.0
    )
    assert cbar.ax.get_ylabel() == "contrast"


# --- Marks: one coordinate mapping, named handles ----------------------------


def test_marks_are_exposed_by_element_id_with_data_coordinates():
    import eyepiece.mpl as mpl

    result = mpl.render(_track())
    ax = result.ax
    assert ax.get_xlim() == (2.0, -2.0)
    for key in ("path/0", "path/1", "obs", "iwa", "owa", "zero", "clock", "tag"):
        assert result.parts[key].axes is ax
    np.testing.assert_allclose(
        result.parts["obs"].get_offsets(), [[0.5, 0.25], [-0.5, -0.25]]
    )
    assert result.parts["clock"].get_transform() is ax.transAxes
    assert result.parts["clock"].get_position() == (0.02, 0.95)
    assert result.parts["tag"].get_transform() is ax.transData
    assert result.parts["iwa"].get_radius() == 0.3
    assert result.parts["owa"].get_radii() == (1.8, 1.8)
    assert result.parts["owa"].get_width() == pytest.approx(0.3)
    np.testing.assert_allclose(result.parts["zero"].get_ydata(), [0.0, 0.0])


def test_errorbar_endpoints_are_named_and_exact():
    import eyepiece.mpl as mpl

    result = mpl.render(_track())
    xerr = np.asarray(result.parts["obs/xerr"].get_segments())
    yerr = np.asarray(result.parts["obs/yerr"].get_segments())
    np.testing.assert_allclose(
        xerr, [[[0.4, 0.25], [0.6, 0.25]], [[-0.7, -0.25], [-0.3, -0.25]]]
    )
    np.testing.assert_allclose(
        yerr, [[[0.5, 0.2], [0.5, 0.3]], [[-0.5, -0.4], [-0.5, -0.1]]]
    )


def test_source_colors_follow_first_encounter_cast_by_default():
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    result = mpl.render(_track(), profile=profile)
    assert result.cast.names == ("b", "c")
    assert result.parts["path/0"].get_color() == profile.colors[0]
    assert result.parts["path/1"].get_color() == profile.colors[1]


def test_explicit_cast_controls_colors_and_rejects_unknown_sources():
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    result = mpl.render(_track(), cast=SourceCast(("c", "b")), profile=profile)
    assert result.parts["path/0"].get_color() == profile.colors[1]
    with pytest.raises(ValueError, match="path/1"):
        mpl.render(_track(), cast=SourceCast(("b",)), profile=profile)


def test_path_opacity_uses_the_shared_weight_mapping():
    import eyepiece.mpl as mpl

    result = mpl.render(_track())
    expected = weight_opacity([1.0, 0.25])
    assert result.parts["path/0"].get_alpha() == pytest.approx(expected[0])
    assert result.parts["path/1"].get_alpha() == pytest.approx(expected[1])


def test_unsupported_region_role_raises_before_drawing():
    import eyepiece.mpl as mpl

    view = TrackView(
        "track",
        AxisSpec("x", "y", (0, 1), (0, 1)),
        (Region("ring", np.array([0.5, 0.5]), 0.2, role="planet"),),
    )
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="ring"):
        mpl.render(view, ax=ax)
    assert len(ax.patches) == 0


def test_result_keeps_the_profile_it_rendered_with(monkeypatch):
    import eyepiece.mpl as mpl

    result = mpl.render(_track())
    used = result.profile

    def fail():
        raise AssertionError("update must not re-snapshot the profile")

    monkeypatch.setattr(mpl, "snapshot_profile", fail)
    moved = replace_elements(_track(), {"clock": Label("clock", "5 d", (0.02, 0.95))})
    result.update(moved)
    assert result.profile is used


# --- Updates -------------------------------------------------------------------


def test_update_reuses_image_and_original_scale(image_sequence):
    import eyepiece.mpl as mpl

    sequence = image_sequence()
    result = mpl.render(sequence.frame(0))
    image = result.parts["image"]
    result.update(sequence.frame(2))
    assert result.parts["image"] is image
    assert tuple(image.get_extent()) == (0, 3, 0, 2)


def test_bounds_hold_through_out_of_order_updates(image_sequence):
    import eyepiece.mpl as mpl

    profile = snapshot_profile()
    sequence = image_sequence()
    fig, ax = plt.subplots()
    result = mpl.render(sequence.frame(2), ax=ax, profile=profile)
    lut = profile.colormaps["intensity"]
    cbar = result.parts["image/colorbar"]
    for index, lut_index in ((0, 0), (1, 128), (2, 255), (0, 0)):
        result.update(sequence.frame(index))
        np.testing.assert_array_equal(_pixel_rgba(fig, ax, 1.5, 0.5), lut[lut_index])
        assert (cbar.mappable.norm.vmin, cbar.mappable.norm.vmax) == (0, 20)
        assert ax.get_xlim() == (0.0, 3.0)


def test_update_moves_marks_in_place_and_keeps_caller_visibility():
    import eyepiece.mpl as mpl

    template = _track()
    result = mpl.render(template)
    handles = dict(result.parts)
    result.parts["iwa"].set_visible(False)
    moved = replace_elements(
        template,
        {
            "obs": Points(
                "obs",
                np.array([[1.0, 0.5], [0.0, 0.0]]),
                source_id="b",
                xerr=np.array([0.1, 0.2]),
                yerr=np.array([0.05, 0.15]),
            ),
            "path/0": Path(
                "path/0", find_element(template, "path/0").xy, "b", visible=(1, 3)
            ),
            "iwa": Region("iwa", np.array([0.1, 0.0]), 0.4),
            "zero": ReferenceLine("zero", "y", 0.5),
            "clock": Label("clock", "3 d", (0.02, 0.95)),
        },
    )
    result.update(moved)
    for key, handle in handles.items():
        assert result.parts[key] is handle
    assert not result.parts["iwa"].get_visible()
    assert result.parts["iwa"].get_radius() == 0.4
    np.testing.assert_allclose(
        result.parts["obs"].get_offsets(), [[1.0, 0.5], [0.0, 0.0]]
    )
    np.testing.assert_allclose(
        result.parts["obs/xerr"].get_segments()[0], [[0.9, 0.5], [1.1, 0.5]]
    )
    path_xy = find_element(template, "path/0").xy
    np.testing.assert_allclose(result.parts["path/0"].get_xydata(), path_xy[1:3])
    np.testing.assert_allclose(result.parts["zero"].get_ydata(), [0.5, 0.5])
    assert result.parts["clock"].get_text() == "3 d"


def test_update_rejects_topology_changes_naming_the_element():
    import eyepiece.mpl as mpl

    template = _track()
    result = mpl.render(template)
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
    renamed = TrackView("track", template.axes, template.marks[:-1])
    with pytest.raises(ValueError, match="tag"):
        result.update(renamed)


def test_update_rejects_image_shape_and_scale_changes(image_sequence):
    import eyepiece.mpl as mpl

    template = image_sequence().frame(0)
    result = mpl.render(template)
    reshaped = ImageView("image", np.zeros((3, 3)), template.axes, template.scale, "s")
    with pytest.raises(ValueError, match="image"):
        result.update(reshaped)
    rescaled = ImageView(
        "image", template.data, template.axes, Scale("linear", 0, 40), "s"
    )
    with pytest.raises(ValueError, match="image"):
        result.update(rescaled)


def test_invalid_second_view_leaves_the_whole_group_unchanged():
    import eyepiece.mpl as mpl

    group = PanelGroup("pair", (_track("a", "a/"), _track("b", "b/")))
    result = mpl.render(group)
    before = {
        key: np.array(part.get_offsets())
        for key, part in result.parts.items()
        if key in ("a/obs", "b/obs")
    }
    clock_before = result.parts["a/clock"].get_text()
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
            "a/clock": Label("a/clock", "99 d", (0.02, 0.95)),
            "b/obs": Points("b/obs", np.zeros((5, 2)), "b"),
        },
    )
    with pytest.raises(ValueError, match="b/obs"):
        result.update(changed)
    for key, offsets in before.items():
        np.testing.assert_array_equal(result.parts[key].get_offsets(), offsets)
    assert result.parts["a/clock"].get_text() == clock_before


def test_clock_labels_are_semantic_parts():
    import eyepiece.mpl as mpl

    template = _track()
    sequence = Sequence(template, np.arange(5.0), "d", (Clock("clock"),))
    result = mpl.render(sequence.frame(0))
    clock = result.parts["clock"]
    result.update(sequence.frame(3))
    assert result.parts["clock"] is clock
    assert clock.get_text() == "3 d"


def test_update_reuses_mapped_image_while_index_is_held(image_sequence, monkeypatch):
    import eyepiece.mpl as mpl

    calls = []
    original = mpl.map_rgba

    def counting(data, **kwargs):
        calls.append(float(np.mean(data)))
        return original(data, **kwargs)

    monkeypatch.setattr(mpl, "map_rgba", counting)
    sequence = image_sequence()
    result = mpl.render(sequence.frame(0))
    for time in (0.5, 1.0, 5.0, 9.9, 10.0, 10.0):
        result.update(sequence.at(time).view)
    assert calls == [0.0, 10.0, 20.0]


# --- Image and colorbar agree; coordinate gaps ----------------------------------


@pytest.mark.parametrize(
    ("scale", "values"),
    [
        (Scale("linear", 0.0, 20.0), np.linspace(0.0, 20.0, 2001)),
        (
            Scale("symmetric", -5.0, 5.0, cmap_role="residual"),
            np.linspace(-5.0, 5.0, 2001),
        ),
        (Scale("log", 1e-3, 1.0, floor=1e-4), np.geomspace(1e-3, 1.0, 2001)),
    ],
)
def test_image_and_colorbar_pick_the_same_lut_entry(scale, values):
    import eyepiece.mpl as mpl

    view = ImageView(
        "image",
        values.reshape(1, -1),
        AxisSpec("x", "y", (0, values.size), (0, 1)),
        scale,
        "q",
    )
    result = mpl.render(view)
    image_rgba = np.asarray(result.parts["image"].get_array())[0]
    mappable = result.parts["image/colorbar"].mappable
    np.testing.assert_array_equal(image_rgba, mappable.to_rgba(values, bytes=True))


def _gap_track(head_xy, path_xy):
    axes = AxisSpec("x", "y", (-1.0, 1.0), (-1.0, 1.0))
    return TrackView(
        "track",
        axes,
        (
            Path("trail", path_xy, source_id="b"),
            Points("head", head_xy, source_id="b", xerr=np.array([0.2])),
        ),
    )


def test_nonfinite_coordinates_are_hidden_gaps_then_restored():
    import eyepiece.mpl as mpl

    path_xy = np.array([[-0.8, -0.8], [np.nan, np.nan], [0.8, -0.8]])
    gap = _gap_track(np.array([[np.nan, np.nan]]), path_xy)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=50)
    result = mpl.render(gap, ax=ax)
    head, trail, xerr = (result.parts[k] for k in ("head", "trail", "head/xerr"))
    background = _pixel_rgba(fig, ax, 0.5, 0.5)
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 0.0, 0.0), background)
    assert len(head.get_offsets()) == 1
    assert not np.isfinite(np.ma.filled(head.get_offsets(), np.nan)).any()
    assert np.isnan(trail.get_xydata()[1]).all()
    np.testing.assert_allclose(trail.get_xydata()[[0, 2]], path_xy[[0, 2]])

    filled = np.array([[-0.8, -0.8], [0.0, -0.8], [0.8, -0.8]])
    result.update(_gap_track(np.array([[0.0, 0.0]]), filled))
    assert result.parts["head"] is head
    assert result.parts["trail"] is trail
    assert result.parts["head/xerr"] is xerr
    np.testing.assert_allclose(np.ma.getdata(head.get_offsets()), [[0.0, 0.0]])
    assert not np.ma.getmaskarray(head.get_offsets()).any()
    np.testing.assert_allclose(trail.get_xydata(), filled)
    np.testing.assert_allclose(xerr.get_segments()[0], [[-0.2, 0.0], [0.2, 0.0]])
    assert not np.array_equal(_pixel_rgba(fig, ax, 0.0, 0.0), background)

    result.update(gap)
    assert not np.isfinite(np.ma.filled(head.get_offsets(), np.nan)).any()
    np.testing.assert_array_equal(_pixel_rgba(fig, ax, 0.0, 0.0), background)


def test_infinite_coordinates_are_gaps_not_far_away_points():
    import eyepiece.mpl as mpl

    path_xy = np.array([[-0.8, -0.8], [np.inf, 0.0], [0.8, -0.8]])
    result = mpl.render(_gap_track(np.array([[np.inf, 0.0]]), path_xy))
    assert np.isnan(result.parts["trail"].get_xydata()[1]).all()
    assert not np.isfinite(
        np.ma.filled(result.parts["head"].get_offsets(), np.nan)
    ).any()
    result.fig.canvas.draw()


# --- Defaults and region fills --------------------------------------------------


def test_default_profile_follows_the_matplotlib_rc_sizes():
    import matplotlib

    import eyepiece.mpl as mpl

    with matplotlib.rc_context({"font.size": 9.0, "lines.linewidth": 0.8}):
        result = mpl.render(_track())
    assert result.profile.text_size_pt == 9.0
    assert result.profile.stroke_width_pt == 0.8
    assert result.parts["clock"].get_fontsize() == 9.0
    assert result.parts["path/0"].get_linewidth() == 0.8


def test_regions_on_images_are_outlines_but_track_regions_fill():
    import eyepiece.mpl as mpl

    ring = Region("iwa", np.array([1.5, 1.0]), 0.5)
    image = ImageView(
        "image",
        np.zeros((2, 3)),
        AxisSpec("x", "y", (0.0, 3.0), (0.0, 2.0)),
        Scale("linear", 0.0, 1.0),
        "signal",
        marks=(ring,),
    )
    on_image = mpl.render(image).parts["iwa"]
    assert on_image.get_facecolor()[3] == 0.0
    assert on_image.get_edgecolor()[3] == 1.0
    on_track = mpl.render(_track()).parts["iwa"]
    assert on_track.get_facecolor()[3] == pytest.approx(0.2)
