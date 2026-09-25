"""Pure prepared-view records: construction, validation, and tree editing."""

import numpy as np
import pytest

from eyepiece.prepared import (
    AxisSpec,
    CurveView,
    ImageView,
    Label,
    PanelGroup,
    Path,
    Points,
    ReferenceLine,
    Region,
    Scale,
    TrackView,
    find_element,
    replace_elements,
)


def _axes(x_limits=(0.0, 4.0), y_limits=(0.0, 4.0)):
    return AxisSpec("x", "y", x_limits, y_limits)


def _scale(vmin=0.0, vmax=1.0):
    return Scale("linear", vmin, vmax)


# --- R2 mask/valid fixture (verbatim from the task brief) -----------------


def test_borrowed_strided_image_preserves_mask():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    data = np.ma.array(source[::-1, ::2], mask=False, copy=False)
    data.mask[0, 0] = True
    view = ImageView(
        "image",
        data,
        AxisSpec("x", "y", (0, 3), (0, 4)),
        Scale("linear", 0, 24),
        "signal",
    )
    assert np.shares_memory(view.data, source)
    assert view.data.dtype == np.float32
    assert not view.valid[0, 0]


# --- ImageView construction, extent, and validity --------------------------


def test_image_view_has_no_separate_extent_field():
    axes = _axes((-1.0, 1.0), (-2.0, 2.0))
    view = ImageView("img", np.zeros((3, 3)), axes, _scale(), "intensity")
    assert not hasattr(view, "extent")
    assert (view.axes.x_limits, view.axes.y_limits) == ((-1.0, 1.0), (-2.0, 2.0))


def test_image_view_plain_array_has_no_valid():
    view = ImageView("img", np.zeros((2, 2)), _axes(), _scale(), "intensity")
    assert view.valid is None


def test_image_view_explicit_valid_only():
    valid = np.array([[True, False], [True, True]])
    view = ImageView(
        "img", np.zeros((2, 2)), _axes(), _scale(), "intensity", valid=valid
    )
    assert view.valid is not None
    assert not view.valid[0, 1]


def test_image_view_mask_and_explicit_valid_combine_with_and():
    data = np.ma.array(np.zeros((2, 2)), mask=False)
    data.mask[0, 0] = True
    explicit = np.array([[True, True], [False, True]])
    view = ImageView("img", data, _axes(), _scale(), "intensity", valid=explicit)
    assert view.valid.tolist() == [[False, True], [False, True]]


def test_image_view_data_shares_memory_when_read_only():
    source = np.zeros((3, 3), dtype=np.float32)
    source.setflags(write=False)
    view = ImageView("img", source, _axes(), _scale(), "intensity")
    assert np.shares_memory(view.data, source)


def test_image_view_negative_stride_preserves_orientation():
    source = np.arange(9, dtype=np.float64).reshape(3, 3)
    flipped = source[::-1]
    view = ImageView("img", flipped, _axes(), _scale(), "intensity")
    assert np.array_equal(view.data, flipped)
    assert np.shares_memory(view.data, source)


# --- Validation: names the offending element id -----------------------------


def test_image_view_rejects_nonnumeric_data():
    with pytest.raises(ValueError, match="img"):
        ImageView("img", np.array([["a", "b"]]), _axes(), _scale(), "intensity")


def test_image_view_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="img"):
        ImageView("img", np.zeros(4), _axes(), _scale(), "intensity")


def test_image_view_rejects_mismatched_valid_shape():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img",
            np.zeros((2, 2)),
            _axes(),
            _scale(),
            "intensity",
            valid=np.ones((3, 3), dtype=bool),
        )


def test_axis_spec_rejects_nonfinite_extent():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img",
            np.zeros((2, 2)),
            _axes((0.0, float("nan"))),
            _scale(),
            "intensity",
        )


def test_axis_spec_rejects_non_increasing_limits():
    with pytest.raises(ValueError, match="img"):
        ImageView("img", np.zeros((2, 2)), _axes((1.0, 1.0)), _scale(), "intensity")


def test_scale_rejects_unknown_kind():
    with pytest.raises(ValueError, match="img"):
        ImageView("img", np.zeros((2, 2)), _axes(), Scale("weird", 0, 1), "intensity")


def test_scale_rejects_degenerate_bounds():
    with pytest.raises(ValueError, match="img"):
        ImageView("img", np.zeros((2, 2)), _axes(), Scale("linear", 1, 1), "intensity")


def test_scale_log_requires_positive_floor():
    with pytest.raises(ValueError, match="img"):
        ImageView("img", np.zeros((2, 2)), _axes(), Scale("log", 0.1, 1.0), "intensity")


def test_scale_log_accepts_positive_floor():
    view = ImageView(
        "img",
        np.zeros((2, 2)),
        _axes(),
        Scale("log", 0.1, 1.0, floor=0.01),
        "intensity",
    )
    assert view.scale.floor == 0.01


def test_scale_log_rejects_nonpositive_vmin():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img", np.zeros((2, 2)), _axes(), Scale("log", -1, 10, floor=1), "intensity"
        )


def test_scale_log_rejects_nonpositive_floor():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img",
            np.zeros((2, 2)),
            _axes(),
            Scale("log", 0.1, 1.0, floor=-1),
            "intensity",
        )


def test_scale_log_rejects_nonfinite_floor():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img",
            np.zeros((2, 2)),
            _axes(),
            Scale("log", 0.1, 1.0, floor=float("nan")),
            "intensity",
        )


def test_scale_rejects_symmetric_bounds_not_centered_on_zero():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img", np.zeros((2, 2)), _axes(), Scale("symmetric", 0, 5), "intensity"
        )


def test_scale_rejects_symmetric_zero_bounds():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img", np.zeros((2, 2)), _axes(), Scale("symmetric", 0, 0), "intensity"
        )


# --- Marks: Path, Points, Region, ReferenceLine, Label ----------------------


def test_path_rejects_malformed_coordinates():
    with pytest.raises(ValueError, match="p1"):
        Path("p1", np.zeros((4, 3)))


def test_path_rejects_nonnumeric_coordinates():
    with pytest.raises(ValueError, match="p1"):
        Path("p1", np.array([["a", "b"], ["c", "d"]]))


def test_path_rejects_nonfinite_coordinates():
    with pytest.raises(ValueError, match="p1"):
        Path("p1", np.array([[0.0, 0.0], [float("nan"), 1.0]]))


def test_points_rejects_malformed_coordinates():
    with pytest.raises(ValueError, match="pts"):
        Points("pts", np.zeros(3))


def test_region_rejects_malformed_center():
    with pytest.raises(ValueError, match="iwa"):
        Region("iwa", np.zeros(3), outer_radius=1.0)


def test_label_rejects_nonfinite_xy():
    with pytest.raises(ValueError, match="lbl"):
        Label("lbl", "companion", (float("nan"), 0.0))


def test_path_preserves_zero_length_visible_interval():
    path = Path("p1", np.zeros((5, 2)), visible=(2, 2))
    assert path.visible == (2, 2)


def test_path_rejects_out_of_range_visible_stop():
    with pytest.raises(ValueError, match="p1"):
        Path("p1", np.zeros((3, 2)), visible=(0, 10))


def test_path_rejects_out_of_range_visible_start_when_stop_is_none():
    with pytest.raises(ValueError, match="p1"):
        Path("p1", np.zeros((3, 2)), visible=(999, None))


def test_path_allows_visible_start_at_vertex_count_when_stop_is_none():
    path = Path("p1", np.zeros((3, 2)), visible=(3, None))
    assert path.visible == (3, None)


def test_path_rejects_negative_weight():
    with pytest.raises(ValueError, match="p1"):
        Path("p1", np.zeros((3, 2)), weight=-1.0)


def test_points_rejects_negative_errors():
    with pytest.raises(ValueError, match="pts"):
        Points("pts", np.zeros((3, 2)), yerr=np.array([1.0, -1.0, 0.5]))


def test_points_rejects_mismatched_error_length():
    with pytest.raises(ValueError, match="pts"):
        Points("pts", np.zeros((3, 2)), xerr=np.array([1.0, 1.0]))


def test_region_rejects_invalid_radii():
    with pytest.raises(ValueError, match="iwa"):
        Region("iwa", np.array([0.0, 0.0]), outer_radius=1.0, inner_radius=1.0)


def test_region_accepts_annulus():
    region = Region("owa", np.array([0.0, 0.0]), outer_radius=2.0, inner_radius=1.0)
    assert region.inner_radius == 1.0


def test_region_rejects_negative_outer_radius():
    with pytest.raises(ValueError, match="iwa"):
        Region("iwa", np.array([0.0, 0.0]), outer_radius=-1.0)


def test_reference_line_rejects_unknown_axis():
    with pytest.raises(ValueError, match="ref"):
        ReferenceLine("ref", axis="z", value=1.0)


def test_reference_line_rejects_nonfinite_value():
    with pytest.raises(ValueError, match="ref"):
        ReferenceLine("ref", axis="x", value=float("inf"))


def test_label_rejects_unknown_space():
    with pytest.raises(ValueError, match="lbl"):
        Label("lbl", "companion", (1.0, 2.0), space="somewhere")


# --- Views embedding marks; unknown mark/view kinds -------------------------


def test_image_view_rejects_unknown_mark_kind():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img",
            np.zeros((2, 2)),
            _axes(),
            _scale(),
            "intensity",
            marks=("not-a-mark",),
        )


def test_curve_view_accepts_typed_marks():
    marks = (Path("trace", np.zeros((3, 2))), Label("lbl", "peak", (0.0, 0.0)))
    view = CurveView("curve", _axes(), marks=marks)
    assert view.marks == marks


def test_track_view_composes_same_mark_vocabulary():
    marks = (Points("obs", np.zeros((2, 2))),)
    view = TrackView("track", _axes(), marks=marks)
    assert view.marks == marks


def test_panel_group_rejects_unknown_view_kind():
    with pytest.raises(ValueError, match="panel"):
        PanelGroup("panel", views=("not-a-view",))


def test_panel_group_rejects_unknown_direction():
    view = CurveView("curve", _axes())
    with pytest.raises(ValueError, match="panel"):
        PanelGroup("panel", views=(view,), direction="diagonal")


# --- Duplicate IDs are rejected, globally within one tree -------------------


def test_image_view_rejects_duplicate_id_between_view_and_mark():
    with pytest.raises(ValueError, match="img"):
        ImageView(
            "img",
            np.zeros((2, 2)),
            _axes(),
            _scale(),
            "intensity",
            marks=(Label("img", "dup", (0.0, 0.0)),),
        )


def test_panel_group_rejects_duplicate_ids_across_sibling_views():
    left = CurveView("shared", _axes())
    right = TrackView("shared", _axes())
    with pytest.raises(ValueError, match="shared"):
        PanelGroup("panel", views=(left, right))


def test_panel_group_rejects_duplicate_ids_in_nested_group():
    inner = PanelGroup("inner", views=(CurveView("a", _axes()),))
    outer_dup = PanelGroup("dup_child", views=(TrackView("a", _axes()),))
    with pytest.raises(ValueError, match="a"):
        PanelGroup("outer", views=(inner, outer_dup))


# --- find_element ------------------------------------------------------------


def test_find_element_locates_nested_mark():
    mark = Label("lbl", "companion", (0.0, 0.0))
    view = ImageView(
        "img", np.zeros((2, 2)), _axes(), _scale(), "intensity", marks=(mark,)
    )
    group = PanelGroup("panel", views=(view,))
    assert find_element(group, "lbl") is mark
    assert find_element(group, "img") is view


def test_find_element_raises_key_error_for_unknown_id():
    view = CurveView("curve", _axes())
    with pytest.raises(KeyError):
        find_element(view, "missing")


# --- replace_elements: validate-before-build, sharing unchanged arrays ------


def test_replace_elements_swaps_mark_and_shares_other_arrays():
    data = np.zeros((2, 2))
    mark = Label("lbl", "before", (0.0, 0.0))
    view = ImageView("img", data, _axes(), _scale(), "intensity", marks=(mark,))
    new_mark = Label("lbl", "after", (0.0, 0.0))
    updated = replace_elements(view, {"lbl": new_mark})
    assert updated.marks[0] is new_mark
    assert updated.data is view.data
    assert updated.axes is view.axes


def test_replace_elements_only_rebuilds_the_changed_branch():
    changed_view = CurveView("a", _axes())
    other_view = TrackView("b", _axes())
    group = PanelGroup("panel", views=(changed_view, other_view))
    replacement = CurveView("a", _axes((0.0, 9.0), (0.0, 9.0)))
    updated = replace_elements(group, {"a": replacement})
    assert updated.views[0] is replacement
    assert updated.views[1] is other_view


def test_replace_elements_rejects_unknown_id():
    view = CurveView("curve", _axes())
    with pytest.raises(KeyError):
        replace_elements(view, {"missing": CurveView("missing", _axes())})


def test_replace_elements_rejects_mismatched_kind():
    view = CurveView("curve", _axes())
    with pytest.raises(TypeError, match="curve"):
        replace_elements(view, {"curve": TrackView("curve", _axes())})


def test_replace_elements_validates_all_before_building_any():
    mark = Label("lbl", "before", (0.0, 0.0))
    view = ImageView(
        "img", np.zeros((2, 2)), _axes(), _scale(), "intensity", marks=(mark,)
    )
    with pytest.raises(TypeError):
        replace_elements(
            view,
            {
                "lbl": Label("lbl", "after", (0.0, 0.0)),
                "img": TrackView("img", _axes()),
            },
        )
    # the valid replacement must not have been applied
    assert view.marks[0] is mark
