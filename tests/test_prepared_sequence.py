"""Replayable sequence states and the shared physical-time evaluator."""

import numpy as np
import pytest

from eyepiece.prepared import (
    ArrayChannel,
    AxisSpec,
    Clock,
    ImageView,
    Label,
    PanelGroup,
    Path,
    PathWindow,
    Points,
    Scale,
    Sequence,
    TrackView,
    find_element,
)


def _axes(x_limits=(0.0, 4.0), y_limits=(0.0, 4.0)):
    return AxisSpec("x", "y", x_limits, y_limits)


def _scale(vmin=0.0, vmax=1.0):
    return Scale("linear", vmin, vmax)


# --- Task-3-brief fixture (image_sequence, in conftest.py) -----------------


def test_hold_uses_acquisition_time(image_sequence):
    sample = image_sequence().at(5.0)
    assert sample.index == 1
    assert sample.acquisition_time == 1.0
    assert sample.physical_time == 5.0
    assert sample.view.data.mean() == 10.0


# --- Left sample-and-hold seeking --------------------------------------------


def test_seeking_before_after_and_between_samples(image_sequence):
    sequence = image_sequence()
    queries = [0.0, 7.0, 2.0, 10.0, 0.0]
    expected_index = [0, 1, 1, 2, 0]
    expected_acquisition = [0.0, 1.0, 1.0, 10.0, 0.0]
    for query, index, acquisition in zip(
        queries, expected_index, expected_acquisition, strict=True
    ):
        sample = sequence.at(query)
        assert sample.index == index
        assert sample.acquisition_time == acquisition
        assert sample.physical_time == query


def test_seek_clamps_out_of_range_but_keeps_raw_physical_time(image_sequence):
    sequence = image_sequence()
    early = sequence.at(-5.0)
    assert early.index == 0
    assert early.acquisition_time == 0.0
    assert early.physical_time == -5.0

    late = sequence.at(1e6)
    assert late.index == 2
    assert late.acquisition_time == 10.0
    assert late.physical_time == 1e6


def test_seek_rejects_nan(image_sequence):
    with pytest.raises(ValueError, match="physical_time"):
        image_sequence().at(float("nan"))


def test_seek_rejects_inf(image_sequence):
    with pytest.raises(ValueError, match="physical_time"):
        image_sequence().at(float("inf"))


def test_frame_matches_at_for_exact_sample_time(image_sequence):
    sequence = image_sequence()
    direct = sequence.frame(1)
    via_at = sequence.at(1.0).view
    assert np.array_equal(direct.data, via_at.data)
    assert direct.scale is via_at.scale


def test_frame_index_out_of_range_raises(image_sequence):
    sequence = image_sequence()
    with pytest.raises(IndexError):
        sequence.frame(3)
    with pytest.raises(IndexError):
        sequence.frame(-1)


def test_frame_index_must_be_integer(image_sequence):
    with pytest.raises(TypeError):
        image_sequence().frame(1.5)


# --- Leading-N arrays are borrowed, unrelated elements share identity -------


def test_leading_axis_image_arrays_are_borrowed_not_copied():
    cube = np.array([np.full((2, 3), v, dtype=np.float32) for v in (0.0, 10.0, 20.0)])
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 3), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
    )
    sequence = Sequence(
        view, np.array([0.0, 1.0, 10.0]), "s", (ArrayChannel("image", "data", cube),)
    )
    frame = sequence.frame(2)
    assert np.shares_memory(frame.data, cube)
    assert frame.data.dtype == np.float32


def test_unrelated_elements_share_identity_across_frames():
    region = Path("trace_of_nothing", np.zeros((3, 2)))
    view = ImageView(
        "image",
        np.zeros((2, 3)),
        AxisSpec("x", "y", (0, 3), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
        marks=(region,),
    )
    cube = np.array([np.full((2, 3), v) for v in (0.0, 10.0, 20.0)])
    sequence = Sequence(
        view, np.array([0.0, 1.0, 10.0]), "s", (ArrayChannel("image", "data", cube),)
    )
    frame0 = sequence.frame(0)
    frame1 = sequence.frame(1)
    mark0 = find_element(frame0, "trace_of_nothing")
    mark1 = find_element(frame1, "trace_of_nothing")
    assert mark0 is mark1
    assert mark0 is region


# --- Times validation: no sorting, no silent repair -------------------------


def test_empty_times_rejected():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError):
        Sequence(view, np.array([]), "s")


def test_nonfinite_times_rejected():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError):
        Sequence(view, np.array([0.0, float("nan"), 1.0]), "s")


def test_duplicate_timestamps_rejected():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError, match="increasing"):
        Sequence(view, np.array([0.0, 0.0, 1.0]), "s")


def test_decreasing_timestamps_rejected_without_sorting():
    """[0, 2, 1] would become valid if sorted; it must be rejected as given."""
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError, match="increasing"):
        Sequence(view, np.array([0.0, 2.0, 1.0]), "s")


# --- sample_kind ---------------------------------------------------------


def test_unsupported_sample_kind_names_it():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(
        ValueError, match="exposure-integrated sample_kind is not supported"
    ):
        Sequence(view, np.array([0.0, 1.0]), "s", sample_kind="exposure-integrated")


# --- ArrayChannel validation -------------------------------------------------


def test_array_channel_length_mismatch_rejected():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    cube = np.zeros((2, 2, 3))
    with pytest.raises(ValueError, match="image"):
        Sequence(
            view,
            np.array([0.0, 1.0, 2.0]),
            "s",
            (ArrayChannel("image", "data", cube),),
        )


def test_array_channel_rejects_unknown_field():
    with pytest.raises(ValueError, match="banana"):
        ArrayChannel("image", "banana", np.zeros((2, 2, 3)))


def test_array_channel_xy_field_requires_points_target():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError, match="image"):
        Sequence(
            view,
            np.array([0.0, 1.0]),
            "s",
            (ArrayChannel("image", "xy", np.zeros((2, 1, 2))),),
        )


def test_array_channel_data_field_requires_image_target():
    points = Points("pts", np.zeros((1, 2)))
    view = TrackView("track", _axes(), marks=(points,))
    with pytest.raises(ValueError, match="pts"):
        Sequence(
            view,
            np.array([0.0, 1.0]),
            "s",
            (ArrayChannel("pts", "data", np.zeros((2, 2, 3))),),
        )


def test_array_channel_points_xy_varies_per_frame():
    head = Points("head", np.zeros((1, 2)))
    view = TrackView("track", _axes(), marks=(head,))
    xy = np.array([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]])
    sequence = Sequence(
        view, np.array([0.0, 1.0, 2.0]), "s", (ArrayChannel("head", "xy", xy),)
    )
    updated = find_element(sequence.frame(2), "head")
    np.testing.assert_allclose(updated.xy, [[2.0, 2.0]])


def test_array_channel_points_xy_carries_gap_frames():
    head = Points("head", np.zeros((1, 2)))
    view = TrackView("track", _axes(), marks=(head,))
    xy = np.array([[[0.0, 0.0]], [[np.nan, np.nan]], [[2.0, 2.0]]])
    sequence = Sequence(
        view, np.array([0.0, 1.0, 2.0]), "s", (ArrayChannel("head", "xy", xy),)
    )
    assert np.all(np.isnan(find_element(sequence.frame(1), "head").xy))
    np.testing.assert_allclose(find_element(sequence.frame(2), "head").xy, [[2, 2]])


def test_image_mask_time_synchronization():
    """Each frame's validity mirrors that frame's own captured mask."""
    plain = np.array([np.full((2, 2), v) for v in (0.0, 10.0, 20.0)])
    cube = np.ma.array(plain, mask=False)
    cube.mask[0, 0, 0] = True
    cube.mask[2, 1, 1] = True
    view = ImageView(
        "image",
        plain[0],
        AxisSpec("x", "y", (0, 2), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
    )
    sequence = Sequence(
        view, np.array([0.0, 1.0, 2.0]), "s", (ArrayChannel("image", "data", cube),)
    )
    frame0 = sequence.frame(0)
    frame1 = sequence.frame(1)
    frame2 = sequence.frame(2)
    assert not frame0.valid[0, 0]
    assert frame1.valid is None or bool(frame1.valid.all())
    assert not frame2.valid[1, 1]


def test_array_channel_valid_field_combines_with_captured_mask():
    plain = np.array([np.full((2, 2), v) for v in (0.0, 10.0)])
    cube = np.ma.array(plain, mask=False)
    cube.mask[0, 0, 0] = True
    valid_cube = np.ones((2, 2, 2), dtype=bool)
    valid_cube[1, 1, 1] = False
    view = ImageView(
        "image",
        plain[0],
        AxisSpec("x", "y", (0, 2), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
    )
    sequence = Sequence(
        view,
        np.array([0.0, 1.0]),
        "s",
        (
            ArrayChannel("image", "data", cube),
            ArrayChannel("image", "valid", valid_cube),
        ),
    )
    frame1 = sequence.frame(1)
    assert not frame1.valid[1, 1]
    assert frame1.valid[0, 0]


# --- PathWindow ---------------------------------------------------------


def _path_track(history=None, n=5):
    path = Path("trail", np.arange(n * 2, dtype=float).reshape(n, 2))
    view = TrackView("track", _axes(), marks=(path,))
    times = np.arange(n, dtype=float)
    return Sequence(view, times, "s", (PathWindow("trail", history=history),))


def test_path_window_growing_prefix():
    sequence = _path_track(history=None, n=5)
    assert find_element(sequence.frame(0), "trail").visible == (0, 1)
    assert find_element(sequence.frame(4), "trail").visible == (0, 5)


def test_path_window_trailing_history():
    sequence = _path_track(history=3, n=5)
    expected = {0: (0, 1), 1: (0, 2), 2: (0, 3), 3: (1, 4), 4: (2, 5)}
    for index, visible in expected.items():
        assert find_element(sequence.frame(index), "trail").visible == visible


def test_path_window_requires_path_target():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError, match="image"):
        Sequence(view, np.array([0.0, 1.0]), "s", (PathWindow("image"),))


def test_path_window_requires_matching_vertex_count():
    path = Path("trail", np.zeros((5, 2)))
    view = TrackView("track", _axes(), marks=(path,))
    with pytest.raises(ValueError, match="trail"):
        Sequence(view, np.array([0.0, 1.0, 2.0]), "s", (PathWindow("trail"),))


# --- Clock ---------------------------------------------------------------


def test_clock_label_reports_acquisition_time():
    label = Label("clock", "", (0.0, 0.0))
    view = ImageView(
        "image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal", marks=(label,)
    )
    sequence = Sequence(view, np.array([0.0, 1.0, 10.0]), "s", (Clock("clock"),))
    assert find_element(sequence.frame(1), "clock").text == "1 s"
    assert find_element(sequence.frame(2), "clock").text == "10 s"


def test_clock_format_drives_frame_at_and_strip_labels():
    label = Label("clock", "", (0.0, 0.0))
    view = ImageView(
        "image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal", marks=(label,)
    )
    clock = Clock("clock", fmt="t = {value:.2f} {unit}")
    sequence = Sequence(view, np.array([0.0, 1.0, 10.0]), "s", (clock,))
    assert find_element(sequence.frame(1), "clock").text == "t = 1.00 s"
    assert find_element(sequence.at(5.0).view, "clock").text == "t = 1.00 s"
    strip = sequence.strip([2, 0])
    assert find_element(strip, "0/clock").text == "t = 10.00 s"
    assert find_element(strip, "1/clock").text == "t = 0.00 s"


def _labels(view):
    if isinstance(view, PanelGroup):
        return [label for child in view.views for label in _labels(child)]
    return [mark for mark in view.marks if isinstance(mark, Label)]


def test_strip_with_a_clock_keeps_one_time_label_per_slot():
    """The Clock label is the slot's time label; strip adds no second one."""
    label = Label("clock", "", (0.5, 0.5))
    view = ImageView(
        "image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal", marks=(label,)
    )
    sequence = Sequence(view, np.array([0.0, 1.0, 10.0]), "s", (Clock("clock"),))
    strip = sequence.strip([2, 0])
    for slot, text in ((0, "10 s"), (1, "0 s")):
        labels = _labels(strip.views[slot])
        assert [lab.id for lab in labels] == [f"{slot}/clock"]
        assert labels[0].text == text
        assert labels[0].xy == (0.5, 0.5)
    with pytest.raises(KeyError):
        find_element(strip, "0/time")


def test_strip_accepts_a_clock_label_named_time():
    """A Clock label with id "time" no longer collides with an added label."""
    label = Label("time", "", (0.0, 0.0))
    view = ImageView(
        "image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal", marks=(label,)
    )
    clock = Clock("time", fmt="t = {value:g} {unit}")
    sequence = Sequence(view, np.array([0.0, 1.0]), "d", (clock,))
    strip = sequence.strip([1, 0])
    assert find_element(strip, "0/time").text == "t = 1 d"
    assert find_element(strip, "1/time").text == "t = 0 d"
    assert [len(_labels(slot)) for slot in strip.views] == [1, 1]


def test_clock_default_format_is_the_compact_text():
    assert Clock("clock").fmt == "{value:g} {unit}"


@pytest.mark.parametrize("fmt", ["{time} s", "{} {unit}", "{value:q}", 5])
def test_clock_rejects_a_bad_format_naming_the_label(fmt):
    with pytest.raises(ValueError, match="clock"):
        Clock("clock", fmt=fmt)


def test_clock_requires_label_target():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    with pytest.raises(ValueError, match="image"):
        Sequence(view, np.array([0.0, 1.0]), "s", (Clock("image"),))


# --- d1: multiple channels on one ImageView compose atomically --------------
#
# `_build_frame` replaces a changed ImageView wholesale to update its
# data/valid (see the module-level comment on `_build_frame`); a Clock
# label or Points head that is a MARK of that same ImageView must still be
# updated each frame, not silently reverted to the template's frame-0
# marks by that wholesale replacement.


def _composed_image_sequence():
    cube = np.array([np.full((2, 2), v) for v in (0.0, 10.0, 20.0)])
    head_xy = np.array([[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]])
    label = Label("clock", "", (0.0, 0.0))
    head = Points("head", np.zeros((1, 2)))
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 2), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
        marks=(label, head),
    )
    channels = (
        ArrayChannel("image", "data", cube),
        ArrayChannel("head", "xy", head_xy),
        Clock("clock"),
    )
    return Sequence(view, np.array([0.0, 1.0, 2.0]), "s", channels)


def test_frame_composes_image_clock_and_head_atomically():
    sequence = _composed_image_sequence()
    for i in range(3):
        frame = sequence.frame(i)
        assert frame.data.mean() == 10.0 * i
        assert find_element(frame, "clock").text == f"{i} s"
        np.testing.assert_allclose(
            find_element(frame, "head").xy, [[float(i), float(i)]]
        )


def test_at_composes_image_clock_and_head_atomically():
    sequence = _composed_image_sequence()
    sample = sequence.at(1.0)
    assert sample.view.data.mean() == 10.0
    assert find_element(sample.view, "clock").text == "1 s"
    np.testing.assert_allclose(find_element(sample.view, "head").xy, [[1.0, 1.0]])


def test_strip_composes_image_clock_and_head_atomically():
    sequence = _composed_image_sequence()
    strip = sequence.strip([2, 0])

    assert find_element(strip, "0/image").data.mean() == 20.0
    assert find_element(strip, "0/clock").text == "2 s"
    np.testing.assert_allclose(find_element(strip, "0/head").xy, [[2.0, 2.0]])

    assert find_element(strip, "1/image").data.mean() == 0.0
    assert find_element(strip, "1/clock").text == "0 s"
    np.testing.assert_allclose(find_element(strip, "1/head").xy, [[0.0, 0.0]])


# --- schedule --------------------------------------------------------------


def test_schedule_endpoints_at_10_and_30_fps(image_sequence):
    sequence = image_sequence()
    ten = sequence.schedule(run_time=1.0, fps=10)
    thirty = sequence.schedule(run_time=1.0, fps=30)
    assert len(ten) == 10
    assert len(thirty) == 30
    assert ten[0] == 0.0
    assert ten[-1] == 10.0
    assert thirty[0] == 0.0
    assert thirty[-1] == 10.0


@pytest.mark.parametrize(("run_time", "fps", "count"), [(0.28, 25, 7), (5 / 12, 12, 5)])
def test_schedule_count_ignores_float_noise(image_sequence, run_time, fps, count):
    """0.28 * 25 is 7.000000000000001 in floating point; that is 7 frames, not 8."""
    schedule = image_sequence().schedule(run_time=run_time, fps=fps)
    assert len(schedule) == count
    assert schedule[-1] == 10.0


def test_schedule_still_rounds_a_real_fraction_up(image_sequence):
    assert len(image_sequence().schedule(run_time=0.281, fps=25)) == 8


def test_schedule_singleton_sequence_is_constant():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    sequence = Sequence(view, np.array([5.0]), "s")
    schedule = sequence.schedule(run_time=0.01, fps=1)
    assert len(schedule) >= 1
    assert np.all(schedule == 5.0)


def test_schedule_rejects_nonpositive_run_time(image_sequence):
    with pytest.raises(ValueError):
        image_sequence().schedule(run_time=0.0, fps=30)


def test_schedule_rejects_nonpositive_fps(image_sequence):
    with pytest.raises(ValueError):
        image_sequence().schedule(run_time=1.0, fps=0)


def test_schedule_rejects_nonfinite_inputs(image_sequence):
    with pytest.raises(ValueError):
        image_sequence().schedule(run_time=float("nan"), fps=30)


# --- d2: `.at` snaps a sample time perturbed by float noise -----------------


def test_schedule_times_recover_their_intended_index_in_order():
    """80 epochs (days), run_time=80/12 at fps=12: every schedule time must
    map back to its own index, in order, with no repeats or skips (the
    reported regression: 9 of 80 were skipped when a schedule time landed
    a few ULPs below the sample time it was meant to reproduce)."""
    n = 80
    times = np.linspace(0.0, 240 / 24, n)
    cube = np.zeros((n, 2, 2))
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 2), (0, 2)),
        Scale("linear", -1, 1),
        "signal",
    )
    sequence = Sequence(view, times, "d", (ArrayChannel("image", "data", cube),))
    schedule = sequence.schedule(run_time=80 / 12, fps=12)
    assert len(schedule) == n
    indices = [sequence.at(float(t)).index for t in schedule]
    assert indices == list(range(n))


def test_at_snaps_a_sample_time_perturbed_one_ulp_below(image_sequence):
    sequence = image_sequence()
    perturbed = float(np.nextafter(1.0, -np.inf))
    assert perturbed < 1.0
    sample = sequence.at(perturbed)
    assert sample.index == 1
    assert sample.acquisition_time == 1.0
    # the raw request itself is still reported unclamped/unsnapped
    assert sample.physical_time == perturbed


def test_at_does_not_snap_a_request_well_below_a_sample(image_sequence):
    sequence = image_sequence()
    sample = sequence.at(0.9)
    assert sample.index == 0
    assert sample.acquisition_time == 0.0


# --- strip -------------------------------------------------------------


def test_strip_prefixes_ids_and_shares_arrays_and_scale():
    cube = np.array([np.full((2, 3), v, dtype=np.float32) for v in (0.0, 10.0, 20.0)])
    view = ImageView(
        "image",
        cube[0],
        AxisSpec("x", "y", (0, 3), (0, 2)),
        Scale("linear", 0, 20),
        "signal",
    )
    sequence = Sequence(
        view, np.array([0.0, 1.0, 10.0]), "s", (ArrayChannel("image", "data", cube),)
    )
    strip = sequence.strip([2, 0, 2])
    assert isinstance(strip, PanelGroup)

    slot0 = find_element(strip, "0/image")
    slot1 = find_element(strip, "1/image")
    slot2 = find_element(strip, "2/image")

    assert np.shares_memory(slot0.data, cube)
    assert slot0.data.mean() == 20.0
    assert slot1.data.mean() == 0.0
    assert slot2.data.mean() == 20.0
    assert slot0.scale is view.scale
    assert slot1.scale is view.scale

    assert find_element(strip, "0/time").text == "10 s"
    assert find_element(strip, "1/time").text == "0 s"
    assert find_element(strip, "2/time").text == "10 s"
    # Without a Clock, strip adds exactly one time label per slot.
    assert [len(_labels(slot)) for slot in strip.views] == [1, 1, 1]


def test_strip_leaves_source_id_unchanged():
    path = Path("trail", np.zeros((2, 2)), source_id="planet-b")
    view = TrackView("track", _axes(), marks=(path,))
    sequence = Sequence(view, np.array([0.0, 1.0]), "s")
    strip = sequence.strip([0, 1])
    prefixed = find_element(strip, "0/trail")
    assert prefixed.source_id == "planet-b"


def test_strip_adds_label_to_first_child_when_template_is_panelgroup():
    left = ImageView("left", np.zeros((2, 2)), _axes(), _scale(0, 1), "signal")
    right = ImageView("right", np.zeros((2, 2)), _axes(), _scale(0, 1), "signal")
    template = PanelGroup("pair", views=(left, right))
    sequence = Sequence(template, np.array([0.0, 1.0]), "s")
    strip = sequence.strip([0, 1])

    slot0 = find_element(strip, "0/pair")
    assert isinstance(slot0, PanelGroup)
    time_label = find_element(strip, "0/time")
    assert time_label in slot0.views[0].marks


def test_strip_handles_repeated_and_out_of_order_indices_uniquely():
    view = ImageView("image", np.zeros((2, 3)), _axes(), _scale(0, 1), "signal")
    sequence = Sequence(view, np.array([0.0, 1.0, 2.0]), "s")
    strip = sequence.strip([2, 2, 0])
    ids = {v.id for v in strip.views}
    assert ids == {"0/image", "1/image", "2/image"}


def test_strip_labels_nested_panelgroup_slot_at_any_depth():
    """A slot whose template is a PanelGroup-of-PanelGroups (a nested grid)
    must still get exactly one added label, on the first marked view
    reached by descending through views[0] at every level -- not raise
    AttributeError from treating an inner PanelGroup as a marked view.
    """
    leaf_a = ImageView("leaf_a", np.zeros((2, 2)), _axes(), _scale(0, 1), "signal")
    leaf_b = ImageView("leaf_b", np.zeros((2, 2)), _axes(), _scale(0, 1), "signal")
    inner = PanelGroup("inner", views=(leaf_a, leaf_b))
    right = ImageView("right", np.zeros((2, 2)), _axes(), _scale(0, 1), "signal")
    template = PanelGroup("outer", views=(inner, right))
    sequence = Sequence(template, np.array([0.0, 1.0]), "s")

    strip = sequence.strip([0, 1])

    # unique prefixed IDs at every depth (find_element would already have
    # raised a construction-time ValueError above if these collided).
    for slot, acquisition in ((0, "0 s"), (1, "1 s")):
        assert find_element(strip, f"{slot}/outer") is not None
        assert find_element(strip, f"{slot}/inner") is not None
        assert find_element(strip, f"{slot}/leaf_a") is not None
        assert find_element(strip, f"{slot}/leaf_b") is not None
        assert find_element(strip, f"{slot}/right") is not None

        time_label = find_element(strip, f"{slot}/time")
        assert time_label.text == acquisition

        leaf_a_prefixed = find_element(strip, f"{slot}/leaf_a")
        leaf_b_prefixed = find_element(strip, f"{slot}/leaf_b")
        assert time_label in leaf_a_prefixed.marks
        assert time_label not in leaf_b_prefixed.marks
