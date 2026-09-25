"""Shared display mapping: normalization, fixed bounds, and RGBA lookup."""

import numpy as np
import pytest

from eyepiece.prepared import (
    Scale,
    map_rgba,
    normalize_values,
    resolve_bounds,
    weight_opacity,
)
from eyepiece.style import snapshot_profile

# --- normalize_values --------------------------------------------------------


def test_signed_normalization_is_not_log_clipping():
    values = np.array([-2.0, 0.0, 2.0, np.nan])
    result = normalize_values(
        values, valid=None, scale=Scale("symmetric", -2, 2, "residual")
    )
    np.testing.assert_allclose(result[:3], [0, 0.5, 1])
    assert result.mask[3]
    assert values[0] == -2


def test_linear_normalization_maps_endpoints():
    values = np.array([0.0, 5.0, 10.0])
    result = normalize_values(values, valid=None, scale=Scale("linear", 0.0, 10.0))
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0])
    assert not np.ma.getmaskarray(result).any()


def test_explicit_invalid_is_masked_even_when_finite():
    values = np.array([1.0, 2.0, 3.0])
    valid = np.array([True, False, True])
    result = normalize_values(values, valid=valid, scale=Scale("linear", 0.0, 4.0))
    assert result.mask.tolist() == [False, True, False]


def test_out_of_range_valid_value_clamps_to_display_edge():
    values = np.array([-5.0, 15.0])
    result = normalize_values(values, valid=None, scale=Scale("linear", 0.0, 10.0))
    np.testing.assert_allclose(result, [0.0, 1.0])
    assert not result.mask.any()


def test_log_valid_zero_is_clipped_not_masked():
    """A valid zero on a log scale renders at the floor; it is not masked.

    This is the distinction the brief calls out: clipping (a display
    choice for an in-range-but-sub-floor valid sample) is not the same
    thing as masking (an invalid or nonfinite sample).
    """
    values = np.array([0.0, np.nan])
    scale = Scale("log", 0.01, 1.0, floor=0.01)
    result = normalize_values(values, valid=None, scale=scale)
    assert not result.mask[0]
    assert result.mask[1]
    # Zero clips to the floor, i.e. the bottom of the display range.
    assert result[0] == pytest.approx(0.0)


def test_log_normalization_maps_in_log_space():
    values = np.array([0.01, 0.1, 1.0])
    scale = Scale("log", 0.01, 1.0, floor=0.01)
    result = normalize_values(values, valid=None, scale=scale)
    np.testing.assert_allclose(result, [0.0, 0.5, 1.0], atol=1e-12)


def test_log_negative_value_clips_to_floor_display():
    values = np.array([-3.0])
    scale = Scale("log", 0.01, 1.0, floor=0.01)
    result = normalize_values(values, valid=None, scale=scale)
    assert not result.mask[0]
    assert result[0] == pytest.approx(0.0)


# --- normalize_values / map_rgba reject an incoherent Scale before mapping ---


def test_normalize_values_rejects_unknown_kind():
    with pytest.raises(ValueError, match="weird"):
        normalize_values(np.array([1.0]), valid=None, scale=Scale("weird", 0.0, 1.0))


def test_normalize_values_rejects_nonfinite_bounds():
    with pytest.raises(ValueError, match="linear"):
        normalize_values(
            np.array([1.0]), valid=None, scale=Scale("linear", 0.0, float("nan"))
        )


def test_normalize_values_rejects_degenerate_bounds():
    with pytest.raises(ValueError, match="linear"):
        normalize_values(np.array([1.0]), valid=None, scale=Scale("linear", 1.0, 1.0))


def test_normalize_values_rejects_symmetric_bounds_not_centered_on_zero():
    with pytest.raises(ValueError, match="symmetric"):
        normalize_values(
            np.array([1.0]), valid=None, scale=Scale("symmetric", 0.0, 5.0)
        )


def test_normalize_values_rejects_log_scale_with_no_floor():
    """The gap this fix closes: this used to raise a bare TypeError."""
    with pytest.raises(ValueError, match="log"):
        normalize_values(np.array([1.0]), valid=None, scale=Scale("log", 1.0, 10.0))


def test_normalize_values_rejects_log_scale_with_nonpositive_floor():
    with pytest.raises(ValueError, match="log"):
        normalize_values(
            np.array([1.0]), valid=None, scale=Scale("log", 0.1, 10.0, floor=-1.0)
        )


def test_normalize_values_rejects_log_scale_with_nonfinite_floor():
    with pytest.raises(ValueError, match="log"):
        normalize_values(
            np.array([1.0]),
            valid=None,
            scale=Scale("log", 0.1, 10.0, floor=float("nan")),
        )


def test_normalize_values_rejects_log_scale_with_nonpositive_vmin():
    """The other gap this fix closes: this used to raise a math domain error."""
    with pytest.raises(ValueError, match="log"):
        normalize_values(
            np.array([1.0]), valid=None, scale=Scale("log", -1.0, 10.0, floor=1.0)
        )


def test_map_rgba_rejects_log_scale_with_nonpositive_vmin():
    profile = snapshot_profile()
    with pytest.raises(ValueError, match="log"):
        map_rgba(
            np.array([1.0]),
            valid=None,
            scale=Scale("log", -1.0, 10.0, floor=1.0),
            profile=profile,
        )


def test_normalize_values_never_mutates_input():
    values = np.array([-2.0, 0.0, 2.0, np.nan])
    before = values.copy()
    normalize_values(values, valid=None, scale=Scale("symmetric", -2, 2))
    np.testing.assert_array_equal(values, before, err_msg="nan-safe compare")
    assert np.isnan(values[3])


# --- resolve_bounds -----------------------------------------------------------


def test_resolve_bounds_scans_frame_by_frame_without_concatenation():
    frames = [
        (np.array([1.0, 2.0]), None),
        (np.array([3.0, -1.0]), None),
    ]
    assert resolve_bounds(frames, kind="linear") == (-1.0, 3.0)


def test_resolve_bounds_masked_finite_outlier_does_not_enter_bounds():
    frames = [
        (np.array([1.0, 1000.0]), np.array([True, False])),
        (np.array([2.0, 3.0]), None),
    ]
    assert resolve_bounds(frames, kind="linear") == (1.0, 3.0)


def test_resolve_bounds_one_valid_frame_among_invalid_frames():
    invalid_frame = (np.full(3, np.nan), None)
    valid_frame = (np.array([2.0, 4.0, 6.0]), np.array([True, True, False]))
    frames = [invalid_frame, valid_frame, invalid_frame]
    assert resolve_bounds(frames, kind="linear") == (2.0, 4.0)


def test_resolve_bounds_rejects_all_invalid_input():
    frames = [(np.full(2, np.nan), None), (np.zeros(2), np.array([False, False]))]
    with pytest.raises(ValueError, match="linear"):
        resolve_bounds(frames, kind="linear")


def test_resolve_bounds_symmetric_is_centered_on_zero():
    frames = [(np.array([-1.0, 3.0]), None)]
    assert resolve_bounds(frames, kind="symmetric") == (-3.0, 3.0)


def test_resolve_bounds_rejects_degenerate_symmetric_input_naming_scale():
    frames = [(np.zeros(3), None)]
    with pytest.raises(ValueError, match="symmetric"):
        resolve_bounds(frames, kind="symmetric")


def test_resolve_bounds_rejects_degenerate_linear_input():
    frames = [(np.full(3, 5.0), None)]
    with pytest.raises(ValueError, match="linear"):
        resolve_bounds(frames, kind="linear")


def test_resolve_bounds_log_ignores_nonpositive_samples():
    frames = [(np.array([-1.0, 0.0, 0.5, 2.0]), None)]
    assert resolve_bounds(frames, kind="log") == (0.5, 2.0)


def test_resolve_bounds_rejects_log_input_with_no_positive_samples():
    frames = [(np.array([-1.0, 0.0]), None)]
    with pytest.raises(ValueError, match="log"):
        resolve_bounds(frames, kind="log")


def test_resolve_bounds_given_explicit_bounds_does_not_scan():
    calls = []

    def _frames():
        calls.append(1)
        yield (np.array([1.0]), None)

    result = resolve_bounds(_frames(), kind="linear", bounds=(0.0, 10.0))
    assert result == (0.0, 10.0)
    assert calls == []


def test_resolve_bounds_rejects_bad_explicit_symmetric_bounds():
    with pytest.raises(ValueError, match="symmetric"):
        resolve_bounds([], kind="symmetric", bounds=(-1.0, 2.0))


def test_resolve_bounds_rejects_bad_explicit_log_bounds():
    with pytest.raises(ValueError, match="log"):
        resolve_bounds([], kind="log", bounds=(-1.0, 2.0))


def test_resolve_bounds_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown"):
        resolve_bounds([(np.array([1.0]), None)], kind="unknown")


# --- weight_opacity -----------------------------------------------------------


def test_weight_opacity_maps_relative_emphasis():
    weights = np.array([0.0, 5.0, 10.0])
    result = weight_opacity(weights)
    np.testing.assert_allclose(result, [0.15, 0.15 + 0.65 * 0.5, 0.80])


def test_weight_opacity_all_zero_weights_are_equal():
    weights = np.zeros(4)
    result = weight_opacity(weights)
    assert np.all(result == result[0])
    assert result[0] == pytest.approx(0.15)


def test_weight_opacity_rejects_negative_weight():
    with pytest.raises(ValueError):
        weight_opacity(np.array([1.0, -0.5]))


def test_weight_opacity_rejects_nonfinite_weight():
    with pytest.raises(ValueError):
        weight_opacity(np.array([1.0, np.nan]))


# --- map_rgba -------------------------------------------------------------


def test_map_rgba_bad_pixels_are_distinct_from_clipped_valid_pixels():
    profile = snapshot_profile()
    data = np.array([-100.0, 0.0, np.nan, 100.0])
    scale = Scale("linear", 0.0, 10.0)
    rgba = map_rgba(data, valid=None, scale=scale, profile=profile)
    assert rgba.dtype == np.uint8
    assert rgba.shape == (4, 4)
    # The out-of-range-but-valid low sample clips to the bottom of the
    # colormap; it must not read as the bad-pixel color.
    assert not np.array_equal(rgba[0], profile.bad_rgba)
    # The nonfinite sample is bad, and reads as the bad-pixel color exactly.
    np.testing.assert_array_equal(rgba[2], profile.bad_rgba)


def test_map_rgba_uses_the_scale_cmap_role():
    profile = snapshot_profile()
    data = np.array([0.0, 1.0])
    intensity = map_rgba(
        data, valid=None, scale=Scale("linear", 0.0, 1.0, "intensity"), profile=profile
    )
    residual = map_rgba(
        data, valid=None, scale=Scale("linear", 0.0, 1.0, "residual"), profile=profile
    )
    assert not np.array_equal(intensity, residual)
