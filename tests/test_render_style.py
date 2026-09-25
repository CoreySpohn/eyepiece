"""SourceCast slot/marker assignment and RenderProfile hwostyle snapshots."""

import hwostyle
import numpy as np
import pytest

from eyepiece.style import RenderProfile, SourceCast, snapshot_profile

# --- SourceCast ---------------------------------------------------------------


def test_source_cast_assigns_slots_in_declaration_order():
    cast = SourceCast(("b", "c"))
    assert cast.slot("b") == 0
    assert cast.slot("c") == 1


def test_source_cast_slots_follow_declaration_order_when_reversed():
    cast = SourceCast(("c", "b"))
    assert cast.slot("c") == 0
    assert cast.slot("b") == 1


def test_source_cast_marker_is_deterministic_by_slot():
    forward = SourceCast(("b", "c"))
    reversed_cast = SourceCast(("c", "b"))
    assert forward.marker("b") == reversed_cast.marker("c")
    assert forward.marker("c") == reversed_cast.marker("b")
    assert forward.marker("b") != forward.marker("c")


def test_source_cast_slot_raises_key_error_for_unknown_name():
    cast = SourceCast(("b", "c"))
    with pytest.raises(KeyError):
        cast.slot("d")


def test_source_cast_marker_raises_key_error_for_unknown_name():
    cast = SourceCast(("b", "c"))
    with pytest.raises(KeyError):
        cast.marker("d")


def test_source_cast_rejects_duplicate_names():
    with pytest.raises(ValueError, match="b"):
        SourceCast(("b", "b"))


# --- snapshot_profile -----------------------------------------------------


def test_snapshot_profile_returns_populated_fields():
    hwostyle.use("dark")
    profile = snapshot_profile()
    assert isinstance(profile, RenderProfile)
    assert len(profile.colors) > 0
    assert "intensity" in profile.colormaps
    assert "residual" in profile.colormaps
    assert profile.colormaps["intensity"].shape == (256, 4)
    assert profile.colormaps["intensity"].dtype == np.uint8
    assert profile.bad_rgba.shape == (4,)
    assert profile.text_color
    assert profile.background_color
    assert profile.reference_color
    assert profile.font_family
    assert profile.text_size_pt == 24
    assert profile.stroke_width_pt == 1.5


def test_snapshot_profile_follows_mode_switch():
    hwostyle.use("dark")
    dark_profile = snapshot_profile()
    hwostyle.use("light")
    light_profile = snapshot_profile()
    assert dark_profile.colors != light_profile.colors
    assert dark_profile.background_color != light_profile.background_color


def test_snapshot_profile_intensity_and_residual_luts_differ():
    hwostyle.use("dark")
    profile = snapshot_profile()
    assert not np.array_equal(
        profile.colormaps["intensity"], profile.colormaps["residual"]
    )


def test_snapshot_profile_bad_rgba_is_distinct_from_lut_entries():
    hwostyle.use("dark")
    profile = snapshot_profile()
    for lut in profile.colormaps.values():
        assert not np.any(np.all(lut == profile.bad_rgba, axis=-1))


def test_snapshot_profile_luts_are_read_only():
    profile = snapshot_profile()
    with pytest.raises(ValueError):
        profile.colormaps["intensity"][0] = [0, 0, 0, 0]


def test_snapshot_profile_colormaps_mapping_is_read_only():
    profile = snapshot_profile()
    with pytest.raises(TypeError):
        profile.colormaps["intensity"] = np.zeros((256, 4), dtype=np.uint8)


def test_snapshot_profile_explicit_overrides_win():
    profile = snapshot_profile(
        font_family="Comic Sans MS", text_size_pt=30, stroke_width_pt=2.0
    )
    assert profile.font_family == "Comic Sans MS"
    assert profile.text_size_pt == 30
    assert profile.stroke_width_pt == 2.0


def test_snapshot_profile_recreation_does_not_mutate_earlier_profile():
    hwostyle.use("dark")
    dark_profile = snapshot_profile()
    dark_colors = dark_profile.colors
    dark_lut = dark_profile.colormaps["intensity"].copy()

    hwostyle.use("light")
    snapshot_profile()

    assert dark_profile.colors == dark_colors
    np.testing.assert_array_equal(dark_profile.colormaps["intensity"], dark_lut)


def test_snapshotting_does_not_call_hwostyle_use_or_change_rcparams():
    import matplotlib

    hwostyle.use("dark")
    mode_before = hwostyle.current_mode()
    rc_before = dict(matplotlib.rcParams)

    snapshot_profile()

    assert hwostyle.current_mode() == mode_before
    assert dict(matplotlib.rcParams) == rc_before
