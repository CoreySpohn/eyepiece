"""The video-safe dark roles: derived from dark, capped for 4:2:0, read on black."""

import itertools
import warnings

import hwostyle
import numpy as np
import pytest
from matplotlib.colors import to_rgb

import eyepiece as ep
import eyepiece.testing as ept
from eyepiece import _style, _video

BLACK = hwostyle.styles.MODE_RC["dark"]["axes.facecolor"]


def _dark_roles():
    from hwostyle.palettes import ALL_ROLE_NAMES, Roles

    roles = Roles("dark")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return {name: roles[name] for name in ALL_ROLE_NAMES}


def _limits(name):
    cap, floor = _video.RELAXED.get(name, (None, None))
    return (
        _video.CHROMA_CAP if cap is None else cap,
        _video.MIN_CONTRAST if floor is None else floor,
    )


def _hue_deg(color):
    return np.degrees(_video._lch(np.asarray(to_rgb(color)))[2])


def test_every_video_role_is_within_the_chroma_cap_and_reads_on_black():
    for name, color in ep.video_roles().items():
        cap, floor = _limits(name)
        assert _video.ycc_chroma(to_rgb(color)) <= cap, name
        assert _video.contrast(color, BLACK) >= floor, name


def test_only_the_gate_forced_roles_are_relaxed_and_never_past_aa():
    assert set(_video.RELAXED) == {"alert", "disk", "prior"}
    for cap, floor in _video.RELAXED.values():
        assert cap is None or cap <= 85.0
        assert floor is None or floor >= 4.5


def test_each_role_keeps_its_hue_and_a_role_inside_the_limits_is_unchanged():
    video = ep.video_roles()
    for name, color in _dark_roles().items():
        if _video._lch(np.asarray(to_rgb(color)))[1] < 0.02:
            assert video[name].lower() == color.lower()  # a gray stays a gray
            continue
        turn = (_hue_deg(video[name]) - _hue_deg(color) + 180.0) % 360.0 - 180.0
        assert abs(turn) < 1.0, name
        cap, floor = _limits(name)
        inside = (
            _video.ycc_chroma(to_rgb(color)) <= cap
            and _video.contrast(color, BLACK) >= floor
        )
        if inside:
            assert video[name].lower() == color.lower(), name


def test_the_dim_roles_are_lifted_and_the_pure_rgb_roles_capped():
    dark, video = _dark_roles(), ep.video_roles()
    for name in ("disk", "reference"):
        assert _video.contrast(dark[name], BLACK) < 4.5
        assert _video.contrast(video[name], BLACK) >= _limits(name)[1]
    for name in ("planet", "star", "alert"):
        assert _video.ycc_chroma(to_rgb(dark[name])) > 95.0
        assert _video.ycc_chroma(to_rgb(video[name])) <= _limits(name)[0]


def test_default_palette_is_unchanged_until_video_is_asked_for():
    hwostyle.use("dark")
    dark = _dark_roles()
    assert ep.role("planet").lower() == dark["planet"].lower()
    assert ep.use_video() is False
    assert ep.role("planet") == ep.video_roles()["planet"]
    assert ep.use_video(False) is True
    assert ep.role("planet").lower() == dark["planet"].lower()


def test_video_applies_in_dark_mode_only():
    hwostyle.use("light")
    light = ep.role("planet")
    ep.use_video()
    assert ep.role("planet") == light
    hwostyle.use("dark")
    assert ep.role("planet") == ep.video_roles()["planet"]


def test_motifs_resolve_in_the_video_mode_by_name_and_when_active():
    star = ep.MotifSet.default()["planet"]
    assert star.resolve("dark-video") == ep.video_roles()["planet"]
    hwostyle.use("dark")
    ep.use_video()
    assert star.color == ep.video_roles()["planet"]


def _findings(motifs, mode):
    return {
        (f.kind, f.keys)
        for f in ept.motif_faults(motifs, mode=mode)
        if f.severity in ("fault", "warning")
    }


def test_gate_the_video_mode_adds_no_motif_fault_or_warning():
    motifs = ep.MotifSet.default()
    assert _findings(motifs, "dark-video") <= _findings(motifs, "dark")
    hues = [k for k, m in motifs.items() if m.is_hue]
    for a, b in itertools.combinations(hues, 2):
        pair = motifs.only(a, b)
        assert _findings(pair, "dark-video") <= _findings(pair, "dark"), (a, b)


def test_active_video_mode_checks_like_the_named_mode():
    motifs = ep.MotifSet.default()
    hwostyle.use("dark")
    ep.use_video()
    assert _findings(motifs, None) == _findings(motifs, "dark-video")
    assert _style.dim_level() == pytest.approx(0.35)
