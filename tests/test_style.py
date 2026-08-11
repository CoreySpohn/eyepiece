"""Call-time style resolution and the zero-style fallback."""

import subprocess
import sys

import hwostyle

from eyepiece import _style


def test_cmap_follows_mode_switch():
    hwostyle.use("dark")
    dark_cm = _style.cmap("intensity")
    with hwostyle.light():
        light_cm = _style.cmap("intensity")
    assert dark_cm is not light_cm


def test_cmap_override_wins():
    assert _style.cmap("intensity", override="viridis").name == "viridis"


def test_color_follows_mode_switch():
    hwostyle.use("dark")
    dark_color = _style.color(0)
    with hwostyle.light():
        light_color = _style.color(0)
    assert dark_color != light_color


def test_color_override_wins():
    assert _style.color(0, override="#123456") == "#123456"


def test_color_index_wraps_past_the_palette_end():
    hwostyle.use("dark")
    assert _style.color(6) == _style.color(0)
    assert _style.color(13) == _style.color(1)


def test_neutral_follows_mode_switch():
    hwostyle.use("dark")
    dark_tone = _style.neutral(0.15)
    with hwostyle.light():
        light_tone = _style.neutral(0.15)
    assert dark_tone != light_tone


def test_zero_style_falls_back_to_light():
    code = (
        "from eyepiece import _style; "
        "print(_style.mode(), _style.cmap('intensity') is not None)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["light", "True"]
