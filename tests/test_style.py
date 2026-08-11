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
        "print(_style.mode(), _style.cmap('intensity') is not None, _style.color(0))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    parts = out.stdout.split()
    assert parts[:2] == ["light", "True"]
    matplotlib_defaults = {
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    }
    assert parts[2] not in matplotlib_defaults


def test_zero_style_with_customized_cycle_follows_it():
    code = (
        "import matplotlib; from cycler import cycler; "
        "matplotlib.rcParams['axes.prop_cycle'] = "
        "cycler(color=['#ff0000', '#00ff00', '#0000ff']); "
        "from eyepiece import _style; "
        "print(_style.mode(), _style.color(0), _style.color(1), _style.color(3))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["light", "#ff0000", "#00ff00", "#ff0000"]
