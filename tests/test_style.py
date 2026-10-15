"""Call-time style resolution and the zero-style fallback."""

import subprocess
import sys

import hwostyle
import pytest
from hwostyle.palettes import ROLE_FIXED_COLORS
from matplotlib.colors import to_hex

from eyepiece import _style


def test_cmap_follows_mode_switch():
    # Probability is oriented by the page (low values at the page end), so its
    # dark-mode map differs from the light one.
    hwostyle.use("dark")
    dark_cm = _style.cmap("probability")
    with hwostyle.light():
        light_cm = _style.cmap("probability")
    assert dark_cm(0.0) != light_cm(0.0)


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


def test_zero_style_with_style_sheet_cycle_follows_it():
    code = (
        "import matplotlib.pyplot as plt; plt.style.use('dark_background'); "
        "from eyepiece import _style; "
        "print(_style.mode(), _style.color(0))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    parts = out.stdout.split()
    assert parts[0] == "light"
    assert parts[1] == "#8dd3c7"


def test_zero_style_with_colorless_cycle_falls_back_to_light():
    code = (
        "import matplotlib; from cycler import cycler; "
        "matplotlib.rcParams['axes.prop_cycle'] = cycler(linestyle=['-', '--', ':']); "
        "from eyepiece import _style; "
        "print(_style.mode(), _style.color(0))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    parts = out.stdout.split()
    assert parts[0] == "light"
    assert parts[1] == _style._light_palette()[0]


def test_zero_style_with_empty_color_list_falls_back_to_light():
    code = (
        "import matplotlib; from cycler import cycler; "
        "matplotlib.rcParams['axes.prop_cycle'] = cycler(color=[]); "
        "from eyepiece import _style; "
        "print(_style.mode(), _style.color(0))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    parts = out.stdout.split()
    assert parts[0] == "light"
    assert parts[1] == _style._light_palette()[0]


def test_public_neutral_endpoints():
    import matplotlib
    from matplotlib.colors import to_rgb

    import eyepiece as ep

    for mode in ("light", "dark"):
        hwostyle.use(mode)
        assert ep.neutral(0.0) == to_rgb(matplotlib.rcParams["axes.facecolor"])
        assert ep.neutral(1.0) == to_rgb(matplotlib.rcParams["text.color"])


def test_public_backing_tracks_background_and_opacity():
    import matplotlib
    from matplotlib.colors import to_rgba

    import eyepiece as ep

    for mode in ("light", "dark"):
        hwostyle.use(mode)
        box = ep.backing(alpha=0.4)
        assert box["facecolor"] == to_rgba(matplotlib.rcParams["axes.facecolor"], 0.4)
        assert box["edgecolor"] == "none"
        assert box is not ep.backing(alpha=0.4)


def test_role_equals_hwostyle_roles_under_each_mode():
    for mode in ("dark", "light"):
        hwostyle.use(mode)
        for name in hwostyle.roles.names:
            assert _style.role(name).lower() == hwostyle.roles[name].lower()


def test_role_without_mode_uses_light_roles():
    code = (
        "from hwostyle.palettes import Roles\n"
        "from eyepiece import _style\n"
        "assert _style.role('alert').lower() == Roles('light').alert.lower()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_role_override_wins_and_unknown_raises():
    import pytest

    assert _style.role("alert", override="#123456") == "#123456"
    with pytest.raises(KeyError, match="No role"):
        _style.role("nonsense")


def test_role_falls_back_to_palette_slot_when_missing(monkeypatch):
    class Missing:
        def __getitem__(self, name):
            raise KeyError(name)

    hwostyle.use("dark")
    monkeypatch.setattr(hwostyle, "roles", Missing(), raising=False)
    assert _style.role("alert") == _style.color(_style._ROLE_FALLBACKS["alert"])
    assert _style.role("model") == _style.color(1)
    assert _style.role("prior") == _style.color(5)
    assert _style.role("positive") == "#1565c0"


@pytest.mark.parametrize("name", ["positive", "negative", "posterior", "truth"])
def test_fixed_role_fallbacks_match_hwostyle_light(name):
    expected = to_hex(ROLE_FIXED_COLORS[name]["light"])
    assert to_hex(_style._ROLE_FALLBACKS[name]) == expected
