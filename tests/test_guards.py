"""The firewall, the banned-import rule, and import purity."""

import pathlib
import subprocess
import sys

import pytest

import eyepiece

SRC = pathlib.Path(eyepiece.__file__).parent
FORBIDDEN_LIBS = (
    "orbix",
    "skyscapes",
    "coronagraphoto",
    "coronachrome",
    "coronalyze",
    "optixstuff",
    "physicaloptix",
    "tiptilt",
    "yippy",
    "photomancy",
    "jaxedith",
    "hwosim",
    "planit",
    "exoverses",
    "pleaserender",
    "jax",
    "xarray",
    "astropy",
)


def test_no_simulation_library_in_import_graph():
    code = (
        "import sys; import eyepiece; "
        "banned = sorted({m for m in sys.modules if m.split('.')[0] in "
        + repr(FORBIDDEN_LIBS)
        + "}); print(','.join(banned))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    banned = out.stdout.strip()
    assert not banned, f"import eyepiece pulled in banned modules: {banned}"


def test_no_frozen_hwostyle_imports():
    for py in SRC.rglob("*.py"):
        text = py.read_text()
        assert "from hwostyle import" not in text, py


def test_import_has_no_rcparams_side_effect():
    code = (
        "import matplotlib; before = dict(matplotlib.rcParams); "
        "import eyepiece; after = dict(matplotlib.rcParams); "
        "print(before == after)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "True"


def test_prepared_import_avoids_matplotlib_and_manim():
    """`import eyepiece.prepared` must not load pyplot, manim, or simulation libs.

    A meta-path finder blocks those modules before the import, so any of
    them being pulled in raises ImportError instead of silently
    succeeding through an already-imported copy.
    """
    code = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        banned = ('manim', 'manim_slides', 'matplotlib.pyplot')\n"
        "        blocked_top = " + repr(FORBIDDEN_LIBS) + "\n"
        "        top = name.split('.')[0]\n"
        "        if name in banned or top in blocked_top:\n"
        "            return self\n"
        "        return None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(f'blocked: {name}')\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "import eyepiece.prepared\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


_EXPECTED_ALL = [
    "ARTIST_KEYS",
    "GLYPHS",
    "PRESETS",
    "Animation",
    "Frame",
    "MosaicResult",
    "PlotResult",
    "SourceStyles",
    "__version__",
    "animate",
    "compare_row",
    "corner",
    "corner_overlay",
    "cov_ellipse",
    "display_limits",
    "extent_arcsec",
    "extent_au",
    "extent_lod",
    "extent_lod_from_pixels",
    "fading_track",
    "file_metadata",
    "hist_vs_pdf",
    "imshow_diverging",
    "imshow_log",
    "label_arcsec",
    "label_au",
    "label_lod",
    "plot_contrast_curve",
    "plot_radial",
    "provenance_fields",
    "provenance_text",
    "radial_profile_plot",
    "rail",
    "record",
    "save_fig",
    "schematic",
    "show_field",
    "sky_fan",
    "stamp",
    "trail",
    "triptych",
]


def test_public_export_list_is_unchanged():
    """Lazy-loading __init__ must keep exactly the current public surface."""
    assert eyepiece.__all__ == _EXPECTED_ALL


def test_lazy_attribute_access_resolves_and_caches():
    """Every exported name resolves via __getattr__ and is then a plain attribute."""
    for name in eyepiece.__all__:
        value = getattr(eyepiece, name)
        assert eyepiece.__dict__[name] is value


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = eyepiece.not_a_real_export


def test_flat_namespace():
    for name in (
        "imshow_log",
        "show_field",
        "compare_row",
        "triptych",
        "save_fig",
        "record",
        "animate",
        "Animation",
        "extent_lod",
        "rail",
        "plot_radial",
        "plot_contrast_curve",
        "radial_profile_plot",
        "PlotResult",
        "ARTIST_KEYS",
        "PRESETS",
        "GLYPHS",
    ):
        assert hasattr(eyepiece, name), name
