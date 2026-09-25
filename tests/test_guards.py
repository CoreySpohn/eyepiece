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


def _blocked_import(statement, banned_names, banned_tops):
    """Run `statement` in a fresh interpreter with some modules unimportable.

    A meta-path finder raises ImportError from `find_spec` for every banned
    module, so importing one fails even if it is installed, instead of
    silently succeeding. (`find_module`-only finders are ignored by
    Python 3.12's import system and would block nothing.)
    """
    code = (
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in " + repr(tuple(banned_names)) + " or (\n"
        "            name.split('.')[0] in " + repr(tuple(banned_tops)) + "\n"
        "        ):\n"
        "            raise ImportError(f'blocked: {name}')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n" + statement + "\nprint('ok')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_prepared_import_avoids_matplotlib_and_manim():
    """`import eyepiece.prepared` must not load pyplot, manim, or simulation libs."""
    out = _blocked_import(
        "import eyepiece.prepared",
        ("matplotlib.pyplot",),
        ("manim", "manim_slides", *FORBIDDEN_LIBS),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_base_and_matplotlib_imports_work_without_manim():
    """`import eyepiece` and `import eyepiece.mpl` never need Manim."""
    out = _blocked_import(
        "import eyepiece, eyepiece.mpl\n"
        "eyepiece.imshow_log, eyepiece.Animation, eyepiece.mpl.render",
        (),
        ("manim", "manim_slides"),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_import_blocker_really_blocks():
    """The guard itself: a banned import in the statement must fail."""
    out = _blocked_import("import csv", ("csv",), ())
    assert out.returncode != 0
    assert "blocked: csv" in out.stderr


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


def test_lazy_exports_table_matches_public_names():
    """_LAZY_EXPORTS must cover exactly __all__ minus the eagerly bound names.

    __version__ is bound eagerly in __init__.py (cheap, no matplotlib), so
    it is the only name __all__ carries that _LAZY_EXPORTS does not.
    """
    eager = {"__version__"}
    assert set(eyepiece._LAZY_EXPORTS) == set(eyepiece.__all__) - eager


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
