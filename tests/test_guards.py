"""The firewall, the banned-import rule, and import purity."""

import pathlib
import subprocess
import sys

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


def test_flat_namespace():
    for name in (
        "imshow_log",
        "show_field",
        "compare_row",
        "save_fig",
        "record",
        "animate",
        "extent_lod",
        "PlotResult",
    ):
        assert hasattr(eyepiece, name), name
