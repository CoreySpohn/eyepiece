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
        "eyepiece.plot.imshow_log, eyepiece.render.Animation, eyepiece.mpl.render",
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


def test_submodule_attribute_access_imports_the_submodule():
    """`ep._style` works on a fresh `import eyepiece`, as scripts use it."""
    out = _blocked_import(
        "import eyepiece as ep\n"
        "assert ep._style.neutral is not None\n"
        "assert callable(ep.mpl.render)",
        (),
        ("manim", "manim_slides"),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = eyepiece.not_a_real_export


_EXPORTS_SURVIVE_SUBMODULE_IMPORTS = """
import importlib
import pkgutil
import types

import eyepiece

# Import every submodule first. Importing a submodule binds its name on the
# package, which would shadow a lazily exported name spelled the same way.
for info in pkgutil.walk_packages(eyepiece.__path__, "eyepiece."):
    try:
        importlib.import_module(info.name)
    except ImportError:
        pass  # an optional renderer (e.g. Manim) that is not installed

bad = []
for name, (module_name, attr_name) in (
    eyepiece._LAZY_EXPORTS
).items():
    value = getattr(eyepiece, name)
    expected = getattr(importlib.import_module(module_name), attr_name)
    if isinstance(value, types.ModuleType) or value is not expected:
        bad.append(name)
for name in eyepiece.__all__:
    if isinstance(getattr(eyepiece, name), types.ModuleType):
        bad.append(name)
print(",".join(sorted(set(bad))))
"""


def test_exports_survive_every_submodule_import():
    """No public name is shadowed by a same-named submodule once it is imported."""
    result = subprocess.run(
        [sys.executable, "-c", _EXPORTS_SURVIVE_SUBMODULE_IMPORTS],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_public_files_have_no_internal_references():
    root = SRC.parent.parent
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "src", "tests", "docs", "README.md"],
        cwd=root,
        text=True,
    ).split("\0")
    result = subprocess.run(
        [sys.executable, str(root / "tools/check_internal_refs.py")]
        + [str(root / name) for name in files if name and (root / name).exists()],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
