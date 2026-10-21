"""Task namespace aliases share objects without eager renderer imports."""

import importlib
import subprocess
import sys
import warnings

import pytest


@pytest.mark.parametrize(
    ("namespace", "canonical", "names"),
    [
        ("eyepiece.style", "eyepiece", ("role", "neutral", "backing")),
        (
            "eyepiece.testing",
            "eyepiece.render",
            ("frame_faults", "assert_frames_pure", "same_picture"),
        ),
    ],
)
def test_task_aliases_are_the_same_public_objects_without_warnings(
    namespace, canonical, names
):
    module = importlib.import_module(namespace)
    owner = importlib.import_module(canonical)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        for name in names:
            assert name in module.__all__
            assert name in dir(module)
            assert getattr(module, name) is getattr(owner, name)
    assert not [w for w in caught if w.category is DeprecationWarning]


@pytest.mark.parametrize("namespace", ["eyepiece.style", "eyepiece.testing"])
def test_importing_task_namespace_does_not_load_alias_render_dependencies(namespace):
    code = f"""
import importlib
import sys

blocked = (
    "eyepiece.render", "eyepiece._render_pool", "eyepiece._render_poster",
    "eyepiece._render_variants", "manim", "torch", "hwoutils",
)

class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise ImportError("eager alias dependency: " + fullname)
        return None

sys.meta_path.insert(0, Blocker())
module = importlib.import_module({namespace!r})
assert not any(name in sys.modules for name in blocked)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "ok"
