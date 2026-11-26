"""Keep optional pytest tooling and Hypothesis profiles predictable."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")


@pytest.mark.parametrize(
    "available",
    [
        "hwoutils.testbudget",
        "hwoutils.internal_refs",
        "hwoutils.precision_guard",
        "no-parent",
    ],
)
@pytest.mark.parametrize("profile", [None, "nightly"])
def test_optional_plugins_and_hypothesis_profiles(available, profile):
    code = """
import importlib.util
import runpy
import sys
from hypothesis import settings

original = importlib.util.find_spec
plugins = {
    "hwoutils.testbudget": "hwoutils.testbudget.plugin",
    "hwoutils.internal_refs": "hwoutils.internal_refs",
    "hwoutils.precision_guard": "hwoutils.precision_guard",
}
def find_spec(name, *args, **kwargs):
    if name in plugins:
        if sys.argv[2] == "no-parent":
            raise ModuleNotFoundError("No module named hwoutils", name="hwoutils")
        return object() if name == sys.argv[2] else None
    return original(name, *args, **kwargs)

importlib.util.find_spec = find_spec
config = runpy.run_path(sys.argv[1])
expected = [plugins[sys.argv[2]]] if sys.argv[2] in plugins else []
assert config["pytest_plugins"] == expected
assert settings.get_profile("quick").max_examples == 25
assert settings.get_profile("nightly").max_examples == 200
assert settings.default.max_examples == int(sys.argv[3])
assert settings.default.deadline is None
"""
    env = os.environ.copy()
    env.pop("HYPOTHESIS_PROFILE", None)
    if profile is not None:
        env["HYPOTHESIS_PROFILE"] = profile
    max_examples = "200" if profile else "25"
    result = subprocess.run(
        [sys.executable, "-c", code, str(CONFTEST), available, max_examples],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_budget_guard_does_not_hide_other_missing_modules(monkeypatch):
    import runpy

    original = importlib.util.find_spec

    def broken_spec(name, *args, **kwargs):
        if name == "hwoutils.testbudget":
            raise ModuleNotFoundError("Missing backend", name="broken_backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", broken_spec)
    with pytest.raises(ModuleNotFoundError, match="Missing backend"):
        runpy.run_path(str(CONFTEST))
