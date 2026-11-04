"""Keep optional budget tooling and Hypothesis profiles predictable."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")


@pytest.mark.parametrize("budget", ["present", "absent", "no-parent"])
@pytest.mark.parametrize("profile", [None, "nightly"])
def test_optional_budget_plugin_and_hypothesis_profiles(budget, profile):
    code = """
import importlib.util
import runpy
import sys
from hypothesis import settings

original = importlib.util.find_spec
def find_spec(name, *args, **kwargs):
    if name == "hwoutils.testbudget":
        if sys.argv[2] == "no-parent":
            raise ModuleNotFoundError("No module named hwoutils", name="hwoutils")
        return object() if sys.argv[2] == "present" else None
    return original(name, *args, **kwargs)

importlib.util.find_spec = find_spec
config = runpy.run_path(sys.argv[1])
expected = ["hwoutils.testbudget.plugin"] if sys.argv[2] == "present" else []
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
    result = subprocess.run(
        [sys.executable, "-c", code, str(CONFTEST), budget, "200" if profile else "25"],
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
