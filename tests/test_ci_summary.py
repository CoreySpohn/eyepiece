"""Check CI summaries without running a test suite or a renderer."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/ci_summary.py"
SPEC = importlib.util.spec_from_file_location("ci_summary", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


@pytest.mark.parametrize(
    ("seconds", "budget", "skip", "options", "expected"),
    [
        (2, 0, "", [], 0),
        (2, 2, "", [], 0),
        (2, 1, "", [], 1),
        (2, 0, "<skipped />", [], 1),
        (2, 0, "<skipped />", ["--allow-skips"], 0),
        (2, 0, '<skipped type="pytest.xfail" />', [], 0),
    ],
)
def test_summary_budget_and_skip_policy(
    tmp_path, monkeypatch, capsys, seconds, budget, skip, options, expected
):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="example" name="test_one" '
        f'time="{seconds}">{skip}</testcase></testsuite></testsuites>'
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), str(junit), "--budget-seconds", str(budget), *options],
    )
    assert SUMMARY.main() == expected
    output = capsys.readouterr().out
    assert "tests: 1" in output
    assert "example::test_one" in output
    assert f"summed test time: {seconds:.1f} s" in output


def test_summary_appends_to_step_summary(tmp_path, monkeypatch):
    junit = tmp_path / "empty.xml"
    junit.write_text("<testsuites><testsuite /></testsuites>")
    target = tmp_path / "summary.md"
    target.write_text("Existing summary\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(junit), "--title", "Slow tier"])
    assert SUMMARY.main() == 0
    assert target.read_text().startswith("Existing summary\n### Slow tier\n")
    assert "tests: 0, skipped: 0" in target.read_text()


def test_missing_junit_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(tmp_path / "missing.xml")])
    assert SUMMARY.main() == 1
    assert "the test step did not run" in capsys.readouterr().out
