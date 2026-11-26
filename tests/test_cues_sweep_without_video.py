"""Keep WAV regression coverage active without optional video executables."""

import shutil
import subprocess
import sys
from pathlib import Path


def test_wav_regression_runs_without_video_tools(tmp_path, monkeypatch):
    # Only Git stays on PATH, for the repository precision guard.
    (tmp_path / "git").symlink_to(shutil.which("git"))
    monkeypatch.setenv("PATH", str(tmp_path))
    test = Path(__file__).with_name("test_cues_sweep_prototype.py")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            # The outer process runs the repository guard with Git on PATH.
            "-p",
            "no:hwoutils.internal_refs",
            f"{test}::test_silent_audio_writes_pcm_under_a_new_nested_path",
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # The repository guard adds its own item, so check the WAV test itself ran.
    assert " passed" in done.stdout, done.stdout + done.stderr
    assert "skipped" not in done.stdout, done.stdout + done.stderr
