"""Keep WAV regression coverage active without optional video executables."""

import subprocess
import sys
from pathlib import Path


def test_wav_regression_runs_without_video_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    test = Path(__file__).with_name("test_cues_sweep_prototype.py")
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{test}::test_silent_audio_writes_pcm_under_a_new_nested_path",
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "1 passed" in done.stdout, done.stdout + done.stderr
